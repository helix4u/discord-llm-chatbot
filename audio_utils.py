import logging
import asyncio
import io
import re
import os # For path operations in download_youtube_video
import gc
import subprocess # For yt-dlp

import torch # For whisper
import whisper # For whisper
from pydub import AudioSegment # For fix_mp3_length
import aiohttp # For tts_request and transcribe_audio_attachment

# Import from config (ensure these are defined in your config.py)
try:
    from config import TTS_API_URL, TTS_VOICE, WHISPER_MODEL # Added WHISPER_MODEL
except ImportError:
    TTS_API_URL = None
    TTS_VOICE = None
    WHISPER_MODEL = "tiny" # Default if not in config
    logging.warning("TTS_API_URL, TTS_VOICE, or WHISPER_MODEL not found in config.py. Using defaults for some.")

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Audio Manipulation and TTS ---

def fix_mp3_length(mp3_data: bytes) -> bytes:
    """
    Re-encode MP3 in-memory via pydub to ensure correct length metadata.
    (Moved from lmcordx.py)
    """
    logger.debug("Attempting to fix MP3 length using pydub.")
    try:
        audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
        output_buffer = io.BytesIO()
        # Standard bitrate, can be made configurable if needed
        audio.export(output_buffer, format="mp3", bitrate="128k") 
        logger.debug("MP3 length fixed successfully.")
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error fixing MP3 length: {e}")
        # Return original data if fixing fails, or handle error as preferred
        return mp3_data


async def tts_request(text: str, speed: float = 1.3) -> bytes | None:
    """
    Send a TTS request to the configured TTS server and return raw MP3 bytes.
    (Moved from lmcordx.py, uses config.TTS_API_URL and config.TTS_VOICE)
    """
    if not TTS_API_URL:
        logger.error("TTS_API_URL is not configured. Cannot make TTS request.")
        return None
    if not TTS_VOICE:
        logger.warning("TTS_VOICE is not configured. TTS may use a default server voice.")

    # Clean text of characters that might interfere with TTS or Markdown
    cleaned_text = re.sub(r'[\*#]+', '', text) 
    payload = {
        "input": cleaned_text,
        "voice": TTS_VOICE,
        "response_format": "mp3", # Or other supported format
        "speed": speed, # Speed parameter, make configurable if desired
    }
    logger.debug(f"Sending TTS request to {TTS_API_URL} with voice {TTS_VOICE} for text: \"{cleaned_text[:50]}...\"")
    try:
        # Create a new session for each request or use a shared one if appropriate
        async with aiohttp.ClientSession() as session:
            async with session.post(TTS_API_URL, json=payload, timeout=30) as resp: # Added timeout
                if resp.status == 200:
                    mp3_raw = await resp.read()
                    if not mp3_raw:
                        logger.error("TTS request returned empty audio data.")
                        return None
                    # Fix MP3 length metadata (important for some players)
                    mp3_fixed = fix_mp3_length(mp3_raw) 
                    logger.info("TTS request successful and MP3 length fixed.")
                    return mp3_fixed
                else:
                    err_txt = await resp.text()
                    logger.error(f"TTS request failed: status={resp.status}, response_text={err_txt}")
                    return None
    except asyncio.TimeoutError:
        logger.error(f"TTS request to {TTS_API_URL} timed out.")
        return None
    except Exception as e:
        logger.error(f"TTS request error: {e}")
        return None

# --- Audio Transcription ---

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using Whisper.
    (Moved from lmcordx.py)
    """
    logger.info(f"Transcribing audio file: {file_path} using Whisper model: {WHISPER_MODEL}.")
    
    model = None # Ensure model is in a defined state
    transcription = ""
    try:
        model = whisper.load_model(WHISPER_MODEL) # Use WHISPER_MODEL from config
        logger.debug(f"Whisper model '{WHISPER_MODEL}' loaded.")
        result = model.transcribe(file_path, fp16=torch.cuda.is_available()) # fp16 if cuda available
        transcription = result["text"]
        logger.info(f"Transcription successful for {file_path}. Text: \"{transcription[:50]}...\"")
    except Exception as e:
        logger.error(f"Error during audio transcription for {file_path}: {e}")
        transcription = f"Error during transcription: {e}" # Return error message
    finally:
        if model:
            # Deleting model and clearing cache can be intensive if done frequently.
            # Consider if model should be loaded once globally if performance is an issue.
            del model 
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        gc.collect() 
        logger.debug("Whisper model and CUDA cache (if applicable) cleaned up.")
    return transcription

async def transcribe_audio_attachment(audio_url: str, session: aiohttp.ClientSession, temp_file_path: str = "temp_audio_attachment.mp3") -> str:
    """
    Downloads an audio attachment and transcribes it.
    (Moved from lmcordx.py, modified to accept session)
    """
    logger.info(f"Downloading audio attachment from: {audio_url}")
    try:
        async with session.get(audio_url, timeout=60) as resp: # Added timeout
            if resp.status == 200:
                audio_data = await resp.read()
                with open(temp_file_path, "wb") as audio_file:
                    audio_file.write(audio_data)
                logger.debug(f"Audio attachment downloaded to {temp_file_path}.")
                # Now transcribe the downloaded file
                transcription = transcribe_audio(temp_file_path)
                return transcription
            else:
                logger.error(f"Failed to download audio file from {audio_url}. Status: {resp.status}, Response: {await resp.text()}")
                return f"Error: Could not download audio (status {resp.status})."
    except asyncio.TimeoutError:
        logger.error(f"Timeout downloading audio attachment from {audio_url}.")
        return "Error: Timeout downloading audio."
    except Exception as e:
        logger.error(f"Error processing audio attachment from {audio_url}: {e}")
        return f"Error processing audio: {e}"
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Temporary audio file {temp_file_path} removed.")
            except OSError as e:
                logger.error(f"Error removing temporary audio file {temp_file_path}: {e}")

# --- YouTube Audio Processing ---

def download_youtube_video(video_url: str, output_dir: str = "downloaded_audio") -> str | None:
    """
    Downloads audio from a YouTube video using yt-dlp.
    (Moved from llmcord_search.py)
    Returns the path to the downloaded audio file or None on failure.
    """
    logger.info(f"Downloading audio from YouTube URL: {video_url}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    
    command = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'mp3', 
        '--audio-quality', '0', 
        '-o', output_template,
        video_url,
        '--no-playlist', # Ensure only single video is downloaded if URL is part of playlist
        '--socket-timeout', '30' # Timeout for network operations within yt-dlp
    ]
    
    logger.debug(f"Executing yt-dlp command: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120) 
        
        output_lines = process.stdout.splitlines()
        downloaded_file_path = None
        for line in output_lines: # Try to find filename from output
            # Common yt-dlp output lines indicating final file path
            destination_markers = ["[ExtractAudio] Destination: ", "[ffmpeg] Destination: ", "[download] Destination: "]
            for marker in destination_markers:
                if marker in line:
                    potential_path = line.split(marker)[1].strip()
                    if potential_path.endswith(".mp3"): # Ensure it's an mp3 file
                        downloaded_file_path = potential_path
                        break
            if downloaded_file_path:
                break
        
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            logger.info(f"Audio successfully downloaded: {downloaded_file_path}")
            return downloaded_file_path
        else: 
            logger.warning("Could not determine exact downloaded filename from yt-dlp output. Fallback: Searching directory.")
            # Fallback: find newest mp3 in the directory (less reliable)
            list_of_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp3")]
            if not list_of_files:
                logger.error("No mp3 files found in output directory after yt-dlp download.")
                return None
            latest_file = max(list_of_files, key=os.path.getctime)
            logger.info(f"Audio successfully downloaded (guessed by newest): {latest_file}")
            return latest_file

    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp command failed for {video_url}. Error: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"yt-dlp command timed out for {video_url}.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during YouTube download for {video_url}: {e}")
        return None

async def transcribe_youtube_video(video_url: str, session: aiohttp.ClientSession) -> str:
    """
    Downloads and transcribes audio from a YouTube video.
    `session` is not used by current download_youtube_video but kept for API consistency.
    """
    logger.info(f"Starting transcription process for YouTube URL: {video_url}")
    downloaded_audio_path = None
    try:
        downloaded_audio_path = await asyncio.to_thread(download_youtube_video, video_url)
        
        if downloaded_audio_path:
            logger.debug(f"Audio downloaded to {downloaded_audio_path}, proceeding with transcription.")
            transcription = await asyncio.to_thread(transcribe_audio, downloaded_audio_path)
            return transcription
        else:
            logger.error(f"Audio download failed for {video_url}, cannot transcribe.")
            return "Error: Audio download failed, cannot transcribe."
    except Exception as e:
        logger.error(f"Error in transcribe_youtube_video for {video_url}: {e}")
        return f"Error during YouTube video transcription: {e}"
    finally:
        # Decide on cleanup strategy for downloaded files.
        # If files are large or numerous, cleanup is good. For debugging, might want to keep them.
        # Example cleanup:
        # if downloaded_audio_path and os.path.exists(downloaded_audio_path):
        #     try:
        #         os.remove(downloaded_audio_path)
        #         logger.debug(f"Temporary YouTube audio file {downloaded_audio_path} removed.")
        #     except OSError as e:
        #         logger.error(f"Error removing temporary YouTube audio file {downloaded_audio_path}: {e}")
        pass # Current: keep files.
