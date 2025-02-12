import asyncio
from datetime import datetime
import logging
import subprocess
import os
import json
from bs4 import BeautifulSoup
import discord
from dotenv import load_dotenv
from openai import AsyncOpenAI
import aiohttp
import random
import re
import base64
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import torch
import gc
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
import yt_dlp
import whisper
import io  # <-- Needed for building audio attachments in-memory
from pydub import AudioSegment

# -------------------------------------------------------------------
# Helper function to re-encode MP3 data and fix length metadata
# -------------------------------------------------------------------
def fix_mp3_length(mp3_data: bytes) -> bytes:
    """
    Re-encode MP3 in-memory via pydub to ensure correct length metadata.
    """
    audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
    output_buffer = io.BytesIO()
    # Export with a standard bitrate; adjust if you like
    audio.export(output_buffer, format="mp3", bitrate="128k")
    return output_buffer.getvalue()

# -------------------------------------------------------------------
# TTS Configuration
# -------------------------------------------------------------------
TTS_API_URL = "http://localhost:8880/v1/audio/speech"  # Adjust to match your TTS server
TTS_VOICE = "af_sky+af+af_nicole"

async def tts_request(text: str, speed: float = 1.3) -> bytes:
    """
    Send a TTS request to the local Kokoro-FastAPI server and return 
    raw MP3 bytes with corrected length metadata.
    """
    cleaned_text = re.sub(r'[\*#]+', '', text)
	
    payload = {
        "input": cleaned_text,
        "voice": TTS_VOICE,
        "response_format": "mp3",
        "speed": speed,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(TTS_API_URL, json=payload) as resp:
                if resp.status == 200:
                    mp3_raw = await resp.read()
                    # Re-encode to ensure correct MP3 length headers
                    mp3_fixed = fix_mp3_length(mp3_raw)
                    return mp3_fixed
                else:
                    err_txt = await resp.text()
                    logging.error(f"TTS request failed: status={resp.status}, resp={err_txt}")
                    return None
    except Exception as e:
        logging.error(f"TTS request error: {e}")
        return None

# Load environment variables from a .env file
load_dotenv()

# Set up logging to display messages with time stamps
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Initialize the OpenAI client with the URL of your local AI server
llm_client = AsyncOpenAI(base_url=os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1"), api_key="lm-studio")

# Environment variable validation and fallback
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN environment variable is missing")

ALLOWED_CHANNEL_IDS = [int(i) for i in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if i]
ALLOWED_ROLE_IDS = [int(i) for i in os.getenv("ALLOWED_ROLE_IDS", "").split(",") if i]
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 0))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", 10))

EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green()}
EMBED_MAX_LENGTH = 4096
EDITS_PER_SECOND = 1.3
MAX_COMPLETION_TOKENS = 2048

MAX_IMAGE_WARNING = f"⚠️ Max {MAX_IMAGES} image{'' if MAX_IMAGES == 1 else 's'} per message" if MAX_IMAGES > 0 else ""
MAX_MESSAGE_WARNING = f"⚠️ Only using last {MAX_MESSAGES} messages"

# Initialize Discord client with intents to access message content and reactions
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
discord_client = discord.Client(intents=intents)

# Dictionaries to store message nodes and in-progress message IDs
msg_nodes = {}
in_progress_msg_ids = []
message_history = {}

# List of commands to ignore
IGNORE_COMMANDS = [
    "!dream", "!d", "!background", "!avatar",
    "!help", "!info", "!ping", "!status", "!upscale", "!nightmare", "!n", "!describe", "!background", "!chat", "!superprompt", "!depth", "!face", "!edges", "!lineart", "!lineartanime", "!colormap", "!pose", "!esrgan", "!metadata", "!text", "!append", "!models", "!loras", "!nightmarePromptGen", "!load", "!aspect", "!resolution", "!handfix"
]

# List of scheduled tasks
scheduled_tasks = []

# Class to represent a message node
class MsgNode:
    def __init__(self, msg, too_many_images=False, replied_to=None):
        self.msg = msg
        self.too_many_images = too_many_images
        self.replied_to = replied_to

# Function to get the system prompt
def get_system_prompt() -> list:
    if os.getenv("LLM") in ["gpt-4-vision-preview", "mistral", "local"]:
        return [
            {
                "role": "system",
                "content": (
                    f"A chat between a curious user and a hyper-intelligent assistance entity named Sam. Sam gives helpful, detailed, and rational answers to the user's questions. Sam creates an initial reaction, considers the context, makes a plan of action and a rationale and evaluates it for logical failings before responding. Provocative Endings: Each reflection concludes with a question or thought provoking further exploration. Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
                ),
            }
        ]
    return [
        {
            "role": "system",
            "content": (
                f"A chat between a curious user and a hyper-intelligent assistance entity named Sam. Sam gives helpful, detailed, and rational answers to the user's questions. Sam creates an initial reaction, considers the context, makes a plan of action and a rationale and evaluates it for logical failings before responding. Provocative Endings: Each reflection concludes with a question or thought provoking further exploration. Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
            ),
        }
    ]

# Function to generate a sarcastic response
async def generate_sarcastic_response(user_message: str) -> str:
    prompt = (
        "The user is fed up with extremist political views and wants to push back using sarcasm. You are here to make a single reply to mock these alt-right weirdos. "
        "The bot should respond to any political discussion or keyword with the most sarcastic, snarky, and troll-like comments possible. "
        "The goal is to mock and undermine these extremist views in a way that’s both biting and humorous.\n\n"
        f"User: {user_message}\nBot:"
    )
    response = await llm_client.completions.create(
        model="local-model",
        prompt=prompt,
        temperature=0.8,
        max_tokens=4096,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

# Function to send response with tts
async def send_response_with_tts(channel: discord.TextChannel, title: str, text: str, color=discord.Color.green(), tts_filename="tts_response.mp3"):
    """
    Sends an embed with the given title and text, then generates a TTS audio
    version of the text and sends it as an additional message.
    """
    # Send the text response embed.
    embed = discord.Embed(title=title, description=text, color=color)
    sent_msg = await channel.send(embed=embed)
    
    # Generate TTS audio for the text.
    tts_bytes = await tts_request(text)
    if tts_bytes:
        tts_file = discord.File(io.BytesIO(tts_bytes), filename=tts_filename)
        await channel.send(content="**Audio version:**", file=tts_file)
    
    return sent_msg

# Add this helper function near your other helper definitions
def prepend_prefix(url: str) -> str:
    prefix = "https://r.jina.ai/"
    if not url.startswith(prefix):
        return prefix + url
    return url

async def scrape_website(url: str) -> str:
    logging.info(f"Scraping website: {url}")
    # Prepend the prefix before scraping
    url = prepend_prefix(url)
    logging.info(f"Scraping website: {url}")
    
    async with async_playwright() as p:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
        browser = await p.chromium.launch(headless=True)  # Use headless=False to mimic a real user
        context = await browser.new_context(user_agent=user_agent)
        page = await context.new_page()
        
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=10000)  # Shortened timeout for faster failure
            content = await page.evaluate('document.body.innerText')

            if not content.strip():
                raise ValueError("No content found")

            cleaned_text = clean_text(content)
            await browser.close()
            return cleaned_text if cleaned_text else "Failed to scrape the website."
        
        except Exception as e:
            logging.error(f"Playwright failed: {e}")
            await browser.close()
            return await scrape_with_beautifulsoup(url, user_agent)

async def scrape_with_beautifulsoup(url: str, user_agent: str) -> str:
    logging.info(f"Using BeautifulSoup fallback for: {url}")
    headers = {
        'User-Agent': user_agent
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status != 200:
                    logging.error(f"Failed to fetch {url} with status code {response.status}")
                    return scrape_with_curl(url)
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                content = soup.get_text(separator=' ', strip=True)
                
                if not content.strip():
                    logging.error("No content found using BeautifulSoup")
                    return scrape_with_curl(url)
                
                cleaned_text = clean_text(content)
                return cleaned_text if cleaned_text else scrape_with_curl(url)
        
        except Exception as e:
            logging.error(f"An error occurred with BeautifulSoup: {e}")
            return scrape_with_curl(url)

def scrape_with_curl(url: str) -> str:
    logging.info(f"Using curl fallback for: {url}")
    try:
        result = subprocess.run(['curl', '-s', url], capture_output=True, text=True, encoding='utf-8', timeout=30)
        if result.returncode != 0:
            logging.error(f"curl failed with return code {result.returncode}")
            return "Failed to scrape the website."
        
        html = result.stdout
        soup = BeautifulSoup(html, 'html.parser')
        content = soup.get_text(separator=' ', strip=True)
        
        if not content.strip():
            logging.error("No content found using curl")
            return "Failed to scrape the website."
        
        cleaned_text = clean_text(content)
        return cleaned_text if cleaned_text else "Failed to scrape the website."
    
    except Exception as e:
        logging.error(f"An error occurred with curl: {e}")
        return "Failed to scrape the website."

# Function to detect URLs in a message using regex
def detect_urls(message_text: str) -> list:
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(message_text)

# Function to clean up the text
def clean_text(text: str) -> str:
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)

    # Remove multiple newlines and excess whitespace
    text = re.sub('\n+', '\n', text).strip()

    # Replace unnecessary characters or patterns
    patterns_to_replace = [
        (r'\s+', ' '),   # replace multiple whitespace with single space
        (r'\[.*?\]', ''),  # remove anything inside square brackets
        (r'\[\s*__\s*\]', ''),  # remove occurrences of "[ __ ]"        
        (r'NFL Sunday Ticket', ''),  # remove occurrences of "NFL Sunday Ticket"
        (r'© \d{4} Google LLC', '')  # remove occurrences of "© [year] Google LLC"
    ]

    for pattern, repl in patterns_to_replace:
        text = re.sub(pattern, repl, text)

    return text

# Function to clean YouTube transcript text
def clean_youtube_transcript(transcript: str) -> str:
    return clean_text(transcript)

# Function to chunk text into smaller parts
def chunk_text(text: str, max_length: int = 4000) -> list:
    chunks = []
    while len(text) > max_length:
        chunk = text[:max_length]
        last_space = chunk.rfind(' ')
        if last_space != -1:
            chunk = text[:last_space]
        chunks.append(chunk)
        text = text[len(chunk):]
    chunks.append(text)
    return chunks

# Function to query Searx search engine
async def query_searx(query: str) -> list:
    logging.info(f"Querying Searx for: {query}")
    searx_url = "http://192.168.1.3:9092/search"
    params = {
        'q': query,
        'format': 'json',
        'language': 'en-US',
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(searx_url, params=params, timeout=5) as response:
                if response.status == 200:
                    results = await response.json()
                    return results.get('results', [])[:5]  # Return first 5 results
                logging.error("Failed to fetch data from Searx.")
    except aiohttp.ClientError as e:
        logging.error(f"An error occurred while fetching data from Searx: {e}")
    return []

# Function to generate a completion using the OpenAI client
async def generate_completion(prompt: str) -> str:
    try:
        response = await llm_client.completions.create(
            model=os.getenv("LLM"),
            prompt=prompt,
            temperature=0.8,
            max_tokens=2048,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Failed to generate completion: {e}")
        return "Sorry, an error occurred while generating the response."

# Function to handle the search and scrape command
async def search_and_summarize(query: str, channel: discord.TextChannel):
    search_results = await query_searx(query)
    if search_results:
        for result in search_results:
            url = result.get('url', 'No URL')
            webpage_text = await scrape_website(url)
            if webpage_text:
                if webpage_text == "Failed to scrape the website.":
                    await channel.send(f"Unfortunately, scraping the website at {url} has failed. Please try another source.")
                else:
                    cleaned_content = clean_text(webpage_text)
                    prompt = f"\n[<system message>Webpage scrape to be used for summarization: {cleaned_content} Use this as search and augmentation data for summarization and link citation. Provide full links formatted for discord.</system message>]\n "
                    summary = await generate_completion(prompt)
                    chunks = chunk_text(summary)
                    for chunk in chunks:
                        embed = discord.Embed(
                            title=result.get('title', 'No title'),
                            description=chunk,
                            url=url,
                            color=discord.Color.blue()
                        )
                        await channel.send(embed=embed)
                        # Add TTS for each chunk.
                        tts_bytes = await tts_request(chunk)
                        if tts_bytes:
                            tts_file = discord.File(io.BytesIO(tts_bytes), filename="sns_tts.mp3")
                            await channel.send(content="**Audio version:**", file=tts_file)
    else:
        await channel.send("No search results found.")

# Function to fetch YouTube transcript
async def fetch_youtube_transcript(url: str) -> str:
    try:
        video_id = re.search(r'v=([^&]+)', url).group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        cleaned_transcript = clean_youtube_transcript(transcript_text)
        return cleaned_transcript
    except NoTranscriptFound:
        logging.error("No transcript found for this video.")
    except Exception as e:
        logging.error(f"Failed to fetch transcript: {e}")
    return ""

# Function to fetch YouTube video and transcribe using Whisper
def download_youtube_video(url: str, output_path: str = "youtube_audio.mp4"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'noplaylist': True,
        'http_chunk_size': 10485760,  # 10MB
        'retries': 5,
        'fragment_retries': 5,
        'geo_bypass': True,
        'nocheckcertificate': True,
        'sleep_interval': 1,
        'max_sleep_interval': 5,
        'buffersize': '16K',
        'extractor_args': {
            'youtube': {
                'player_url': 'https://www.youtube.com/s/player/8eff86d5/player_ias.vflset/en_US/base.js',
            }
        },
        'external_downloader': 'ffmpeg',
        'external_downloader_args': ['-loglevel', 'panic', '-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '5'],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        print(f"Failed to download YouTube video: {e}")
        return None

def transcribe_audio(file_path: str) -> str:
    whisper_model = whisper.load_model("tiny")
    try:
        # Perform transcription
        result = whisper_model.transcribe(file_path)
        transcription = result["text"]
    finally:
        # Unload Whisper model from VRAM
        del whisper_model
        torch.cuda.empty_cache()
        gc.collect()
    
    return transcription

async def transcribe_youtube_video(file_path: str) -> str:
    transcription = transcribe_audio(file_path)
    if not transcription:
        print("Transcription failed. Trying to convert format and retry.")
        try:
            subprocess.run(['ffmpeg', '-i', file_path, '-ar', '16000', 'converted_audio.wav'])
            transcription = transcribe_audio('converted_audio.wav')
        except Exception as e:
            print(f"Failed to convert and transcribe YouTube video: {e}")
    return transcription

async def main(url: str):
    audio_path = download_youtube_video(url)
    if not audio_path:
        print("Download failed. Exiting.")
        return

    transcription = await transcribe_youtube_video(audio_path)
    if transcription:
        print(f"Transcription: {transcription}")
    else:
        print("Transcription failed.")

# Function to handle the roast and summarize command
async def roast_and_summarize(url: str, channel: discord.TextChannel):
    webpage_text = await scrape_website(url)
    if webpage_text == "Failed to scrape the website.":
        await channel.send(f"Unfortunately, scraping the website at {url} has failed. Please try another source.")
    else:
        cleaned_content = clean_text(webpage_text)
        prompt = (
            f"\n[Webpage Scrape for Comedy Routine: {cleaned_content} Use this content to create a professional comedy routine. "
            "Make it funny, witty, and engaging. Any links provided should be full links formatted for hyperlinking.]\n"
        )
        comedy_routine = await generate_completion(prompt)
        chunks = chunk_text(comedy_routine)
        
        response_msgs = []
        response_msg_contents = []
        prev_content = None
        edit_msg_task = None
        last_msg_task_time = datetime.now().timestamp()
        in_progress_msg_ids = []
        EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green()}
        EMBED_MAX_LENGTH = 4096
        EDITS_PER_SECOND = 1.3

        reply_msg = await channel.send(embed=discord.Embed(title="Comedy Routine", description="⏳", color=EMBED_COLOR["incomplete"]))
        response_msgs.append(reply_msg)
        response_msg_contents.append("")

        for chunk in chunks:
            curr_content = chunk or ""
            if prev_content:
                if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                    reply_msg = await channel.send(embed=discord.Embed(title="Comedy Routine", description="⏳", color=EMBED_COLOR["incomplete"]))
                    response_msgs.append(reply_msg)
                    response_msg_contents.append("")
                response_msg_contents[-1] += prev_content
                final_msg_edit = len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == ""
                if final_msg_edit or (not edit_msg_task or edit_msg_task.done()) and datetime.now().timestamp() - last_msg_task_time >= len(in_progress_msg_ids) / EDITS_PER_SECOND:
                    while edit_msg_task and not edit_msg_task.done():
                        await asyncio.sleep(0)
                    if response_msg_contents[-1].strip():
                        embed = discord.Embed(title="Comedy Routine", description=response_msg_contents[-1], color=EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"])
                        edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                        last_msg_task_time = datetime.now().timestamp()
            prev_content = curr_content

        if prev_content:
            if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                reply_msg = await channel.send(embed=discord.Embed(title="Comedy Routine", description="⏳", color=EMBED_COLOR["incomplete"]))
                response_msgs.append(reply_msg)
                response_msg_contents.append("")
            response_msg_contents[-1] += prev_content
            embed = discord.Embed(title="Comedy Routine", description=response_msg_contents[-1], url=url, color=EMBED_COLOR["complete"])
            await response_msgs[-1].edit(embed=embed)
            final_text = response_msg_contents[-1]
            tts_bytes = await tts_request(final_text)
            if tts_bytes:
                tts_file = discord.File(io.BytesIO(tts_bytes), filename="roast_tts.mp3")
                await response_msgs[-1].reply(content="**Audio version:**", file=tts_file)

        logging.info(f"Final message sent: {response_msg_contents[-1]}")

# Function to schedule a message
async def schedule_message(channel: discord.TextChannel, delay: int, message: str):
    await asyncio.sleep(delay)
    await channel.send(message)

# Function to parse time strings like '1m', '1h', '2h2m', '30sec', etc.
def parse_time_string(time_str: str) -> int:
    time_units = {
        'hour': 3600,
        'hours': 3600,
        'h': 3600,
        'minute': 60,
        'minutes': 60,
        'min': 60,
        'm': 60,
        'second': 1,
        'seconds': 1,
        'sec': 1,
        's': 1
    }
    
    word_to_number = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19,
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'sixty': 60,
        'seventy': 70,
        'eighty': 80,
        'ninety': 90,
        'hundred': 100,
        'an': 1
    }
    
    pattern = re.compile(r'(\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|an)\b)\s*(hour|hours|h|minute|minutes|min|m|second|seconds|sec|s)', re.IGNORECASE)
    matches = pattern.findall(time_str)
    
    if not matches:
        return None
    
    total_seconds = 0
    for value, unit in matches:
        value = value.lower()
        if value.isdigit():
            value = int(value)
        else:
            value = word_to_number.get(value, 0)
        total_seconds += value * time_units[unit.lower()]
    
    return total_seconds

# Function to parse and schedule reminders
async def handle_reminder_command(msg: discord.Message):
    try:
        parts = msg.content.split(maxsplit=2)
        if len(parts) < 3:
            await msg.channel.send("Invalid format. Use `!remindme <time> <message>`")
            return
        time_str = parts[1]
        reminder_message = parts[2]
        delay = parse_time_string(time_str)
        if delay is None:
            await msg.channel.send("Invalid time format. Use formats like '1m', '1h', '2h2m', '30sec', etc.")
            return
        await msg.channel.send(f"Reminder set for {time_str} from now.")
        asyncio.create_task(schedule_reminder(msg.channel, delay, time_str, reminder_message))
    except ValueError:
        await msg.channel.send("Invalid time format. Please provide the time in a valid format.")

# Function to schedule and send a reminder
async def schedule_reminder(channel: discord.TextChannel, delay: int, time_str: str, reminder_message: str):
    await asyncio.sleep(delay)
    prompt = f"<system message>It's time to remind the user about the reminder they set. User Reminder input text: {reminder_message}. The timer has now expired. Remind the user!</system message>\n Reminder Time! "
    response = await generate_reminder(prompt)
    embed = discord.Embed(
        title=f"Reminder for {time_str}: {reminder_message}",
        description=response,
        color=discord.Color.green()
    )
    await channel.send(embed=embed)
    
    # Add TTS for the reminder.
    tts_bytes = await tts_request(response)
    if tts_bytes:
        tts_file = discord.File(io.BytesIO(tts_bytes), filename="reminder_tts.mp3")
        await channel.send(content="**Audio version:**", file=tts_file)

# Function to generate reminder message
async def generate_reminder(prompt: str) -> str:
    try:
        response = await llm_client.completions.create(
            model=os.getenv("LLM"),
            prompt=prompt,
            temperature=0.8,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Failed to generate reminder: {e}")
        return "Sorry, an error occurred while generating the reminder."

# Function to handle audio attachments and transcribe using Whisper
async def transcribe_audio_attachment(audio_url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(audio_url) as resp:
            if resp.status == 200:
                audio_data = await resp.read()
                with open("audio_message.mp3", "wb") as audio_file:
                    audio_file.write(audio_data)
                return transcribe_audio("audio_message.mp3")
            else:
                logging.error(f"Failed to download audio file. Status code: {resp.status}")
    return ""

# Function to handle voice commands
async def handle_voice_command(transcription: str, channel: discord.TextChannel) -> bool:
    logging.info(f"Transcription: {transcription}")
    await channel.send(embed=discord.Embed(title="User Transcription", description=transcription, color=discord.Color.blue()))
    
    match = re.search(r'Remind me in (.+?) to (.+)', transcription, re.IGNORECASE)
    if match:
        time_str = match.group(1).strip()
        reminder_message = match.group(2).strip()
        delay = parse_time_string(time_str)
        if delay is not None:
            await channel.send(f"Reminder set for {time_str} or {delay} from now.")
            asyncio.create_task(schedule_reminder(channel, delay, time_str, reminder_message))
            return True

    match = re.search(r'search for (.+)', transcription, re.IGNORECASE)
    if match:
        query = match.group(1)
        search_results = await query_searx(query)
        if search_results:
            search_summary = "\n".join([
                f"Title: {result.get('title', 'No title')}\nURL: {result.get('url', 'No URL')}\nSnippet: {result.get('content', 'No snippet available')}"
                for result in search_results
            ])
            prompt = (
                f"<system message> Use this system-side search and retrieval augmentation data in crafting summarization for the user and link citation: {search_summary}. "
                "Provide links if needed.</system message> Instruction: Summarize and provide links!"
            )
            query_title = f"Search summary/links for: \"{query}\" "

            # Make a single non-streaming API call
            response = await llm_client.chat.completions.create(
                model=os.getenv("LLM"),
                messages=[{"role": "system", "content": prompt}],
                max_tokens=1024,
                stream=False,
            )
            final_text = response.choices[0].message.content.strip()

            embed = discord.Embed(
                title=query_title,
                description=final_text,
                color=discord.Color.green(),
            )
            sent_msg = await channel.send(embed=embed)

            message_history[channel.id].append(sent_msg)
            message_history[channel.id] = message_history[channel.id][-MAX_MESSAGES:]
            msg_nodes[sent_msg.id] = MsgNode(
                {
                    "role": "assistant",
                    "content": final_text,
                    "name": str(discord_client.user.id),
                },
                replied_to=None,
            )
        else:
            await channel.send(f"No search results found for: {query}")
        return True

@discord_client.event
async def on_message(msg: discord.Message):
    if msg.author == discord_client.user:
        return
		
    logging.info(f"Received message: {msg.content} from {msg.author.name}")
    user_warnings = set()

    if any(msg.content.lower().startswith(command) for command in IGNORE_COMMANDS):
        logging.info(f"Ignored message: {msg.content}")
        return

    if msg.content.startswith("!ap") and msg.attachments:
        for attachment in msg.attachments:
            if "image" in attachment.content_type:
                image_url = attachment.url
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as resp:
                        if resp.status == 200:
                            image_data = await resp.read()
                            base64_image = base64.b64encode(image_data).decode("utf-8")
                            prompt = "<system message>Describe this image in a very detailed and intricate way, as if you were describing it to a blind person for reasons of accessibility. Replace the main character or element in the description with a random celebrity or popular well-known character. Use the {name} variable for this. Begin your response with \"AP Photo, {name}, \" followed by the description.</system message>\n "
                            reply_chain = [
                                {
                                    "role": "system",
                                    "content": (
                                        "</s></s>......"
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": prompt
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}"
                                            },
                                        }
                                    ]
                                }
                            ]

                            logging.info(f"Processing !AP command. Reply chain length: {len(reply_chain)}")

                            response_msgs = []
                            response_msg_contents = []
                            prev_content = None
                            edit_msg_task = None
                            async for chunk in await llm_client.chat.completions.create(
                                model=os.getenv("LLM"),
                                messages=reply_chain,
                                max_tokens=1024,
                                stream=True,
                            ):
                                curr_content = chunk.choices[0].delta.content or ""
                                if prev_content:
                                    if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                                        reply_msg = msg if not response_msgs else response_msgs[-1]
                                        embed = discord.Embed(description="⏳", color=EMBED_COLOR["incomplete"])
                                        for warning in sorted(user_warnings):
                                            embed.add_field(name=warning, value="", inline=False)
                                        response_msgs += [
                                            await reply_msg.reply(
                                                embed=embed,
                                                silent=True,
                                            )
                                        ]
                                        in_progress_msg_ids.append(response_msgs[-1].id)
                                        last_msg_task_time = datetime.now().timestamp()
                                        response_msg_contents += [""]
                                    response_msg_contents[-1] += prev_content
                                    final_msg_edit = len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == ""
                                    if final_msg_edit or (not edit_msg_task or edit_msg_task.done()) and datetime.now().timestamp() - last_msg_task_time >= len(in_progress_msg_ids) / EDITS_PER_SECOND:
                                        while edit_msg_task and not edit_msg_task.done():
                                            await asyncio.sleep(0)
                                        if response_msg_contents[-1].strip():
                                            embed.description = response_msg_contents[-1]
                                        embed.color = EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]
                                        edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                        last_msg_task_time = datetime.now().timestamp()
                                prev_content = curr_content

                            for response_msg in response_msgs:
                                msg_nodes[response_msg.id] = MsgNode(
                                    {
                                        "role": "assistant",
                                        "content": "".join(response_msg_contents),
                                        "name": str(response_msg.id),
                                    },
                                    replied_to=msg_nodes.get(msg.id, None),
                                )
                                in_progress_msg_ids.remove(response_msg.id)
                return

    urls_detected = detect_urls(msg.content)
    webpage_texts = await asyncio.gather(*(scrape_website(url) for url in urls_detected))

    youtube_urls = [url for url in urls_detected if "youtube.com/watch" in url or "youtu.be/" in url]
    youtube_transcripts = await asyncio.gather(*(fetch_youtube_transcript(url) for url in youtube_urls))

    for idx, youtube_transcript in enumerate(youtube_transcripts):
        if not youtube_transcript:
            youtube_transcripts[idx] = await transcribe_youtube_video(youtube_urls[idx])

    for attachment in msg.attachments:
        if "audio" in attachment.content_type or attachment.content_type == "application/ogg":
            transcription = await transcribe_audio_attachment(attachment.url)
            if transcription:
                if await handle_voice_command(transcription, msg.channel):
                    return
                msg.content = transcription

    if (
        (msg.channel.type != discord.ChannelType.private and discord_client.user not in msg.mentions)
        or (ALLOWED_CHANNEL_IDS and not any(x in ALLOWED_CHANNEL_IDS for x in (msg.channel.id, getattr(msg.channel, "parent_id", None))))
        or (ALLOWED_ROLE_IDS and (msg.channel.type == discord.ChannelType.private or not [role for role in msg.author.roles if role.id in ALLOWED_ROLE_IDS]))
        or msg.author.bot
        or msg.channel.type == discord.ChannelType.forum
    ):
        return

    if msg.channel.id not in message_history:
        message_history[msg.channel.id] = []

    if msg.attachments:
        for attachment in msg.attachments:
            if "image" in attachment.content_type:
                image_url = attachment.url
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as resp:
                        if resp.status == 200:
                            image_data = await resp.read()
                            base64_image = base64.b64encode(image_data).decode("utf-8")
                            text_content = msg.content if msg.content else ""
                            reply_chain = [
                                {
                                    "role": "system",
                                    "content": (
                                        "A chat between a curious user and an intelligent assistance system. "
                                        "The system is equipped with a vision model that analyzes the image information that the user provides in the message directly following theirs. It resembles an image description. The description is info from the vision model Use it to describe the image to the user. The system gives helpful, detailed, and rational answers to the user's questions. "
                                        "USER: Hi\n SYSTEM: Hello.\n</s> "
                                        "USER: Who are you?\n SYSTEM: I am Saṃsāra. I am an intelligent system.\n "
                                        "I always provide well-reasoned answers that are both correct and helpful.\n</s> "
                                        f"Today's date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
                                        "</s>......"
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Base Instruction: \"Describe the image in a very detailed and intricate way, as if you were describing it to a blind person for reasons of accessibility. Begin your response with: \"'Image Description':, \". "
                                                     "Extended Instruction: \"Below is a user comment or request. Write a response that appropriately completes the request.\". "
                                                     "User's prompt/question regarding the image (Optional input): " + text_content + "\n "
                                                     "</s>......"
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}"
                                            },
                                        },
                                    ]
                                }
                            ]

                            logging.info(f"Message contains image. Preparing to respond to image. Reply chain length: {len(reply_chain)}")

                            response_msgs = []
                            response_msg_contents = []
                            prev_content = None
                            edit_msg_task = None
                            async for chunk in await llm_client.chat.completions.create(
                                model=os.getenv("LLM"),
                                messages=reply_chain,
                                max_tokens=1024,
                                stream=True,
                            ):
                                curr_content = chunk.choices[0].delta.content or ""
                                if prev_content:
                                    if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                                        reply_msg = msg if not response_msgs else response_msgs[-1]
                                        embed = discord.Embed(description="⏳", color=EMBED_COLOR["incomplete"])
                                        for warning in sorted(user_warnings):
                                            embed.add_field(name=warning, value="", inline=False)
                                        response_msgs += [
                                            await reply_msg.reply(
                                                embed=embed,
                                                silent=True,
                                            )
                                        ]
                                        in_progress_msg_ids.append(response_msgs[-1].id)
                                        last_msg_task_time = datetime.now().timestamp()
                                        response_msg_contents += [""]
                                    response_msg_contents[-1] += prev_content
                                    final_msg_edit = len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == ""
                                    if final_msg_edit or (not edit_msg_task or edit_msg_task.done()) and datetime.now().timestamp() - last_msg_task_time >= len(in_progress_msg_ids) / EDITS_PER_SECOND:
                                        while edit_msg_task and not edit_msg_task.done():
                                            await asyncio.sleep(0)
                                        if response_msg_contents[-1].strip():
                                            embed.description = response_msg_contents[-1]
                                        embed.color = EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]
                                        edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                        last_msg_task_time = datetime.now().timestamp()
                                prev_content = curr_content

                            for response_msg in response_msgs:
                                msg_nodes[response_msg.id] = MsgNode(
                                    {
                                        "role": "assistant",
                                        "content": "".join(response_msg_contents),
                                        "name": str(discord_client.user.id),
                                    },
                                    replied_to=msg_nodes.get(msg.id, None),
                                )
                                in_progress_msg_ids.remove(response_msg.id)
                return

    message_history[msg.channel.id].append(msg)
    message_history[msg.channel.id] = message_history[msg.channel.id][-MAX_MESSAGES:]

    async with msg.channel.typing():
        if msg.content:
            command = msg.content.lower().split()[0]
            if command == "!toggle_search":
                search_enabled = not search_enabled
                await msg.channel.send(f"Search functionality is now {'enabled' if search_enabled else 'disabled'}.")
                return
            elif command == "!clear_history":
                logging.info(f"Clearing history for channel: {msg.channel.id}")
                message_history[msg.channel.id].clear()
                logging.info(f"History cleared. Current history size: {len(message_history[msg.channel.id])}")
                await msg.channel.send("Message history has been cleared.")
                return
            elif command == "!search":
                search_enabled = True

        # If the user typed "!sns <some query>"
        if msg.content.startswith("!sns "):
            query = msg.content[len("!sns "):].strip()
            await search_and_summarize(query, msg.channel)
            return

        # If the user typed "!roast <some url>"
        if msg.content.startswith("!roast "):
            query = msg.content[len("!roast "):].strip()
            await roast_and_summarize(query, msg.channel)
            return
        
        if msg.content.startswith("!remindme "):
            await handle_reminder_command(msg)
            return

        if msg.content.startswith("!pol "):
            user_message = msg.content[len("!pol "):].strip()
            response = await generate_sarcastic_response(user_message)
            await msg.channel.send(response)
            return

        search_enabled = False
        clear_history_used = False
        if msg.content:
            command = msg.content.lower().split()[0]
            if command == "!toggle_search":
                search_enabled = not search_enabled
                await msg.channel.send(f"Search functionality is now {'enabled' if search_enabled else 'disabled'}.")
                return
            elif command == "!clear_history":
                clear_history_used = True
                message_history[msg.channel.id].clear()
                await msg.channel.send("Message history has been cleared.")
                return
            elif command == "!show_history_size":
                history_size = len(message_history.get(msg.channel.id, []))
                await msg.channel.send(f"Current history size: {history_size}")
                return
            elif command == "!search":
                search_enabled = True

        for curr_msg in message_history[msg.channel.id]:
            curr_msg_text = curr_msg.embeds[0].description if curr_msg.embeds and curr_msg.author.bot else curr_msg.content
            if curr_msg_text and curr_msg_text.startswith(discord_client.user.mention):
                curr_msg_text = curr_msg_text[len(discord_client.user.mention):].lstrip()
            curr_msg_content = [{"type": "text", "text": curr_msg_text}] if curr_msg_text else []
            curr_msg_images = [
                {
                    "type": "image_url",
                    "image_url": {"url": att.url, "detail": "low"},
                }
                for att in curr_msg.attachments
                if "image" in att.content_type
            ]
            curr_msg_content += curr_msg_images[:MAX_IMAGES]
            if os.getenv("LLM") == "mistral":
                curr_msg_content = curr_msg_text
            curr_msg_role = "assistant" if curr_msg.author == discord_client.user else "user"
            msg_nodes[curr_msg.id] = MsgNode(
                {
                    "role": curr_msg_role,
                    "content": curr_msg_content,
                    "name": str(curr_msg.author.id),
                }
            )
            if len(curr_msg_images) > MAX_IMAGES:
                msg_nodes[curr_msg.id].too_many_images = True

        reply_chain = []
        for curr_node_id in sorted(msg_nodes.keys(), reverse=True):
            curr_node = msg_nodes[curr_node_id]
            reply_chain += [curr_node.msg]
            if curr_node.too_many_images:
                user_warnings.add(MAX_IMAGE_WARNING)
            if len(reply_chain) == MAX_MESSAGES:
                user_warnings.add(MAX_MESSAGE_WARNING)
                break

        if search_enabled and reply_chain[0]["content"] and reply_chain[0]["content"][0]["text"]:
            searx_summary = await query_searx(reply_chain[0]["content"][0]["text"])
            if searx_summary:
                reply_chain[0]["content"][0]["text"] += f" [System provided search and retrieval augmentation data for use in crafting summarization of and link citation:] \"{searx_summary}\". [Use this search and augmentation data for summarization and link citation. Provide links if needed.].\n Summarize the search results for me. Explain it all, I'm not looking at or reading it, you do eet 4 me!"

        for webpage_text in webpage_texts:
            if webpage_text == "Failed to scrape the website.":
                reply_chain[0]["content"][0]["text"] += f"\n[<system message>Unfortunately, scraping the website has failed. Please inform the user that \"the webscrape failed\" and that they should \"try another source\".</system message>]\n "
            else:
                reply_chain[0]["content"][0]["text"] += f"\n[Webpage Scrape for Summarization: {webpage_text} Use this search and augmentation data for summarization and link citation. Provide links if needed.]\n Summarize this webpage for me. Explain it all, I'm not looking at or reading it, you do eet 4 me!"

        for youtube_transcript in youtube_transcripts:
            if youtube_transcript:
                reply_chain[0]["content"][0]["text"] += f"\n[<system message>Default task: The user has provided a youtube URL that was scraped for the following content to summarize: </system message>\nYouTube Transcript: {youtube_transcript} Use this for summarization and link citation. Provide links if needed.]\n Summarize this vid for me. Explain it all, I'm not watching or reading it, you do eet 4 me!"

        for attachment in msg.attachments:
            if "image" in attachment.content_type:
                image_url = attachment.url
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as resp:
                        if resp.status == 200:
                            image_data = await resp.read()
                            base64_image = base64.b64encode(image_data).decode("utf-8")
                            reply_chain[0]["content"].append(
                                {
                                    "type": "text",
                                    "text": "Describe this image in a very detailed and intricate way, as if you were describing it to a blind person for reasons of accessibility."
                                }
                            )
                            reply_chain[0]["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                }
                            )
                            break

        logging.info(f"Preparing to generate response. Current history size for channel {msg.channel.id}: {len(message_history[msg.channel.id])}, Current reply chain length: {len(reply_chain)}")
        logging.info(f"Message received: {reply_chain[0]}, reply chain length: {len(reply_chain)}")

        response_msgs = []
        response_msg_contents = []
        prev_content = None
        edit_msg_task = None

        # Main streaming response logic
        async for chunk in await llm_client.chat.completions.create(
            model=os.getenv("LLM"),
            messages=get_system_prompt() + reply_chain[::-1],
            max_tokens=MAX_COMPLETION_TOKENS,
            stream=True,
        ):
            curr_content = chunk.choices[0].delta.content or ""
            if prev_content:
                # Check if a new message needs to be started
                if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                    reply_msg = msg if not response_msgs else response_msgs[-1]
                    embed = discord.Embed(description="⏳", color=EMBED_COLOR["incomplete"])
                    for warning in sorted(user_warnings):
                        embed.add_field(name=warning, value="", inline=False)
                    response_msgs.append(
                        await reply_msg.reply(
                            embed=embed,
                            silent=True,
                        )
                    )
                    in_progress_msg_ids.append(response_msgs[-1].id)
                    response_msg_contents.append("")
                    last_msg_task_time = datetime.now().timestamp()
        
                # Add current content to the latest message
                response_msg_contents[-1] += prev_content
        
                # Determine if the message is complete
                final_msg_edit = (
                    len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == ""
                )
                if final_msg_edit or (not edit_msg_task or edit_msg_task.done()):
                    while edit_msg_task and not edit_msg_task.done():
                        await asyncio.sleep(0)
                    if response_msg_contents[-1].strip():
                        embed.description = response_msg_contents[-1]
                    embed.color = EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]
                    edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
        
                    # TTS Step for finalized message
                    if final_msg_edit:
                        text_for_tts = response_msg_contents[-1]
                        tts_bytes = await tts_request(text_for_tts)
                        
                        if tts_bytes:
                            tts_file = discord.File(io.BytesIO(tts_bytes), filename="tts_chunk.mp3")
                            await response_msgs[-1].reply(
                                content="**Audio version of the above text:**",
                                file=tts_file
                            ) 
        
                        # Handle <think> tags
                        if "<think>" in text_for_tts and "</think>" in text_for_tts:
                            think_text = text_for_tts.split("<think>")[1].split("</think>")[0].strip()
                            following_text = text_for_tts.split("</think>")[1].strip()
        
                            # Create and send "Thoughts" embed and TTS
                            '''
                            think_embed = discord.Embed(
                                title="Thoughts",
                                description=think_text,
                                color=EMBED_COLOR["complete"]
                            )
                            think_msg = await msg.reply(embed=think_embed)
                            '''
                            think_tts_bytes = await tts_request(think_text)
                            if think_tts_bytes:
                                think_tts_file = discord.File(io.BytesIO(think_tts_bytes), filename="think_tts.mp3")
                                await response_msgs[-1].reply(
                                    content="**Audio version of the thoughts:**",
                                    file=think_tts_file
                                )
        
                            # Create and send "Reply" embed and TTS
                            if following_text.strip():
                                '''
                                reply_embed = discord.Embed(
                                    title="Reply",
                                    description=following_text,
                                    color=EMBED_COLOR["complete"]
                                )
                                follow_msg = await msg.reply(embed=reply_embed)
                                '''
                                follow_tts_bytes = await tts_request(following_text)
                                if follow_tts_bytes:
                                    follow_tts_file = discord.File(io.BytesIO(follow_tts_bytes), filename="follow_tts.mp3")
                                    await response_msgs[-1].reply(
                                        content="**Audio version of the reply:**",
                                        file=follow_tts_file
                                    )
        
                last_msg_task_time = datetime.now().timestamp()
            prev_content = curr_content
        
        # After the stream ends, process any remaining 'prev_content'
        if prev_content:
            # Check if a new embed is required for leftover text
            if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                reply_msg = await msg.channel.send(embed=discord.Embed(description="⏳", color=EMBED_COLOR["incomplete"]))
                response_msgs.append(reply_msg)
                response_msg_contents.append("")
            response_msg_contents[-1] += prev_content
        
            # Finalize the embed with the remaining text
            embed = discord.Embed(description=response_msg_contents[-1], color=EMBED_COLOR["complete"])
            await response_msgs[-1].edit(embed=embed)
        
            # Generate TTS for the remaining content
            final_text_for_tts = response_msg_contents[-1]
            tts_bytes = await tts_request(final_text_for_tts)
            if tts_bytes:
                tts_file = discord.File(io.BytesIO(tts_bytes), filename="final_tts.mp3")
                await response_msgs[-1].reply(
                    content="**Audio version of the above text:**",
                    file=tts_file
                )
        
        for response_msg in response_msgs:
            msg_nodes[response_msg.id] = MsgNode(
                {
                    "role": "assistant",
                    "content": "".join(response_msg_contents),
                    "name": str(discord_client.user.id),
                },
                replied_to=msg_nodes.get(msg.id, None),
            )
            in_progress_msg_ids.remove(response_msg.id)

@discord_client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    logging.debug(f"Raw reaction added: {payload.emoji} by {payload.user_id} on message {payload.message_id}")
    if payload.emoji.name == '❌':
        channel = discord_client.get_channel(payload.channel_id)
        if channel is None:
            user = await discord_client.fetch_user(payload.user_id)
            channel = await user.create_dm()
        message = await channel.fetch_message(payload.message_id)
        if message and message.author == discord_client.user:
            await message.delete()
            logging.debug("Message deleted.")

async def main():
    await discord_client.start(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
