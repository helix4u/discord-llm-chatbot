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
import io  # For building audio attachments in-memory

#
# ===========================
#        TTS SETTINGS
# ===========================
#
# Below are the variables/functions for your Kokoro-FastAPI TTS server.
# We will only generate ONE final TTS at the end of the entire reply,
# ensuring the file is a single unified MP3 that mobile clients can play fully.
#

TTS_API_URL = "http://localhost:8880/v1/audio/speech"  # Your Kokoro-FastAPI endpoint
TTS_VOICE = "af_sky+af+af_nicole"                      # Desired voice or model

async def tts_request(text: str, speed: float = 1.3) -> bytes:
    """
    Send a TTS request to the local Kokoro-FastAPI server and return the raw MP3 bytes.
    We'll call this once at the end of the response stream, so we produce a single unified MP3.
    """
    payload = {
        "input": text,
        "voice": TTS_VOICE,
        "response_format": "mp3",
        "speed": speed,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(TTS_API_URL, json=payload) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    err_txt = await resp.text()
                    logging.error(f"TTS request failed: {resp.status}, resp={err_txt}")
                    return None
    except Exception as e:
        logging.error(f"TTS request error: {e}")
        return None

#
# ================================
#         LOAD ENV & LOGGING
# ================================
#

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

#
# ===============================
#         OPENAI CLIENT
# ===============================
#
# Points to your local LLM endpoint (e.g. LM Studio).
#

llm_client = AsyncOpenAI(
    base_url=os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1"),
    api_key="lm-studio"
)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN environment variable is missing")

#
# ===============================
#     DISCORD GLOBAL VARS
# ===============================
#

ALLOWED_CHANNEL_IDS = [int(i) for i in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if i]
ALLOWED_ROLE_IDS = [int(i) for i in os.getenv("ALLOWED_ROLE_IDS", "").split(",") if i]
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 0))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", 10))

EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green()}
EMBED_MAX_LENGTH = 4096
EDITS_PER_SECOND = 1.3
MAX_COMPLETION_TOKENS = 2048

MAX_IMAGE_WARNING = (f"⚠️ Max {MAX_IMAGES} image{'' if MAX_IMAGES == 1 else 's'} per message"
                     if MAX_IMAGES > 0 else "")
MAX_MESSAGE_WARNING = f"⚠️ Only using last {MAX_MESSAGES} messages"

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
discord_client = discord.Client(intents=intents)

msg_nodes = {}
in_progress_msg_ids = []
message_history = {}

#
# ===============================
#     IGNORED COMMANDS
# ===============================
#

IGNORE_COMMANDS = [
    "!dream", "!d", "!background", "!avatar",
    "!help", "!info", "!ping", "!status", "!upscale", "!nightmare", "!n", "!describe", "!background", "!chat",
    "!superprompt", "!depth", "!face", "!edges", "!lineart", "!lineartanime", "!colormap", "!pose", "!esrgan",
    "!metadata", "!text", "!append", "!models", "!loras", "!nightmarePromptGen", "!load", "!aspect", "!resolution",
    "!handfix"
]

scheduled_tasks = []

#
# ===============================
#       MESSAGE HISTORY
# ===============================
#

class MsgNode:
    def __init__(self, msg, too_many_images=False, replied_to=None):
        self.msg = msg
        self.too_many_images = too_many_images
        self.replied_to = replied_to

#
# ===============================
#     SYSTEM PROMPT LOGIC
# ===============================
#

def get_system_prompt() -> list:
    """
    Return a system prompt used to 'seed' the conversation style.
    """
    if os.getenv("LLM") in ["gpt-4-vision-preview", "mistral", "local"]:
        return [
            {
                "role": "system",
                "content": (
                    "A chat between a curious user and a hyper-intelligent assistance system. "
                    "The system gives helpful, detailed, and rational answers to the user's questions. "
                    f"Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
                    "USER: Hi\n SYSTEM: Hello.\n</s> "
                    "USER: Who are you?\n SYSTEM: I am a snarky, yet intelligent system named Saṃsāra, or Sam.\n"
                    "I always provide well-reasoned answers that are both correct and helpful, sometimes snarky or witty.\n</s> "
                    "</s>......"
                ),
            }
        ]
    # Default else:
    return [
        {
            "role": "system",
            "content": (
                "A chat between a curious user and a hyper-intelligent assistance system. "
                "The assistant gives helpful, detailed, and rational answers to the user's questions. "
                "USER: Hi\n SYSTEM: Hello.\n</s> "
                "USER: Who are you?\n SYSTEM: I am a snarky, yet intelligent system named Saṃsāra, or Sam.\n "
                "I always provide well-reasoned answers that are both correct and helpful and sometimes snarky or witty.\n</s> "
                f"Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
                "</s>......"
            ),
        }
    ]


#
# ===============================
#      HELPER LLM FUNCTIONS
# ===============================
#

async def generate_sarcastic_response(user_message: str) -> str:
    """
    Make a single sarcastic or mocking response about alt-right topics.
    """
    prompt = (
        "The user is fed up with extremist political views and wants to push back using sarcasm. "
        "You are here to make a single reply to mock these alt-right weirdos. "
        "Use the most sarcastic, snarky, troll-like comments. Make it biting and humorous.\n\n"
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

#
# ===============================
#       SCRAPING FUNCTIONS
# ===============================
#

async def scrape_website(url: str) -> str:
    """
    Try scraping the website with Playwright, fallback to BeautifulSoup/cURL on failure.
    """
    logging.info(f"Scraping website: {url}")
    async with async_playwright() as p:
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=user_agent)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=10000)
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
    headers = {'User-Agent': user_agent}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status != 200:
                    logging.error(f"Failed to fetch {url} status code={response.status}")
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
        result = subprocess.run(
            ['curl', '-s', url],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )
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

#
# ===============================
#       TEXT CLEANUP
# ===============================
#

def detect_urls(message_text: str) -> list:
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        r'[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(message_text)

def clean_text(text: str) -> str:
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = re.sub('\n+', '\n', text).strip()

    # Some patterns to remove or unify
    patterns_to_replace = [
        (r'\s+', ' '),
        (r'\[.*?\]', ''),
        (r'\[\s*__\s*\]', ''),
        (r'NFL Sunday Ticket', ''),
        (r'© \d{4} Google LLC', '')
    ]
    for pattern, repl in patterns_to_replace:
        text = re.sub(pattern, repl, text)

    return text

#
# ===============================
#        SEARCH WITH SEARX
# ===============================
#

async def query_searx(query: str) -> list:
    """
    Query a local Searx instance for web search results
    """
    logging.info(f"Querying Searx for: {query}")
    searx_url = "http://192.168.1.3:9092/search"  # Adjust if needed
    params = {'q': query, 'format': 'json', 'language': 'en-US'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(searx_url, params=params, timeout=5) as response:
                if response.status == 200:
                    results = await response.json()
                    return results.get('results', [])[:5]
                logging.error("Failed to fetch data from Searx.")
    except aiohttp.ClientError as e:
        logging.error(f"An error occurred while fetching data from Searx: {e}")
    return []

async def generate_completion(prompt: str) -> str:
    """
    Simple text-completion call to the local model (non-streaming).
    """
    try:
        response = await llm_client.completions.create(
            model="local-model",
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

#
# ===============================
#     YOUTUBE HELPER FUNCS
# ===============================
#

def clean_youtube_transcript(transcript: str) -> str:
    return clean_text(transcript)

async def fetch_youtube_transcript(url: str) -> str:
    """
    Use youtube_transcript_api to fetch text, fallback to empty if not found.
    """
    try:
        video_id = re.search(r'v=([^&]+)', url).group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        joined_text = " ".join([entry['text'] for entry in transcript])
        return clean_youtube_transcript(joined_text)
    except NoTranscriptFound:
        logging.error("No transcript found for this video.")
    except Exception as e:
        logging.error(f"Failed to fetch transcript: {e}")
    return ""

def download_youtube_video(url: str, output_path: str = "youtube_audio.mp4"):
    """
    Use yt_dlp to download audio for local transcription with Whisper.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'user_agent': (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ),
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
        'external_downloader_args': [
            '-loglevel', 'panic',
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5'
        ],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        print(f"Failed to download YouTube video: {e}")
        return None

def transcribe_audio(file_path: str) -> str:
    """
    Run Whisper on local GPU/CPU to transcribe an audio file.
    """
    whisper_model = whisper.load_model("tiny")
    try:
        result = whisper_model.transcribe(file_path)
        return result["text"]
    finally:
        del whisper_model
        torch.cuda.empty_cache()
        gc.collect()

async def transcribe_youtube_video(file_path: str) -> str:
    """
    Attempt to transcribe a local .mp4 or .wav using Whisper.
    """
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
    """
    Quick test function if you just want to run a direct script.
    """
    audio_path = download_youtube_video(url)
    if not audio_path:
        print("Download failed. Exiting.")
        return
    transcription = await transcribe_youtube_video(audio_path)
    if transcription:
        print(f"Transcription: {transcription}")
    else:
        print("Transcription failed.")

#
# ===============================
#        COMMAND HANDLERS
# ===============================
#

async def roast_and_summarize(url: str, channel: discord.TextChannel):
    """
    Example command: "!roast <url>"
    Scrapes a website, generates a comedic routine, and streams it in chunks.
    """
    webpage_text = await scrape_website(url)
    if webpage_text == "Failed to scrape the website.":
        await channel.send(
            f"Unfortunately, scraping the website at {url} has failed. Please try another source."
        )
    else:
        cleaned_content = clean_text(webpage_text)
        prompt = (
            f"\n[Webpage Scrape for Comedy Routine: {cleaned_content} "
            "Use this content to create a professional comedy routine. "
            "Make it funny, witty, and engaging. Provide full hyperlink formatting.]\n"
        )
        comedy_routine = await generate_completion(prompt)
        # chunk the comedic text
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

        reply_msg = await channel.send(
            embed=discord.Embed(
                title="Comedy Routine",
                description="⏳",
                color=EMBED_COLOR["incomplete"]
            )
        )
        response_msgs.append(reply_msg)
        response_msg_contents.append("")

        for chunk in chunks:
            curr_content = chunk or ""
            if prev_content:
                if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                    reply_msg = await channel.send(
                        embed=discord.Embed(
                            title="Comedy Routine",
                            description="⏳",
                            color=EMBED_COLOR["incomplete"]
                        )
                    )
                    response_msgs.append(reply_msg)
                    response_msg_contents.append("")
                response_msg_contents[-1] += prev_content
                final_msg_edit = (
                    len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH
                    or curr_content == ""
                )
                if (final_msg_edit or (not edit_msg_task or edit_msg_task.done()) 
                    and datetime.now().timestamp() - last_msg_task_time 
                    >= len(in_progress_msg_ids) / EDITS_PER_SECOND):
                    while edit_msg_task and not edit_msg_task.done():
                        await asyncio.sleep(0)
                    if response_msg_contents[-1].strip():
                        embed = discord.Embed(
                            title="Comedy Routine",
                            description=response_msg_contents[-1],
                            color=EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]
                        )
                        edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                        last_msg_task_time = datetime.now().timestamp()
            prev_content = curr_content

        if prev_content:
            if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                reply_msg = await channel.send(
                    embed=discord.Embed(
                        title="Comedy Routine",
                        description="⏳",
                        color=EMBED_COLOR["incomplete"]
                    )
                )
                response_msgs.append(reply_msg)
                response_msg_contents.append("")
            response_msg_contents[-1] += prev_content
            embed = discord.Embed(
                title="Comedy Routine",
                description=response_msg_contents[-1],
                url=url,
                color=EMBED_COLOR["complete"]
            )
            await response_msgs[-1].edit(embed=embed)
        logging.info(f"Final message sent: {response_msg_contents[-1]}")

async def schedule_message(channel: discord.TextChannel, delay: int, message: str):
    await asyncio.sleep(delay)
    await channel.send(message)

def parse_time_string(time_str: str) -> int:
    """
    Parse time like '1h', '2h30m', '45s', 'one hour', etc. Return seconds.
    """
    time_units = {
        'hour': 3600, 'hours': 3600, 'h': 3600,
        'minute': 60, 'minutes': 60, 'min': 60, 'm': 60,
        'second': 1, 'seconds': 1, 'sec': 1, 's': 1
    }
    word_to_number = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
        'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
        'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
        'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'an': 1
    }
    pattern = re.compile(
        r'(\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|'
        r'twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|'
        r'nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|'
        r'ninety|hundred|an)\b)\s*(hour|hours|h|minute|minutes|min|m|second|seconds|sec|s)',
        re.IGNORECASE
    )
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
            await msg.channel.send(
                "Invalid time format. Use '1m', '1h', '2h2m', '30sec', etc."
            )
            return
        await msg.channel.send(f"Reminder set for {time_str} from now.")
        asyncio.create_task(schedule_reminder(msg.channel, delay, time_str, reminder_message))
    except ValueError:
        await msg.channel.send("Invalid time format. Please provide the time in a valid format.")

async def schedule_reminder(channel: discord.TextChannel, delay: int, time_str: str, reminder_message: str):
    await asyncio.sleep(delay)
    prompt = (
        f"<system message>It's time to remind the user about the reminder they set. "
        f"User Reminder: {reminder_message}. The timer is up. Remind them!</system message>\n"
        "Reminder Time!"
    )
    response = await generate_reminder(prompt)
    embed = discord.Embed(
        title=f"Reminder for {time_str}: {reminder_message}",
        description=response,
        color=discord.Color.green()
    )
    await channel.send(embed=embed)

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

async def transcribe_audio_attachment(audio_url: str) -> str:
    """
    Download user-uploaded audio for transcription via Whisper.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(audio_url) as resp:
            if resp.status == 200:
                audio_data = await resp.read()
                with open("audio_message.mp3", "wb") as audio_file:
                    audio_file.write(audio_data)
                return transcribe_audio("audio_message.mp3")
            else:
                logging.error(f"Failed to download audio file. status={resp.status}")
    return ""

async def handle_voice_command(transcription: str, channel: discord.TextChannel) -> bool:
    """
    Example: If user says 'Remind me in 2 minutes to feed the cat', parse that.
    Or 'search for ...' triggers a Searx query.
    """
    logging.info(f"Transcription: {transcription}")
    await channel.send(embed=discord.Embed(
        title="User Transcription",
        description=transcription,
        color=discord.Color.blue()
    ))

    # Attempt to parse "Remind me in X to Y"
    match = re.search(r'Remind me in (.+?) to (.+)', transcription, re.IGNORECASE)
    if match:
        time_str = match.group(1).strip()
        reminder_message = match.group(2).strip()
        delay = parse_time_string(time_str)
        if delay is not None:
            await channel.send(f"Reminder set for {time_str} or {delay} seconds from now.")
            asyncio.create_task(schedule_reminder(channel, delay, time_str, reminder_message))
            return True

    # Attempt to parse "search for <something>"
    match = re.search(r'search for (.+)', transcription, re.IGNORECASE)
    if match:
        query = match.group(1)
        search_results = await query_searx(query)
        if search_results:
            search_summary = "\n".join([
                f"Title: {r.get('title', 'No title')}\n"
                f"URL: {r.get('url', 'No URL')}\n"
                f"Snippet: {r.get('content', 'No snippet')}"
                for r in search_results
            ])
            prompt = (
                f"<system message>Use this system-side search data to craft summarization with link citation:\n"
                f"{search_summary}\nProvide links if needed.</system message>"
            )
            query_title = f"Search summary/links for: \"{query}\""
            response_msgs = []
            response_msg_contents = []
            prev_content = None
            edit_msg_task = None
            last_msg_task_time = datetime.now().timestamp()
            in_progress_msg_ids = []

            async for chunk in await llm_client.chat.completions.create(
                model=os.getenv("LLM"),
                messages=[{"role": "system", "content": prompt}],
                max_tokens=1024,
                stream=True,
            ):
                curr_content = chunk.choices[0].delta.content or ""
                if prev_content:
                    if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                        reply_msg = await channel.send(
                            embed=discord.Embed(
                                title=query_title,
                                description="⏳",
                                color=EMBED_COLOR["incomplete"]
                            )
                        )
                        response_msgs.append(reply_msg)
                        response_msg_contents.append("")
                    response_msg_contents[-1] += prev_content
                    final_msg_edit = (
                        len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH
                        or curr_content == ""
                    )
                    if final_msg_edit:
                        while edit_msg_task and not edit_msg_task.done():
                            await asyncio.sleep(0)
                        if response_msg_contents[-1].strip():
                            embed = discord.Embed(
                                title=query_title,
                                description=response_msg_contents[-1],
                                color=EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]
                            )
                            edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                    last_msg_task_time = datetime.now().timestamp()
                prev_content = curr_content

            if prev_content:
                if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                    reply_msg = await channel.send(
                        embed=discord.Embed(
                            title=query_title,
                            description="⏳",
                            color=EMBED_COLOR["incomplete"]
                        )
                    )
                    response_msgs.append(reply_msg)
                    response_msg_contents.append("")
                response_msg_contents[-1] += prev_content
                embed = discord.Embed(
                    title=query_title,
                    description=response_msg_contents[-1],
                    color=EMBED_COLOR["complete"]
                )
                await response_msgs[-1].edit(embed=embed)

            for response_msg in response_msgs:
                message_history[channel.id].append(response_msg)
                message_history[channel.id] = message_history[channel.id][-MAX_MESSAGES:]
                msg_nodes[response_msg.id] = MsgNode(
                    {
                        "role": "assistant",
                        "content": "".join(response_msg_contents),
                        "name": str(discord_client.user.id),
                    },
                    replied_to=None,
                )
        else:
            await channel.send(f"No search results found for: {query}")
        return True
    return False

#
# ===============================
#       DISCORD ON_MESSAGE
# ===============================
#

@discord_client.event
async def on_message(msg: discord.Message):
    """
    Main message handler. Streams the LLM's response, then does a SINGLE
    TTS at the end with the entire text, so mobile users get the full audio.
    """
    # Skip if it's our own bot message (prevents re-processing attachments, infinite loops)
    if msg.author == discord_client.user:
        return

    logging.info(f"Received message: {msg.content} from {msg.author.name}")
    user_warnings = set()

    # Skip if message starts with an ignored command
    if any(msg.content.lower().startswith(command) for command in IGNORE_COMMANDS):
        logging.info(f"Ignored command: {msg.content}")
        return

    #
    # Check for specific commands (like !ap, etc.) or do your normal logic
    # We'll omit that for brevity, or you can keep your existing code
    #

    # Detect any posted URLs
    urls_detected = detect_urls(msg.content)
    webpage_texts = await asyncio.gather(*(scrape_website(url) for url in urls_detected))

    # Check for YouTube links and attempt transcript or Whisper fallback
    youtube_urls = [u for u in urls_detected if "youtube.com/watch" in u or "youtu.be/" in u]
    youtube_transcripts = await asyncio.gather(*(fetch_youtube_transcript(u) for u in youtube_urls))
    for idx, yt_transcript in enumerate(youtube_transcripts):
        if not yt_transcript:
            youtube_transcripts[idx] = await transcribe_youtube_video(youtube_urls[idx])

    # If there's an audio attachment from a user, transcribe it
    for attachment in msg.attachments:
        if ("audio" in attachment.content_type) or (attachment.content_type == "application/ogg"):
            transcription = await transcribe_audio_attachment(attachment.url)
            if transcription:
                # If it triggers a voice command like "Remind me..."
                if await handle_voice_command(transcription, msg.channel):
                    return
                # Otherwise, treat the transcribed text as part of user's input
                msg.content = transcription

    # Additional gating logic:
    # if user didn't mention the bot, not in allowed channels, or is a bot, skip
    if (
        (msg.channel.type != discord.ChannelType.private and discord_client.user not in msg.mentions)
        or (ALLOWED_CHANNEL_IDS and not any(
            x in ALLOWED_CHANNEL_IDS for x in (msg.channel.id, getattr(msg.channel, "parent_id", None))
        ))
        or (ALLOWED_ROLE_IDS and (
            msg.channel.type == discord.ChannelType.private
            or not [role for role in msg.author.roles if role.id in ALLOWED_ROLE_IDS]
        ))
        or msg.author.bot
        or msg.channel.type == discord.ChannelType.forum
    ):
        return

    # Keep a message history
    if msg.channel.id not in message_history:
        message_history[msg.channel.id] = []
    message_history[msg.channel.id].append(msg)
    message_history[msg.channel.id] = message_history[msg.channel.id][-MAX_MESSAGES:]

    # Handle custom commands
    if msg.content.startswith("!remindme "):
        await handle_reminder_command(msg)
        return

    if msg.content.startswith("!pol "):
        user_message = msg.content[len("!pol "):].strip()
        response = await generate_sarcastic_response(user_message)
        await msg.channel.send(response)
        return

    if msg.content.startswith("!roast "):
        url = msg.content[len("!roast "):].strip()
        await roast_and_summarize(url, msg.channel)
        return

    #
    # Build up the conversation from message history
    #
    reply_chain = []
    for posted_msg in message_history[msg.channel.id]:
        # If the bot used an embed, the text is in embed.description
        # else it's normal .content
        posted_msg_text = (posted_msg.embeds[0].description if (posted_msg.embeds and posted_msg.author.bot)
                           else posted_msg.content)
        if posted_msg_text and posted_msg_text.startswith(discord_client.user.mention):
            posted_msg_text = posted_msg_text[len(discord_client.user.mention):].lstrip()

        msg_content = [{"type": "text", "text": posted_msg_text}] if posted_msg_text else []
        # If there's images in the attachments, add them (up to MAX_IMAGES)
        msg_images = [
            {"type": "image_url", "image_url": {"url": att.url, "detail": "low"}}
            for att in posted_msg.attachments
            if ("image" in att.content_type)
        ]
        msg_content += msg_images[:MAX_IMAGES]
        if len(msg_images) > MAX_IMAGES:
            user_warnings.add(MAX_IMAGE_WARNING)

        role = "assistant" if posted_msg.author == discord_client.user else "user"
        node_data = {
            "role": role,
            "content": msg_content,
            "name": str(posted_msg.author.id),
        }
        msg_nodes[posted_msg.id] = MsgNode(node_data)
        reply_chain.append(node_data)
        if len(reply_chain) == MAX_MESSAGES:
            user_warnings.add(MAX_MESSAGE_WARNING)
            break

    # Insert augmentation data from web pages or YouTube transcripts
    if reply_chain:
        if webpage_texts:
            for w_text in webpage_texts:
                if w_text == "Failed to scrape the website.":
                    reply_chain[-1]["content"][0]["text"] += (
                        "\n[Scrape failed: please try another source.]\n"
                    )
                else:
                    reply_chain[-1]["content"][0]["text"] += (
                        f"\n[Webpage content: {w_text} Summarize or provide info.]\n"
                    )
        for y_text in youtube_transcripts:
            if y_text:
                reply_chain[-1]["content"][0]["text"] += (
                    f"\n[YouTube transcript: {y_text} Summarize or provide info.]\n"
                )

    # Generate the model's response in streaming fashion
    response_msgs = []
    response_msg_contents = []
    prev_content = None
    edit_msg_task = None

    async with msg.channel.typing():
        async for chunk in await llm_client.chat.completions.create(
            model=os.getenv("LLM"),
            messages=get_system_prompt() + reply_chain[::-1],
            max_tokens=MAX_COMPLETION_TOKENS,
            stream=True,
        ):
            curr_content = chunk.choices[0].delta.content or ""
            if prev_content:
                # If we haven't posted anything yet or the last embed is too large:
                if (not response_msgs
                    or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH):
                    reply_msg = await msg.reply(embed=discord.Embed(
                        description="⏳",
                        color=EMBED_COLOR["incomplete"]
                    ), silent=True)
                    response_msgs.append(reply_msg)
                    response_msg_contents.append("")

                # Accumulate text
                response_msg_contents[-1] += prev_content

                # We only "finalize" visually if the chunk ended or max embed length is exceeded
                final_msg_edit = (
                    (len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH)
                    or (curr_content == "")
                )
                if final_msg_edit or (not edit_msg_task or edit_msg_task.done()):
                    while edit_msg_task and not edit_msg_task.done():
                        await asyncio.sleep(0)
                    # If there's text to show:
                    if response_msg_contents[-1].strip():
                        embed = discord.Embed(
                            description=response_msg_contents[-1],
                            color=EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]
                        )
                        edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

            prev_content = curr_content

        # After the stream ends, handle leftover if any
        if prev_content:
            if (not response_msgs
                or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH):
                reply_msg = await msg.reply(embed=discord.Embed(
                    description="⏳",
                    color=EMBED_COLOR["incomplete"]
                ), silent=True)
                response_msgs.append(reply_msg)
                response_msg_contents.append("")
            response_msg_contents[-1] += prev_content
            # Final "complete" embed
            embed = discord.Embed(
                description=response_msg_contents[-1],
                color=EMBED_COLOR["complete"]
            )
            await response_msgs[-1].edit(embed=embed)

    #
    # Now we unify EVERYTHING into one final text for TTS. That ensures a single MP3 with the entire content.
    #
    final_text = "".join(response_msg_contents)

    # Generate TTS for the entire final text
    if final_text.strip():
        tts_bytes = await tts_request(final_text)
        if tts_bytes:
            # Create an in-memory file
            tts_file = discord.File(io.BytesIO(tts_bytes), filename="full_reply.mp3")
            # Attach the single, full MP3 for mobile/others:
            await response_msgs[-1].reply(
                content="**Audio version (full)**:",
                file=tts_file
            )

    # Store everything for reference
    for response_msg in response_msgs:
        msg_nodes[response_msg.id] = MsgNode(
            {
                "role": "assistant",
                "content": final_text,  # the entire final text
                "name": str(discord_client.user.id),
            },
            replied_to=msg_nodes.get(msg.id, None),
        )
        if response_msg.id in in_progress_msg_ids:
            in_progress_msg_ids.remove(response_msg.id)

@discord_client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    logging.debug(f"Raw reaction: {payload.emoji} by {payload.user_id} on {payload.message_id}")
    if payload.emoji.name == '❌':
        channel = discord_client.get_channel(payload.channel_id)
        if channel is None:
            user = await discord_client.fetch_user(payload.user_id)
            channel = await user.create_dm()
        message = await channel.fetch_message(payload.message_id)
        if message and message.author == discord_client.user:
            await message.delete()
            logging.debug("Message deleted.")

#
# ===============================
#          STARTUP
# ===============================
#

async def run_discord_bot():
    await discord_client.start(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    asyncio.run(run_discord_bot())
