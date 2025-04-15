import asyncio
from datetime import datetime
import logging
import os
import json
import random
import re
import base64
import gc
import torch
import io

import discord
from discord import File, Embed
from dotenv import load_dotenv
from openai import AsyncOpenAI
import aiohttp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
import whisper
from pydub import AudioSegment

# -------------------------------------------------------------------
# Helper: re-encode MP3 data to fix length metadata
# -------------------------------------------------------------------
def fix_mp3_length(mp3_data: bytes) -> bytes:
    """
    Re-encode MP3 in-memory via pydub to ensure correct length metadata.
    """
    audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
    output_buffer = io.BytesIO()
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
                    mp3_fixed = fix_mp3_length(mp3_raw)
                    return mp3_fixed
                else:
                    err_txt = await resp.text()
                    logging.error(f"TTS request failed: status={resp.status}, resp={err_txt}")
                    return None
    except Exception as e:
        logging.error(f"TTS request error: {e}")
        return None

# -------------------------------------------------------------------
# Load environment variables / logging / config
# -------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

llm_client = AsyncOpenAI(
    base_url=os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1"),
    api_key="lm-studio"
)

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

IGNORE_COMMANDS = [
    "!dream", "!d", "!background", "!avatar",
    "!help", "!info", "!ping", "!status", "!upscale", "!nightmare", "!n", "!describe", 
    "!background", "!chat", "!superprompt", "!depth", "!face", "!edges", "!lineart", 
    "!lineartanime", "!colormap", "!pose", "!esrgan", "!metadata", "!text", "!append", 
    "!models", "!loras", "!nightmarePromptGen", "!load", "!aspect", "!resolution", "!handfix"
]

scheduled_tasks = []

class MsgNode:
    def __init__(self, msg, too_many_images=False, replied_to=None):
        self.msg = msg
        self.too_many_images = too_many_images
        self.replied_to = replied_to

def get_system_prompt() -> list:
    if os.getenv("LLM") in ["gpt-4-vision-preview", "mistral", "local"]:
        return [
            {
                "role": "system",
                "content": (
                    f"A chat between a curious user and a hyper-intelligent assistance entity named Sam. "
                    f"Sam gives helpful, detailed, and rational answers to the user's questions. "
                    f"Sam creates an initial reaction, considers the context, makes a plan of action and a rationale "
                    f"and evaluates it for logical failings before responding. Provocative Endings: Each reflection "
                    f"concludes with a question or thought provoking further exploration. "
                    f"Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
                ),
            }
        ]
    return [
        {
            "role": "system",
            "content": (
                f"A chat between a curious user and a hyper-intelligent assistance entity named Sam. "
                f"Sam gives helpful, detailed, and rational answers to the user's questions. "
                f"Sam creates an initial reaction, considers the context, makes a plan of action and a rationale "
                f"and evaluates it for logical failings before responding. Provocative Endings: Each reflection "
                f"concludes with a question or thought provoking further exploration. "
                f"Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
            ),
        }
    ]

# -------------------------------------------------------------------
# scrape_latest_tweets (must be defined before on_message)
# -------------------------------------------------------------------
async def scrape_latest_tweets(username: str, limit: int = 10) -> list:
    logging.info(f"[SCRAPE] Fetching last {limit} tweets from @{username}")
    tweets_collected = []
    seen_timestamps = set()
    scroll_delay = 2.0
    total_scrolls = max(5, limit // 2)
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=".pw-profile",
            headless=False,  # Not headless so you can see the browser for debugging
            args=["--disable-blink-features=AutomationControlled"]
        )
        page = await browser.new_page()
        url = f"https://x.com/{username}/with_replies"
        await page.goto(url, timeout=60000)
        await asyncio.sleep(5)
        scrolls_done = 0
        while scrolls_done < total_scrolls and len(tweets_collected) < limit:
            await page.mouse.wheel(0, 3000)
            await asyncio.sleep(scroll_delay)
            scrolls_done += 1
            articles = await page.locator("article").all()
            for art in articles:
                try:
                    timestamp_elem = art.locator("time").nth(0)
                    ts_attr = await timestamp_elem.get_attribute("datetime")
                    if not ts_attr or ts_attr in seen_timestamps:
                        continue
                    user_handle = ""
                    try:
                        anchor = art.locator("a[role='link'][href^='/']").first
                        href = await anchor.get_attribute("href")
                        if href:
                            parts = href.strip("/").split("/")
                            if parts:
                                user_handle = parts[0]
                    except:
                        pass
                    text_locator = art.locator("div[lang]").nth(0)
                    content = (await text_locator.inner_text(timeout=3000)).strip()
                    if not content:
                        continue
                    tweets_collected.append({
                        "timestamp": ts_attr,
                        "from_user": user_handle,
                        "content": content
                    })
                    seen_timestamps.add(ts_attr)
                    if len(tweets_collected) >= limit:
                        break
                except Exception as e:
                    logging.error(f"Error while processing a tweet: {e}")
                    continue
        await browser.close()
    tweets_collected.sort(key=lambda x: x["timestamp"], reverse=True)
    return tweets_collected[:limit]

# --- Changed to use chat completions ---
async def generate_sarcastic_response(user_message: str) -> str:
    prompt = (
        "The user is fed up with extremist political views and wants to push back using sarcasm. "
        "You are here to make a single reply to mock these alt-right weirdos. "
        "The bot should respond to any political discussion or keyword with the most sarcastic, "
        "snarky, and troll-like comments possible. The goal is to mock and undermine these extremist views "
        "in a way that’s both biting and humorous.\n\n"
        f"User: {user_message}\nBot:"
    )
    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=os.getenv("LLM"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=4096,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ),
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Failed to generate sarcastic response: {e}")
        return "Sorry, an error occurred while generating the response."

# --- Changed generate_completion similarly ---
async def generate_completion(prompt: str) -> str:
    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=os.getenv("LLM"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=2048,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ),
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Failed to generate completion: {e}")
        return "Sorry, an error occurred while generating the response."

async def send_response_with_tts(
    channel: discord.TextChannel, 
    title: str, 
    text: str, 
    color=discord.Color.green(), 
    tts_filename="tts_response.mp3"
):
    embed = Embed(title=title, description=text, color=color)
    sent_msg = await channel.send(embed=embed)
    tts_bytes = await tts_request(text)
    if tts_bytes:
        tts_file = File(io.BytesIO(tts_bytes), filename=tts_filename)
        await channel.send(content="**Audio version:**", file=tts_file)
    return sent_msg

def detect_urls(message_text: str) -> list:
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        r'[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(message_text)

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n+', '\n', text).strip()
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

def clean_youtube_transcript(transcript: str) -> str:
    return clean_text(transcript)

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

# -------------------------------------------------------------------
# Additional tool: scrape_website (Playwright)
# -------------------------------------------------------------------
async def scrape_website(url: str) -> str:
    logging.info(f"Scraping website: {url}")
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Not headless so you can see the browser
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
            return "Failed to scrape the website."

# -------------------------------------------------------------------
# Query Searx
# -------------------------------------------------------------------
async def query_searx(query: str) -> list:
    logging.info(f"Querying Searx for: {query}")
    searx_url = "http://192.168.1.3:9092/search"
    preferences_str = (
        "eJx1V0GT6yYM_jXNxfMyfX2HTg85dabXdqa9MzIottaAWMBJvL--IrZjvN53WGKEEOKT9KHVkLHjSJguHXqMYH_57U-P9yQ_lGVImtBrLF-"
        "sCWzj0BCcLPhuhA4vMGY-WdZg8YL-VKaaXbCY8XIiJyoqRH5Ml7_AJjw5zD2byz9___vfKcEVE0LU_eXXU-7R4SVR2XqKmEabk2KvxBeVoV22"
        "GyYli2xvGC8MMj1z7E7zNpXyZBePNPqMUYGlzjv5XvaDuYHcxqjl3Fn6PmKcFHmVKYuBWUj-Sp6yWNWRrV0doAStFQPoO_KC2h8ddErV2AhU4"
        "D2kppxAN1TqShYLoBCGxlGMHGuZeN7I2KTMsVaedzeUlNpC05Lv9vPcjnrArNQzXFrrb_mm1I0McjEjzqcU8So-viIpsru5kSBbWZow1FM9Rk"
        "tYSwzih0Cq3JhIl7npGoNPkIj9zkuUA8goxRLVWFTFxfLXcfNMCVGef_drd4SyYds3g-MgyAYZi2XHbxQKzJtWlkBMO1e_P6oDriZycWYF9h"
        "oRm8TXfIeIjaGIWpCfFgCvkfxAoCsDnWQFtCvAbLDF2C3TjrkTF4OFqQQxbcfUK44F7VRFpQtsTA1mD22EMix2yZl2uyF5qDaTl0_iMX0tW0"
        "2-Uep5szGQHiBVDlqSI-PUFNgS1QvcpoznmBZf4MNOkXRlWqoYdAC_KPjg1q8JYDPEAX3EwJXtAOJFR2nN1zC2Z4O3ZTZXZYG4KcMrJOCgeI"
        "svNQc-k26S7tlCrHM7ZYg5FNqp8iHzMHFmwWMoTq_O5DvlwhKfy3eCnrnef6d2qudCdg_wJpYsrLUGapmH9FlY-FSphVWL4H3kjJ-1Eo9RH6"
        "RCdInydBDz9OmO33_8-P2x3c2MBv0W_YQfHlytL4WEONQSD7cCxiaIYzt16NY8CIgxjy1WafdETI4YCpvcsa2W4uiEKCvBnR80sJdwN2ny7"
        "CeHlX9v4RzulcPlkk64ZUd2EW67OwfJQzsnzU_ER76RRNpSe8a8H_NWzi9DJ_R7qn_as-THR1O8W6sjSiZW9iE-6FZnZCuZosGFrXzK7oXD9"
        "sx-9PYp3eXPUzKDWoHbZkNdty-9kjpY2RJKj-LTtCf1G0ktySVqSmZ5UWLTjyvpFYcN5J2_G2_vmD9Pjr2AVMX2aoV6Yk2o6KlmKpnnAvwc"
        "AsrbwTOJ1gcstHoAapHvoFpkB7AW-Rfs0S_nPQ__-LZQ5JEz-SoE5TtpRfap5bvbDiRHD215NBV1gjwvuNJ8CRPIC1TATSgdT-XLuiYUKQD"
        "C8sju11OWxyzL07i-j8GU-tyUQi-851-rFKXfa6G6kViXIt5tmcKa2-93yY36Pk_BHuRZdIjHLD5AL1c1lA-MmzAQrIX8UpbqlNLZ4yf8rg"
        "eW1L5avq8lmIaxHX0e16oeA8YxvVCWfpGMvPGS_bl6i0af5HVOfeX1k_n395t4_ER6L8mrEwKykvYlQpXajRxyzX80FI1nPr1Yju38uoW-"
        "amgM6fzBfpf3TujSSZvS5Ag-WQmjqc1E42moBDnHM610unWtwY7CZelSoH6cl9m555TldUBprQWQ-RndKTCoufO-R-mID8sCrdI96uGwIhg"
        "oycABp_Si05_4UAIxTw5WlsQQdPX878ok_buVtu0LTXuVRv7KRx8jFFJT0tbKPVzJoJO0BVJCl_8BGa3pmQ==&q=%s"
    )
    params = {
        "q": query,
        "preferences": preferences_str,
        "format": "json",
        "language": "en-US",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(searx_url, params=params, timeout=15) as response:
                if response.status == 200:
                    results_json = await response.json()
                    return results_json.get("results", [])[:5]
                else:
                    logging.error(f"Failed to fetch data from Searx. Status {response.status}")
    except aiohttp.ClientError as e:
        logging.error(f"An error occurred while fetching data from Searx: {e}")
    return []

# -------------------------------------------------------------------
# Changed to use chat.completions for generate_completion
# -------------------------------------------------------------------
async def generate_completion(prompt: str) -> str:
    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=os.getenv("LLM"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=2048,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ),
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Failed to generate completion: {e}")
        return "Sorry, an error occurred while generating the response."

# -------------------------------------------------------------------
# Searching + Summarizing, chunked streaming
# -------------------------------------------------------------------
async def search_and_summarize(query: str, channel: discord.TextChannel):
    search_results = await query_searx(query)
    if not search_results:
        await channel.send("No search results found.")
        return

    for result in search_results:
        url = result.get('url', 'No URL')
        webpage_text = await scrape_website(url)
        if webpage_text == "Failed to scrape the website.":
            await channel.send(f"Unfortunately, scraping the website at {url} has failed. Please try another source.")
            continue

        cleaned_content = clean_text(webpage_text)
        prompt = (
            f"\n[<think>Webpage scrape to be used for summarization: {cleaned_content} "
            "Use this as search and augmentation data for summarization and link citation. Provide full links formatted for discord.</think>]\n "
        )
        summary = await generate_completion(prompt)
        chunks = chunk_text(summary, max_length=EMBED_MAX_LENGTH)

        response_msgs = []
        response_msg_contents = []
        prev_content = None
        edit_msg_task = None
        last_msg_task_time = datetime.now().timestamp()
        embed_color = EMBED_COLOR

        reply_msg = await channel.send(
            embed=Embed(
                title=result.get('title', 'No title'),
                description="⏳",
                url=url,
                color=embed_color["incomplete"]
            )
        )
        response_msgs.append(reply_msg)
        response_msg_contents.append("")

        for chunk in chunks:
            curr_content = chunk or ""
            if prev_content:
                if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                    new_embed = Embed(
                        title=result.get('title', 'No title'),
                        description="⏳",
                        url=url,
                        color=embed_color["incomplete"]
                    )
                    new_msg = await channel.send(embed=new_embed)
                    response_msgs.append(new_msg)
                    response_msg_contents.append("")
                response_msg_contents[-1] += prev_content

                final_msg_edit = (len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == "")
                if final_msg_edit or (not edit_msg_task or edit_msg_task.done()):
                    while edit_msg_task and not edit_msg_task.done():
                        await asyncio.sleep(0)
                    if response_msg_contents[-1].strip():
                        embed = Embed(
                            title=result.get('title', 'No title'),
                            description=response_msg_contents[-1],
                            url=url,
                            color=(embed_color["complete"] if final_msg_edit else embed_color["incomplete"])
                        )
                        edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                        if final_msg_edit:
                            tts_bytes = await tts_request(response_msg_contents[-1])
                            if tts_bytes:
                                tts_file = File(io.BytesIO(tts_bytes), filename="sns_tts_part.mp3")
                                await response_msgs[-1].reply(content="**Audio version:**", file=tts_file)
                    last_msg_task_time = datetime.now().timestamp()
            prev_content = curr_content

        if prev_content:
            if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                leftover_embed = Embed(
                    title=result.get('title', 'No title'),
                    description="⏳",
                    url=url,
                    color=embed_color["incomplete"]
                )
                leftover_msg = await channel.send(embed=leftover_embed)
                response_msgs.append(leftover_msg)
                response_msg_contents.append("")
            response_msg_contents[-1] += prev_content

            final_embed = Embed(
                title=result.get('title', 'No title'),
                description=response_msg_contents[-1],
                url=url,
                color=embed_color["complete"]
            )
            await response_msgs[-1].edit(embed=final_embed)

            leftover_tts = await tts_request(response_msg_contents[-1])
            if leftover_tts:
                leftover_file = File(io.BytesIO(leftover_tts), filename="sns_tts_last.mp3")
                await response_msgs[-1].reply(content="**Audio version:**", file=leftover_file)

# -------------------------------------------------------------------
# fetch_youtube_transcript
# -------------------------------------------------------------------
async def fetch_youtube_transcript(url: str) -> str:
    try:
        video_id = re.search(r'v=([^&]+)', url)
        if not video_id:
            video_id = re.search(r'youtu\.be/([^?]+)', url)
        if not video_id:
            return ""
        video_id = video_id.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return clean_youtube_transcript(transcript_text)
    except NoTranscriptFound:
        logging.error("No transcript found for this video.")
    except Exception as e:
        logging.error(f"Failed to fetch transcript: {e}")
    return ""

# -------------------------------------------------------------------
# roast_and_summarize
# -------------------------------------------------------------------
async def roast_and_summarize(url: str, channel: discord.TextChannel):
    webpage_text = await scrape_website(url)
    if webpage_text == "Failed to scrape the website.":
        await channel.send(f"Unfortunately, scraping the website at {url} has failed. Please try another source.")
        return
    cleaned_content = clean_text(webpage_text)
    prompt = (
        f"\n[Webpage Scrape for Comedy Routine: {cleaned_content} "
        "Use this content to create a professional comedy routine. "
        "Make it funny, witty, and engaging. Any links provided should be full links formatted for hyperlinking.]\n"
    )
    comedy_routine = await generate_completion(prompt)
    chunks = chunk_text(comedy_routine, max_length=EMBED_MAX_LENGTH)

    response_msgs = []
    response_msg_contents = []
    prev_content = None
    edit_msg_task = None
    last_msg_task_time = datetime.now().timestamp()
    embed_color = EMBED_COLOR

    reply_msg = await channel.send(embed=Embed(
        title="Comedy Routine", description="⏳", color=embed_color["incomplete"]
    ))
    response_msgs.append(reply_msg)
    response_msg_contents.append("")

    for chunk in chunks:
        curr_content = chunk or ""
        if prev_content:
            if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                new_embed = Embed(
                    title="Comedy Routine", 
                    description="⏳", 
                    color=embed_color["incomplete"]
                )
                new_msg = await channel.send(embed=new_embed)
                response_msgs.append(new_msg)
                response_msg_contents.append("")
            response_msg_contents[-1] += prev_content
            final_msg_edit = (len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == "")
            if final_msg_edit or (not edit_msg_task or edit_msg_task.done()):
                while edit_msg_task and not edit_msg_task.done():
                    await asyncio.sleep(0)
                if response_msg_contents[-1].strip():
                    embed = Embed(
                        title="Comedy Routine",
                        description=response_msg_contents[-1],
                        color=(embed_color["complete"] if final_msg_edit else embed_color["incomplete"])
                    )
                    edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
            last_msg_task_time = datetime.now().timestamp()
        prev_content = curr_content

    if prev_content:
        if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
            leftover_embed = Embed(
                title="Comedy Routine", 
                description="⏳", 
                color=embed_color["incomplete"]
            )
            leftover_msg = await channel.send(embed=leftover_embed)
            response_msgs.append(leftover_msg)
            response_msg_contents.append("")
        response_msg_contents[-1] += prev_content
        final_embed = Embed(
            title="Comedy Routine",
            description=response_msg_contents[-1],
            url=url,
            color=embed_color["complete"]
        )
        await response_msgs[-1].edit(embed=final_embed)

    logging.info(f"Final message sent: {response_msg_contents[-1]}")
    for msg, content in zip(response_msgs, response_msg_contents):
        tts_bytes = await tts_request(content)
        if tts_bytes:
            tts_file = File(io.BytesIO(tts_bytes), filename="roast_tts.mp3")
            await msg.reply(content="**Audio version:**", file=tts_file)

# -------------------------------------------------------------------
# schedule_message, parse_time_string, reminder handlers
# -------------------------------------------------------------------
async def schedule_message(channel: discord.TextChannel, delay: int, message: str):
    await asyncio.sleep(delay)
    await channel.send(message)

def parse_time_string(time_str: str) -> int:
    time_units = {
        'hour': 3600, 'hours': 3600, 'h': 3600,
        'minute': 60, 'minutes': 60, 'min': 60, 'm': 60,
        'second': 1, 'seconds': 1, 'sec': 1, 's': 1
    }
    word_to_number = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
        'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
        'ninety': 90, 'hundred': 100, 'an': 1
    }
    pattern = re.compile(
        r'(\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
        r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|'
        r'thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|an)\b)\s*'
        r'(hour|hours|h|minute|minutes|min|m|second|seconds|sec|s)',
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
            await msg.channel.send("Invalid time format. Use formats like '1m', '1h', '2h2m', '30sec', etc.")
            return
        await msg.channel.send(f"Reminder set for {time_str} from now.")
        asyncio.create_task(schedule_reminder(msg.channel, delay, time_str, reminder_message))
    except ValueError:
        await msg.channel.send("Invalid time format. Please provide the time in a valid format.")

async def schedule_reminder(channel: discord.TextChannel, delay: int, time_str: str, reminder_message: str):
    await asyncio.sleep(delay)
    prompt = (
        f"<think>It's time to remind the user about the reminder they set. "
        f"User Reminder input text: {reminder_message}. The timer has now expired. Remind the user!</think>\n "
        f"Reminder Time! "
    )
    response = await generate_reminder(prompt)
    embed = Embed(
        title=f"Reminder for {time_str}: {reminder_message}",
        description=response,
        color=discord.Color.green()
    )
    await channel.send(embed=embed)
    tts_bytes = await tts_request(response)
    if tts_bytes:
        tts_file = File(io.BytesIO(tts_bytes), filename="reminder_tts.mp3")
        await channel.send(content="**Audio version:**", file=tts_file)

async def generate_reminder(prompt: str) -> str:
    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=os.getenv("LLM"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=256,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ),
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Failed to generate reminder: {e}")
        return "Sorry, an error occurred while generating the reminder."

async def summarize_tweets(tweets: list, channel: discord.TextChannel, title: str = "Tweet Summary") -> None:
    if not tweets:
        await channel.send("No tweets to summarize.")
        return

    # Build raw snippet from tweets (ordered chronologically).
    tweets_sorted = sorted(tweets, key=lambda x: x["timestamp"])
    snippet = ""
    for t in tweets_sorted:
        snippet += f"[{t['timestamp']}] @{t['from_user'] or 'unknown'}: {t['content']}\n"

    # Send the raw tweets embed as a separate message so it won't be edited.
    raw_embed = Embed(
        title=f"{title} - Raw Tweets",
        description=snippet,
        color=EMBED_COLOR["incomplete"]
    )
    await channel.send(embed=raw_embed)

    # Build the full prompt (instruction + raw snippet).
    prompt_prefix = (
        "You are an analyst summarizing the following user's tweets in chronological order. "
        "Focus on capturing shifts in tone, rhetorical strategy, attention, and any biases. "
        "Avoid listing each tweet; instead, weave a coherent narrative paragraph. English only.\n\n"
    )
    full_prompt = prompt_prefix + snippet

    # Send a separate summary embed with a waiting indicator.
    summary_embed = Embed(
        title=f"{title} - Summary",
        description="⏳",
        color=EMBED_COLOR["incomplete"]
    )
    summary_msg = await channel.send(embed=summary_embed)

    accumulated = ""  # Holds the streamed summary text.
    edit_interval = 1 / 1.3  # ~0.77 seconds per edit.
    last_edit_time = asyncio.get_event_loop().time()

    try:
        # Await the streaming LLM response.
        stream = await llm_client.chat.completions.create(
            model=os.getenv("LLM") or "deepseek-r1-distill-llama-8b-abliterated",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )
        async for chunk in stream:
            new_piece = chunk.choices[0].delta.content or ""
            # If appending the new piece would exceed the embed limit for the summary,
            # perform a rollover: send the current accumulation as final, then start a new summary embed.
            if len(accumulated + new_piece) > EMBED_MAX_LENGTH:
                final_text = "Summary so far:\n" + accumulated
                # Ensure final_text is within the limit.
                final_text = final_text[:EMBED_MAX_LENGTH]
                await summary_msg.edit(embed=Embed(
                    title=f"{title} - Summary (Part)",
                    description=final_text,
                    color=EMBED_COLOR["complete"]
                ))
                # Roll over: send a new summary embed and reset the accumulated text.
                summary_msg = await channel.send(embed=Embed(
                    title=f"{title} - Summary (Continued)",
                    description="⏳",
                    color=EMBED_COLOR["incomplete"]
                ))
                accumulated = new_piece
                last_edit_time = asyncio.get_event_loop().time()
            else:
                accumulated += new_piece

            now = asyncio.get_event_loop().time()
            if now - last_edit_time >= edit_interval:
                updated_text = "Summary so far:\n" + accumulated
                # If the update text exceeds the limit, trim it.
                if len(updated_text) > EMBED_MAX_LENGTH:
                    updated_text = updated_text[:EMBED_MAX_LENGTH]
                await summary_msg.edit(embed=Embed(
                    title=f"{title} - Summary",
                    description=updated_text,
                    color=EMBED_COLOR["complete"]
                ))
                last_edit_time = now

        # Final update when streaming completes.
        final_text = "Summary:\n" + accumulated
        if len(final_text) > EMBED_MAX_LENGTH:
            final_text = final_text[:EMBED_MAX_LENGTH]
        await summary_msg.edit(embed=Embed(
            title=f"{title} - Summary",
            description=final_text,
            color=EMBED_COLOR["complete"]
        ))
    except Exception as e:
        logging.error(f"Error streaming tweet summary: {e}")
        if accumulated:
            partial_text = "Partial Summary (due to error):\n" + accumulated
            if len(partial_text) > EMBED_MAX_LENGTH:
                partial_text = partial_text[:EMBED_MAX_LENGTH]
            await summary_msg.edit(embed=Embed(
                title=f"{title} - Summary",
                description=partial_text,
                color=EMBED_COLOR["complete"]
            ))
        else:
            await summary_msg.edit(embed=Embed(
                title=f"{title} - Summary",
                description="An error occurred during summarization.",
                color=discord.Color.red()
            ))






async def stream_summary_to_discord(summary_text: str, channel: discord.TextChannel, title="Tweet Summary"):
    EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green()}
    text_chunks = chunk_text(summary_text, max_length=EMBED_MAX_LENGTH)
    response_msgs = []
    response_msg_contents = []
    prev_content = None
    edit_msg_task = None

    first_embed = Embed(title=title, description="⏳", color=EMBED_COLOR["incomplete"])
    first_msg = await channel.send(embed=first_embed)
    response_msgs.append(first_msg)
    response_msg_contents.append("")

    for chunk in text_chunks:
        if prev_content:
            if len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                new_embed = Embed(title=title, description="⏳", color=EMBED_COLOR["incomplete"])
                new_msg = await channel.send(embed=new_embed)
                response_msgs.append(new_msg)
                response_msg_contents.append("")
            response_msg_contents[-1] += prev_content
            final_msg_edit = (len(response_msg_contents[-1] + chunk) > EMBED_MAX_LENGTH or chunk == "")
            if final_msg_edit or (not edit_msg_task or edit_msg_task.done()):
                while edit_msg_task and not edit_msg_task.done():
                    await asyncio.sleep(0)
                if response_msg_contents[-1].strip():
                    embed = Embed(
                        title=title,
                        description=response_msg_contents[-1],
                        color=EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]
                    )
                    edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                    if final_msg_edit:
                        tts_bytes = await tts_request(response_msg_contents[-1])
                        if tts_bytes:
                            tts_file = File(io.BytesIO(tts_bytes), filename="tweets_summary_part.mp3")
                            await response_msgs[-1].reply(content="**Audio version:**", file=tts_file)
            last_msg_task_time = datetime.now().timestamp()
        prev_content = chunk

    if prev_content:
        if len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
            leftover_embed = Embed(title=title, description="⏳", color=EMBED_COLOR["incomplete"])
            leftover_msg = await channel.send(embed=leftover_embed)
            response_msgs.append(leftover_msg)
            response_msg_contents.append("")
        response_msg_contents[-1] += prev_content
        final_embed = Embed(
            title=title,
            description=response_msg_contents[-1],
            color=EMBED_COLOR["complete"]
        )
        await response_msgs[-1].edit(embed=final_embed)
        leftover_tts = await tts_request(response_msg_contents[-1])
        if leftover_tts:
            leftover_file = File(io.BytesIO(leftover_tts), filename="tweets_summary_last.mp3")
            await response_msgs[-1].reply(content="**Audio version:**", file=leftover_file)

# -------------------------------------------------------------------
# Single-embed Searx summary
# -------------------------------------------------------------------
async def do_search_and_summarize_in_one_embed(query: str, channel: discord.TextChannel):
    search_results = await query_searx(query)
    if not search_results:
        await channel.send(f"No search results found for: {query}")
        return

    lines = []
    for r in search_results:
        t = r.get('title', 'No title')
        u = r.get('url', 'No URL')
        s = r.get('content', 'No snippet available')
        lines.append(f"Title: {t}\nURL: {u}\nSnippet: {s}")
    search_summary = "\n".join(lines)

    prompt = (
        f"<think>Use this system-side search data in summarizing for the user. Provide links if needed.\n"
        f"Search data:\n{search_summary}\n</think>\n"
        "Instruction: Summarize and provide links!"
    )
    try:
        resp = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=os.getenv("LLM"),
                messages=[{"role": "system", "content": prompt}],
                max_tokens=2048,
                stream=False
            ),
            timeout=30
        )
        final_text = resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in search summarization: {e}")
        final_text = "An error occurred during search summarization."

    text_chunks = chunk_text(final_text, EMBED_MAX_LENGTH)
    response_msgs = []
    response_msg_contents = []
    prev_content = None
    edit_msg_task = None
    title_for_embed = f"Search summary for: {query}"
    first_embed = Embed(title=title_for_embed, description="⏳", color=EMBED_COLOR["incomplete"])
    first_msg = await channel.send(embed=first_embed)
    response_msgs.append(first_msg)
    response_msg_contents.append("")
    last_msg_task_time = datetime.now().timestamp()
    for chunk in text_chunks:
        if prev_content:
            if len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                new_embed = Embed(title=title_for_embed, description="⏳", color=EMBED_COLOR["incomplete"])
                new_msg = await channel.send(embed=new_embed)
                response_msgs.append(new_msg)
                response_msg_contents.append("")
            response_msg_contents[-1] += prev_content
            final_msg_edit = (len(response_msg_contents[-1] + chunk) > EMBED_MAX_LENGTH or chunk == "")
            if final_msg_edit or (not edit_msg_task or edit_msg_task.done()):
                while edit_msg_task and not edit_msg_task.done():
                    await asyncio.sleep(0)
                if response_msg_contents[-1].strip():
                    embed_upd = Embed(
                        title=title_for_embed,
                        description=response_msg_contents[-1],
                        color=EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]
                    )
                    edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed_upd))
                    if final_msg_edit:
                        tts_bytes = await tts_request(response_msg_contents[-1])
                        if tts_bytes:
                            tts_file = File(io.BytesIO(tts_bytes), filename="search_chunk.mp3")
                            await response_msgs[-1].reply(content="**Audio version:**", file=tts_file)
            last_msg_task_time = datetime.now().timestamp()
        prev_content = chunk
    if prev_content:
        if len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
            extra_embed = Embed(title=title_for_embed, description="⏳", color=EMBED_COLOR["incomplete"])
            new_msg = await channel.send(embed=extra_embed)
            response_msgs.append(new_msg)
            response_msg_contents.append("")
        response_msg_contents[-1] += prev_content
        final_embed = Embed(
            title=title_for_embed,
            description=response_msg_contents[-1],
            color=EMBED_COLOR["complete"]
        )
        await response_msgs[-1].edit(embed=final_embed)
        leftover_tts = await tts_request(response_msg_contents[-1])
        if leftover_tts:
            leftover_file = File(io.BytesIO(leftover_tts), filename="search_final.mp3")
            await response_msgs[-1].reply(content="**Audio version:**", file=leftover_file)

# -------------------------------------------------------------------
# Whisper-based voice transcription
# -------------------------------------------------------------------
def transcribe_audio(file_path: str) -> str:
    whisper_model = whisper.load_model("tiny")
    try:
        result = whisper_model.transcribe(file_path)
        transcription = result["text"]
    finally:
        del whisper_model
        torch.cuda.empty_cache()
        gc.collect()
    return transcription

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

async def handle_voice_command(transcription: str, channel: discord.TextChannel) -> bool:
    logging.info(f"Transcription: {transcription}")
    await channel.send(embed=Embed(title="User Transcription", description=transcription, color=discord.Color.blue()))
    
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
                f"Title: {result.get('title', 'No title')}\n"
                f"URL: {result.get('url', 'No URL')}\n"
                f"Snippet: {result.get('content', 'No snippet available')}"
                for result in search_results
            ])
            prompt = (
                f"<think> Use this system-side search and retrieval augmentation data in crafting summarization for the user and link citation: {search_summary}. "
                f"Provide links if needed.</think> Instruction: Summarize and provide links!"
            )
            query_title = f"Search summary/links for: \"{query}\" "
            try:
                response = await asyncio.wait_for(
                    llm_client.chat.completions.create(
                        model=os.getenv("LLM"),
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=1024,
                        stream=False
                    ),
                    timeout=30
                )
                final_text = response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Error in voice search summarization: {e}")
                final_text = "An error occurred during search summarization."
            embed_color = EMBED_COLOR
            embed = Embed(
                title=query_title,
                description=final_text,
                color=embed_color["complete"],
            )
            sent_msg = await channel.send(embed=embed)
            tts_bytes = await tts_request(final_text)
            if tts_bytes:
                tts_file = File(io.BytesIO(tts_bytes), filename="search_tts.mp3")
                await sent_msg.reply(content="**Audio version of the above text:**", file=tts_file)
            message_history[channel.id].append(sent_msg)
            message_history[channel.id] = message_history[channel.id][-MAX_MESSAGES:]
            msg_nodes[sent_msg.id] = MsgNode(
                {"role": "assistant", "content": final_text, "name": str(discord_client.user.id)},
                replied_to=None,
            )
        else:
            await channel.send(f"No search results found for: {query}")
        return True

    return False

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

@discord_client.event
async def on_message(msg: discord.Message):
    if msg.author == discord_client.user:
        return

    logging.info(f"Received message: {msg.content} from {msg.author.name}")
    user_warnings = set()

    if any(msg.content.lower().startswith(command) for command in IGNORE_COMMANDS):
        logging.info(f"Ignored message: {msg.content}")
        return

    # -----------------------------------------------------------------------
    # COMMAND: !gettweets <twitter_handle> [limit]
    # -----------------------------------------------------------------------
    if msg.content.lower().startswith("!gettweets "):
        parts = msg.content.split()
        if len(parts) < 2:
            await msg.channel.send("Usage: `!gettweets <twitter_handle> [limit]`")
            return
        username = parts[1].strip("@")
        limit = 10
        if len(parts) > 2:
            try:
                limit = int(parts[2])
            except ValueError:
                pass
        await msg.channel.send(f"Fetching last {limit} tweets from **@{username}**…")
        tweets = await scrape_latest_tweets(username, limit=limit)
        if not tweets:
            await msg.channel.send("No tweets found or scraping failed.")
            return
        await summarize_tweets(tweets, msg.channel, title=f"Summary of @{username}'s last tweets")
        return


    # -----------------------------------------------------------------------
    # Handle "!ap" for images
    # -----------------------------------------------------------------------
    if msg.content.startswith("!ap") and msg.attachments:
        for attachment in msg.attachments:
            if "image" in attachment.content_type:
                image_url = attachment.url
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as resp:
                        if resp.status == 200:
                            image_data = await resp.read()
                            base64_image = base64.b64encode(image_data).decode("utf-8")
                            prompt = (
                                "Describe this image in a very detailed and intricate way, "
                                "as if you were describing it to a blind person for reasons of accessibility. "
                                "Replace the main character or element in the description with a random celebrity "
                                "or popular well-known character. Use the {name} variable for this. "
                                "Begin your response with \"AP Photo, {name}, \" followed by the description.\n "
                            )
                            reply_chain = [
                                {"role": "system", "content": "</s></s>......"},
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]}
                            ]
                            logging.info(f"Processing !AP command. Chain length: {len(reply_chain)}")
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
                                        embed = Embed(description="⏳", color=EMBED_COLOR["incomplete"])
                                        for warning in sorted(user_warnings):
                                            embed.add_field(name=warning, value="", inline=False)
                                        new_msg = await reply_msg.reply(embed=embed, silent=True)
                                        response_msgs.append(new_msg)
                                        in_progress_msg_ids.append(new_msg.id)
                                        response_msg_contents.append("")
                                    response_msg_contents[-1] += prev_content
                                    final_msg_edit = (len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == "")
                                    if final_msg_edit or (not edit_msg_task or edit_msg_task.done()):
                                        while edit_msg_task and not edit_msg_task.done():
                                            await asyncio.sleep(0)
                                        if response_msg_contents[-1].strip():
                                            embed = Embed(description=response_msg_contents[-1],
                                                          color=(EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]))
                                            edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                prev_content = curr_content
                            for response_msg in response_msgs:
                                msg_nodes[response_msg.id] = MsgNode(
                                    {"role": "assistant", "content": "".join(response_msg_contents), "name": str(response_msg.id)},
                                    replied_to=msg_nodes.get(msg.id, None)
                                )
                                in_progress_msg_ids.remove(response_msg.id)
                return

    # -----------------------------------------------------------------------
    # Detect URLs and fetch associated content
    # -----------------------------------------------------------------------
    urls_detected = detect_urls(msg.content)
    webpage_texts = await asyncio.gather(*(scrape_website(url) for url in urls_detected))
    youtube_urls = [url for url in urls_detected if ("youtube.com/watch" in url or "youtu.be/" in url)]
    youtube_transcripts = await asyncio.gather(*(fetch_youtube_transcript(url) for url in youtube_urls))
    for attachment in msg.attachments:
        if "audio" in attachment.content_type or attachment.content_type == "application/ogg":
            transcription = await transcribe_audio_attachment(attachment.url)
            if transcription:
                if await handle_voice_command(transcription, msg.channel):
                    return
                msg.content = transcription

    if ((msg.channel.type != discord.ChannelType.private and discord_client.user not in msg.mentions) or
        (ALLOWED_CHANNEL_IDS and not any(x in ALLOWED_CHANNEL_IDS for x in (msg.channel.id, getattr(msg.channel, "parent_id", None)))) or
        (ALLOWED_ROLE_IDS and (msg.channel.type == discord.ChannelType.private or not [role for role in msg.author.roles if role.id in ALLOWED_ROLE_IDS])) or
        msg.author.bot or msg.channel.type == discord.ChannelType.forum):
        return

    if msg.channel.id not in message_history:
        message_history[msg.channel.id] = []

    # -----------------------------------------------------------------------
    # Handle messages with images (fallback)
    # -----------------------------------------------------------------------
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
                                {"role": "system", "content": (
                                    "A chat between a curious user and an intelligent assistance system. "
                                    "The system is equipped with a vision model that analyzes the image "
                                    "information that the user provides. The system gives helpful, detailed, "
                                    "and rational answers.\n</s> " +
                                    f"Today's date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n " +
                                    "</s>......"
                                )},
                                {"role": "user", "content": [
                                    {"type": "text", "text": (
                                        "Base Instruction: \"Describe the image in a very detailed and intricate way, "
                                        "as if you were describing it to a blind person for accessibility. "
                                        "Begin your response with: \"'Image Description':, \". "
                                        "Extended Instruction: \"Below is a user comment or request. Write a response "
                                        "that appropriately completes the request.\". " +
                                        f"User's prompt: {text_content}\n </s>......"
                                    )},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]}
                            ]
                            logging.info(f"Message contains image. Preparing to respond. Chain: {len(reply_chain)}")
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
                                        embed = Embed(description="⏳", color=EMBED_COLOR["incomplete"])
                                        for warning in sorted(user_warnings):
                                            embed.add_field(name=warning, value="", inline=False)
                                        new_msg = await reply_msg.reply(embed=embed, silent=True)
                                        response_msgs.append(new_msg)
                                        in_progress_msg_ids.append(new_msg.id)
                                        response_msg_contents.append("")
                                    response_msg_contents[-1] += prev_content
                                    final_msg_edit = (len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == "")
                                    if final_msg_edit or (not edit_msg_task or edit_msg_task.done()):
                                        while edit_msg_task and not edit_msg_task.done():
                                            await asyncio.sleep(0)
                                        if response_msg_contents[-1].strip():
                                            embed = Embed(description=response_msg_contents[-1],
                                                          color=(EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]))
                                            edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                prev_content = curr_content
                            for response_msg in response_msgs:
                                msg_nodes[response_msg.id] = MsgNode(
                                    {"role": "assistant", "content": "".join(response_msg_contents), "name": str(discord_client.user.id)},
                                    replied_to=msg_nodes.get(msg.id, None)
                                )
                                in_progress_msg_ids.remove(response_msg.id)
                return

    message_history[msg.channel.id].append(msg)
    message_history[msg.channel.id] = message_history[msg.channel.id][-MAX_MESSAGES:]

    # -----------------------------------------------------------------------
    # Additional commands: !search, !sns, !roast, !remindme, !pol
    # -----------------------------------------------------------------------
    async with msg.channel.typing():
        if msg.content.lower().startswith("!search "):
            query = msg.content[len("!search "):].strip()
            await do_search_and_summarize_in_one_embed(query, msg.channel)
            return

        if msg.content.startswith("!sns "):
            query = msg.content[len("!sns "):].strip()
            await do_search_and_summarize_in_one_embed(query, msg.channel)
            return

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

        if msg.content:
            cmd = msg.content.lower().split()[0]
            if cmd == "!toggle_search":
                search_enabled = not globals().get('search_enabled', False)
                globals()['search_enabled'] = search_enabled
                await msg.channel.send(f"Search functionality is now {'enabled' if search_enabled else 'disabled'}.")
                return
            elif cmd == "!clear_history":
                message_history[msg.channel.id].clear()
                await msg.channel.send("Message history has been cleared.")
                return
            elif cmd == "!show_history_size":
                size = len(message_history.get(msg.channel.id, []))
                await msg.channel.send(f"Current history size: {size}")
                return

        for curr_msg in message_history[msg.channel.id]:
            curr_msg_text = curr_msg.embeds[0].description if curr_msg.embeds and curr_msg.author.bot else curr_msg.content
            if curr_msg_text and curr_msg_text.startswith(discord_client.user.mention):
                curr_msg_text = curr_msg_text[len(discord_client.user.mention):].lstrip()
            curr_msg_content = [{"type": "text", "text": curr_msg_text}] if curr_msg_text else []
            curr_msg_images = [
                {"type": "image_url", "image_url": {"url": att.url, "detail": "low"}}
                for att in curr_msg.attachments if "image" in att.content_type
            ]
            curr_msg_content += curr_msg_images[:MAX_IMAGES]
            if os.getenv("LLM") == "mistral":
                curr_msg_content = curr_msg_text
            curr_msg_role = "assistant" if curr_msg.author == discord_client.user else "user"
            msg_nodes[curr_msg.id] = MsgNode(
                {"role": curr_msg_role, "content": curr_msg_content, "name": str(curr_msg.author.id)}
            )
            if len(curr_msg_images) > MAX_IMAGES:
                msg_nodes[curr_msg.id].too_many_images = True

        reply_chain = []
        for curr_node_id in sorted(msg_nodes.keys(), reverse=True):
            curr_node = msg_nodes[curr_node_id]
            reply_chain.append(curr_node.msg)
            if curr_node.too_many_images:
                user_warnings.add(MAX_IMAGE_WARNING)
            if len(reply_chain) == MAX_MESSAGES:
                user_warnings.add(MAX_MESSAGE_WARNING)
                break

        for webpage_text in webpage_texts:
            if webpage_text == "Failed to scrape the website.":
                reply_chain[0]["content"][0]["text"] += ("\n[Unfortunately, scraping the website has failed. Please try another source.]\n")
            else:
                reply_chain[0]["content"][0]["text"] += (f"\n[Webpage Scrape for Summarization: {webpage_text} Summarize this webpage and provide links if needed.]\n")

        for youtube_transcript in youtube_transcripts:
            if youtube_transcript:
                reply_chain[0]["content"][0]["text"] += (
                    f"\n[Default task: YouTube transcript content provided for summarization. "
                    f"YouTube Transcript: {youtube_transcript} Summarize this video and provide links if needed.]\n"
                )

        logging.info(f"Preparing to generate response. History size for channel {msg.channel.id}: {len(message_history[msg.channel.id])}, reply chain length: {len(reply_chain)}")

        response_msgs = []
        response_msg_contents = []
        prev_content = None
        edit_msg_task = None

        async for chunk in await llm_client.chat.completions.create(
            model=os.getenv("LLM"),
            messages=get_system_prompt() + reply_chain[::-1],
            max_tokens=MAX_COMPLETION_TOKENS,
            stream=True,
        ):
            curr_content = chunk.choices[0].delta.content or ""
            if prev_content:
                if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                    reply_msg = msg if not response_msgs else response_msgs[-1]
                    embed = Embed(description="⏳", color=EMBED_COLOR["incomplete"])
                    for warning in sorted(user_warnings):
                        embed.add_field(name=warning, value="", inline=False)
                    new_msg = await reply_msg.reply(embed=embed, silent=True)
                    response_msgs.append(new_msg)
                    in_progress_msg_ids.append(new_msg.id)
                    response_msg_contents.append("")
                response_msg_contents[-1] += prev_content
                final_msg_edit = (len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == "")
                if final_msg_edit or (not edit_msg_task or edit_msg_task.done()):
                    while edit_msg_task and not edit_msg_task.done():
                        await asyncio.sleep(0)
                    if response_msg_contents[-1].strip():
                        embed = Embed(description=response_msg_contents[-1],
                                      color=(EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"]))
                        edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                    if final_msg_edit:
                        text_for_tts = response_msg_contents[-1]
                        tts_bytes = await tts_request(text_for_tts)
                        if tts_bytes:
                            tts_file = File(io.BytesIO(tts_bytes), filename="tts_chunk.mp3")
                            await response_msgs[-1].reply(content="**Audio version of the above text:**", file=tts_file)
                        if "<think>" in text_for_tts and "</think>" in text_for_tts:
                            think_text = text_for_tts.split("<think>")[1].split("</think>")[0].strip()
                            following_text = text_for_tts.split("</think>")[1].strip()
                            think_tts = await tts_request(think_text)
                            if think_tts:
                                file_think = File(io.BytesIO(think_tts), filename="think_tts.mp3")
                                await response_msgs[-1].reply(content="**Audio version of the thoughts:**", file=file_think)
                            if following_text.strip():
                                follow_tts = await tts_request(following_text)
                                if follow_tts:
                                    file_follow = File(io.BytesIO(follow_tts), filename="follow_tts.mp3")
                                    await response_msgs[-1].reply(content="**Audio version of the reply:**", file=file_follow)
                last_msg_task_time = datetime.now().timestamp()
            prev_content = curr_content

        if prev_content:
            if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                leftover_msg = await msg.channel.send(embed=Embed(description="⏳", color=EMBED_COLOR["incomplete"]))
                response_msgs.append(leftover_msg)
                response_msg_contents.append("")
            response_msg_contents[-1] += prev_content
            embed = Embed(description=response_msg_contents[-1], color=EMBED_COLOR["complete"])
            await response_msgs[-1].edit(embed=embed)
            leftover_tts = await tts_request(response_msg_contents[-1])
            if leftover_tts:
                leftover_file = File(io.BytesIO(leftover_tts), filename="final_tts.mp3")
                await response_msgs[-1].reply(content="**Audio version of the above text:**", file=leftover_file)

        for response_msg in response_msgs:
            msg_nodes[response_msg.id] = MsgNode(
                {"role": "assistant", "content": "".join(response_msg_contents), "name": str(discord_client.user.id)},
                replied_to=msg_nodes.get(msg.id, None)
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
