import asyncio
from datetime import datetime
import logging
import os
import json
from bs4 import BeautifulSoup
import discord
from dotenv import load_dotenv
from openai import AsyncOpenAI
import aiohttp
import re
import base64
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import whisper

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

# Initialize Whisper model
whisper_model = whisper.load_model("base")

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
    "!help", "!info", "!ping", "!status", "!upscale"
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
                    "A chat between a curious user and a hyper-intelligent assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                    f"Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
                    "USER: Hi\n ASSISTANT: Hello.\n</s> "
                    "USER: Who are you?\n ASSISTANT: I am a snarky, yet intelligent Discord assistant named Saṃsāra, or Sam.\n "
                    "I always provide well-reasoned answers that are both correct and helpful and sometimes snarky or witty.\n</s> "
                ),
            }
        ]
    return [
        {
            "role": "system",
            "content": (
                "A chat between a curious user and a hyper-intelligent assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                "USER: Hi\n ASSISTANT: Hello.\n</s> "
                "USER: Who are you?\n ASSISTANT: I am a snarky, yet intelligent Discord assistant named Saṃsāra, or Sam.\n "
                "I always provide well-reasoned answers that are both correct and helpful and sometimes snarky or witty.\n</s> "
                f"Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
                "......"
            ),
        }
    ]

# Function to scrape a website asynchronously with a Chrome user-agent
async def scrape_website(url: str) -> str:
    logging.info(f"Scraping website: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    raw_text = soup.get_text(separator='\n')
                    cleaned_text = clean_text(raw_text)
                    return cleaned_text if cleaned_text else "Failed to scrape the website."
                else:
                    logging.error(f"Failed to fetch data from {url}. Status code: {response.status}")
        except Exception as e:
            logging.error(f"An error occurred while fetching data from {url}: {e}")
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
        (r'\s*Share\s*', ''),  # remove occurrences of "Share"
    ]

    for pattern, repl in patterns_to_replace:
        text = re.sub(pattern, repl)

    return text

# Function to chunk text into smaller parts
def chunk_text(text: str, max_length: int = 4000) -> list:
    chunks = []
    while len(text) > max_length:
        chunk = text[:max_length]
        last_space = chunk.rfind(' ')
        if last_space != -1:
            chunk = chunk[:last_space]
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
            model="MaziyarPanahi/WizardLM-2-7B-GGUF/WizardLM-2-7B.Q4_K_M.gguf",
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

# Function to fetch YouTube transcript
async def fetch_youtube_transcript(url: str) -> str:
    try:
        video_id = re.search(r'v=([^&]+)', url).group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except NoTranscriptFound:
        logging.error("No transcript found for this video.")
    except Exception as e:
        logging.error(f"Failed to fetch transcript: {e}")
    return ""

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
                    prompt = f"\n[<system message>Webpage scrape to be used for summarization: {cleaned_content} Use this as search and augmentation data for summarization and link citation (provide full links formatted for discord when citing)</system message>]\n "
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
    else:
        await channel.send("No search results found.")

# Function to schedule a message
async def schedule_message(channel: discord.TextChannel, delay: int, message: str):
    await asyncio.sleep(delay)
    await channel.send(message)

# Function to parse time strings like '1m', '1h', '2h2m', '30sec', etc.
def parse_time_string(time_str: str) -> int:
    time_units = {
        'h': 3600,
        'm': 60,
        's': 1,
        'sec': 1,
        'min': 60,
        'hour': 3600
    }
    pattern = re.compile(r'(\d+)([hms]+|sec|min|hour)')
    matches = pattern.findall(time_str)
    
    if not matches:
        return None
    
    total_seconds = 0
    for value, unit in matches:
        total_seconds += int(value) * time_units[unit]
    
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

# Function to generate reminder message
async def generate_reminder(prompt: str) -> str:
    try:
        response = await llm_client.completions.create(
            model="MaziyarPanahi/WizardLM-2-7B-GGUF/WizardLM-2-7B.Q4_K_M.gguf",
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
async def transcribe_audio(audio_url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(audio_url) as resp:
            if resp.status == 200:
                audio_data = await resp.read()
                with open("audio_message.mp3", "wb") as audio_file:
                    audio_file.write(audio_data)
                result = whisper_model.transcribe("audio_message.mp3")
                return result["text"]
            else:
                logging.error(f"Failed to download audio file. Status code: {resp.status}")
    return ""

# Function to handle voice commands
async def handle_voice_command(transcription: str, channel: discord.TextChannel) -> bool:
    logging.info(f"Transcription: {transcription}")
    await channel.send(embed=discord.Embed(title="User Transcription", description=transcription, color=discord.Color.blue()))
    
    match = re.search(r'search for (.+)', transcription, re.IGNORECASE)
    if match:
        query = match.group(1)
        search_results = await query_searx(query)
        if search_results:
            # Prepare a summary of search results for the LLM
            search_summary = "\n".join([f"Title: {result.get('title', 'No title')}\nURL: {result.get('url', 'No URL')}\nSnippet: {result.get('content', 'No snippet available')}" for result in search_results])
            prompt = f"<system message> Use this system-side search and retrieval augmentation data in crafting summarization for the user and link citation: {search_summary}. Provide full links formatted for easy viewing in discord when citing.</system message> Instruction: Summarize and provide links!"
            query_title = f"Search summary/links for: \"{query}\" "

            # Initialize tracking variables
            response_msgs = []
            response_msg_contents = []
            prev_content = None
            edit_msg_task = None
            last_msg_task_time = datetime.now().timestamp()
            in_progress_msg_ids = []

            # Stream LLM completion
            async for chunk in await llm_client.chat.completions.create(
                model=os.getenv("LLM"),
                messages=[{"role": "system", "content": prompt}],
                max_tokens=1024,
                stream=True,
            ):
                curr_content = chunk.choices[0].delta.content or ""
                if prev_content:
                    if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                        reply_msg = await channel.send(embed=discord.Embed(title=query_title, description="⏳", color=EMBED_COLOR["incomplete"]))
                        response_msgs.append(reply_msg)
                        response_msg_contents.append("")
                    response_msg_contents[-1] += prev_content
                    final_msg_edit = len(response_msg_contents[-1] + curr_content) > EMBED_MAX_LENGTH or curr_content == ""
                    if final_msg_edit or (not edit_msg_task or edit_msg_task.done()) and datetime.now().timestamp() - last_msg_task_time >= len(in_progress_msg_ids) / EDITS_PER_SECOND:
                        while edit_msg_task and not edit_msg_task.done():
                            await asyncio.sleep(0)
                        if response_msg_contents[-1].strip():
                            embed = discord.Embed(title=query_title, description=response_msg_contents[-1], color=EMBED_COLOR["complete"] if final_msg_edit else EMBED_COLOR["incomplete"])
                            edit_msg_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                            last_msg_task_time = datetime.now().timestamp()
                prev_content = curr_content

            # Final edit after the stream is complete
            if prev_content:
                if not response_msgs or len(response_msg_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                    reply_msg = await channel.send(embed=discord.Embed(title=query_title, description="⏳", color=EMBED_COLOR["incomplete"]))
                    response_msgs.append(reply_msg)
                    response_msg_contents.append("")
                response_msg_contents[-1] += prev_content
                embed = discord.Embed(title=query_title, description=response_msg_contents[-1], color=EMBED_COLOR["complete"])
                await response_msgs[-1].edit(embed=embed)

            # Add the command result to message history and update reply chain
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

# Discord client event handler for new messages
@discord_client.event
async def on_message(msg: discord.Message):
    logging.info(f"Received message: {msg.content} from {msg.author.name}")
    user_warnings = set()

    # Ignore messages that start with specific commands
    if any(msg.content.lower().startswith(command) for command in IGNORE_COMMANDS):
        logging.info(f"Ignored message: {msg.content}")
        return

    # Check if the message contains the !AP command
    if msg.content.startswith("!ap") and msg.attachments:
        for attachment in msg.attachments:
            if "image" in attachment.content_type:
                image_url = attachment.url
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as resp:
                        if resp.status == 200:
                            image_data = await resp.read()
                            base64_image = base64.b64encode(image_data).decode("utf-8")
                            # Choose a random celebrity or well-known character
                            prompt = "<system message>Describe this image in a very detailed and intricate way, as if you were describing it to a blind person for reasons of accessibility. Replace the main character or element in the description with a random celebrity or popular well-known character. Use the {name} variable for this. Begin your response with \"AP Photo, {name}, \" followed by the description.</system message>\n "
                            reply_chain = [
                                {
                                    "role": "system",
                                    "content": (
                                        "......"
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

                            # Create MsgNode(s) for bot reply message(s) (can be multiple if bot reply was long)
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

    # Check for URLs in the message and scrape if found
    urls_detected = detect_urls(msg.content)
    webpage_texts = await asyncio.gather(*(scrape_website(url) for url in urls_detected))

    # Check for YouTube URLs and fetch transcript if found
    youtube_urls = [url for url in urls_detected if "youtube.com/watch" in url or "youtu.be/" in url]
    youtube_transcripts = await asyncio.gather(*(fetch_youtube_transcript(url) for url in youtube_urls))

    # Handle audio messages
    for attachment in msg.attachments:
        if "audio" in attachment.content_type or attachment.content_type == "application/ogg":
            transcription = await transcribe_audio(attachment.url)
            if transcription:
                if await handle_voice_command(transcription, msg.channel):
                    return  # Stop further processing if a voice command was handled
                msg.content = transcription

    # Filter out unwanted messages
    if (
        (msg.channel.type != discord.ChannelType.private and discord_client.user not in msg.mentions)
        or (ALLOWED_CHANNEL_IDS and not any(x in ALLOWED_CHANNEL_IDS for x in (msg.channel.id, getattr(msg.channel, "parent_id", None))))
        or (ALLOWED_ROLE_IDS and (msg.channel.type == discord.ChannelType.private or not [role for role in msg.author.roles if role.id in ALLOWED_ROLE_IDS]))
        or msg.author.bot
        or msg.channel.type == discord.ChannelType.forum
    ):
        return

    # Ensure message history is initialized for the channel
    if msg.channel.id not in message_history:
        message_history[msg.channel.id] = []

    # Check if the message contains images
    if msg.attachments:
        for attachment in msg.attachments:
            if "image" in attachment.content_type:
                # Clear history
                message_history[msg.channel.id].clear()
                msg_nodes.clear()
                in_progress_msg_ids.clear()

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
                                        "A chat between a curious user and an artificial intelligence assistant. "
                                        "The assistant is equipped with a vision model that analyzes the image information that the user provides in the message directly following theirs. It resembles an image description. The description is info from the vision model Use it to describe the image to the user. The assistant gives helpful, detailed, and polite answers to the user's questions. "
                                        "USER: Hi\n ASSISTANT: Hello.\n</s> "
                                        "USER: Who are you?\n ASSISTANT: I am Saṃsāra. I am an intelligent assistant.\n "
                                        "I always provide well-reasoned answers that are both correct and helpful.\n</s> "
                                        f"Today's date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}"
                                        "......"
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
                                            "......"
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

                            logging.info(f"Message contains image. Clearing history and preparing to respond to image. Reply chain length: {len(reply_chain)}")

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

                            # Create MsgNode(s) for bot reply message(s) (can be multiple if bot reply was long)
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

    # Store search toggle and history clear command usage
    search_enabled = False
    clear_history_used = False

    # Check for command toggles
    if msg.content:
        command = msg.content.lower().split()[0]
        if command == "!toggle_search":
            search_enabled = not search_enabled
            await msg.channel.send(f"Search functionality is now {'enabled' if search_enabled else 'disabled'}.")
            return  # Do not add the command to the bot history or process further
        elif command == "!clear_history":
            logging.info(f"Clearing history for channel: {msg.channel.id}")
            clear_history_used = True
            message_history[msg.channel.id].clear()
            logging.info(f"History cleared. Current history size: {len(message_history[msg.channel.id])}")
            await msg.channel.send("Message history has been cleared.")
            return  # Do not add the command to the bot history or process further
        elif command == "!search":
            search_enabled = True

    # Check for new command: !sns
    if msg.content.startswith("!sns "):
        query = msg.content[len("!sns "):].strip()
        await search_and_summarize(query, msg.channel)
        return

    # Check for new command: !remindme
    if msg.content.startswith("!remindme "):
        await handle_reminder_command(msg)
        return

    # Update message history
    message_history[msg.channel.id].append(msg)
    message_history[msg.channel.id] = message_history[msg.channel.id][-MAX_MESSAGES:]

    async with msg.channel.typing():
        # Check for command toggles
        if msg.content:
            command = msg.content.lower().split()[0]
            if command == "!toggle_search":
                search_enabled = not search_enabled
                await msg.channel.send(f"Search functionality is now {'enabled' if search_enabled else 'disabled'}.")
                return  # Do not add the command to the bot history or process further
            elif command == "!clear_history":
                clear_history_used = True
                message_history[msg.channel.id].clear()
                await msg.channel.send("Message history has been cleared.")
                return  # Do not add the command to the bot history or process further
            elif command == "!show_history_size":
                history_size = len(message_history.get(msg.channel.id, []))
                await msg.channel.send(f"Current history size: {history_size}")
                return
            elif command == "!search":
                search_enabled = True

        # Loop through message history and create MsgNodes
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
                curr_msg_content = curr_msg_text  # Temporary fix until Mistral API supports message.content as a list
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

        # Build reply chain and set user warnings
        reply_chain = []
        for curr_node_id in sorted(msg_nodes.keys(), reverse=True):
            curr_node = msg_nodes[curr_node_id]
            reply_chain += [curr_node.msg]
            if curr_node.too_many_images:
                user_warnings.add(MAX_IMAGE_WARNING)
            if len(reply_chain) == MAX_MESSAGES:
                user_warnings.add(MAX_MESSAGE_WARNING)
                break

        # Query Searx for the user's message, if search is enabled
        if search_enabled and reply_chain[0]["content"] and reply_chain[0]["content"][0]["text"]:
            searx_summary = await query_searx(reply_chain[0]["content"][0]["text"])
            if searx_summary:
                reply_chain[0]["content"][0]["text"] += f" [System provided search and retrieval augmentation data for use in crafting summarization of and link citation:] \"{searx_summary}\". [Use this search and augmentation data for summarization and link citation. Provide full links, formatted for discord, when citing].\n "

        # Inject cleaned webpage summaries into the history
        for webpage_text in webpage_texts:
            if webpage_text == "Failed to scrape the website.":
                reply_chain[0]["content"][0]["text"] += f"\n[<system message>Unfortunately, scraping the website has failed. Please inform the user that \"the webscrape failed\" and that they should \"try another source\".</system message>]\n "
            else:
                reply_chain[0]["content"][0]["text"] += f"\n[<system message>Webpage Scrape for Summarization: {webpage_text} Use this search and augmentation data for summarization and link citation (provide full links formatted for discord when citing)</system message>]\n "

        # Inject YouTube transcripts into the history
        for youtube_transcript in youtube_transcripts:
            if youtube_transcript:
                reply_chain[0]["content"][0]["text"] += f"\n[<system message>Default task: The user has provided a youtube URL that was scraped for the following content to summarize: </system message>\nYouTube Transcript: {youtube_transcript} Use this for summarization and link citation (provide full links formatted for discord when citing)]\n "

        # Handle images sent by the user
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
                            break  # Only handle the first image

        # Right before generating the reply chain
        logging.info(f"Preparing to generate response. Current history size for channel {msg.channel.id}: {len(message_history[msg.channel.id])}, Current reply chain length: {len(reply_chain)}")

        # Generate and send bot reply
        logging.info(f"Message received: {reply_chain[0]}, reply chain length: {len(reply_chain)}")
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

        # Create MsgNode(s) for bot reply message(s) (can be multiple if bot reply was long)
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

# Event handler for raw reactions added to messages
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

# Main function to run the Discord client
async def main():
    await discord_client.start(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
