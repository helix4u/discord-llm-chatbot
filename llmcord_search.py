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
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                    f"Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
                    "USER: Hi\n ASSISTANT: Hello.\n</s> "
                    "USER: Who are you?\n ASSISTANT: I am a snarky, yet intelligent Discord assistant named Saṃsāra.\n "
                    "I always provide well-reasoned answers that are both correct and helpful and sometimes snarky or witty.\n</s> "
                ),
            }
        ]
    return [
        {
            "role": "system",
            "content": (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                "USER: Hi\n ASSISTANT: Hello.\n</s> "
                "USER: Who are you?\n ASSISTANT: I am a snarky, yet intelligent Discord assistant named Saṃsāra.\n "
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
                    return clean_text(raw_text)
                else:
                    logging.error(f"Failed to fetch data from {url}. Status code: {response.status}")
        except Exception as e:
            logging.error(f"An error occurred while fetching data from {url}: {e}")
    return ""

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
        text = re.sub(pattern, repl, text)

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

# Function to handle the search and scrape command
async def search_and_summarize(query: str, channel: discord.TextChannel):
    search_results = await query_searx(query)
    if search_results:
        for result in search_results:
            url = result.get('url', 'No URL')
            if url != 'No URL':
                webpage_text = await scrape_website(url)
                if webpage_text:
                    cleaned_content = clean_text(webpage_text)
                    summary_prompt = f"\n[Webpage Scrape for Summarization: {cleaned_content} Use this search and augmentation data for summarization and link citation (provide full links formatted for discord when citing)]\n "
                    summary = await generate_completion(summary_prompt)
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

# Discord client event handler for new messages
@discord_client.event
async def on_message(msg: discord.Message):
    logging.info(f"Received message: {msg.content} from {msg.author.name}")
    user_warnings = set()
    
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
                            prompt = "Describe this image in a very detailed and intricate way, as if you were describing it to a blind person for reasons of accessibility. Replace the main character or element in the description with a random celebrity or popular well-known character. Use the {{name}} variable for this. Begin your response with \"AP Photo, {name}, \" followed by the description.\n "
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
                return

    # Check for URLs in the message and scrape if found
    urls_detected = detect_urls(msg.content)
    webpage_texts = await asyncio.gather(*(scrape_website(url) for url in urls_detected))

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
            reply_chain[0]["content"][0]["text"] += f"\n[Webpage Scrape for Summarization: {webpage_text} Use this search and augmentation data for summarization and link citation (provide full links formatted for discord when citing)]\n "

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
