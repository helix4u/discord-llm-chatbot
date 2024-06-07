import asyncio
from datetime import datetime
import logging
import os
import json
from bs4 import BeautifulSoup
import discord
from dotenv import load_dotenv
from openai import AsyncOpenAI
import aiohttp  # For asynchronous web requests
import re  # For URL detection
import requests
import base64  # For image encoding

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Initialize the OpenAI client with your local AI server details
llm_client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Function to query Searx search engine
async def query_searx(query):
    print(f"Querying Searx for: {query}")
    searx_url = "http://192.168.1.3:9092/search"
    params = {
        'q': query,
        'format': 'json',
        'language': 'en-US',
    }
    try:
        response = requests.get(searx_url, params=params, timeout=5)
        if response.status_code == 200:
            results = response.json()
            if 'results' in results:
                summarized_info = []
                for result in results['results'][:10]:
                    title = result.get('title', 'No title')
                    url = result.get('url', 'No URL')
                    content = result.get('content', 'No content')
                    summarized_info.append(f"Title: {title}\nURL: {url}\nContent: {content}")
                return "\n\n".join(summarized_info)
            else:
                print("No results found in Searx response.")
        else:
            print("Failed to fetch data from Searx.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data: {e}")
    return None


# Function to generate a completion using the OpenAI client
def generate_completion(prompt):
    response = llm_client.completions.create(
        model="MaziyarPanahi/WizardLM-2-7B-GGUF/WizardLM-2-7B.Q4_K_M.gguf",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

# Load configuration for different LLM models
LLM_CONFIG = {
    "gpt": {
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": "https://api.openai.com/v1",
    },
    "mistral": {
        "api_key": os.environ["MISTRAL_API_KEY"],
        "base_url": "https://api.mistral.ai/v1",
    },
    "local": {
        "api_key": "Not used",
        "base_url": os.environ["LOCAL_SERVER_URL"],
    },
}
LLM_VISION_SUPPORT = "vision" in os.environ["LLM"]
MAX_COMPLETION_TOKENS = 2048

# Load allowed channel and role IDs from environment variables
ALLOWED_CHANNEL_IDS = [int(i) for i in os.environ["ALLOWED_CHANNEL_IDS"].split(",") if i]
ALLOWED_ROLE_IDS = [int(i) for i in os.environ["ALLOWED_ROLE_IDS"].split(",") if i]
MAX_IMAGES = int(os.environ["MAX_IMAGES"]) if LLM_VISION_SUPPORT else 0
MAX_IMAGE_WARNING = f"⚠️ Max {MAX_IMAGES} image{'' if MAX_IMAGES == 1 else 's'} per message" if MAX_IMAGES > 0 else ""
MAX_MESSAGES = int(os.environ["MAX_MESSAGES"])
MAX_MESSAGE_WARNING = f"⚠️ Only using last {MAX_MESSAGES} messages"

EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green()}
EMBED_MAX_LENGTH = 4096
EDITS_PER_SECOND = 1.3

# Initialize Discord client with intents to access message content
intents = discord.Intents.default()
intents.message_content = True
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
def get_system_prompt():
    if os.environ["LLM"] == "gpt-4-vision-preview" or "mistral" in os.environ["LLM"] or "local" in os.environ["LLM"]:
        # Temporary fix until gpt-4-vision-preview, Mistral API and LM Studio support message.name
        return [
            {
                "role": "system",
                "content": f"{os.environ['CUSTOM_SYSTEM_PROMPT']}\nToday's date: {datetime.now().strftime('%B %d %Y')}",
            }
        ]
    return [
        {
            "role": "system",
            "content": f"{os.environ['CUSTOM_SYSTEM_PROMPT']}\nUser's names are their Discord IDs and should be typed as '<@ID>'.\nToday's date: {datetime.now().strftime('%B %d %Y')}",
        }
    ]

# Function to scrape a website asynchronously
async def scrape_website(url):
    print(f"Scraping website: {url}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    text = soup.get_text()
                    return text
                else:
                    print("Failed to fetch data from website.")
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
    return None

# Function to detect URLs in a message using regex
def detect_urls(message_text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    urls = url_pattern.findall(message_text)
    return urls

# Discord client event handler for new messages
@discord_client.event
async def on_message(msg):
    logging.info(f"Received message: {msg.content} from {msg.author.name}")
    # The initial message filtering remains the same...

    # Check for URLs in the message and scrape if found
    urls_detected = detect_urls(msg.content)
    webpage_texts = []
    for url in urls_detected:
        webpage_text = await scrape_website(url)
        if webpage_text:
            webpage_texts.append(webpage_text)

    # Filter out unwanted messages
    if (
        (msg.channel.type != discord.ChannelType.private and discord_client.user not in msg.mentions)
        or (ALLOWED_CHANNEL_IDS and not any(x in ALLOWED_CHANNEL_IDS for x in (msg.channel.id, getattr(msg.channel, "parent_id", None))))
        or (ALLOWED_ROLE_IDS and (msg.channel.type == discord.ChannelType.private or not [role for role in msg.author.roles if role.id in ALLOWED_ROLE_IDS]))
        or msg.author.bot
        or msg.channel.type == discord.ChannelType.forum
    ):
        return

    # Store search toggle and history clear command usage
    search_enabled = False
    clear_history_used = False

    # Check for command toggles
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

    # Update message history
    message_history.setdefault(msg.channel.id, [])
    message_history[msg.channel.id].append(msg)
    message_history[msg.channel.id] = message_history[msg.channel.id][-MAX_MESSAGES:]

    async with msg.channel.typing():
        # Check for command toggles
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
            if curr_msg_text.startswith(discord_client.user.mention):
                curr_msg_text = curr_msg_text[len(discord_client.user.mention) :].lstrip()
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
            if "mistral" in os.environ["LLM"]:
                # Temporary fix until Mistral API supports message.content as a list
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

        # Build reply chain and set user warnings
        reply_chain = []
        user_warnings = set()
        for curr_node_id in sorted(msg_nodes.keys(), reverse=True):
            curr_node = msg_nodes[curr_node_id]
            reply_chain += [curr_node.msg]
            if curr_node.too_many_images:
                user_warnings.add(MAX_IMAGE_WARNING)
            if len(reply_chain) == MAX_MESSAGES:
                user_warnings.add(MAX_MESSAGE_WARNING)
                break

        # Query Searx for the user's message, if search is enabled
        if search_enabled:
            searx_summary = await query_searx(reply_chain[0]["content"][0]["text"])
            if searx_summary:
                reply_chain[0]["content"][0]["text"] += f" [Search and retrieval augmentation data for summarization and link citation (provide full links formatted for discord when citing): {searx_summary}]"
        
        # Inject webpage summaries into the history
        for webpage_text in webpage_texts:
            reply_chain[0]["content"][0]["text"] += f"\n[Webpage Scrape for Summarization: {webpage_text}]"

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
            model=os.environ["LLM"],
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
                replied_to=msg_nodes[msg.id],
            )
            in_progress_msg_ids.remove(response_msg.id)

# Main function to run the Discord client
async def main():
    await discord_client.start(os.environ["DISCORD_BOT_TOKEN"])

if __name__ == "__main__":
    asyncio.run(main())
