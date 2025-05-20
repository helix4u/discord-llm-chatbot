import os
import logging
import discord # For discord.Color
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

# Environment Variables
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1")
LLM = os.getenv("LLM", "local-model") # Default to "local-model" if not set
TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:8880/v1/audio/speech")
TTS_VOICE = os.getenv("TTS_VOICE", "af_sky+af+af_nicole")
ALLOWED_CHANNEL_IDS_STR = os.getenv("ALLOWED_CHANNEL_IDS", "")
ALLOWED_CHANNEL_IDS = [int(id_str) for id_str in ALLOWED_CHANNEL_IDS_STR.split(',') if id_str.strip()] if ALLOWED_CHANNEL_IDS_STR else []
ALLOWED_ROLE_IDS_STR = os.getenv("ALLOWED_ROLE_IDS", "")
ALLOWED_ROLE_IDS = [int(id_str) for id_str in ALLOWED_ROLE_IDS_STR.split(',') if id_str.strip()] if ALLOWED_ROLE_IDS_STR else []
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 0)) # Default to 0 (no limit) if not set or invalid
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", 10)) # Default to 10 if not set or invalid
SEARX_URL = os.getenv("SEARX_URL")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny") # Added Whisper model configuration

# Shared Constants
EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green()}
EMBED_MAX_LENGTH = 4096  # Discord embed description limit
EDITS_PER_SECOND = 1.3
MAX_COMPLETION_TOKENS = 2048 # As it was in lmcordx.py

# Dynamic warning messages based on MAX_IMAGES and MAX_MESSAGES
MAX_IMAGE_WARNING = (f"⚠️ Max {MAX_IMAGES} image{'' if MAX_IMAGES == 1 else 's'} per message"
                     if MAX_IMAGES > 0 else "Image processing is available.") # Adjusted for clarity if 0
MAX_MESSAGE_WARNING = f"⚠️ Only using last {MAX_MESSAGES} messages for context."

# Using the more extensive list from lmcordx.py
IGNORE_COMMANDS = [
    "!dream", "!d", "!background", "!avatar",
    "!help", "!info", "!ping", "!status", "!upscale", "!nightmare", "!n", "!describe", 
    "!background", "!chat", "!superprompt", "!depth", "!face", "!edges", "!lineart", 
    "!lineartanime", "!colormap", "!pose", "!esrgan", "!metadata", "!text", "!append", 
    "!models", "!loras", "!nightmarePromptGen", "!load", "!aspect", "!resolution", "!handfix",
    # Added commands that were in the original config.py's IGNORE_COMMANDS for completeness,
    # though they might be duplicates or covered by the above.
    "!toggle_search", "!clear_history", "!show_history_size" 
]
# Remove duplicates by converting to set and back to list
IGNORE_COMMANDS = sorted(list(set(IGNORE_COMMANDS)))


# Logging Setup
logging.basicConfig(
    level=logging.INFO, # Changed from DEBUG in lmcordx to INFO as a general config
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", # Added logger name
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__) # Create a logger for config.py

# Initial llm_client setup
# API key is hardcoded as "lm-studio" as per original script, this could also be an env var
llm_client = AsyncOpenAI(base_url=LOCAL_SERVER_URL, api_key="lm-studio")

# Validate essential configurations and log them
if not DISCORD_BOT_TOKEN:
    logger.critical("CRITICAL: DISCORD_BOT_TOKEN is not set. Please set it in your .env file.")
    # Consider exiting here if the token is essential for any script importing this config
    # exit() 
else:
    logger.info("DISCORD_BOT_TOKEN loaded.")

if not LOCAL_SERVER_URL:
    logger.critical("CRITICAL: LOCAL_SERVER_URL is not set. Please set it in your .env file.")
    # exit()
else:
    logger.info(f"LOCAL_SERVER_URL set to: {LOCAL_SERVER_URL}")

if not LLM:
    logger.warning("WARNING: LLM model name is not set in .env. Using default 'local-model'.")
else:
    logger.info(f"LLM model set to: {LLM}")

if ALLOWED_CHANNEL_IDS:
    logger.info(f"Bot will respond in these channels without mention: {ALLOWED_CHANNEL_IDS}")
else:
    logger.info("Bot will only respond to direct mentions or commands in all channels (unless role restricted).")

if ALLOWED_ROLE_IDS:
    logger.info(f"Bot will only respond to users with these roles: {ALLOWED_ROLE_IDS}")
else:
    logger.info("Bot will respond to all users (role restrictions disabled).")

# These checks are more about usage in specific scripts, but can be logged here for awareness
if not TTS_API_URL:
    logger.warning("WARNING: TTS_API_URL is not set. TTS features will be disabled in scripts that use it.")
else:
    logger.info(f"TTS_API_URL set to: {TTS_API_URL}")

if not SEARX_URL:
    logger.warning("WARNING: SEARX_URL is not set. Search features will be disabled in scripts that use it.")
else:
    logger.info(f"SEARX_URL set to: {SEARX_URL}")

logger.info(f"Whisper model set to: {WHISPER_MODEL}") # Log whisper model
logger.info(f"Max images per message: {MAX_IMAGES if MAX_IMAGES > 0 else 'Unlimited'}")
logger.info(f"Max messages in history: {MAX_MESSAGES}")
logger.info(f"Ignore commands list: {IGNORE_COMMANDS}")
