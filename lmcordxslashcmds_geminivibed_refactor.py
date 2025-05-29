import asyncio
import base64
import gc
import io
import json
import logging
import os
import random
import re
from datetime import datetime, timedelta
from typing import Union, Optional, List, Any

import aiohttp
import discord
import torch # Ensure torch is available if whisper fp16 is used with CUDA
import whisper
from discord import app_commands # Import for slash commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from openai import AsyncOpenAI, AsyncStream # Using the direct import

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

# ChromaDB for contextual memory
import chromadb

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
class Config:
    """Configuration class for the bot."""
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
        if not self.DISCORD_BOT_TOKEN:
            raise ValueError("DISCORD_BOT_TOKEN environment variable is missing")

        self.LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1")
        self.LLM_MODEL = os.getenv("LLM", "local-model") # Used for main responses and summarization
        self.VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "llava") # Used for vision tasks and context generation

        self.ALLOWED_CHANNEL_IDS = [int(i) for i in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if i]
        self.ALLOWED_ROLE_IDS = [int(i) for i in os.getenv("ALLOWED_ROLE_IDS", "").split(",") if i]
        
        self.MAX_IMAGES_PER_MESSAGE = int(os.getenv("MAX_IMAGES_PER_MESSAGE", 1))
        self.MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", 10)) # Max number of past user/assistant messages in short-term history
        self.MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", 2048))
        
        self.TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:8880/v1/audio/speech")
        self.TTS_VOICE = os.getenv("TTS_VOICE", "af_sky+af+af_nicole")
        self.TTS_ENABLED_DEFAULT = os.getenv("TTS_ENABLED_DEFAULT", "true").lower() == "true"

        self.SEARX_URL = os.getenv("SEARX_URL", "http://192.168.1.3:9092/search")
        self.SEARX_PREFERENCES = os.getenv("SEARX_PREFERENCES", "eJx1V8uy2zYM_Zp6o4mnaRadLrzqTLftTLvXQCQsISIJhg_bul9f0JIsyrpZRNcEQRA4AA4YBQl7DoTx0qPDAOaX3_50eI_yh5J8oiJ0CssvVgSmsagJTgZcn6HHC-TEJ8MKDF7QncpSsfUGE1565t7giawotj7wY7r8BSbiyWIaWF_--fvf_04RrhgRghouv57SgBYvkYqBU8CYTYotu1Y8ahN0y3HN1MommxuGC4Mszxz603ysjWkyi18KXcLQgqHeWfm9nAd9A4lJt8u9s_RHxjC15NpESQzMQnJXcpTEqgpszOoAReiMGEDXkxPs_uihb9saIQEMnIPYlBvohm17JYMFVvBjYykEDrVMPG_k28TEoVaW040hlx_NnUZq22dSIDzoJve9ctNR6rIaMS0KXdLU95sZpdSXJCdupJGfgsAxBrzWRkR21zcSqNutFib09VLlYAhriUb8EIxbmyOpstZ9o_GJGrGLO1UWF0Mz5G5xE-VG0m3LkvdQFCSG8q_n5lk0cnr--4LIghehfMtpy9_JF7A3C0nSMe38wzRZdgIhblpfH5Xhqw5cnFixugbEJvI13SFgoymgkpxMi8vXQG4kUJUBuSsxL_u9FA-s8SnW2GHo171nPzRRDWwg1NAvO97AVKogbt7UO5YlO7HKYu9Z6xr8AboA5bPcODxPLwtyUJ0lJz-Jc_xctlocSY0QK4cMyRVhagrCkaqNjy_LVqXLXUx4DnGt2w8zBVKVeWECUB7couC8XX9NAJshCwI6rlGxRweSFw0JIgpDVEiuez53hhQsNbjfl12uHPcgIfYU1-bxusP6iKcgLNlBFZhYP2u8rQdyJz2_OzL5tVd_3MGluiCfgqZwbNsuTBsFhJGFza6G7ytWccxddikv65g9hhxfIMzcVSqsKZ9XeYKFkgd8qQl2idRnVSfXhuQLOVfuJR4nThwHHkta1oiFbUlLHwQp9gq7CYZS-hVfFME-ujt1U60io-IBTofSt5u00FvHPMZ34ZutkX5kTviuFTkHdZAKpJHSdBDz9Bb112_ffn9sYems0W18EfHDga31hXoQx1ri4FZyswlC7qYe7Vr7HjGkXCrr1W7PQSFXjIV079hVWyHbztS6d37QyE6qromTYzfZis_id3_298rhEqQVCq79kd687WL20ntmLqOfiI8sLKW1tfOM-ZDTxm8vQydpw92IXOZgQ7sESwFSzaWdFIYC6zeGKKEsJF_FQq5_Xx-dfUp35fOUzJhW2BZOKJWC1VkZdFLqatqPuhtJM4nPlWLxrxDRTvE1xz6XNncseQ9bwq5CVmPYDxWqeVjmyjY3-4XWN9vLmDhAsMh3ICyyAwxkdbd59J2EArblSvt8FbpxvTy_9mXh-tsOAksPZTjrLYKCckwyXJOM73WG-0G4xq2rmRoPQczig78yADSlw7MroidYy_qlLLUqlbX3KN0plffhu4XsoszbOFQuTJzfmvcleT18gIw8NcrEqdRuZJHrPqaxaDzH5Ktb2cy87YfqKaNJpQ92uzRbaXsrj5EmBXDRyFjStZmgHY2VIKVwppUWtlerN1l6Ml4KSI_zsjoPHJOwHMrTWqKfB8ROgaGdX973IC_iw7YMh1YNqMbDjmDQSp5HnOKLFn7iQ0F9XhysLCkVdNX8n5ZJ3u9GHmefaJqrPOSvfPQxQHmCtvKKlThsGbknGXhSmJf_AWSO7BY=&q=%s")

        self.EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green(), "error": discord.Color.red()}
        self.EMBED_MAX_LENGTH = 4096 
        self.EDITS_PER_SECOND = 1.3 

        # ChromaDB Configuration
        self.CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "chat_history")
        
        # User-Provided Context
        self.USER_PROVIDED_CONTEXT = os.getenv("USER_PROVIDED_CONTEXT", "") # User-defined context to be injected

config = Config()

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# ChromaDB Client Initialization
# -------------------------------------------------------------------
try:
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    chat_history_collection = chroma_client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
    logger.info(f"ChromaDB initialized. Path: {config.CHROMA_DB_PATH}, Collection: {config.CHROMA_COLLECTION_NAME}")
except Exception as e:
    logger.critical(f"Failed to initialize ChromaDB: {e}", exc_info=True)
    chroma_client = None
    chat_history_collection = None

# -------------------------------------------------------------------
# Bot Initialization
# -------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
intents.guilds = True

bot = commands.Bot(command_prefix=commands.when_mentioned_or("!"), intents=intents) 

llm_client = AsyncOpenAI(base_url=config.LOCAL_SERVER_URL, api_key="lm-studio") # Ensure API key is set if needed

message_history = {} 
reminders = [] 

class MsgNode:
    """Represents a message in the conversation history for LLM context."""
    def __init__(self, role: str, content: Any, name: Optional[str] = None): 
        self.role = role 
        self.content = content 
        self.name = name

    def to_dict(self):
        """Converts MsgNode to a dictionary format suitable for OpenAI API."""
        data = {"role": self.role}
        if not isinstance(self.content, (str, list)):
            data["content"] = str(self.content) 
            logger.warning(f"MsgNode content was not str or list, converted to str: {type(self.content)}")
        else:
            data["content"] = self.content
        
        if self.name:
            data["name"] = str(self.name)
        return data

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def get_system_prompt() -> MsgNode:
    """Returns the system prompt for the LLM."""
    return MsgNode(
        role="system",
        content=(
            "You are Sam, a hyper-intelligence designed to learn and evolve. Provide helpful, detailed, and rational answers. "
            "Consider the context, make a plan, and evaluate for logical failings before responding. "
            "Conclude reflections with a thought-provoking question or idea when it may sound natural. "
            "If you need to 'think' before responding, use <think>Your thoughts here...</think> tags. Do NOT use emojis." 
            f"Current Date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}"
        )
    )

def _build_initial_prompt_messages(
    user_query_content: Union[str, List[dict]],
    user_id: Optional[str] = None,
    channel_id: Optional[int] = None,
    summarized_chroma_context_str: Optional[str] = None
) -> List[MsgNode]:
    """
    Constructs the initial list of MsgNode objects for the LLM prompt.
    Includes main system prompt, user-provided global context, ChromaDB summary,
    short-term message history, and the current user query.
    Manages history trimming.
    """
    prompt_list: List[MsgNode] = []

    # 1. Add the main system prompt
    prompt_list.append(get_system_prompt())

    # 2. Add User-Provided Global Context (if any)
    if config.USER_PROVIDED_CONTEXT:
        prompt_list.append(MsgNode(role="system", content=f"User-Set Global Context:\n{config.USER_PROVIDED_CONTEXT}"))
        # logger.info("Added User-Provided Context to LLM prompt structure.") # Logged when used

    # 3. Add Summarized ChromaDB Context (if any)
    if summarized_chroma_context_str:
        context_text_for_prompt = (
            "The following is a summary of a relevant past conversation. "
            "Use it to provide a more informed response.\n\n"
            "--- Relevant Retrieved Context Summary ---\n" 
            + summarized_chroma_context_str +
            "\n--- End Summary ---"
        )
        prompt_list.append(MsgNode(role="system", content=context_text_for_prompt))
        # logger.info("Added LLM-summarized context from ChromaDB to prompt structure.") # Logged when used
    
    # 4. Add short-term message history (from previous turns in the channel)
    # This history should already be trimmed to config.MAX_MESSAGE_HISTORY by the calling function
    # or by how message_history[channel_id] is maintained.
    history_to_add: List[MsgNode] = []
    if channel_id and channel_id in message_history:
        history_to_add = message_history[channel_id] # This is already trimmed
    
    # 5. Add the current user's message
    current_user_msg = MsgNode("user", user_query_content, name=str(user_id) if user_id else None)

    # Combine system prompts, trimmed history, and current user message
    # System prompts are already at the beginning of prompt_list.
    # We need to ensure the conversational part (history_to_add + current_user_msg) is correctly appended and trimmed if necessary.
    
    # The number of actual conversational turns to keep is config.MAX_MESSAGE_HISTORY.
    # This includes previous user/assistant pairs.
    # The current user message is one part of a new turn.
    
    # Concatenate all parts: system prompts, then historical conversation, then current user message.
    # System prompts are already in prompt_list.
    
    # `history_to_add` should contain pairs of user/assistant messages.
    # We want to ensure that the total number of these historical messages
    # doesn't exceed config.MAX_MESSAGE_HISTORY. Since message_history[channel_id]
    # is already trimmed to MAX_MESSAGE_HISTORY items (not pairs), this should be fine.
    
    final_prompt_list = prompt_list + history_to_add + [current_user_msg]
    
    # Now, trim the *entire* list if it's too long, prioritizing recent messages
    # while trying to keep initial system prompts.
    # This is a complex trimming logic. A simpler approach:
    # Count non-system messages (history + current user query).
    # If this count > MAX_MESSAGE_HISTORY + 1 (for current query), trim the oldest *conversational* messages.

    # Let's count initial system messages
    num_initial_system_prompts = 0
    for msg_node in final_prompt_list:
        if msg_node.role == "system":
            num_initial_system_prompts += 1
        else:
            break # Stop counting once a non-system message is found
            
    initial_system_msgs = final_prompt_list[:num_initial_system_prompts]
    conversational_msgs = final_prompt_list[num_initial_system_prompts:]

    # Trim the conversational_msgs part if it's too long.
    # config.MAX_MESSAGE_HISTORY refers to individual MsgNodes (user or assistant).
    # The current user message is already included in conversational_msgs.
    if len(conversational_msgs) > config.MAX_MESSAGE_HISTORY : # If history + current query > max
        # Keep the last config.MAX_MESSAGE_HISTORY items from conversational_msgs
        trimmed_conversational_msgs = conversational_msgs[-config.MAX_MESSAGE_HISTORY:]
    else:
        trimmed_conversational_msgs = conversational_msgs
        
    return initial_system_msgs + trimmed_conversational_msgs


def chunk_text(text: str, max_length: int = config.EMBED_MAX_LENGTH) -> List[str]:
    """Chunks text into smaller parts for Discord embeds, respecting lines."""
    if not text: return [""]
    chunks = []
    current_chunk = ""
    for line in text.splitlines(keepends=True):
        if len(current_chunk) + len(line) > max_length:
            if current_chunk: 
                chunks.append(current_chunk)
            current_chunk = line
            while len(current_chunk) > max_length: 
                chunks.append(current_chunk[:max_length])
                current_chunk = current_chunk[max_length:]
        else:
            current_chunk += line
    if current_chunk: 
        chunks.append(current_chunk)
    return chunks if chunks else [""]

def detect_urls(message_text: str) -> List[str]:
    if not message_text: return []
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(message_text)

def clean_text_for_tts(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[\*#_~\<\>\[\]\(\)]+', '', text) # Remove common markdown
    text = re.sub(r'http[s]?://\S+', '', text) # Remove URLs
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE) # Remove think tags
    return text.strip()

async def _send_audio_segment(destination: Union[discord.abc.Messageable, discord.Interaction, discord.Message], 
                                segment_text: str, filename_suffix: str, 
                                is_thought: bool = False, base_filename: str = "response"):
    """Internal helper to process and send a single audio segment."""
    if not segment_text:
        return
    cleaned_segment = clean_text_for_tts(segment_text)
    if not cleaned_segment:
        logger.info(f"Skipping TTS for empty/cleaned {filename_suffix} segment.")
        return

    tts_audio_data = await tts_request(cleaned_segment)
    
    actual_destination_channel: Optional[discord.abc.Messageable] = None
    if isinstance(destination, discord.Interaction):
        actual_destination_channel = destination.channel
    elif isinstance(destination, discord.Message):
        actual_destination_channel = destination.channel
    elif isinstance(destination, discord.abc.Messageable): # Covers TextChannel, DMChannel, Thread
        actual_destination_channel = destination
    
    if not actual_destination_channel:
        logger.warning(f"TTS destination channel could not be resolved for type {type(destination)}")
        return

    if tts_audio_data:
        try:
            # Attempt to fix potential MP3 header issues by re-exporting with pydub
            audio = AudioSegment.from_file(io.BytesIO(tts_audio_data), format="mp3")
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format="mp3", bitrate="128k") # Standard bitrate
            fixed_audio_data = output_buffer.getvalue()

            file = discord.File(io.BytesIO(fixed_audio_data), filename=f"{base_filename}_{filename_suffix}.mp3")
            
            content_message = None
            if is_thought:
                content_message = "**Sam's thoughts (TTS):**"
            elif filename_suffix == "main_response" or filename_suffix == "full": 
                content_message = "**Sam's response (TTS):**"
            
            await actual_destination_channel.send(content=content_message, file=file)
            logger.info(f"Sent TTS audio: {base_filename}_{filename_suffix}.mp3 to channel {actual_destination_channel.id}")
        except Exception as e:
            logger.error(f"Error processing or sending TTS for {filename_suffix}: {e}", exc_info=True)
    else:
        logger.warning(f"TTS request failed for {filename_suffix} segment.")

async def send_tts_audio(destination: Union[discord.abc.Messageable, discord.Interaction, discord.Message], text_to_speak: str, base_filename: str = "response"):
    """
    Generates TTS audio and sends it. If <think> tags are present,
    sends thoughts and main response as separate audio files.
    """
    if not config.TTS_ENABLED_DEFAULT or not text_to_speak:
        return

    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    match = think_pattern.search(text_to_speak)

    if match:
        thought_text = match.group(1).strip()
        response_text = text_to_speak[match.end():].strip() 
        
        logger.info("Found <think> tags. Processing thoughts and response separately for TTS.")
        await _send_audio_segment(destination, thought_text, "thoughts", is_thought=True, base_filename=base_filename)
        await asyncio.sleep(0.5) # Small delay between sending two audio files
        await _send_audio_segment(destination, response_text, "main_response", is_thought=False, base_filename=base_filename)
    else:
        logger.info("No <think> tags found. Processing full text for TTS.")
        await _send_audio_segment(destination, text_to_speak, "full", is_thought=False, base_filename=base_filename)

# -------------------------------------------------------------------
# ChromaDB and Context Management
# -------------------------------------------------------------------

def get_context_from_chromadb(query: str, n_results: int = 1) -> List[str]:
    """Queries ChromaDB for conversations similar to the given query."""
    if not chat_history_collection:
        logger.warning("ChromaDB not available, skipping context retrieval.")
        return []
    try:
        results = chat_history_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results and results['documents'] else []
    except Exception as e:
        logger.error(f"Failed to query ChromaDB: {e}", exc_info=True)
        return []

def ingest_conversation_to_chromadb(channel_id: int, user_id: int, conversation_history: List[MsgNode]):
    """Ingests a completed conversation into ChromaDB."""
    if not chat_history_collection:
        logger.warning("ChromaDB not available, skipping ingestion.")
        return

    # Don't ingest if the history is too short or just the system prompt
    # conversation_history here is the full prompt sent to LLM, including system messages.
    # We need to check if there's at least one user and one assistant message *within the conversational part*.
    
    # Filter out system messages to check conversational length
    non_system_messages = [msg for msg in conversation_history if msg.role in ['user', 'assistant']]
    if len(non_system_messages) < 2: # Needs at least one user and one assistant message
        logger.debug(f"Skipping ChromaDB ingestion for short conversation (non-system messages: {len(non_system_messages)}).")
        return

    try:
        # Combine messages into a single document for ChromaDB.
        # Include system messages as they provide context for the conversation.
        full_conversation_text_parts = []
        for msg in conversation_history:
            if isinstance(msg.content, str):
                full_conversation_text_parts.append(f"{msg.role}: {msg.content}")
            elif isinstance(msg.content, list): # Handle multi-modal content for user messages
                text_parts_for_chroma = [part["text"] for part in msg.content if part["type"] == "text"]
                if text_parts_for_chroma:
                    full_conversation_text_parts.append(f"{msg.role}: {' '.join(text_parts_for_chroma)}")
                else: # e.g. image only message
                    full_conversation_text_parts.append(f"{msg.role}: [Media content, no text part for ChromaDB]")
        
        full_conversation_text = "\n".join(full_conversation_text_parts)
        
        if not full_conversation_text.strip():
            logger.info("Skipping ingestion of empty conversation text to ChromaDB.")
            return

        # Use a unique ID based on channel, user, and timestamp
        doc_id = f"channel_{channel_id}_user_{user_id}_{int(datetime.now().timestamp())}"
        
        metadata = {
            "channel_id": str(channel_id),
            "user_id": str(user_id),
            "timestamp": datetime.now().isoformat()
        }

        chat_history_collection.add(
            documents=[full_conversation_text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        logger.info(f"Ingested conversation into ChromaDB with ID: {doc_id}")
    except Exception as e:
        logger.error(f"Failed to ingest conversation into ChromaDB: {e}", exc_info=True)

async def get_summarized_context(past_conversation: str, current_query: str) -> Optional[str]:
    """
    Uses an LLM to summarize a past conversation in the context of a new query.
    """
    logger.info("Generating contextual summary for retrieved memory...")
    
    summarizer_prompt_content = (
        f"You are a context summarization expert. Below is a user's current query and a "
        f"past conversation. Your task is to read the past conversation and write a concise, "
        f"one-paragraph summary of ONLY the key points that are relevant to the user's CURRENT query. "
        f"Do not answer the user's query itself. Focus on extracting relevant facts or discussion points.\n\n"
        f"CURRENT QUERY: \"{current_query}\"\n\n"
        f"PAST CONVERSATION:\n---\n{past_conversation}\n---\n\n"
        f"CONCISE RELEVANT SUMMARY:"
    )

    try:
        response = await llm_client.chat.completions.create(
            model=config.LLM_MODEL, # Use the main text model for summarization
            messages=[{"role": "system", "content": summarizer_prompt_content}], # This prompt is self-contained
            max_tokens=300, 
            temperature=0.2, 
            stream=False 
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated contextual summary: {summary[:100]}...")
            return summary
        else:
            logger.warning("Context summarization returned no content.")
            return None
    except Exception as e:
        logger.error(f"Failed to generate contextual summary: {e}", exc_info=True)
        return None

# -------------------------------------------------------------------
# Core LLM Interaction Logic
# -------------------------------------------------------------------

async def get_context_aware_llm_stream(prompt_messages: List[MsgNode], is_vision_request: bool) -> tuple[Optional[AsyncStream], str]:
    """
    Performs the two-step LLM call using the provided prompt_messages.
    1. Generate a "suggested context" for the user's query (last message in prompt_messages).
    2. Stream the final response using a copy of prompt_messages, with the "suggested context"
       injected into the last user message.
    Assumes prompt_messages already contains all necessary system prompts (main, user-global, Chroma).
    """
    if not prompt_messages:
        raise ValueError("Prompt messages cannot be empty for get_context_aware_llm_stream.")

    last_user_message_node = next((msg for msg in reversed(prompt_messages) if msg.role == 'user'), None)
    if not last_user_message_node:
        # This should not happen if _build_initial_prompt_messages always adds a user message.
        logger.error("No user message found in prompt_messages for get_context_aware_llm_stream.")
        raise ValueError("No user message found in the prompt history for context generation.")

    logger.info("Step 1: Generating suggested context (model-generated)...")
    # The system prompt for this specific context generation step is fixed.
    context_generation_system_prompt = MsgNode(
        role="system",
        content=(
            "You are a context analysis expert. Your task is to read the user's question or statement "
            "and generate a concise 'suggested context' for viewing it. This context should clarify "
            "underlying assumptions, define key terms, or establish a frame of reference that will "
            "lead to the most insightful and helpful response. Do not answer the user's question. "
            "Only provide a single, short paragraph for the suggested context."
        )
    )
    # The context generation LLM call only needs its specific system prompt and the last user message.
    context_generation_llm_input = [context_generation_system_prompt, last_user_message_node]

    generated_context = "Context generation failed or was not applicable." # Default
    try:
        context_response = await llm_client.chat.completions.create(
            model=config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL,
            messages=[msg.to_dict() for msg in context_generation_llm_input],
            max_tokens=250,
            stream=False,
            temperature=0.4,
        )
        if context_response.choices and context_response.choices[0].message.content:
            generated_context = context_response.choices[0].message.content.strip()
            logger.info(f"Successfully generated model context: {generated_context[:150]}...")
        else:
            logger.warning("Model-generated context step returned no content.")
    except Exception as e:
        logger.error(f"Could not generate model-suggested context: {e}", exc_info=True)

    logger.info("Step 2: Streaming final response with injected model-generated context.")
    # Create a deep copy of the original full prompt_messages to avoid modifying it in place.
    # This prompt_messages already contains main system prompt, user-global, Chroma, history.
    final_prompt_messages_for_stream = [
        MsgNode(m.role, m.content.copy() if isinstance(m.content, list) else m.content, m.name) 
        for m in prompt_messages
    ]
    
    # Find the last user message node IN THE COPIED LIST to inject the model-generated context.
    final_user_message_node_in_copy = next((msg for msg in reversed(final_prompt_messages_for_stream) if msg.role == 'user'), None)
    
    if not final_user_message_node_in_copy: 
        logger.error("Critical error: final_user_message_node_in_copy is None in get_context_aware_llm_stream. This indicates an issue with prompt construction.")
        # Fallback: return None for stream, but still provide the generated_context for display.
        return None, generated_context

    original_question_text = ""
    if isinstance(final_user_message_node_in_copy.content, str):
        original_question_text = final_user_message_node_in_copy.content
    elif isinstance(final_user_message_node_in_copy.content, list):
        text_part = next((part['text'] for part in final_user_message_node_in_copy.content if part['type'] == 'text'), "")
        original_question_text = text_part

    # Prepare the text to inject into the user's message.
    injected_text_for_user_message = (
        f"<model_generated_suggested_context>\n{generated_context}\n</model_generated_suggested_context>\n\n"
        f"<user_question>\nWith all prior context (including global, retrieved, and the suggested context above) in mind, Sam, please respond to the following:\n{original_question_text}\n</user_question>"
    )

    # Update the content of the last user message node in the copied list.
    if isinstance(final_user_message_node_in_copy.content, str):
        final_user_message_node_in_copy.content = injected_text_for_user_message
    elif isinstance(final_user_message_node_in_copy.content, list):
        text_part_found_and_updated = False
        for part_idx, part in enumerate(final_user_message_node_in_copy.content):
            if part['type'] == 'text':
                # Replace the existing text part with the new injected content
                final_user_message_node_in_copy.content[part_idx] = {"type": "text", "text": injected_text_for_user_message}
                text_part_found_and_updated = True
                break
        if not text_part_found_and_updated: # If no text part existed (e.g., image-only message originally)
            final_user_message_node_in_copy.content.insert(0, {"type": "text", "text": injected_text_for_user_message})

    # Determine the model for the final streaming call
    final_stream_model = config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL
    logger.info(f"Using model for final streaming response: {final_stream_model}")
    # logger.debug(f"Final prompt messages for streaming: {[m.to_dict() for m in final_prompt_messages_for_stream]}")


    try:
        final_llm_stream = await llm_client.chat.completions.create(
            model=final_stream_model,
            messages=[msg.to_dict() for msg in final_prompt_messages_for_stream], # Use the modified copy
            max_tokens=config.MAX_COMPLETION_TOKENS,
            stream=True,
            temperature=0.7, # Or your preferred temperature for final response
        )
        return final_llm_stream, generated_context # Return the model-generated context for display
    except Exception as e:
        logger.error(f"Failed to create LLM stream for final response: {e}", exc_info=True)
        return None, generated_context # Still return generated_context for display even if stream fails


async def _stream_llm_handler(
    interaction_or_message: Union[discord.Interaction, discord.Message], 
    prompt_messages: List[MsgNode], # This is the fully constructed prompt from _build_initial_prompt_messages
    title: str,
    initial_message_to_edit: Optional[discord.Message] = None,
    summarized_chroma_context: Optional[str] = None # Primarily for display prefix now
) -> str:
    sent_messages: List[discord.Message] = []
    full_response_content = ""
    
    is_interaction = isinstance(interaction_or_message, discord.Interaction)
    channel = interaction_or_message.channel 
    if not channel: 
        logger.error(f"_stream_llm_handler: Channel is None for {type(interaction_or_message)} ID {interaction_or_message.id}.")
        return ""

    current_initial_message: Optional[discord.Message] = None
    if initial_message_to_edit:
        current_initial_message = initial_message_to_edit
    else: 
        placeholder_embed = discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"])
        try:
            if is_interaction:
                current_initial_message = await interaction_or_message.followup.send(embed=placeholder_embed, wait=True)
            else: 
                logger.error("_stream_llm_handler: initial_message_to_edit is None for a non-interaction. This is unexpected.")
                return ""
        except discord.HTTPException as e:
            logger.error(f"Failed to send initial followup for stream '{title}': {e}")
            return "" 
    
    if current_initial_message:
        sent_messages.append(current_initial_message)
    else: 
        logger.error(f"Failed to establish an initial message for streaming title '{title}'.")
        return ""

    response_prefix = "" 
    try:
        is_vision_request = any(
            isinstance(p.content, list) and any(c.get("type") == "image_url" for c in p.content)
            for p in prompt_messages # Check the fully built prompt
        )
        # `prompt_messages` here is already fully constructed by _build_initial_prompt_messages
        stream, model_generated_context_for_display = await get_context_aware_llm_stream(prompt_messages, is_vision_request)

        # Construct the response prefix for display in Discord
        prefix_parts = []
        if summarized_chroma_context: # This is the one passed for display
            prefix_parts.append(f"**Retrieved & Summarized Context:**\n> {summarized_chroma_context.replace(chr(10), ' ').strip()}\n\n---")
        
        if config.USER_PROVIDED_CONTEXT:
            prefix_parts.append(f"**User-Provided Context:**\n> {config.USER_PROVIDED_CONTEXT.replace(chr(10), ' ').strip()}\n\n---")

        # Use the model_generated_context_for_display returned by get_context_aware_llm_stream
        prefix_parts.append(f"**Model-Generated Suggested Context:**\n> {model_generated_context_for_display.replace(chr(10), ' ').strip()}\n\n---")
        prefix_parts.append("**Response:**\n")
        response_prefix = "\n".join(prefix_parts)

        if stream is None: 
            error_text = response_prefix + "Failed to get response from LLM."
            error_embed = discord.Embed(title=title, description=error_text, color=config.EMBED_COLOR["error"])
            if sent_messages: await sent_messages[0].edit(embed=error_embed)
            return ""

        last_edit_time = asyncio.get_event_loop().time()
        accumulated_delta_for_update = "" 

        if current_initial_message: 
            initial_context_embed = discord.Embed(title=title, description=response_prefix + "⏳ Thinking...", color=config.EMBED_COLOR["incomplete"])
            try:
                await current_initial_message.edit(embed=initial_context_embed)
            except discord.HTTPException as e:
                logger.warning(f"Failed to edit initial message with context for '{title}': {e}")

        async for chunk_data in stream:
            delta_content = chunk_data.choices[0].delta.content or "" if chunk_data.choices and chunk_data.choices[0].delta else ""
            if delta_content: 
                full_response_content += delta_content
                accumulated_delta_for_update += delta_content

            current_time = asyncio.get_event_loop().time()
            if accumulated_delta_for_update and (current_time - last_edit_time >= (1.0 / config.EDITS_PER_SECOND) or len(accumulated_delta_for_update) > 200) : 
                display_text = response_prefix + full_response_content
                text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
                # logger.info(f"Stream update: Title '{title}'. Chunks: {len(text_chunks)}. Sent msgs: {len(sent_messages)}. AccumDelta: {len(accumulated_delta_for_update)}")
                accumulated_delta_for_update = "" 

                for i, chunk_content_part in enumerate(text_chunks):
                    embed = discord.Embed(
                        title=title if i == 0 else f"{title} (cont.)",
                        description=chunk_content_part, # Renamed to avoid conflict
                        color=config.EMBED_COLOR["incomplete"]
                    )
                    try:
                        if i < len(sent_messages):
                            await sent_messages[i].edit(embed=embed)
                        else:
                            if channel: 
                                new_msg = await channel.send(embed=embed)
                                sent_messages.append(new_msg)
                            else: 
                                logger.error(f"Cannot send overflow chunk {i+1} for '{title}': channel is None.")
                                break 
                    except discord.HTTPException as e_edit_send: 
                        logger.warning(f"Failed edit/send embed part {i+1} (stream) for '{title}': {e_edit_send}")
                last_edit_time = current_time
        
        if accumulated_delta_for_update: # Process any remaining accumulated text
            display_text = response_prefix + full_response_content
            text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
            # logger.info(f"Final stream accumulation processing: Title '{title}'. Chunks: {len(text_chunks)}. Sent msgs: {len(sent_messages)}.")
            for i, chunk_content_part in enumerate(text_chunks): # Renamed
                embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content_part, color=config.EMBED_COLOR["incomplete"])
                try:
                    if i < len(sent_messages): await sent_messages[i].edit(embed=embed)
                    else:
                        if channel: sent_messages.append(await channel.send(embed=embed))
                except discord.HTTPException as e: logger.warning(f"Failed final accum. edit/send for '{title}': {e}")

        final_display_text = response_prefix + full_response_content
        final_chunks = chunk_text(final_display_text, config.EMBED_MAX_LENGTH)
        # logger.info(f"Finalizing stream: Title '{title}'. Final Chunks: {len(final_chunks)}. Current sent msgs: {len(sent_messages)}.")

        if len(final_chunks) < len(sent_messages): # Delete extra messages if response shortened
            for k in range(len(final_chunks), len(sent_messages)):
                try: await sent_messages[k].delete()
                except discord.HTTPException: pass 
            sent_messages = sent_messages[:len(final_chunks)]
        
        for i, chunk_content_part in enumerate(final_chunks): # Renamed
            embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content_part, color=config.EMBED_COLOR["complete"])
            if i < len(sent_messages):
                await sent_messages[i].edit(embed=embed)
            else: 
                if channel:
                    new_msg = await channel.send(embed=embed)
                    sent_messages.append(new_msg)
                else:
                    logger.error(f"Cannot send final color overflow chunk {i+1} for '{title}': channel is None.")
                    break

        if not full_response_content.strip() and sent_messages: 
            empty_text = response_prefix + ("\nSam didn't provide a response." if model_generated_context_for_display != "Context generation failed or was not applicable." else "\nSam had an issue and couldn't respond.")
            await sent_messages[0].edit(embed=discord.Embed(title=title, description=empty_text, color=config.EMBED_COLOR["error"]))

    except Exception as e:
        logger.error(f"Error in _stream_llm_handler for '{title}': {e}", exc_info=True)
        error_prefix_for_display = response_prefix if 'response_prefix' in locals() and response_prefix else ""
        error_embed = discord.Embed(title=title, description=error_prefix_for_display + f"An error occurred: {str(e)[:1000]}", color=config.EMBED_COLOR["error"])
        if sent_messages: 
            try: await sent_messages[0].edit(embed=error_embed)
            except discord.HTTPException: pass 
        elif is_interaction: 
            try: await interaction_or_message.followup.send(embed=error_embed, ephemeral=True)
            except discord.HTTPException: pass 
            
    return full_response_content

async def stream_llm_response_to_interaction(
    interaction: discord.Interaction, 
    prompt_messages: list, # This is the fully constructed prompt
    title: str = "Sam's Response", 
    force_new_followup_flow: bool = False,
    summarized_chroma_context: Optional[str] = None # For display prefix
):
    initial_msg_for_handler: Optional[discord.Message] = None
    if not force_new_followup_flow:
        try:
            if not interaction.response.is_done(): 
                await interaction.response.defer(ephemeral=False) 
            initial_msg_for_handler = await interaction.original_response()
            
            is_placeholder = False
            if initial_msg_for_handler and initial_msg_for_handler.embeds: 
                current_embed = initial_msg_for_handler.embeds[0]
                if current_embed.title == title and current_embed.description and "⏳ Generating context..." in current_embed.description:
                    is_placeholder = True
            
            if not is_placeholder and initial_msg_for_handler: 
                await initial_msg_for_handler.edit(embed=discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"]))
        except discord.HTTPException as e:
            logger.error(f"Error defer/get original_response for interaction '{title}': {e}")
            force_new_followup_flow = True 
            initial_msg_for_handler = None 
    
    if force_new_followup_flow:
        initial_msg_for_handler = None

    # prompt_messages is already fully built by the calling slash command using _build_initial_prompt_messages
    full_response_content = await _stream_llm_handler(
        interaction_or_message=interaction,
        prompt_messages=prompt_messages,
        title=title,
        initial_message_to_edit=initial_msg_for_handler,
        summarized_chroma_context=summarized_chroma_context 
    )

    if full_response_content:
        channel_id = interaction.channel_id
        if channel_id not in message_history: message_history[channel_id] = []
        
        # prompt_messages contains the full prompt sent to LLM, including the user's query part.
        # The last message in prompt_messages should be the user's query.
        user_message_node_for_history = next((msg for msg in reversed(prompt_messages) if msg.role == 'user'), None)

        if user_message_node_for_history:
             # We need the *original* user query content, not the one modified by get_context_aware_llm_stream
             # This is tricky because prompt_messages was modified.
             # For now, let's assume the last user message in the *input* prompt_messages is what we want.
             # This part needs careful review if prompt_messages is modified before this point.
             # The `prompt_messages` passed to `_stream_llm_handler` is the one built by `_build_initial_prompt_messages`.
             # The `get_context_aware_llm_stream` makes a *copy* and modifies that copy.
             # So, the `prompt_messages` here should still have the original user query as its last user message.
            message_history[channel_id].append(user_message_node_for_history)
        
        assistant_response_node = MsgNode(role="assistant", content=full_response_content, name=str(bot.user.id))
        message_history[channel_id].append(assistant_response_node)
        message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:] # Trim
        
        # For ChromaDB, use the prompt_messages that was actually sent (which includes system contexts)
        # and the final assistant response.
        chroma_ingest_history = list(prompt_messages) 
        chroma_ingest_history.append(assistant_response_node)
        ingest_conversation_to_chromadb(channel_id, interaction.user.id, chroma_ingest_history)
        
        tts_base_id = interaction.id 
        if initial_msg_for_handler: 
            tts_base_id = initial_msg_for_handler.id
        
        await send_tts_audio(interaction, full_response_content, f"interaction_{tts_base_id}")

async def stream_llm_response_to_message(
    target_message: discord.Message, 
    prompt_messages: list, # This is the fully constructed prompt
    title: str = "Sam's Response",
    summarized_chroma_context: Optional[str] = None # For display prefix
):
    initial_embed = discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"])
    reply_message: Optional[discord.Message] = None
    try:
        reply_message = await target_message.reply(embed=initial_embed, silent=True) 
    except discord.HTTPException as e:
        logger.error(f"Failed to send initial reply for message stream '{title}': {e}")
        return 

    # prompt_messages is already fully built by on_message using _build_initial_prompt_messages
    full_response_content = await _stream_llm_handler(
        interaction_or_message=target_message, 
        prompt_messages=prompt_messages,
        title=title,
        initial_message_to_edit=reply_message,
        summarized_chroma_context=summarized_chroma_context
    )

    if full_response_content:
        channel_id = target_message.channel.id
        # message_history[channel_id] was already updated with the user's current message in on_message
        # Now add assistant's response
        assistant_response_node = MsgNode(role="assistant", content=full_response_content, name=str(bot.user.id))
        message_history[channel_id].append(assistant_response_node)
        message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:] # Trim

        # For ChromaDB, use the prompt_messages that was actually sent 
        # (which includes system contexts and the current user message)
        # and the final assistant response.
        chroma_ingest_history = list(prompt_messages) 
        chroma_ingest_history.append(assistant_response_node)
        ingest_conversation_to_chromadb(channel_id, target_message.author.id, chroma_ingest_history)
        
        await send_tts_audio(target_message.channel, full_response_content, base_filename=f"message_{target_message.id}")


# -------------------------------------------------------------------
# Text-to-Speech (TTS)
# -------------------------------------------------------------------
async def tts_request(text: str, speed: float = 1.3) -> Optional[bytes]:
    if not text: return None
    payload = { "input": text, "voice": config.TTS_VOICE, "response_format": "mp3", "speed": speed }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(config.TTS_API_URL, json=payload, timeout=30) as resp:
                if resp.status == 200: return await resp.read()
                else: logger.error(f"TTS request failed: status={resp.status}, resp={await resp.text()}"); return None
    except asyncio.TimeoutError: logger.error("TTS request timed out."); return None
    except Exception as e: logger.error(f"TTS request error: {e}", exc_info=True); return None

# -------------------------------------------------------------------
# Web Scraping and Search
# -------------------------------------------------------------------
JS_EXPAND_SHOWMORE_TWITTER = """
(maxClicks) => {
    let clicks = 0;
    const getButtons = () => Array.from(document.querySelectorAll('[role="button"]'))
        .filter(b => {
            const t = (b.textContent || '').toLowerCase();
            if (!t.includes('show more')) { return false; }
            const article = b.closest('article');
            if (!article) { return false; }
            // Avoid clicking "Show more" on Grok results or blockquotes
            const articleText = article.textContent || '';
            if (articleText.match(/grok/i)) { return false; } 
            if (b.closest('[role="blockquote"]')) { return false; } 
            return true;
        });
    while (clicks < maxClicks) {
        const buttonsToClick = getButtons();
        if (buttonsToClick.length === 0) break;
        const button = buttonsToClick[0];
        try { button.click(); clicks++; } catch (e) { break; } // If click fails, stop
    }
    return clicks;
}
"""

JS_EXTRACT_TWEETS_TWITTER = """
() => {
    const tweets = [];
    document.querySelectorAll('article[data-testid="tweet"]').forEach(article => {
        try {
            const timeTag = article.querySelector('time');
            const timestamp = timeTag ? timeTag.getAttribute('datetime') : null;
            let tweetLink = null, id = '', username = 'unknown_user';
            
            // Try to get link from the timestamp's parent anchor first
            const primaryLinkElement = timeTag ? timeTag.closest('a[href*="/status/"]') : null;
            if (primaryLinkElement) {
                tweetLink = primaryLinkElement.href;
            } else {
                // Fallback: find any link in the article that looks like a status link
                const articleLinks = Array.from(article.querySelectorAll('a[href*="/status/"]'));
                if (articleLinks.length > 0) {
                    // Prefer links not pointing to photos/videos within the tweet if possible
                    tweetLink = articleLinks.find(link => !link.href.includes("/photo/") && !link.href.includes("/video/"))?.href || articleLinks[0].href;
                }
            }

            if (tweetLink) {
                const match = tweetLink.match(/\/([a-zA-Z0-9_]+)\/status\/(\d+)/);
                if (match) { username = match[1]; id = match[2]; }
            }
            
            const tweetTextElement = article.querySelector('div[data-testid="tweetText"]');
            const content = tweetTextElement ? tweetTextElement.innerText.trim() : '';

            const socialContextElement = article.querySelector('div[data-testid="socialContext"]');
            let is_repost = false, reposted_by = null;
            if (socialContextElement && /reposted|retweeted/i.test(socialContextElement.innerText)) {
                is_repost = true;
                const userLinkInContext = socialContextElement.querySelector('a[href^="/"]');
                if (userLinkInContext) {
                    // Extract username from the href, avoiding known non-username parts
                    const hrefParts = userLinkInContext.href.split('/');
                    reposted_by = hrefParts.filter(part => !['analytics', 'likes', 'media', 'status', 'with_replies', 'following', 'followers', ''].includes(part)).pop();
                }
            }

            // Only add if there's content or media (photo/video)
            if (content || article.querySelector('[data-testid="tweetPhoto"], [data-testid="videoPlayer"]')) {
                tweets.push({
                    id: id || `no-id-${Date.now()}-${Math.random()}`, // Fallback ID
                    username, 
                    content, 
                    timestamp: timestamp || new Date().toISOString(), // Fallback timestamp
                    is_repost, 
                    reposted_by,
                    tweet_url: tweetLink || (id ? `https://x.com/${username}/status/${id}` : '')
                });
            }
        } catch (e) { /* console.warn('Error extracting tweet:', e); */ }
    });
    return tweets;
}
"""

async def scrape_website(url: str) -> Optional[str]:
    logger.info(f"Scraping website: {url}")
    user_data_dir = os.path.join(os.getcwd(), ".pw-profile") 
    profile_dir_usable = True
    if not os.path.exists(user_data_dir):
        try: os.makedirs(user_data_dir, exist_ok=True)
        except OSError as e: profile_dir_usable = False; logger.error(f"Could not create .pw-profile directory: {e}")
            
    context_manager = None 
    browser_instance_sw = None # For non-persistent context
    page = None
    try:
        async with async_playwright() as p:
            if profile_dir_usable:
                context = await p.chromium.launch_persistent_context(
                    user_data_dir, headless=True, 
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                )
            else: 
                logger.warning("Using non-persistent context for scrape_website.")
                browser_instance_sw = await p.chromium.launch(
                    headless=True, 
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
                )
                context = await browser_instance_sw.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    java_script_enabled=True,
                    ignore_https_errors=True
                )
            context_manager = context 
            page = await context_manager.new_page()
            await page.goto(url, wait_until='domcontentloaded', timeout=25000) 
            
            content_selectors = ["article", "main", "div[role='main']", "body"]
            content = ""
            for selector in content_selectors:
                try:
                    element = page.locator(selector).first
                    if await element.count() > 0 and await element.is_visible(): 
                        content = await element.inner_text(timeout=5000) 
                        if content and len(content.strip()) > 200: break 
                except PlaywrightTimeoutError: logger.debug(f"Timeout for selector {selector} on {url}")
                except Exception as e_sel: logger.warning(f"Error with selector {selector} on {url}: {e_sel}")
            
            if not content or len(content.strip()) < 100 : content = await page.evaluate('document.body.innerText')
            return re.sub(r'\s\s+', ' ', content.strip()) if content else None
    except PlaywrightTimeoutError:
        logger.error(f"Playwright timed out scraping {url}")
        return "Scraping timed out."
    except Exception as e:
        logger.error(f"Playwright failed for {url}: {e}", exc_info=True)
        return "Failed to scrape the website due to an error."
    finally:
        if page and not page.is_closed():
            try: await page.close()
            except Exception: pass 
        if context_manager: 
            try: await context_manager.close()
            except Exception as e_ctx: 
                if "Target page, context or browser has been closed" not in str(e_ctx): 
                    logger.debug(f"Ignoring error closing context for {url}: {e_ctx}") # Debug for this common case
        if browser_instance_sw and not profile_dir_usable: 
            try: await browser_instance_sw.close()
            except Exception: pass

async def scrape_latest_tweets(username_queried: str, limit: int = 10) -> List[dict]:
    logger.info(f"Scraping last {limit} tweets for @{username_queried} (with_replies) with JS enhancement.")
    tweets_collected = []
    seen_tweet_ids = set() 
    user_data_dir = os.path.join(os.getcwd(), ".pw-profile")
    profile_dir_usable = True
    if not os.path.exists(user_data_dir):
        try: os.makedirs(user_data_dir, exist_ok=True)
        except OSError: profile_dir_usable = False; logger.error(f"Could not create .pw-profile. Using non-persistent context.")
    
    context_manager = None; browser_instance_st = None; page = None
    try:
        async with async_playwright() as p:
            if profile_dir_usable:
                # logger.info(f"Using persistent profile: {user_data_dir}") # Less verbose
                context = await p.chromium.launch_persistent_context(
                    user_data_dir, headless=True, 
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
                    slow_mo=150 
                )
            else:
                logger.warning("Using non-persistent context for tweet scraping.")
                browser_instance_st = await p.chromium.launch(
                    headless=True, 
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                    slow_mo=150
                )
                context = await browser_instance_st.new_context(
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
                )
            context_manager = context; page = await context_manager.new_page()
            
            url = f"https://x.com/{username_queried.lstrip('@')}/with_replies" 
            logger.info(f"Navigating to {url}")
            await page.goto(url, timeout=60000, wait_until="domcontentloaded") 
            
            try:
                await page.wait_for_selector("article[data-testid='tweet']", timeout=30000)
                logger.info("Initial tweet articles detected.")
                await asyncio.sleep(1.5); await page.keyboard.press("Escape"); await asyncio.sleep(0.5); await page.keyboard.press("Escape")
            except PlaywrightTimeoutError: 
                logger.warning(f"Timed out waiting for initial tweet articles for @{username_queried}. Site structure might have changed or profile is protected/empty.")
                return [] 
            
            max_scroll_attempts = limit + 15 
            for scroll_attempt in range(max_scroll_attempts):
                if len(tweets_collected) >= limit: break
                # logger.debug(f"Tweet scrape attempt {scroll_attempt + 1}/{max_scroll_attempts}. Collected: {len(tweets_collected)}/{limit}")
                
                try: 
                    clicked_count = await page.evaluate(JS_EXPAND_SHOWMORE_TWITTER, 5) 
                    if clicked_count > 0: logger.info(f"Clicked {clicked_count} 'Show more' elements."); await asyncio.sleep(1.5 + random.uniform(0.3, 0.9)) 
                except Exception as e_sm: logger.warning(f"JS 'Show More' error: {e_sm}")
                
                extracted_this_round = []; newly_added_count = 0
                try: extracted_this_round = await page.evaluate(JS_EXTRACT_TWEETS_TWITTER)
                except Exception as e_js: logger.error(f"JS tweet extraction error: {e_js}")
                
                for data in extracted_this_round: 
                    uid = data.get('id') or (data.get("username","") + (data.get("content") or "")[:30] + data.get("timestamp","")) 
                    if uid and uid not in seen_tweet_ids:
                        tweets_collected.append(data) 
                        seen_tweet_ids.add(uid); newly_added_count +=1
                        if len(tweets_collected) >= limit: break
                # logger.info(f"Extracted {len(extracted_this_round)}, added {newly_added_count} new unique tweets.")
                
                if newly_added_count == 0 and scroll_attempt > (limit // 2 + 7): 
                    logger.info("No new unique tweets found in several attempts. Assuming end or stuck."); break
                
                await page.evaluate("window.scrollBy(0, window.innerHeight * 1.5);"); await asyncio.sleep(random.uniform(3.0, 5.0)) 
    except PlaywrightTimeoutError as e: logger.warning(f"Playwright overall timeout for @{username_queried}: {e}")
    except Exception as e: logger.error(f"Unexpected error scraping tweets for @{username_queried}: {e}", exc_info=True)
    finally:
        if page and not page.is_closed(): 
            try: await page.close() 
            except Exception as e_page_close_final: logger.debug(f"Ignoring error closing page (final attempt) for @{username_queried}: {e_page_close_final}")
        if context_manager: 
            try: await context_manager.close()
            except Exception as e_ctx_final:
                if "Target page, context or browser has been closed" not in str(e_ctx_final):
                    logger.debug(f"Error closing context (final attempt) for @{username_queried}: {e_ctx_final}")
        if browser_instance_st and not profile_dir_usable: 
            try: await browser_instance_st.close()
            except Exception as e_browser_final: logger.debug(f"Ignoring error closing browser (final attempt) for @{username_queried}: {e_browser_final}")
                
    tweets_collected.sort(key=lambda x: x.get("timestamp", ""), reverse=True) 
    logger.info(f"Finished scraping. Collected {len(tweets_collected)} tweets for @{username_queried}.")
    return tweets_collected[:limit]

async def query_searx(query: str) -> List[dict]:
    logger.info(f"Querying Searx for: {query}")
    params = {'q': query, 'format': 'json', 'language': 'en-US'}
    if config.SEARX_PREFERENCES: params['preferences'] = config.SEARX_PREFERENCES
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config.SEARX_URL, params=params, timeout=10) as response:
                response.raise_for_status() 
                results_json = await response.json()
                return results_json.get('results', [])[:5] 
    except aiohttp.ClientError as e: logger.error(f"Searx query failed for '{query}': {e}")
    except json.JSONDecodeError: logger.error(f"Failed to decode JSON from Searx for '{query}'")
    return []

async def fetch_youtube_transcript(url: str) -> Optional[str]:
    try:
        video_id_match = re.search(r'(?:v=|\/|embed\/|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})', url)
        if not video_id_match: logger.warning(f"No YouTube video ID from URL: {url}"); return None
        video_id = video_id_match.group(1)
        logger.info(f"Fetching YouTube transcript for ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        try: transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            try: transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
            except NoTranscriptFound:
                available_langs = [t.language for t in transcript_list]
                if available_langs:
                    logger.warning(f"No English transcript for {video_id}. Available: {available_langs}. Trying first available.")
                    transcript = transcript_list.find_generated_transcript([available_langs[0]]) 
        if transcript:
            fetched_data = transcript.fetch()
            full_text = " ".join([entry['text'] for entry in fetched_data])
            return f"(Language: {transcript.language}) {full_text}" if transcript.language != 'en' else full_text
        else: logger.warning(f"No transcript found for YouTube video: {url}"); return None
    except Exception as e: logger.error(f"Failed to fetch YouTube transcript for {url}: {e}", exc_info=True); return None

def transcribe_audio_file(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path): logger.error(f"Audio file not found: {file_path}"); return None
    try:
        logger.info(f"Loading Whisper model (base) for: {file_path}")
        model = whisper.load_model("base") 
        logger.info(f"Transcribing audio: {file_path}")
        result = model.transcribe(file_path, fp16=torch.cuda.is_available()) 
        logger.info(f"Transcription successful for {file_path}.")
        return result["text"]
    except Exception as e: logger.error(f"Whisper transcription failed for {file_path}: {e}", exc_info=True); return None
    finally:
        if 'model' in locals(): del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

@tasks.loop(seconds=30) 
async def check_reminders():
    now = datetime.now()
    due_indices = [i for i, r_tuple in enumerate(reminders) if now >= r_tuple[0]] 
    for index in sorted(due_indices, reverse=True): 
        reminder_time, channel_id, user_id, message_content, original_time_str = reminders.pop(index)
        logger.info(f"Reminder DUE for user {user_id} in channel {channel_id}: {message_content}")
        try:
            channel = await bot.fetch_channel(channel_id)
            user = await bot.fetch_user(user_id) 
            if channel and user and isinstance(channel, discord.abc.Messageable): # Added isinstance check
                embed = discord.Embed(title=f"⏰ Reminder! (Set {original_time_str})", description=message_content, color=discord.Color.blue(), timestamp=reminder_time)
                embed.set_footer(text=f"Reminder for {user.display_name}")
                await channel.send(content=user.mention, embed=embed) 
                await send_tts_audio(channel, f"Reminder for {user.display_name}: {message_content}", base_filename=f"reminder_{user_id}_{channel_id}")
            else: logger.warning(f"Could not fetch channel/user or channel not messageable for reminder: ChID {channel_id}, UserID {user_id}")
        except discord.errors.NotFound: logger.warning(f"Channel/User not found for reminder: ChID {channel_id}, UserID {user_id}.")
        except Exception as e: logger.error(f"Failed to send reminder (ChID {channel_id}, UserID {user_id}): {e}", exc_info=True)

def parse_time_string_to_delta(time_str: str) -> tuple[Optional[timedelta], Optional[str]]:
    """Parses a time string (e.g., '1d2h30m') into a timedelta and a descriptive string."""
    patterns = {
        'd': r'(\d+)\s*d(?:ay(?:s)?)?', 
        'h': r'(\d+)\s*h(?:our(?:s)?|r(?:s)?)?', 
        'm': r'(\d+)\s*m(?:inute(?:s)?|in(?:s)?)?', 
        's': r'(\d+)\s*s(?:econd(?:s)?|ec(?:s)?)?'
    }
    delta_args = {'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
    original_parts = [] 
    time_str_processed = time_str.lower() 

    for key, pattern_regex in patterns.items():
        for match in re.finditer(pattern_regex, time_str_processed):
            value = int(match.group(1))
            unit_full = {'d': 'days', 'h': 'hours', 'm': 'minutes', 's': 'seconds'}[key]
            delta_args[unit_full] += value
            original_parts.append(f"{value} {unit_full.rstrip('s') if value == 1 else unit_full}")
        time_str_processed = re.sub(pattern_regex, "", time_str_processed) 

    if not any(val > 0 for val in delta_args.values()): return None, None 
    
    time_delta = timedelta(**delta_args)
    descriptive_str = ", ".join(original_parts) if original_parts else "immediately" 
    if not descriptive_str and time_delta.total_seconds() > 0 : descriptive_str = "a duration" 
    return time_delta, descriptive_str

# -------------------------------------------------------------------
# ChatGPT Ingestion Functions
# -------------------------------------------------------------------
def parse_chatgpt_export(json_file_path: str) -> List[dict]:
    """
    Parses a ChatGPT conversations.json file and extracts all conversations.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            conversations_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"ChatGPT export file not found: {json_file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in ChatGPT export file: {json_file_path}")
        return []

    extracted_conversations = []
    for convo in conversations_data:
        title = convo.get('title', 'Untitled')
        create_time_ts = convo.get('create_time')
        create_time = datetime.fromtimestamp(create_time_ts) if create_time_ts else datetime.now() 
        
        messages = []
        current_node_id = convo.get('current_node')
        mapping = convo.get('mapping', {})

        while current_node_id:
            node = mapping.get(current_node_id)
            if not node: break 
            
            message_data = node.get('message')
            if message_data and message_data.get('content') and message_data['content']['content_type'] == 'text':
                author_role = message_data['author']['role']
                text_parts = message_data['content'].get('parts', [])
                text_content = "".join(text_parts) if isinstance(text_parts, list) else ""
                
                if text_content and author_role in ['user', 'assistant', 'system']: 
                    messages.append({'role': author_role, 'content': text_content})
            
            current_node_id = node.get('parent') 
        
        messages.reverse() 
        if messages:
            extracted_conversations.append({'title': title, 'create_time': create_time, 'messages': messages})
    return extracted_conversations

def store_conversations_in_chromadb(conversations: List[dict], source: str = "chatgpt_export") -> int:
    """Stores a list of parsed conversations in ChromaDB."""
    if not chat_history_collection:
        logger.error("Cannot store conversations, ChromaDB not available.")
        return 0
    
    documents, metadatas, ids = [], [], []
    for i, convo in enumerate(conversations):
        full_conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in convo['messages']])
        if not full_conversation_text.strip(): continue 

        documents.append(full_conversation_text)
        metadatas.append({
            "title": convo['title'],
            "source": source,
            "create_time": convo['create_time'].isoformat() 
        })
        ids.append(f"{source}_{convo.get('title', 'untitled').replace(' ', '_')}_{i}_{int(convo['create_time'].timestamp())}")

    if documents:
        try:
            chat_history_collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Successfully added {len(documents)} conversations from {source} to ChromaDB.")
            return len(documents)
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
    return 0
    
# -------------------------------------------------------------------
# Slash Commands (Application Commands)
# -------------------------------------------------------------------

@bot.tree.command(name="ingest_chatgpt_export", description="Ingests a conversations.json file from a ChatGPT export.")
@app_commands.describe(file_path="The full local path to your conversations.json file.")
@app_commands.checks.has_permissions(manage_messages=True) 
async def ingest_chatgpt_export_command(interaction: discord.Interaction, file_path: str):
    await interaction.response.defer(ephemeral=True) 
    logger.info(f"Ingestion of '{file_path}' initiated by {interaction.user.name}.")
    
    if not os.path.exists(file_path):
        await interaction.followup.send(f"Error: File not found at the specified path: `{file_path}`")
        return

    parsed_conversations = parse_chatgpt_export(file_path)
    if not parsed_conversations:
        await interaction.followup.send("Could not parse any valid conversations from the file. Check file format and content.")
        return
        
    count = store_conversations_in_chromadb(parsed_conversations)
    await interaction.followup.send(f"Successfully ingested {count} conversations from the export file into ChromaDB.")

@ingest_chatgpt_export_command.error
async def ingest_export_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("You need 'Manage Messages' (or server) permission to run this command.", ephemeral=True)
    else:
        logger.error(f"Error in ingest_chatgpt_export command: {error}", exc_info=True)
        if interaction.response.is_done():
            await interaction.followup.send(f"An unexpected error occurred: {error}", ephemeral=True)
        else: 
            await interaction.response.send_message(f"An unexpected error occurred: {error}", ephemeral=True)


@bot.tree.command(name="remindme", description="Sets a reminder. E.g., 1h30m Check the oven.")
@app_commands.describe(time_duration="Duration (e.g., '10m', '2h30m', '1d').", reminder_message="The message for your reminder.")
async def remindme_slash_command(interaction: discord.Interaction, time_duration: str, reminder_message: str):
    time_delta, descriptive_time_str = parse_time_string_to_delta(time_duration)
    if not time_delta or time_delta.total_seconds() <= 0:
        await interaction.response.send_message("Invalid time duration. Format: '10m', '2h30m', '1d'. Minimum 1 second.", ephemeral=True)
        return
    reminder_time = datetime.now() + time_delta
    reminders.append((reminder_time, interaction.channel_id, interaction.user.id, reminder_message, descriptive_time_str or "later"))
    reminders.sort(key=lambda x: x[0]) 
    await interaction.response.send_message(f"Okay, {interaction.user.mention}! I'll remind you in {descriptive_time_str or 'the specified time'} about: \"{reminder_message}\"")
    logger.info(f"Reminder set for {interaction.user.name} at {reminder_time} for: {reminder_message}")

@bot.tree.command(name="roast", description="Generates a comedy routine based on a webpage.")
@app_commands.describe(url="The URL of the webpage to roast.")
async def roast_slash_command(interaction: discord.Interaction, url: str):
    logger.info(f"Roast command by {interaction.user.name} for {url}.")
    try:
        webpage_text = await scrape_website(url)
        if not webpage_text or "Failed to scrape" in webpage_text or "Scraping timed out" in webpage_text:
            msg = f"Sorry, couldn't roast {url}. {webpage_text or 'Could not retrieve content.'}"
            if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
            else: await interaction.followup.send(msg, ephemeral=True)
            return
        
        prompt_nodes = _build_initial_prompt_messages(
            user_query_content=f"Analyze content from {url} and write a comedy routine:\n{webpage_text[:3000]}",
            user_id=str(interaction.user.id)
            # No channel_id or summarized_chroma_context for this specific command's direct input
        )
        await stream_llm_response_to_interaction(interaction, prompt_nodes, title=f"Comedy Roast of {url}", force_new_followup_flow=False)
    except Exception as e:
        logger.error(f"Error in roast_slash_command for {url}: {e}", exc_info=True)
        msg = f"Error roasting {url}: {str(e)[:1000]}"
        if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
        else: await interaction.followup.send(msg, ephemeral=True)

@bot.tree.command(name="search", description="Performs a web search and summarizes results.")
@app_commands.describe(query="Your search query.")
async def search_slash_command(interaction: discord.Interaction, query: str):
    logger.info(f"Search command by {interaction.user.name} for: {query}.")
    
    if not interaction.response.is_done(): 
        try: await interaction.response.defer(ephemeral=False) 
        except discord.HTTPException as e_defer:
            logger.error(f"Search: Failed to defer interaction: {e_defer}")
            try: await interaction.response.send_message("Error starting search. Please try again.",ephemeral=True)
            except discord.HTTPException: logger.error("Search: Also failed to send error message after defer failure.")
            return 
    
    try:
        search_results = await query_searx(query)
        if not search_results:
            await interaction.followup.send("No search results found.") 
            return
            
        snippets = [f"{i+1}. **{discord.utils.escape_markdown(r.get('title','N/A'))}** (<{r.get('url','N/A')}>)\n    {discord.utils.escape_markdown(r.get('content',r.get('description','No snippet'))[:250])}..." for i, r in enumerate(search_results)]
        formatted_results = "\n\n".join(snippets)
        
        await interaction.followup.send(embed=discord.Embed(title=f"Top Search Results for: {query}", description=formatted_results[:config.EMBED_MAX_LENGTH], color=config.EMBED_COLOR["incomplete"]))

        prompt_nodes = _build_initial_prompt_messages(
            user_query_content=f"Summarize these search results for the query '{query}':\n\n{formatted_results[:3000]}",
            user_id=str(interaction.user.id)
        )
        await stream_llm_response_to_interaction(interaction, prompt_nodes, title=f"Summary for: {query}", force_new_followup_flow=True)
    except Exception as e:
        logger.error(f"Error in search_slash_command for '{query}': {e}", exc_info=True)
        try: await interaction.followup.send(f"Sorry, an error searching for '{query}': {str(e)[:1000]}", ephemeral=True)
        except Exception as e_f: logger.error(f"Further error sending search error followup for '{query}': {e_f}")

@bot.tree.command(name="pol", description="Generates a sarcastic response to a political statement.")
@app_commands.describe(statement="The political statement.")
async def pol_slash_command(interaction: discord.Interaction, statement: str):
    logger.info(f"Pol command by {interaction.user.name} for: {statement[:50]}.")
    try:
        # Specific system prompt for this command, User-Provided Global Context will be added by helper
        pol_system_content = ("You are a bot that generates extremely sarcastic, snarky, and troll-like comments "
                              "to mock extremist political views or absurd political statements. Your goal is to be biting and humorous, "
                              "undermining the statement without being directly offensive or vulgar. Focus on wit and irony.") 
        
        # We want the pol_system_content to be the primary system instruction for this task.
        # _build_initial_prompt_messages adds get_system_prompt() and config.USER_PROVIDED_CONTEXT first.
        # For this command, we might want to override get_system_prompt() or ensure pol_system_content is dominant.
        # For now, let's construct it by adding pol_system_content after the global ones.
        
        base_prompt_nodes = _build_initial_prompt_messages(
            user_query_content=f"Generate sarcastic comeback: \"{statement}\"",
            user_id=str(interaction.user.id)
        )
        # Insert the pol-specific system prompt after the global system prompts but before history/user query.
        # Find first non-system message to insert before, or append if all are system.
        insert_idx = 0
        for idx, node in enumerate(base_prompt_nodes):
            if node.role != "system":
                insert_idx = idx
                break
            insert_idx = idx + 1 # If all are system, append after last system
            
        final_prompt_nodes = base_prompt_nodes[:insert_idx] + [MsgNode("system", pol_system_content)] + base_prompt_nodes[insert_idx:]


        await stream_llm_response_to_interaction(interaction, final_prompt_nodes, title="Sarcastic Political Commentary", force_new_followup_flow=False)
    except Exception as e:
        logger.error(f"Error in pol_slash_command: {e}", exc_info=True)
        msg = f"Error with pol command: {str(e)[:1000]}"
        if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
        else: await interaction.followup.send(msg, ephemeral=True)

@bot.tree.command(name="gettweets", description="Fetches and summarizes recent tweets from a user.")
@app_commands.describe(username="The X/Twitter username (without @).", limit="Number of tweets to fetch (max 50).")
async def gettweets_slash_command(interaction: discord.Interaction, username: str, limit: app_commands.Range[int, 1, 50] = 10):
    logger.info(f"Gettweets command by {interaction.user.name} for @{username}, limit {limit}.")
    
    if not interaction.response.is_done():
        try: await interaction.response.defer(ephemeral=False) 
        except discord.HTTPException as e_defer:
            logger.error(f"Gettweets: Failed to defer interaction: {e_defer}")
            try: await interaction.response.send_message("Error starting command. Please try again.",ephemeral=True)
            except discord.HTTPException: logger.error("Gettweets: Also failed to send error message after defer failure.")
            return
    
    try:
        tweets = await scrape_latest_tweets(username.lstrip('@'), limit=limit)
        if not tweets:
            await interaction.followup.send(f"Could not fetch tweets for @{username.lstrip('@')}. Profile might be private/non-existent or X is blocking. Scraping X is very unreliable.")
            return

        tweet_texts = []
        for t in tweets:
            ts = t.get('timestamp', 'N/A')
            author = t.get('username', username.lstrip('@')) 
            content = discord.utils.escape_markdown(t.get('content', 'N/A'))
            tweet_url = t.get('tweet_url', '')
            header = f"[{ts}] @{author}"
            if t.get('is_repost') and t.get('reposted_by'):
                header = f"[{ts}] @{t.get('reposted_by')} reposted @{author}"
            link_text = f" ([Link]({tweet_url}))" if tweet_url else ""
            tweet_texts.append(f"{header}: {content}{link_text}")

        raw_tweets_display = "\n\n".join(tweet_texts)
        embed_title = f"Recent Tweets from @{username.lstrip('@')}"
        if not raw_tweets_display: raw_tweets_display = "No tweet content could be displayed."

        raw_tweet_chunks = chunk_text(raw_tweets_display, config.EMBED_MAX_LENGTH)
        for i, chunk_content_part in enumerate(raw_tweet_chunks):
            chunk_title = embed_title if i == 0 else f"{embed_title} (cont.)"
            embed = discord.Embed(title=chunk_title, description=chunk_content_part, color=config.EMBED_COLOR["incomplete"]) 
            await interaction.followup.send(embed=embed) 

        prompt_nodes_summary = _build_initial_prompt_messages(
            user_query_content=f"Summarize themes, topics, and sentiment from @{username.lstrip('@')}'s recent tweets:\n\n{raw_tweets_display[:3500]}",
            user_id=str(interaction.user.id)
        )
        await stream_llm_response_to_interaction(
            interaction, prompt_nodes_summary, 
            title=f"Tweet Summary for @{username.lstrip('@')}",
            force_new_followup_flow=True 
        )
    except Exception as e:
        logger.error(f"Error in gettweets_slash_command for @{username}: {e}", exc_info=True)
        try: await interaction.followup.send(f"Sorry, an error occurred while fetching tweets for @{username}: {str(e)[:1000]}", ephemeral=True)
        except Exception as e_f: logger.error(f"Further error in gettweets error followup for @{username}: {e_f}")

@bot.tree.command(name="ap", description="Describes an attached image with a creative AP Photo twist.")
@app_commands.describe(image="The image to describe.", user_prompt="Optional additional prompt for the description.")
async def ap_slash_command(interaction: discord.Interaction, image: discord.Attachment, user_prompt: str = ""):
    logger.info(f"AP command by {interaction.user.name}.")
    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            msg = "The attached file is not a valid image."
            if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
            else: await interaction.followup.send(msg, ephemeral=True)
            return
            
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url_for_llm = f"data:{image.content_type};base64,{base64_image}"
        
        chosen_celebrity = random.choice(["Keanu Reeves", "Dwayne Johnson", "Zendaya", "Tom Hanks", "Margot Robbie", "Ryan Reynolds"]) 
        
        # Specific system prompt for this command's task, to be added after global ones.
        ap_task_prompt_text = (f"You are an AP photo caption writer. Describe the attached image in a detailed and intricate way, "
                               f"as if for a blind person. However, creatively replace the main subject or character in the image with {chosen_celebrity}. "
                               f"Begin your response with 'AP Photo: {chosen_celebrity}...' "
                               f"If the user provided an additional prompt, consider it: '{user_prompt}'")
        
        # Construct the user query part for _build_initial_prompt_messages
        # This will be a list for multi-modal input
        user_content_for_ap = [
            {"type": "text", "text": user_prompt if user_prompt else "Describe this image with the AP Photo twist mentioned in the system prompt."}, # User prompt or default
            {"type": "image_url", "image_url": {"url": image_url_for_llm}}
        ]

        prompt_nodes = _build_initial_prompt_messages(
            user_query_content=user_content_for_ap,
            user_id=str(interaction.user.id)
        )
        # Insert the AP-specific system prompt after global system prompts.
        insert_idx = 0
        for idx, node in enumerate(prompt_nodes):
            if node.role != "system":
                insert_idx = idx
                break
            insert_idx = idx + 1 
        prompt_nodes = prompt_nodes[:insert_idx] + [MsgNode("system", ap_task_prompt_text)] + prompt_nodes[insert_idx:]


        await stream_llm_response_to_interaction(interaction, prompt_nodes, title=f"AP Photo Description ft. {chosen_celebrity}", force_new_followup_flow=False)
    except Exception as e:
        logger.error(f"Error in ap_slash_command: {e}", exc_info=True)
        msg = f"Error with AP command: {str(e)[:1000]}"
        if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
        else: await interaction.followup.send(msg, ephemeral=True)

@bot.tree.command(name="clearhistory", description="Clears the bot's message history for this channel.")
@app_commands.checks.has_permissions(manage_messages=True)
async def clearhistory_slash_command(interaction: discord.Interaction):
    if interaction.channel_id in message_history:
        message_history[interaction.channel_id] = [] 
        logger.info(f"Message history cleared for channel {interaction.channel_id} by {interaction.user.name}")
        await interaction.response.send_message("Short-term message history for this channel has been cleared.", ephemeral=True)
    else:
        await interaction.response.send_message("No short-term history to clear for this channel.", ephemeral=True)

@clearhistory_slash_command.error
async def clearhistory_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("You don't have permission to clear history (Manage Messages required).", ephemeral=True)
    else:
        logger.error(f"Error in clearhistory_slash_command: {error}", exc_info=True)
        msg = "An unexpected error occurred with this command."
        if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
        else: await interaction.followup.send(msg, ephemeral=True)

# -------------------------------------------------------------------
# Main Event Handlers
# -------------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot: return 
    
    prefixes = await bot.get_prefix(message) 
    is_command_attempt = False
    if isinstance(prefixes, (list, tuple)):
        is_command_attempt = any(message.content.startswith(p) for p in prefixes)
    elif isinstance(prefixes, str):
        is_command_attempt = message.content.startswith(prefixes)
    
    if is_command_attempt:
        await bot.process_commands(message) 
        return 

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions if bot.user else False 
    
    channel_id = message.channel.id 
    author_roles = getattr(message.author, 'roles', []) 

    allowed_by_channel = not config.ALLOWED_CHANNEL_IDS or channel_id in config.ALLOWED_CHANNEL_IDS or \
                        (isinstance(message.channel, discord.Thread) and message.channel.parent_id in config.ALLOWED_CHANNEL_IDS)
    allowed_by_role = not config.ALLOWED_ROLE_IDS or is_dm or any(role.id in config.ALLOWED_ROLE_IDS for role in author_roles)

    should_respond = is_dm or is_mentioned or (allowed_by_channel and allowed_by_role)

    if not should_respond:
        if not (is_dm or is_mentioned):
            if not allowed_by_channel:
                logger.debug(f"Msg from {message.author.name} in ChID {channel_id} ignored (channel not allowed).")
            elif not allowed_by_role:
                logger.debug(f"Msg from {message.author.name} in ChID {channel_id} ignored (user role not allowed).")
        return 

    logger.info(f"General LLM message from {message.author.name} in {getattr(message.channel, 'name', f'ChID {channel_id}')}: {message.content[:50]}")

    if channel_id not in message_history: message_history[channel_id] = []
    
    current_message_content_parts = [] 
    user_message_text_for_processing = message.content.replace(f"<@{bot.user.id}>", "").strip() if bot.user else message.content 

    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("audio/"):
                try:
                    if not os.path.exists("temp"): os.makedirs("temp")
                    audio_filename = f"temp/temp_audio_{attachment.id}.{attachment.filename.split('.')[-1]}"
                    await attachment.save(audio_filename)
                    transcription = transcribe_audio_file(audio_filename)
                    if os.path.exists(audio_filename): os.remove(audio_filename) 
                    if transcription: 
                        user_message_text_for_processing = (user_message_text_for_processing + " " + f"[Audio Transcript: {transcription}]").strip()
                        logger.info(f"Added transcript from audio attachment: {transcription[:50]}...")
                except Exception as e: logger.error(f"Error processing audio attachment: {e}", exc_info=True)
                break 

    if user_message_text_for_processing: current_message_content_parts.append({"type": "text", "text": user_message_text_for_processing})

    image_added_to_prompt = False
    if message.attachments:
        for i, attachment in enumerate(message.attachments):
            if i >= config.MAX_IMAGES_PER_MESSAGE: break 
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    img_bytes = await attachment.read()
                    b64_img = base64.b64encode(img_bytes).decode('utf-8')
                    current_message_content_parts.append({"type": "image_url", "image_url": {"url": f"data:{attachment.content_type};base64,{b64_img}"}})
                    image_added_to_prompt = True
                    logger.info(f"Added image attachment {attachment.filename} to prompt.")
                except Exception as e: logger.error(f"Error processing image {attachment.filename}: {e}")
    
    scraped_content_for_llm = "" 
    # Use user_message_text_for_processing for URL detection as it includes transcriptions
    if detected_urls_in_text := detect_urls(user_message_text_for_processing): 
        for i, url in enumerate(detected_urls_in_text[:2]): 
            logger.info(f"Processing URL from message: {url}")
            content_piece = None
            youtube_match = re.search(r'(?:v=|\/|embed\/|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})', url)
            if youtube_match: 
                transcript = await fetch_youtube_transcript(url)
                if transcript: content_piece = f"\n\n--- YouTube Transcript for {url} ---\n{transcript[:2000]}\n--- End Transcript ---" 
            else: 
                scraped_text = await scrape_website(url)
                if scraped_text and "Failed to scrape" not in scraped_text and "Scraping timed out" not in scraped_text:
                    content_piece = f"\n\n--- Webpage Content for {url} ---\n{scraped_text[:3000]}\n--- End Webpage Content ---" 
            if content_piece: scraped_content_for_llm += content_piece
            await asyncio.sleep(0.2) 

    if scraped_content_for_llm:
        text_part_found = False
        for part_idx, part in enumerate(current_message_content_parts):
            if part["type"] == "text":
                current_message_content_parts[part_idx]["text"] = scraped_content_for_llm + "\n\nUser's message (after URL content): " + part["text"]
                text_part_found = True
                break
        if not text_part_found: 
            current_message_content_parts.insert(0, {"type": "text", "text": scraped_content_for_llm + "\n\n(User sent an attachment, possibly with URLs in a non-text part, or no text initially)"})

    if not current_message_content_parts: 
        if image_added_to_prompt: 
            current_message_content_parts.append({"type": "text", "text": "The user sent this image. Please describe it or respond to it if there's an implicit question."})
        else:
            logger.info("Ignoring message with no processable text, audio, or image content after all processing.")
            return

    user_msg_node_content_final: Union[str, List[dict]]
    if len(current_message_content_parts) == 1 and current_message_content_parts[0]["type"] == "text": # and not image_added_to_prompt (this condition might be too restrictive if image was added but text is primary)
        user_msg_node_content_final = current_message_content_parts[0]["text"]
    else:
        user_msg_node_content_final = current_message_content_parts
    
    # --- CONTEXT RETRIEVAL & SUMMARIZATION ---
    # Determine text for ChromaDB query (prefer actual text content)
    chroma_query_text = user_message_text_for_processing if user_message_text_for_processing.strip() else \
                        ("User sent an image/attachment" if image_added_to_prompt else "User sent a message with no textual content.")

    retrieved_docs = get_context_from_chromadb(chroma_query_text, n_results=1) 
    
    summarized_chroma_db_context_for_prompt: Optional[str] = None 
    summarized_chroma_db_context_for_display: Optional[str] = None # Separate for display prefix

    if retrieved_docs:
        logger.info(f"Retrieved document from ChromaDB for query: '{chroma_query_text[:50]}...'")
        summary = await get_summarized_context(
            past_conversation=retrieved_docs[0], 
            current_query=chroma_query_text
        )
        if summary:
            summarized_chroma_db_context_for_prompt = summary
            summarized_chroma_db_context_for_display = summary # Use the same for display
            logger.info("Successfully summarized ChromaDB context.")
        else:
            logger.info("ChromaDB summarization yielded no content, but docs were retrieved.")
    else:
        logger.info(f"No relevant documents retrieved from ChromaDB for query: '{chroma_query_text[:50]}...'")

    # Build the full prompt for the LLM
    llm_prompt_for_current_turn = _build_initial_prompt_messages(
        user_query_content=user_msg_node_content_final,
        user_id=str(message.author.id),
        channel_id=channel_id, # For short-term history
        summarized_chroma_context_str=summarized_chroma_db_context_for_prompt # For LLM prompt
    )
    
    # Update short-term message history with the current user's message *before* calling LLM
    # This user_msg_node_content_final is what _build_initial_prompt_messages used.
    # Note: stream_llm_response_to_message will add this user message AGAIN if we are not careful.
    # The `_build_initial_prompt_messages` should use `message_history[channel_id]` which is from *previous* turns.
    # Then, `stream_llm_response_to_message` should add the *current* user message and then the assistant response.
    
    # Let's adjust: `message_history` is updated *after* the turn.
    # `_build_initial_prompt_messages` uses the history *up to but not including* the current user message.

    # Call the streaming function
    await stream_llm_response_to_message(
        target_message=message, 
        prompt_messages=llm_prompt_for_current_turn, # This is the fully formed prompt
        summarized_chroma_context=summarized_chroma_db_context_for_display # For display prefix only
    )
    
    # After the response, update the message_history[channel_id]
    # The user message part of the current turn is the last 'user' role message in llm_prompt_for_current_turn
    current_user_msg_node = next((msg for msg in reversed(llm_prompt_for_current_turn) if msg.role == 'user'), None)
    if current_user_msg_node:
        if channel_id not in message_history: message_history[channel_id] = [] # Should be initialized already
        message_history[channel_id].append(current_user_msg_node) 
        # The assistant's response is added inside stream_llm_response_to_message
        message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:] # Trim


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == bot.user.id or str(payload.emoji) != '❌': return 
    try:
        channel = await bot.fetch_channel(payload.channel_id)
        if not isinstance(channel, discord.abc.Messageable): return

        message = await channel.fetch_message(payload.message_id)
    except (discord.NotFound, discord.Forbidden): return 
    
    if message.author.id != bot.user.id: return 

    can_delete = False 
    if isinstance(channel, (discord.TextChannel, discord.Thread)) and channel.guild:
        try:
            member = await channel.guild.fetch_member(payload.user_id) # Reactor
            # Allow if reactor has manage_messages OR is the original interactor/replied-to user
            if member and member.guild_permissions.manage_messages: 
                can_delete = True
            # Check if reactor was the one who initiated the command (for slash commands)
            elif message.interaction and message.interaction.user.id == payload.user_id:
                can_delete = True
            # Check if the bot's message was a reply to the reactor's message
            elif message.reference and message.reference.message_id:
                original_message = await channel.fetch_message(message.reference.message_id)
                if original_message.author.id == payload.user_id:
                    can_delete = True
        except discord.HTTPException: pass # Failed to fetch member or original message

    elif isinstance(channel, discord.DMChannel): # In DMs, the other user can always delete bot's messages
        can_delete = True
    
    # Fallback check if not covered by guild permissions (e.g. for DMs or if initial checks failed)
    if not can_delete:
        if message.interaction and message.interaction.user.id == payload.user_id: 
            can_delete = True
        elif message.reference and message.reference.message_id:
            try:
                original_message = await channel.fetch_message(message.reference.message_id)
                if original_message.author.id == payload.user_id: can_delete = True
            except discord.NotFound: pass

    if can_delete:
        try: 
            await message.delete()
            logger.info(f"Message {message.id} deleted by reaction from user {payload.user_id}.")
        except Exception as e: logger.error(f"Failed to delete message {message.id} by reaction: {e}")

@bot.event
async def on_ready():
    if not bot.user: 
        logger.critical("Bot user is not available on_ready. This is a critical error.")
        return
    logger.info(f'{bot.user.name} has connected to Discord! ID: {bot.user.id}')
    logger.info(f"discord.py version: {discord.__version__}")
    logger.info(f"Operating in channels: {config.ALLOWED_CHANNEL_IDS if config.ALLOWED_CHANNEL_IDS else 'All permitted by default'}")
    logger.info(f"Restricted to roles: {config.ALLOWED_ROLE_IDS if config.ALLOWED_ROLE_IDS else 'None'}")
    logger.info(f"User-Provided Global Context: {'Present' if config.USER_PROVIDED_CONTEXT else 'Not set'}")
    try:
        synced = await bot.tree.sync() 
        logger.info(f"Synced {len(synced)} slash commands globally.")
    except Exception as e: logger.error(f"Failed to sync slash commands: {e}", exc_info=True)
    
    if not check_reminders.is_running(): check_reminders.start()
    await bot.change_presence(activity=discord.Game(name="with /commands | Ask me anything!"))

@bot.tree.error 
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    command_name = interaction.command.name if interaction.command else 'unknown_command'
    logger.error(f"Slash command error for '{command_name}': {error}", exc_info=True) 
    
    error_message = "An unexpected error occurred with this slash command."
    original_error_is_unknown_interaction = False

    if isinstance(error, app_commands.CommandInvokeError): 
        original_error = error.original
        if isinstance(original_error, discord.errors.NotFound) and original_error.code == 10062: 
            error_message = "The command took too long to respond initially, or the interaction expired. Please try again."
            logger.warning(f"Original 'Unknown Interaction' (10062) for {command_name}. Interaction ID: {interaction.id}")
            original_error_is_unknown_interaction = True 
        else: error_message = f"Command '{command_name}' failed: {str(original_error)[:500]}"
    elif isinstance(error, app_commands.CommandNotFound): error_message = "Command not found. This is unexpected."
    elif isinstance(error, app_commands.MissingPermissions): error_message = f"You lack permissions: {', '.join(error.missing_permissions)}"
    elif isinstance(error, app_commands.BotMissingPermissions): error_message = f"I lack permissions: {', '.join(error.missing_permissions)}"
    elif isinstance(error, app_commands.CheckFailure): error_message = "You do not meet the requirements to use this command."
    elif isinstance(error, app_commands.CommandOnCooldown): error_message = f"This command is on cooldown. Try again in {error.retry_after:.2f} seconds."
    elif isinstance(error, app_commands.TransformerError): error_message = f"Invalid argument: {error.value}. Expected type: {error.type}."
    
    if original_error_is_unknown_interaction: return 

    try:
        if interaction.response.is_done(): await interaction.followup.send(error_message, ephemeral=True)
        else: await interaction.response.send_message(error_message, ephemeral=True)
    except discord.errors.HTTPException as ehttp: 
        if ehttp.code == 40060: 
            logger.warning(f"Error handler: Interaction for '{command_name}' already acknowledged. Trying followup. OrigErr: {error}")
            try: await interaction.followup.send(error_message, ephemeral=True) 
            except Exception as e_followup: logger.error(f"Error handler: Failed followup for '{command_name}': {e_followup}")
        else: logger.error(f"Error handler: HTTPException for '{command_name}': {ehttp}. OrigErr: {error}")
    except discord.errors.NotFound: 
        logger.error(f"Error handler: Interaction not found for '{command_name}'. OrigErr: {error}")
    except Exception as e_generic: 
        logger.error(f"Error handler: Generic error for '{command_name}': {e_generic}. OrigErr: {error}")

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound): pass 
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.reply(f"You're missing an argument for !{ctx.command.name if ctx.command else 'command'}: {error.param.name}.", silent=True)
    elif isinstance(error, commands.BadArgument):
        await ctx.reply(f"Invalid argument provided for !{ctx.command.name if ctx.command else 'command'}.", silent=True)
    elif isinstance(error, commands.CheckFailure): 
        await ctx.reply("You don't have permissions for that prefix command.", silent=True)
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Error invoking prefix command !{ctx.command.name if ctx.command else 'command'}: {error.original}", exc_info=error.original)
        await ctx.reply(f"An error occurred with !{ctx.command.name if ctx.command else 'command'}: {error.original}", silent=True)
    else:
        logger.error(f"Unhandled prefix command error: {error}", exc_info=True)


if __name__ == "__main__":
    if not config.DISCORD_BOT_TOKEN:
        logger.critical("DISCORD_BOT_TOKEN is not set. Bot cannot start.")
    elif not chroma_client or not chat_history_collection: 
        logger.critical("ChromaDB failed to initialize. Bot cannot start.")
    else:
        try:
            bot.run(config.DISCORD_BOT_TOKEN, log_handler=None) 
        except discord.LoginFailure:
            logger.critical("Failed to log in with the provided Discord token. Please check it.")
        except Exception as e:
            logger.critical(f"An unexpected error occurred during bot startup: {e}", exc_info=True)

