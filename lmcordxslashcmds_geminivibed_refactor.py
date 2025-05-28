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

import aiohttp
import discord
import torch # Ensure torch is available if whisper fp16 is used with CUDA
import whisper
from discord import app_commands # Import for slash commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from openai import AsyncOpenAI
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

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
        self.LLM_MODEL = os.getenv("LLM", "local-model") 
        self.VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "llava") # Ensure this is a valid model name for your setup

        self.ALLOWED_CHANNEL_IDS = [int(i) for i in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if i]
        self.ALLOWED_ROLE_IDS = [int(i) for i in os.getenv("ALLOWED_ROLE_IDS", "").split(",") if i]
        
        self.MAX_IMAGES_PER_MESSAGE = int(os.getenv("MAX_IMAGES_PER_MESSAGE", 1))
        self.MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", 10))
        self.MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", 2048))
        
        self.TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:8880/v1/audio/speech")
        self.TTS_VOICE = os.getenv("TTS_VOICE", "af_sky+af+af_nicole")
        self.TTS_ENABLED_DEFAULT = os.getenv("TTS_ENABLED_DEFAULT", "true").lower() == "true"

        self.SEARX_URL = os.getenv("SEARX_URL", "http://192.168.1.3:9092/search")
        self.SEARX_PREFERENCES = os.getenv("SEARX_PREFERENCES", "eJx1V0GT6yYM_jXNxfMyfX2HTg85dabXdqa9MzIottaAWMBJvL--IrZjvN53WGKEEOKT9KHVkLHjSJguHXqMYH_57U-P9yQ_lGVImtBrLF-sCWzj0BCcLPhuhA4vMGY-WdZg8YL-VKaaXbCY8XIiJyoqRH5Ml7_AJjw5zD2byz9___vfKcEVE0LU_eXXU-7R4SVR2XqKmEabk2KvxBeVoV22GyYli2xvGC8MMj1z7E7zNpXyZBePNPqMUYGlzjv5XvaDuYHcxqjl3Fn6PmKcFHmVKYuBWUj-Sp6yWNWRrV0doAStFQPoO_KC2h8ddErV2AhU4D2kppxAN1TqShYLoBCGxlGMHGuZeN7I2KTMsVaedzeUlNpC05Lv9vPcjnrArNQzXFrrb_mm1I0McjEjzqcU8So-viIpsru5kSBbWZow1FM9RktYSwzih0Cq3JhIl7npGoNPkIj9zkuUA8goxRLVWFTFxfLXcfNMCVGef_drd4SyYds3g-MgyAYZi2XHbxQKzJtWlkBMO1e_P6oDriZycWYF9hoRm8TXfIeIjaGIWpCfFgCvkfxAoCsDnWQFtCvAbLDF2C3TjrkTF4OFqQQxbcfUK44F7VRFpQtsTA1mD22EMix2yZl2uyF5qDaTl0_iMX0tW02-Uep5szGQHiBVDlqSI-PUFNgS1QvcpoznmBZf4MNOkXRlWqoYdAC_KPjg1q8JYDPEAX3EwJXtAOJFR2nN1zC2Z4O3ZTZXZYG4KcMrJOCgeIsvNQc-k26S7tlCrHM7ZYg5FNqp8iHzMHFmwWMoTq_O5DvlwhKfy3eCnrnef6d2qudCdg_wJpYsrLUGapmH9FlY-FSphVWL4H3kjJ-1Eo9RH6RCdInydBDz9OmO33_8-P2x3c2MBv0W_YQfHlytL4WEONQSD7cCxiaIYzt16NY8CIgxjy1WafdETI4YCpvcsa2W4uiEKCvBnR80sJdwN2ny7CeHlX9v4RzulcPlkk64ZUd2EW67OwfJQzsnzU_ER76RRNpSe8a8H_NWzi9DJ_R7qn_as-THR1O8W6sjSiZW9iE-6FZnZCuZosGFrXzK7oXD9sx-9PYp3eXPUzKDWoHbZkNdty-9kjpY2RJKj-LTtCf1G0ktySVqSmZ5UWLTjyvpFYcN5J2_G2_vmD9Pjr2AVMX2aoV6Yk2o6KlmKpnnAvwcAsrbwTOJ1gcstHoAapHvoFpkB7AW-Rfs0S_nPQ__-LZQ5JEz-SoE5TtpRfap5bvbDiRHD215NBV1gjwvuNJ8CRPIC1TATSgdT-XLuiYUKQDC8sju11OWxyzL07i-j8GU-tyUQi-851-rFKXfa6G6kViXIt5tmcKa2-93yY36Pk_BHuRZdIjHLD5AL1c1lA-MmzAQrIX8UpbqlNLZ4yf8rgeW1L5avq8lmIaxHX0e16oeA8YxvVCWfpGMvPGS_bl6i0af5HVOfeX1k_n395t4_ER6L8mrEwKykvYlQpXajRxyzX80FI1nPr1Yju38uoW-amgM6fzBfpf3TujSSZvS5Ag-WQmjqc1E42moBDnHM610unWtwY7CZelSoH6cl9m555TldUBprQWQ-RndKTCoufO-R-mID8sCrdI96uGwIhgoycABp_Si05_4UAIxTw5WlsQQdPX878ok_buVtu0LTXuVRv7KRx8jFFJT0tbKPVzJoJO0BVJCl_8BGa3pmQ==")


        self.EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green(), "error": discord.Color.red()}
        self.EMBED_MAX_LENGTH = 4096 
        self.EDITS_PER_SECOND = 1.3 

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
# Bot Initialization
# -------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
intents.guilds = True

bot = commands.Bot(command_prefix=commands.when_mentioned_or("!"), intents=intents) 

llm_client = AsyncOpenAI(base_url=config.LOCAL_SERVER_URL, api_key="lm-studio")

message_history = {} 
reminders = [] 

class MsgNode:
    """Represents a message in the conversation history for LLM context."""
    def __init__(self, role: str, content: any, name: str = None): 
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
            "If you need to 'think' before responding, use <think>Your thoughts here...</think> tags. Don't use emojis unless asked." 
            f"Current Date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}"
        )
    )

def chunk_text(text: str, max_length: int = config.EMBED_MAX_LENGTH) -> list:
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


def detect_urls(message_text: str) -> list:
    if not message_text: return []
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(message_text)

def clean_text_for_tts(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[\*#_~\<\>\[\]\(\)]+', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    return text.strip()

async def _send_audio_segment(destination: discord.abc.Messageable, segment_text: str, filename_suffix: str, is_thought: bool = False, base_filename: str = "response"):
    """Internal helper to process and send a single audio segment."""
    if not segment_text:
        return
    cleaned_segment = clean_text_for_tts(segment_text)
    if not cleaned_segment:
        logger.info(f"Skipping TTS for empty/cleaned {filename_suffix} segment.")
        return

    logger.info(f"Requesting TTS for {filename_suffix}: {cleaned_segment[:100]}...")
    tts_audio_data = await tts_request(cleaned_segment)
    if tts_audio_data:
        try:
            audio = AudioSegment.from_file(io.BytesIO(tts_audio_data), format="mp3")
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format="mp3", bitrate="128k")
            fixed_audio_data = output_buffer.getvalue()

            file = discord.File(io.BytesIO(fixed_audio_data), filename=f"{base_filename}_{filename_suffix}.mp3")
            
            content_message = None
            if is_thought:
                content_message = "**Sam's thoughts:**"
            elif filename_suffix == "main_response" or filename_suffix == "full": 
                content_message = "**Sam's response:**"

            if isinstance(destination, discord.InteractionMessage):
                 await destination.channel.send(content=content_message, file=file) 
            elif hasattr(destination, 'send'): 
                await destination.send(content=content_message, file=file)
            else:
                logger.warning(f"Cannot send TTS to destination of type {type(destination)}")

            logger.info(f"Sent TTS audio: {base_filename}_{filename_suffix}.mp3")
        except Exception as e:
            logger.error(f"Error processing or sending TTS for {filename_suffix}: {e}", exc_info=True)
    else:
        logger.warning(f"TTS request failed for {filename_suffix} segment.")


async def send_tts_audio(destination: discord.abc.Messageable, text_to_speak: str, base_filename: str = "response"):
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
        await asyncio.sleep(0.5) 
        await _send_audio_segment(destination, response_text, "main_response", is_thought=False, base_filename=base_filename)
    else:
        logger.info("No <think> tags found. Processing full text for TTS.")
        await _send_audio_segment(destination, text_to_speak, "full", is_thought=False, base_filename=base_filename)

# -------------------------------------------------------------------
# Core LLM Interaction Logic (NEW TWO-STEP PROCESS)
# -------------------------------------------------------------------

async def get_context_aware_llm_stream(prompt_messages: list[MsgNode], is_vision_request: bool):
    """
    Performs the two-step LLM call:
    1. Generate a "suggested context" for the user's query.
    2. Stream the final response using the original prompt enhanced with the generated context.
    """
    if not prompt_messages:
        raise ValueError("Prompt messages cannot be empty.")

    last_user_message_node = next((msg for msg in reversed(prompt_messages) if msg.role == 'user'), None)
    if not last_user_message_node:
        raise ValueError("No user message found in the prompt history.")

    # --- Step 1: Generate the Suggested Context ---
    logger.info("Step 1: Generating suggested context...")
    context_generation_prompt = [
        MsgNode(
            role="system",
            content=(
                "You are a context analysis expert. Your task is to read the user's question or statement "
                "and generate a concise 'suggested context' for viewing it. This context should clarify "
                "underlying assumptions, define key terms, or establish a frame of reference that will "
                "lead to the most insightful and helpful response. Do not answer the user's question. "
                "Only provide a single, short paragraph for the suggested context."
            )
        ),
        last_user_message_node  # Use the last user message as the basis for context
    ]

    generated_context = "Context generation failed or was not applicable."
    try:
        # Use a non-streaming call for the context generation
        context_response = await llm_client.chat.completions.create(
            model=config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL,
            messages=[msg.to_dict() for msg in context_generation_prompt],
            max_tokens=250,  # Limit context size
            stream=False,
            temperature=0.4,
        )
        if context_response.choices and context_response.choices[0].message.content:
            generated_context = context_response.choices[0].message.content.strip()
            logger.info(f"Successfully generated context: {generated_context[:150]}...")
        else:
            logger.warning("Context generation returned no content.")

    except Exception as e:
        logger.error(f"Could not generate suggested context: {e}", exc_info=True)
        # We will proceed with the generated_context holding the error message.

    # --- Step 2: Prepare and Stream the Final Response ---
    logger.info("Step 2: Streaming final response with injected context.")
    
    # Create a deep copy of prompt messages to avoid modifying the original history
    final_prompt_messages = [MsgNode(m.role, m.content, m.name) for m in prompt_messages]
    
    # Find the last user message in the copied list to modify it
    final_user_message_node = next((msg for msg in reversed(final_prompt_messages) if msg.role == 'user'), None)

    # Extract the original question text
    original_question = ""
    if isinstance(final_user_message_node.content, str):
        original_question = final_user_message_node.content
    elif isinstance(final_user_message_node.content, list):
        # For vision requests, find the text part
        text_part = next((part['text'] for part in final_user_message_node.content if part['type'] == 'text'), "")
        original_question = text_part

    # Construct the new text content with the injected context
    injected_prompt_text = (
        f"<model_generated_suggested_context>\n"
        f"{generated_context}\n"
        f"</model_generated_suggested_context>\n\n"
        f"<user_question>\n"
        f"With that context in mind, please respond to the following:\n"
        f"{original_question}\n"
        f"</user_question>"
    )

    # Update the content of the last user message node
    if isinstance(final_user_message_node.content, str):
        final_user_message_node.content = injected_prompt_text
    elif isinstance(final_user_message_node.content, list):
        # Update the text part, keeping image parts intact
        text_part_found = False
        for part in final_user_message_node.content:
            if part['type'] == 'text':
                part['text'] = injected_prompt_text
                text_part_found = True
                break
        if not text_part_found: # Should not happen if logic is correct
            final_user_message_node.content.insert(0, {"type": "text", "text": injected_prompt_text})

    # Determine the model for the final response
    current_model = config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL
    logger.info(f"Using model for final streaming: {current_model}")

    # Create the final stream
    final_stream = await llm_client.chat.completions.create(
        model=current_model,
        messages=[msg.to_dict() for msg in final_prompt_messages],
        max_tokens=config.MAX_COMPLETION_TOKENS,
        stream=True,
        temperature=0.7,
    )

    return final_stream, generated_context


async def stream_llm_response_to_interaction(
    interaction: discord.Interaction,
    prompt_messages: list,
    title: str = "Sam's Response",
    is_edit_of_original_response: bool = False 
):
    """Streams LLM response to a Discord interaction (slash command), using the two-step context process."""
    message_to_edit = None 
    original_interaction_message_id = None 

    if not interaction.response.is_done():
        try:
            await interaction.response.defer(ephemeral=False) 
            message_to_edit = await interaction.original_response()
            original_interaction_message_id = message_to_edit.id
        except discord.errors.InteractionResponded: 
            logger.warning("Interaction already responded before explicit defer, trying to get original response.")
            try:
                message_to_edit = await interaction.original_response()
                original_interaction_message_id = message_to_edit.id
            except discord.NotFound:
                logger.error("Could not get original response after InteractionResponded error.")
                response_embed_fallback = discord.Embed(title=title, description="⏳ Thinking...", color=config.EMBED_COLOR["incomplete"])
                message_to_edit = await interaction.followup.send(embed=response_embed_fallback, wait=True)
                original_interaction_message_id = message_to_edit.id


    if not message_to_edit: 
        logger.warning("No message to edit after defer attempts, sending new followup for streaming.")
        response_embed_init = discord.Embed(title=title, description="⏳ Thinking...", color=config.EMBED_COLOR["incomplete"])
        message_to_edit = await interaction.followup.send(embed=response_embed_init, wait=True)
        original_interaction_message_id = message_to_edit.id
    
    initial_embed = discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"])
    await message_to_edit.edit(embed=initial_embed)

    full_response_content = ""
    accumulated_chunk = ""
    last_edit_time = asyncio.get_event_loop().time()
    embed_count = 0 

    try:
        is_vision_request = any(isinstance(p.content, list) and any(c.get("type") == "image_url" for c in p.content) for p in prompt_messages)
        stream, generated_context = await get_context_aware_llm_stream(prompt_messages, is_vision_request)

        response_embed = discord.Embed(title=title, color=config.EMBED_COLOR["incomplete"])
        context_display = f"**Model-Generated Suggested Context:**\n> {generated_context.replace(chr(10), ' ')}\n\n---\n**Response:**\n"
        response_embed.description = context_display
        await message_to_edit.edit(embed=response_embed)

        async for chunk_data in stream:
            delta_content = ""
            if chunk_data.choices and len(chunk_data.choices) > 0:
                choice = chunk_data.choices[0]
                if choice.delta:
                    delta_content = choice.delta.content or ""
            
            full_response_content += delta_content
            accumulated_chunk += delta_content

            current_time = asyncio.get_event_loop().time()
            
            if accumulated_chunk and (current_time - last_edit_time >= (1.0 / config.EDITS_PER_SECOND) or len(accumulated_chunk) > 150):
                try:
                    response_embed.description += accumulated_chunk
                    
                    if len(response_embed.description) > config.EMBED_MAX_LENGTH:
                        response_embed.description = response_embed.description[:config.EMBED_MAX_LENGTH]
                    
                    await message_to_edit.edit(embed=response_embed)
                    last_edit_time = current_time
                    accumulated_chunk = "" 

                except discord.errors.NotFound:
                    logger.warning("Failed to edit message during stream, it might have been deleted.")
                    return
                except discord.errors.HTTPException as e:
                    logger.error(f"HTTPException during stream edit: {e}. Len: {len(response_embed.description)}")
                    await asyncio.sleep(0.5)

        if accumulated_chunk:
             response_embed.description += accumulated_chunk

        response_embed.description = response_embed.description[:config.EMBED_MAX_LENGTH].strip()
        response_embed.color = config.EMBED_COLOR["complete"]
        if not full_response_content.strip():
            response_embed.description += "\nNo response or error."
            response_embed.color = config.EMBED_COLOR["error"]
        
        await message_to_edit.edit(embed=response_embed)
        
        channel_id = interaction.channel_id
        if channel_id not in message_history: message_history[channel_id] = []
        message_history[channel_id].append(MsgNode(role="assistant", content=full_response_content, name=str(bot.user.id)))
        message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:]

        tts_base_filename = f"interaction_{original_interaction_message_id or interaction.id}"
        await send_tts_audio(interaction.channel, full_response_content, base_filename=tts_base_filename) 

    except Exception as e:
        logger.error(f"Error streaming LLM response to interaction: {e}", exc_info=True)
        error_embed = discord.Embed(title=title, description=f"An error occurred: {str(e)[:1000]}", color=config.EMBED_COLOR["error"])
        try:
            await message_to_edit.edit(embed=error_embed)
        except discord.errors.NotFound: pass
    return message_to_edit


async def stream_llm_response_to_message(
    target_message: discord.Message, 
    prompt_messages: list, 
    title: str = "Sam's Response"
):
    """Streams LLM response as a reply to a regular Discord message, using the two-step context process."""
    initial_embed = discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"])
    current_reply_message = await target_message.reply(embed=initial_embed, silent=True) 

    full_response_content = ""
    accumulated_chunk = ""
    last_edit_time = asyncio.get_event_loop().time()
    embed_count = 0

    try:
        is_vision_request = any(isinstance(p.content, list) and any(c.get("type") == "image_url" for c in p.content) for p in prompt_messages)
        stream, generated_context = await get_context_aware_llm_stream(prompt_messages, is_vision_request)

        response_embed = discord.Embed(title=title, color=config.EMBED_COLOR["incomplete"])
        context_display = f"**Model-Generated Suggested Context:**\n> {generated_context.replace(chr(10), ' ')}\n\n---\n**Response:**\n"
        response_embed.description = context_display
        await current_reply_message.edit(embed=response_embed)

        async for chunk_data in stream:
            delta_content = ""
            if chunk_data.choices and len(chunk_data.choices) > 0:
                choice = chunk_data.choices[0]
                if choice.delta:
                    delta_content = choice.delta.content or ""
            
            full_response_content += delta_content
            accumulated_chunk += delta_content

            current_time = asyncio.get_event_loop().time()
            
            if accumulated_chunk and (current_time - last_edit_time >= (1.0 / config.EDITS_PER_SECOND) or len(accumulated_chunk) > 150):
                try:
                    response_embed.description += accumulated_chunk

                    if len(response_embed.description) > config.EMBED_MAX_LENGTH:
                        response_embed.description = response_embed.description[:config.EMBED_MAX_LENGTH]

                    await current_reply_message.edit(embed=response_embed)
                    last_edit_time = current_time
                    accumulated_chunk = "" 
                    
                except discord.errors.NotFound:
                    logger.warning("Failed to edit message during stream, it might have been deleted.")
                    return
                except discord.errors.HTTPException as e:
                    logger.error(f"HTTPException during stream edit: {e}. Len: {len(response_embed.description)}")
                    await asyncio.sleep(0.5) 
            
        if accumulated_chunk:
            response_embed.description += accumulated_chunk
            
        response_embed.description = response_embed.description[:config.EMBED_MAX_LENGTH].strip()
        response_embed.color = config.EMBED_COLOR["complete"]
        if not full_response_content.strip():
            response_embed.description += "\nNo response or error."
            response_embed.color = config.EMBED_COLOR["error"]
        
        await current_reply_message.edit(embed=response_embed)
        
        channel_id = target_message.channel.id
        if channel_id not in message_history: message_history[channel_id] = []
        message_history[channel_id].append(MsgNode(role="assistant", content=full_response_content, name=str(bot.user.id)))
        message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:]

        await send_tts_audio(target_message.channel, full_response_content, base_filename=f"message_{target_message.id}")

    except Exception as e:
        logger.error(f"Error streaming LLM response to message: {e}", exc_info=True)
        error_embed = discord.Embed(title=title, description=f"An error occurred: {str(e)[:1000]}", color=config.EMBED_COLOR["error"])
        try:
            await current_reply_message.edit(embed=error_embed) 
        except discord.errors.NotFound: pass
    return current_reply_message 

# -------------------------------------------------------------------
# Text-to-Speech (TTS)
# -------------------------------------------------------------------
async def tts_request(text: str, speed: float = 1.3) -> bytes | None:
    if not text: return None
    payload = {
        "input": text,
        "voice": config.TTS_VOICE,
        "response_format": "mp3",
        "speed": speed,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(config.TTS_API_URL, json=payload, timeout=30) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    logger.error(f"TTS request failed: status={resp.status}, resp={await resp.text()}")
                    return None
    except asyncio.TimeoutError:
        logger.error("TTS request timed out.")
        return None
    except Exception as e:
        logger.error(f"TTS request error: {e}", exc_info=True)
        return None

# -------------------------------------------------------------------
# Web Scraping and Search
# -------------------------------------------------------------------
# JavaScript for expanding "Show more" buttons safely
# UPDATED to directly check for "Grok" text within the parent article to avoid all Grok links.
JS_EXPAND_SHOWMORE_TWITTER = """
(maxClicks) => {
    let clicks = 0;
    const getButtons = () => Array.from(document.querySelectorAll('[role="button"]'))
        .filter(b => {
            const t = (b.textContent || '').toLowerCase();
            // 1. Must be a "show more" button
            if (!t.includes('show more')) {
                return false;
            }

            const article = b.closest('article');
            if (!article) {
                return false;
            }

            // 2. THE DEFINITIVE GROK CHECK:
            // If the article's text content mentions "Grok", it is a summary card. IGNORE IT.
            const articleText = article.textContent || '';
            if (articleText.match(/grok/i)) {
                return false;
            }

            // 3. Must NOT be inside a quoted tweet
            if (b.closest('[role="blockquote"]')) {
                return false;
            }

            return true;
        });

    while (clicks < maxClicks) {
        const buttonsToClick = getButtons();
        if (buttonsToClick.length === 0) {
            break;
        }
        const button = buttonsToClick[0];
        try {
            button.click();
            clicks++;
        } catch (e) {
            break;
        }
    }
    return clicks;
}
"""

# JavaScript for extracting tweet data, adapted from user's x-scrape.py
JS_EXTRACT_TWEETS_TWITTER = """
() => {
    const tweets = [];
    document.querySelectorAll('article[data-testid="tweet"]').forEach(article => {
        try {
            const timeTag = article.querySelector('time');
            const timestamp = timeTag ? timeTag.getAttribute('datetime') : null;
            
            let tweetLink = null;
            let id = '';
            let username = 'unknown_user'; 

            const primaryLinkElement = timeTag ? timeTag.closest('a[href*="/status/"]') : null;
            if (primaryLinkElement) {
                tweetLink = primaryLinkElement.href;
            } else {
                const articleLinks = Array.from(article.querySelectorAll('a[href*="/status/"]'));
                if (articleLinks.length > 0) {
                    for(let link of articleLinks){
                        if(!link.href.includes("/photo/") && !link.href.includes("/video/")){
                            tweetLink = link.href;
                            break;
                        }
                    }
                    if(!tweetLink && articleLinks.length > 0) tweetLink = articleLinks[0].href;
                }
            }

            if (tweetLink) {
                const match = tweetLink.match(/\/([a-zA-Z0-9_]+)\/status\/(\d+)/);
                if (match) {
                    username = match[1]; 
                    id = match[2];
                }
            }

            const tweetTextElement = article.querySelector('div[data-testid="tweetText"]');
            const content = tweetTextElement ? tweetTextElement.innerText.trim() : '';

            const socialContextElement = article.querySelector('div[data-testid="socialContext"]');
            let is_repost = false; 
            let reposted_by = null; 
            if (socialContextElement && (socialContextElement.innerText.toLowerCase().includes('reposted') || socialContextElement.innerText.toLowerCase().includes('retweeted'))) {
                is_repost = true;
                const userLinkInContext = socialContextElement.querySelector('a[href^="/"]');
                 if (userLinkInContext) {
                    const hrefParts = userLinkInContext.href.split('/');
                    if (hrefParts.length > 0) {
                        for(let i = hrefParts.length -1; i >=0; i--){
                            if(!['analytics', 'likes', 'media', 'status', 'with_replies', 'following', 'followers'].includes(hrefParts[i])){
                                reposted_by = hrefParts[i];
                                break;
                            }
                        }
                    }
                 }
            }
            
            if (content || article.querySelector('[data-testid="tweetPhoto"], [data-testid="videoPlayer"]')) { 
                tweets.push({
                    id: id || `no-id-${Date.now()}-${Math.random()}`, 
                    username: username, 
                    content: content,
                    timestamp: timestamp || new Date().toISOString(),
                    is_repost: is_repost,
                    reposted_by: reposted_by, 
                    tweet_url: tweetLink || (id ? `https://x.com/${username}/status/${id}` : '')
                });
            }
        } catch (e) {
            // console.warn('Error extracting a tweet with JS:', e);
        }
    });
    return tweets;
}
"""


async def scrape_website(url: str) -> str | None:
    logger.info(f"Scraping website: {url}")
    user_data_dir = os.path.join(os.getcwd(), ".pw-profile") 
    profile_dir_usable = False
    if os.path.exists(user_data_dir):
        profile_dir_usable = True
    else:
        try:
            os.makedirs(user_data_dir)
            profile_dir_usable = True
            logger.info(f"Created Playwright profile directory: {user_data_dir}")
        except OSError as e:
            logger.error(f"Could not create .pw-profile directory: {e}. Using non-persistent context for scrape_website.")
            
    context_manager = None 
    browser_instance_sw = None
    page = None
    try:
        async with async_playwright() as p:
            if profile_dir_usable:
                context = await p.chromium.launch_persistent_context(
                    user_data_dir,
                    headless=False, 
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                )
                context_manager = context
            else: 
                 browser_instance_sw = await p.chromium.launch(
                    headless=False, 
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
                        if content and len(content.strip()) > 200: 
                            break
                except PlaywrightTimeoutError:
                    logger.debug(f"Timeout for selector {selector} on {url}")
                    continue 
                except Exception as e_sel:
                    logger.warning(f"Error with selector {selector} on {url}: {e_sel}")
            
            if not content or len(content.strip()) < 100 : 
                content = await page.evaluate('document.body.innerText')
            
            cleaned_content = re.sub(r'\s\s+', ' ', content.strip()) 
            return cleaned_content if cleaned_content else None
            
    except PlaywrightTimeoutError:
        logger.error(f"Playwright timed out scraping {url}")
        return "Scraping timed out."
    except Exception as e:
        logger.error(f"Playwright failed for {url}: {e}", exc_info=True)
        return "Failed to scrape the website due to an error."
    finally:
        if page and not page.is_closed():
            try: await page.close()
            except Exception as e_page_close: logger.warning(f"Ignoring error closing page for {url}: {e_page_close}")
        
        if context_manager:
            try: await context_manager.close()
            except Exception as e_context_close:
                if "Target page, context or browser has been closed" in str(e_context_close):
                    logger.warning(f"Context for {url} was already closed: {e_context_close}")
                else:
                    logger.error(f"Error closing context for {url}: {e_context_close}", exc_info=True)
        
        if browser_instance_sw and not profile_dir_usable: 
            try: await browser_instance_sw.close()
            except Exception as e_browser_close: logger.warning(f"Ignoring error closing non-persistent browser for {url}: {e_browser_close}")


async def scrape_latest_tweets(username_queried: str, limit: int = 5) -> list:
    logger.info(f"Scraping last {limit} tweets for @{username_queried} with JS enhancement.")
    tweets_collected = []
    seen_tweet_ids = set()
    
    user_data_dir = os.path.join(os.getcwd(), ".pw-profile")
    profile_dir_usable = os.path.exists(user_data_dir) or os.makedirs(user_data_dir, exist_ok=True)

    context_manager = None
    browser_instance_st = None 
    page = None

    try:
        async with async_playwright() as p:
            if profile_dir_usable:
                logger.info(f"Using persistent profile: {user_data_dir}")
                context = await p.chromium.launch_persistent_context(
                    user_data_dir, headless=False,
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
                    slow_mo=150 
                )
                context_manager = context
            else:
                logger.warning("Profile directory not usable. Proceeding with non-persistent context for tweet scraping.")
                browser_instance_st = await p.chromium.launch(
                    headless=False, 
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                    slow_mo=150
                )
                context = await browser_instance_st.new_context(
                     user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
                )
                context_manager = context
            
            page = await context_manager.new_page()
            url = f"https://x.com/{username_queried.lstrip('@')}/with_replies"
            logger.info(f"Navigating to {url}")
            await page.goto(url, timeout=60000, wait_until="domcontentloaded") 
            
            try:
                await page.wait_for_selector("article[data-testid='tweet']", timeout=30000) 
                logger.info("Initial tweet articles detected.")
                await asyncio.sleep(1.5) 
                await page.keyboard.press("Escape") 
                await asyncio.sleep(0.5)
                await page.keyboard.press("Escape")
            except PlaywrightTimeoutError:
                logger.warning(f"Timed out waiting for initial tweet articles for @{username_queried}. Page might be structured differently or not loaded.")
                screenshot_path = f"error_screenshot_tweet_article_load_{username_queried}_{datetime.now():%Y%m%d_%H%M%S}.png"
                await page.screenshot(path=screenshot_path)
                logger.info(f"Screenshot on tweet article timeout: {screenshot_path}")
                return [] 

            max_scroll_attempts = limit + 10 
            for scroll_attempt in range(max_scroll_attempts):
                if len(tweets_collected) >= limit: break
                
                logger.debug(f"Tweet scrape attempt {scroll_attempt + 1}. Collected: {len(tweets_collected)}/{limit}")

                try:
                    clicked_count = await page.evaluate(JS_EXPAND_SHOWMORE_TWITTER, 3) 
                    if clicked_count > 0:
                        logger.info(f"Clicked {clicked_count} 'Show more' elements via JS.")
                        await asyncio.sleep(1.5 + random.uniform(0.2, 0.8)) 
                except Exception as e_sm:
                    logger.warning(f"JS error during 'Show More' expansion: {e_sm}")

                extracted_this_round = []
                try:
                    extracted_this_round = await page.evaluate(JS_EXTRACT_TWEETS_TWITTER)
                except Exception as e_js_extract:
                    logger.error(f"JS error during tweet extraction: {e_js_extract}")

                newly_added_count = 0
                for data in extracted_this_round:
                    tweet_id = data.get('id')
                    unique_signature = tweet_id if tweet_id else (data.get("username","") + (data.get("content") or "")[:30] + data.get("timestamp",""))

                    if unique_signature and unique_signature not in seen_tweet_ids:
                        tweets_collected.append({
                            "timestamp": data.get("timestamp", datetime.now().isoformat()),
                            "content": data.get("content", "N/A"),
                            "user": username_queried.lstrip('@'), 
                            "original_author": data.get("username", "unknown_user"), 
                            "id": tweet_id,
                            "url": data.get("tweet_url", "")
                        })
                        seen_tweet_ids.add(unique_signature)
                        newly_added_count +=1
                        if len(tweets_collected) >= limit: break
                
                logger.info(f"Extracted {len(extracted_this_round)} items via JS, added {newly_added_count} new unique tweets.")

                if newly_added_count == 0 and scroll_attempt > (limit // 2 + 5): 
                    logger.info("No new unique tweets found in several recent attempts. Assuming end of relevant content or stuck.")
                    break
                
                await page.evaluate("window.scrollBy(0, window.innerHeight * 1.5);") 
                await asyncio.sleep(random.uniform(3.5, 5.5)) 
            
    except PlaywrightTimeoutError as e_timeout:
        logger.warning(f"Playwright overall timeout during tweet scraping for @{username_queried}: {e_timeout}")
        if page: 
            screenshot_path = f"error_screenshot_gettweets_timeout_{username_queried}_{datetime.now():%Y%m%d_%H%M%S}.png"
            try: await page.screenshot(path=screenshot_path)
            except Exception as e_ss: logger.error(f"Failed to save screenshot: {e_ss}")
            logger.info(f"Screenshot attempted to {screenshot_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred scraping tweets for @{username_queried}: {e}", exc_info=True)
        if page: 
            screenshot_path = f"error_screenshot_gettweets_exception_{username_queried}_{datetime.now():%Y%m%d_%H%M%S}.png"
            try: await page.screenshot(path=screenshot_path)
            except Exception as e_ss: logger.error(f"Failed to save screenshot: {e_ss}")
            logger.info(f"Screenshot attempted to {screenshot_path}")
    finally:
        if page and not page.is_closed():
            try: await page.close()
            except Exception as e_pc: logger.warning(f"Ignoring error closing page for @{username_queried}: {e_pc}")
        
        if context_manager:
            try: await context_manager.close()
            except Exception as e_cc:
                if "Target page, context or browser has been closed" in str(e_cc):
                    logger.warning(f"Context for @{username_queried} was already closed: {e_cc}")
                else:
                    logger.error(f"Error closing context for @{username_queried}: {e_cc}", exc_info=True)
        
        if browser_instance_st and not profile_dir_usable: 
            try: await browser_instance_st.close()
            except Exception as e_bc: logger.warning(f"Ignoring error closing non-persistent browser for @{username_queried}: {e_bc}")
            
    tweets_collected.sort(key=lambda x: x.get("timestamp", ""), reverse=True) 
    logger.info(f"Finished scraping. Collected {len(tweets_collected)} tweets for @{username_queried}.")
    return tweets_collected[:limit]


async def query_searx(query: str) -> list:
    logger.info(f"Querying Searx for: {query}")
    params = {'q': query, 'format': 'json', 'language': 'en-US'}
    if config.SEARX_PREFERENCES: 
        params['preferences'] = config.SEARX_PREFERENCES
        
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config.SEARX_URL, params=params, timeout=10) as response:
                response.raise_for_status() 
                results_json = await response.json()
                return results_json.get('results', [])[:5] 
    except aiohttp.ClientError as e:
        logger.error(f"Searx query failed for '{query}': {e}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from Searx for query: {query}")
    return []

# -------------------------------------------------------------------
# YouTube and Audio Transcription
# -------------------------------------------------------------------
async def fetch_youtube_transcript(url: str) -> str | None:
    try:
        video_id_match = re.search(r'(?:v=|\/|embed\/|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})', url)
        if not video_id_match:
            logger.warning(f"Could not extract YouTube video ID from URL: {url}")
            return None
        video_id = video_id_match.group(1)
        logger.info(f"Fetching transcript for YouTube video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        transcript = None
        try: 
            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
        except NoTranscriptFound: pass 

        if not transcript: 
            try:
                transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
            except NoTranscriptFound: pass 

        if not transcript: 
            available_langs = [t.language for t in transcript_list]
            logger.warning(f"No English transcript for {video_id}. Available: {available_langs}. Trying first available.")
            if available_langs:
                 transcript = transcript_list.find_generated_transcript([available_langs[0]]) 

        if transcript:
            fetched_transcript_data = transcript.fetch()
            full_text = " ".join([entry['text'] for entry in fetched_transcript_data])
            return f"(Language: {transcript.language}) {full_text}" if transcript.language != 'en' else full_text
        else:
            logger.warning(f"No transcript found at all for YouTube video: {url}")
            return None

    except Exception as e: 
        logger.error(f"Failed to fetch YouTube transcript for {url}: {e}", exc_info=True)
    return None

def transcribe_audio_file(file_path: str) -> str | None:
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found for transcription: {file_path}")
        return None
    try:
        logger.info(f"Loading Whisper model (base) for transcription...")
        model = whisper.load_model("base") 
        logger.info(f"Transcribing audio file: {file_path}")
        result = model.transcribe(file_path, fp16=torch.cuda.is_available())
        transcribed_text = result["text"]
        logger.info(f"Transcription successful for {file_path}.")
        return transcribed_text
    except Exception as e:
        logger.error(f"Whisper transcription failed for {file_path}: {e}", exc_info=True)
        return None
    finally:
        if 'model' in locals(): del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# -------------------------------------------------------------------
# Reminder Task
# -------------------------------------------------------------------
@tasks.loop(seconds=30) 
async def check_reminders():
    now = datetime.now()
    due_reminders_indices = []
    
    # logger.debug(f"Checking reminders... {len(reminders)} pending.") # Optional: verbose logging
    for i, reminder_tuple in enumerate(reminders): 
        reminder_time, channel_id, user_id, message_content, original_time_str = reminder_tuple
        # logger.debug(f"Checking reminder: Time {reminder_time}, Now {now}") # Optional
        if now >= reminder_time:
            logger.info(f"Reminder DUE for user {user_id} in channel {channel_id}: {message_content}")
            try:
                channel = await bot.fetch_channel(channel_id) # Use fetch_channel
                user = await bot.fetch_user(user_id) 
                
                channel_name_for_log = "DM" if isinstance(channel, discord.DMChannel) else getattr(channel, 'name', f"ID:{channel_id}")

                if channel and user:
                    logger.info(f"Sending reminder to {user.name} in {channel_name_for_log}: {message_content}")
                    embed = discord.Embed(
                        title=f"⏰ Reminder! (Set {original_time_str})", 
                        description=message_content,
                        color=discord.Color.blue(),
                        timestamp=reminder_time 
                    )
                    embed.set_footer(text=f"Reminder for {user.display_name}")
                    await channel.send(content=user.mention, embed=embed)
                    await send_tts_audio(channel, f"Reminder for {user.display_name}: {message_content}", base_filename=f"reminder_{user_id}_{channel_id}")
                else:
                    if not channel: logger.warning(f"Could not fetch channel for reminder: ChID {channel_id}")
                    if not user: logger.warning(f"Could not fetch user for reminder: UserID {user_id}")
                due_reminders_indices.append(i)
            except discord.errors.NotFound:
                 logger.warning(f"Channel or User not found for reminder: ChID {channel_id}, UserID {user_id}. Removing reminder.")
                 due_reminders_indices.append(i) # Also mark for removal if not found
            except Exception as e:
                logger.error(f"Failed to send reminder (ChID {channel_id}, UserID {user_id}): {e}", exc_info=True)
                # Decide if you want to keep the reminder for a retry or remove it. For now, removing.
                due_reminders_indices.append(i) 
    
    if due_reminders_indices:
        logger.info(f"Removing {len(due_reminders_indices)} due reminders.")
        for index in sorted(due_reminders_indices, reverse=True):
            reminders.pop(index)


def parse_time_string_to_delta(time_str: str) -> tuple[timedelta | None, str | None]:
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
            
    if not any(val > 0 for val in delta_args.values()): 
        return None, None
        
    time_delta = timedelta(**delta_args)
    descriptive_str = ", ".join(original_parts) if original_parts else "immediately" 
    if not descriptive_str and time_delta.total_seconds() > 0: 
        descriptive_str = "a duration"

    return time_delta, descriptive_str


# -------------------------------------------------------------------
# Slash Commands (Application Commands)
# -------------------------------------------------------------------

@bot.tree.command(name="remindme", description="Sets a reminder. E.g., 1h30m Check the oven.")
@app_commands.describe(
    time_duration="Duration (e.g., '10m', '2h30m', '1d').",
    reminder_message="The message for your reminder."
)
async def remindme_slash_command(interaction: discord.Interaction, time_duration: str, reminder_message: str):
    time_delta, descriptive_time_str = parse_time_string_to_delta(time_duration)

    if not time_delta or time_delta.total_seconds() <= 0:
        await interaction.response.send_message("Invalid time duration. Please use formats like '10m', '2h30m', '1d'. Minimum 1 second.", ephemeral=True)
        return

    reminder_time = datetime.now() + time_delta
    reminders.append((reminder_time, interaction.channel_id, interaction.user.id, reminder_message, descriptive_time_str))
    
    await interaction.response.send_message(f"Okay, {interaction.user.mention}! I'll remind you in {descriptive_time_str} about: \"{reminder_message}\"")
    logger.info(f"Reminder set via slash command for {interaction.user.name} at {reminder_time} for: {reminder_message}")

@bot.tree.command(name="roast", description="Generates a comedy routine based on a webpage.")
@app_commands.describe(url="The URL of the webpage to roast.")
async def roast_slash_command(interaction: discord.Interaction, url: str):
    logger.info(f"Roast command invoked by {interaction.user.name} for {url}.")
    # Deferral is now handled inside stream_llm_response_to_interaction
    try:
        webpage_text = await scrape_website(url)
        if not webpage_text or "Failed to scrape" in webpage_text or "Scraping timed out" in webpage_text:
            # Need to defer before sending a followup if we don't stream
            await interaction.response.defer(ephemeral=True)
            await interaction.followup.send(f"Sorry, I couldn't properly roast {url}. {webpage_text if webpage_text else 'Could not retrieve content.'}")
            return

        prompt_nodes = [
            get_system_prompt(),
            MsgNode(role="user", content=f"Analyze the content from {url} (provided below) and write a short, witty, and engaging comedy routine about it. Keep it light-hearted and observational.\n\nWebsite Content:\n{webpage_text[:3000]}")
        ]
        await stream_llm_response_to_interaction(interaction, prompt_nodes, title=f"Comedy Roast of {url}")
    except Exception as e_cmd:
        logger.error(f"Error during roast_slash_command execution for {url}: {e_cmd}", exc_info=True)
        if not interaction.response.is_done():
            try:
                await interaction.response.send_message(f"Sorry, an error occurred while roasting {url}: {str(e_cmd)[:1000]}", ephemeral=True)
            except Exception as e_resp_err:
                logger.error(f"Further error sending error response for roast {url}: {e_resp_err}")


@bot.tree.command(name="search", description="Performs a web search and summarizes results.")
@app_commands.describe(query="Your search query.")
async def search_slash_command(interaction: discord.Interaction, query: str):
    logger.info(f"Search command invoked by {interaction.user.name} for query: {query}.")
    try:
        await interaction.response.defer(thinking=True, ephemeral=False)
        logger.info(f"Search command deferred successfully for query: {query}.")
    except Exception as e_defer:
        logger.error(f"Error during defer in search for query '{query}': {e_defer}", exc_info=True)
        return
    
    try:
        search_results = await query_searx(query)
        if not search_results:
            await interaction.followup.send("No search results found for your query.")
            return
            
        search_snippets = []
        for i, r in enumerate(search_results):
            title = r.get('title', 'N/A')
            res_url = r.get('url', 'N/A') 
            snippet_text = r.get('content', r.get('description', 'No snippet available.'))
            search_snippets.append(f"{i+1}. **{discord.utils.escape_markdown(title)}** (<{res_url}>)\n    {discord.utils.escape_markdown(snippet_text[:250])}...")

        formatted_results = "\n\n".join(search_snippets)
        embed = discord.Embed(title=f"Top Search Results for: {query}", description=formatted_results[:config.EMBED_MAX_LENGTH], color=config.EMBED_COLOR["incomplete"])
        await interaction.followup.send(embed=embed) 

        prompt_nodes = [
            get_system_prompt(),
            MsgNode(role="user", content=f"Please provide a concise summary of the following search results for the query '{query}':\n\n{formatted_results[:3000]}") 
        ]
        
        # This will use the two-step context process now
        await stream_llm_response_to_interaction(interaction, prompt_nodes, title=f"Summary for: {query}", is_edit_of_original_response=True)

    except Exception as e_cmd:
        logger.error(f"Error during search_slash_command execution for query '{query}' after defer: {e_cmd}", exc_info=True)
        try:
            await interaction.followup.send(f"Sorry, an error occurred while searching for '{query}': {str(e_cmd)[:1000]}", ephemeral=True)
        except Exception as e_followup_err:
            logger.error(f"Further error sending error followup for search '{query}': {e_followup_err}")


@bot.tree.command(name="pol", description="Generates a sarcastic response to a political statement.")
@app_commands.describe(statement="The political statement.")
async def pol_slash_command(interaction: discord.Interaction, statement: str):
    logger.info(f"Pol command invoked by {interaction.user.name} with statement: {statement[:50]}.")
    # Deferral is now handled inside stream_llm_response_to_interaction
    try:
        system_content = (
            "You are a bot that generates extremely sarcastic, snarky, and troll-like comments "
            "to mock extremist political views or absurd political statements. Your goal is to be biting and humorous, "
            "undermining the statement without being directly offensive or vulgar. Focus on wit and irony."
        )
        prompt_nodes = [
            MsgNode(role="system", content=system_content),
            MsgNode(role="user", content=f"Generate a sarcastic comeback to this political statement: \"{statement}\"")
        ]
        # This will NOT use the two-step process because we are overriding the system prompt.
        # To use the two-step process here, the logic would need to be more complex.
        # For now, this specific command will behave as it did before.
        await stream_llm_response_to_interaction(interaction, prompt_nodes, title="Sarcastic Political Commentary")
    except Exception as e_cmd:
        logger.error(f"Error during pol_slash_command execution after defer: {e_cmd}", exc_info=True)
        if not interaction.response.is_done():
            await interaction.response.send_message(f"Sorry, an error occurred: {str(e_cmd)[:1000]}", ephemeral=True)


@bot.tree.command(name="gettweets", description="Fetches and summarizes recent tweets from a user.")
@app_commands.describe(username="The X/Twitter username (without @).", limit="Number of tweets to fetch (max 50).")
async def gettweets_slash_command(interaction: discord.Interaction, username: str, limit: app_commands.Range[int, 1, 50] = 10):
    logger.info(f"Gettweets command invoked by {interaction.user.name} for @{username}.")
    try:
        await interaction.response.defer(thinking=True, ephemeral=False)
        logger.info(f"Gettweets command deferred successfully for @{username}.")
    except discord.errors.NotFound as e_nf_defer: 
        logger.error(f"NotFound (Unknown Interaction) error during defer in gettweets for @{username}: {e_nf_defer}. Interaction ID: {interaction.id}. This interaction is likely lost.")
        return 
    except Exception as e_defer: 
        logger.error(f"Generic error during defer in gettweets for @{username}: {e_defer}", exc_info=True)
        return

    try:
        tweets = await scrape_latest_tweets(username.lstrip('@'), limit=limit)

        if not tweets:
            await interaction.followup.send(f"Could not fetch tweets for @{username.lstrip('@')}. The profile might be private, non-existent, or X is blocking the request. Scraping X is very unreliable.")
            return

        tweet_texts = [
            f"[{t.get('timestamp', 'N/A')}] @{t.get('original_author', username.lstrip('@'))}: {discord.utils.escape_markdown(t.get('content', 'N/A'))}" 
            for t in tweets
        ]
        raw_tweets_display = "\n\n".join(tweet_texts)

        embed_title = f"Recent Tweets from @{username.lstrip('@')}"
        if not raw_tweets_display: raw_tweets_display = "No tweet content could be displayed."

        raw_tweet_chunks = chunk_text(raw_tweets_display, config.EMBED_MAX_LENGTH)
        for i, chunk in enumerate(raw_tweet_chunks):
            chunk_title = embed_title if i == 0 else f"{embed_title} (cont.)"
            embed = discord.Embed(title=chunk_title, description=chunk, color=config.EMBED_COLOR["incomplete"])
            await interaction.followup.send(embed=embed) 

        prompt_nodes_summary = [
            get_system_prompt(),
            MsgNode(role="user", content=f"Summarize the key themes, topics, and overall sentiment from the following recent tweets by @{username.lstrip('@')}:\n\n{raw_tweets_display[:3500]}")
        ]
        
        # This will now use the two-step context process
        await stream_llm_response_to_interaction(
            interaction, 
            prompt_nodes_summary, 
            title=f"Tweet Summary for @{username.lstrip('@')}",
            is_edit_of_original_response=True 
        )

    except Exception as e_cmd:
        logger.error(f"Error during gettweets_slash_command execution for @{username} after defer: {e_cmd}", exc_info=True)
        try:
            await interaction.followup.send(f"Sorry, an error occurred while fetching tweets for @{username}: {str(e_cmd)[:1000]}", ephemeral=True)
        except Exception as e_followup_err:
            logger.error(f"Further error sending error followup for @{username}: {e_followup_err}")


@bot.tree.command(name="ap", description="Describes an attached image with a creative AP Photo twist.")
@app_commands.describe(image="The image to describe.", user_prompt="Optional additional prompt for the description.")
async def ap_slash_command(interaction: discord.Interaction, image: discord.Attachment, user_prompt: str = ""):
    logger.info(f"AP command invoked by {interaction.user.name}.")
    # Deferral is now handled inside stream_llm_response_to_interaction
    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            await interaction.response.defer(ephemeral=True)
            await interaction.followup.send("The attached file is not a valid image.")
            return

        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url_for_llm = f"data:{image.content_type};base64,{base64_image}"

        celebrities = ["Keanu Reeves", "Dwayne 'The Rock' Johnson", "Zendaya", "Tom Hanks", "Margot Robbie", "Ryan Reynolds", "Awkwafina", "Idris Elba", "Beyoncé", "Leonardo DiCaprio"]
        chosen_celebrity = random.choice(celebrities)

        llm_prompt_text = (
            f"You are an AP photo caption writer. Describe the attached image in a detailed and intricate way, "
            f"as if for a blind person. However, creatively replace the main subject or character in the image with {chosen_celebrity}. "
            f"Begin your response with 'AP Photo: {chosen_celebrity}...' "
            f"If the user provided an additional prompt, consider it: '{user_prompt}'"
        )
        
        prompt_nodes = [
            # The two-step process expects the standard system prompt, so we don't use get_system_prompt() here
            # for this custom command. It will behave as before.
            MsgNode(
                role="user",
                content=[ 
                    {"type": "text", "text": llm_prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url_for_llm}}
                ]
            )
        ]
        
        await stream_llm_response_to_interaction(interaction, prompt_nodes, title=f"AP Photo Description ft. {chosen_celebrity}")
    except Exception as e_cmd:
        logger.error(f"Error during ap_slash_command execution after defer: {e_cmd}", exc_info=True)
        if not interaction.response.is_done():
            await interaction.response.send_message(f"Sorry, an error occurred with the AP command: {str(e_cmd)[:1000]}", ephemeral=True)


@bot.tree.command(name="clearhistory", description="Clears the bot's message history for this channel.")
@app_commands.checks.has_permissions(manage_messages=True)
async def clearhistory_slash_command(interaction: discord.Interaction):
    if interaction.channel_id in message_history:
        message_history[interaction.channel_id] = []
        logger.info(f"Message history cleared for channel {interaction.channel_id} by {interaction.user.name} via slash command.")
        await interaction.response.send_message("Message history for this channel has been cleared.", ephemeral=True)
    else:
        await interaction.response.send_message("No history to clear for this channel.", ephemeral=True)

@clearhistory_slash_command.error
async def clearhistory_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("You don't have permission to clear history (Manage Messages required).", ephemeral=True)
    else:
        logger.error(f"Error in clearhistory_slash_command: {error}", exc_info=True)
        if not interaction.response.is_done():
            await interaction.response.send_message("An unexpected error occurred with this command.", ephemeral=True)
        else:
            await interaction.followup.send("An unexpected error occurred with this command.", ephemeral=True)


# -------------------------------------------------------------------
# Main Event Handlers
# -------------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot: 
        return

    await bot.process_commands(message) 
    
    prefixes = await bot.get_prefix(message)
    is_command_attempt = False
    if isinstance(prefixes, str): 
        if message.content.startswith(prefixes):
            is_command_attempt = True
    elif isinstance(prefixes, (list, tuple)): 
        if any(message.content.startswith(p) for p in prefixes):
            is_command_attempt = True
    
    if is_command_attempt:
        return


    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions

    if not (is_dm or is_mentioned):
        if config.ALLOWED_CHANNEL_IDS and message.channel.id not in config.ALLOWED_CHANNEL_IDS:
            is_thread_in_allowed_channel = isinstance(message.channel, discord.Thread) and message.channel.parent_id in config.ALLOWED_CHANNEL_IDS
            if not is_thread_in_allowed_channel:
                return
    
    if config.ALLOWED_ROLE_IDS and not is_dm and not any(role.id in config.ALLOWED_ROLE_IDS for role in message.author.roles):
        logger.debug(f"Message from {message.author.name} without allowed role for general response. Ignoring.")
        return
    
    channel_display_name = ""
    if is_dm:
        channel_display_name = f"DM with {message.author.name}"
    elif hasattr(message.channel, 'name'):
        channel_display_name = message.channel.name
    else:
        channel_display_name = f"Channel ID {message.channel.id}"

    logger.info(f"General message for LLM from {message.author.name} in {channel_display_name}: {message.content[:50]}")


    if message.channel.id not in message_history:
        message_history[message.channel.id] = []

    current_message_content_parts = []
    user_message_text = message.content

    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("audio/"):
                try:
                    if not os.path.exists("temp"): os.makedirs("temp")
                    audio_filename = f"temp/temp_audio_{attachment.id}.{attachment.filename.split('.')[-1]}"
                    await attachment.save(audio_filename)
                    logger.info(f"Transcribing audio attachment: {audio_filename}")
                    transcription = transcribe_audio_file(audio_filename)
                    if os.path.exists(audio_filename): os.remove(audio_filename) 
                    if transcription:
                        user_message_text = (user_message_text + " " + transcription).strip()
                        await message.reply(f"*Transcribed audio: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"*", silent=True)
                except Exception as e:
                    logger.error(f"Error processing audio attachment: {e}", exc_info=True)
                break 

    if user_message_text: 
        current_message_content_parts.append({"type": "text", "text": user_message_text})

    image_added_to_current_message = False
    if message.attachments:
        for i, attachment in enumerate(message.attachments):
            if i >= config.MAX_IMAGES_PER_MESSAGE: break 
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    image_bytes = await attachment.read()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    image_url_for_llm = f"data:{attachment.content_type};base64,{base64_image}"
                    current_message_content_parts.append({"type": "image_url", "image_url": {"url": image_url_for_llm}})
                    image_added_to_current_message = True
                    logger.info(f"Added image attachment for LLM processing: {attachment.filename}")
                except Exception as e:
                    logger.error(f"Error processing image attachment {attachment.filename}: {e}", exc_info=True)

    detected_urls_in_text = detect_urls(user_message_text) 
    scraped_content_for_llm = "" 
    if detected_urls_in_text:
        for i, url in enumerate(detected_urls_in_text):
            if i >= 2 : break 
            logger.info(f"Processing URL from message: {url}")
            content_piece = None
            
            youtube_match = re.search(r'(?:v=|\/|embed\/|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})', url)
            if youtube_match: # Check if it's a YouTube URL
                transcript = await fetch_youtube_transcript(url)
                if transcript:
                    content_piece = f"\n\n--- YouTube Transcript for {url} ---\n{transcript[:1500]}...\n--- End Transcript ---" 
            else: # General website scraping
                scraped_text = await scrape_website(url)
                if scraped_text and "Failed to scrape" not in scraped_text and "Scraping timed out" not in scraped_text:
                     content_piece = f"\n\n--- Webpage Content for {url} ---\n{scraped_text[:1500]}...\n--- End Webpage Content ---"
            
            if content_piece:
                scraped_content_for_llm += content_piece
            await asyncio.sleep(0.2) 

    if scraped_content_for_llm:
        text_part_found = False
        for part in current_message_content_parts:
            if part["type"] == "text":
                part["text"] = scraped_content_for_llm + "\n\nUser's message: " + part["text"]
                text_part_found = True
                break
        if not text_part_found: 
            current_message_content_parts.insert(0, {"type": "text", "text": scraped_content_for_llm + "\n\n(User sent an attachment, possibly with URLs in a non-text part, or no text)"})


    if not current_message_content_parts: 
        if image_added_to_current_message: 
             current_message_content_parts.append({"type": "text", "text": "The user sent this image. Please describe it or respond to it if there's an implicit question."})
        else:
            logger.info("Ignoring message with no processable text, audio, or image content after all processing.")
            return

    user_msg_node_content = current_message_content_parts if len(current_message_content_parts) > 1 or image_added_to_current_message else current_message_content_parts[0]["text"]
    
    message_history[message.channel.id].append(MsgNode(role="user", content=user_msg_node_content, name=str(message.author.id)))
    message_history[message.channel.id] = message_history[message.channel.id][-config.MAX_MESSAGE_HISTORY:]

    llm_conversation_history = [get_system_prompt()] + message_history[message.channel.id]
    
    await stream_llm_response_to_message(message, llm_conversation_history)

@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    """Handles deleting bot messages via reaction."""
    # Ignore reactions from the bot itself
    if payload.user_id == bot.user.id:
        return

    # Check if the reaction is the one we're looking for (❌)
    if str(payload.emoji) != '❌':
        return

    try:
        channel = await bot.fetch_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
    except discord.NotFound:
        logger.warning(f"Could not find message or channel for reaction cleanup: msg_id={payload.message_id}")
        return
    except discord.Forbidden:
        logger.warning(f"Lacking permissions to fetch message for reaction cleanup in channel: {payload.channel_id}")
        return

    # Only proceed if the message was sent by the bot
    if message.author.id != bot.user.id:
        return

    reacting_user = await bot.fetch_user(payload.user_id)
    can_delete = True

    # Check for admin/mod permissions
    if isinstance(channel, discord.TextChannel):
        member = await channel.guild.fetch_member(payload.user_id)
        if member and member.guild_permissions.manage_messages:
            can_delete = True
            logger.info(f"Admin '{reacting_user.name}' authorized to delete bot message {message.id}.")

    # Check if the reacting user was the one who triggered the bot's response
    if not can_delete and message.reference and message.reference.message_id:
        try:
            original_message = await channel.fetch_message(message.reference.message_id)
            if original_message.author.id == payload.user_id:
                can_delete = True
                logger.info(f"Original author '{reacting_user.name}' authorized to delete bot message {message.id}.")
        except discord.NotFound:
            logger.warning(f"Could not find the original message that bot replied to: {message.reference.message_id}")
    
    # This covers slash command interactions where the user is directly available
    if not can_delete and message.interaction and message.interaction.user.id == payload.user_id:
        can_delete = True
        logger.info(f"Interaction initiator '{reacting_user.name}' authorized to delete bot message {message.id}.")


    if can_delete:
        try:
            await message.delete()
            logger.info(f"Message {message.id} deleted by {reacting_user.name} using ❌ reaction.")
        except discord.Forbidden:
            logger.error(f"Failed to delete message {message.id}. Bot lacks 'Manage Messages' permission.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while deleting message {message.id}: {e}", exc_info=True)


# -------------------------------------------------------------------
# Bot Startup & Error Handling
# -------------------------------------------------------------------
@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord! ID: {bot.user.id}')
    logger.info(f"discord.py version: {discord.__version__}")
    logger.info(f"Operating in channels: {config.ALLOWED_CHANNEL_IDS if config.ALLOWED_CHANNEL_IDS else 'All permitted by default'}")
    logger.info(f"Restricted to roles: {config.ALLOWED_ROLE_IDS if config.ALLOWED_ROLE_IDS else 'None'}")
    
    try:
        synced = await bot.tree.sync() 
        logger.info(f"Synced {len(synced)} slash commands globally.")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}", exc_info=True)

    if not check_reminders.is_running():
        check_reminders.start()
    await bot.change_presence(activity=discord.Game(name="with /commands | Ask me anything!"))

@bot.tree.error 
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    command_name = interaction.command.name if interaction.command else 'unknown_command'
    logger.error(f"Slash command error for '{command_name}': {error}", exc_info=True)
    
    error_message = "An unexpected error occurred with this slash command."
    original_error_is_unknown_interaction = False

    if isinstance(error, app_commands.CommandInvokeError): 
        original_error = error.original
        if isinstance(original_error, discord.errors.NotFound) and original_error.code == 10062: # Unknown Interaction
             error_message = "The command took too long to respond initially, or the interaction expired. Please try again."
             logger.warning(f"Original 'Unknown Interaction' error for {command_name}. Interaction ID: {interaction.id}")
             original_error_is_unknown_interaction = True
        else:
            error_message = f"Command '{command_name}' failed: {str(original_error)[:500]}"
    elif isinstance(error, app_commands.CommandNotFound): 
        error_message = "Command not found. This is unexpected."
    elif isinstance(error, app_commands.MissingPermissions):
        error_message = f"You lack the required permissions: {', '.join(error.missing_permissions)}"
    elif isinstance(error, app_commands.BotMissingPermissions):
        error_message = f"I lack the required permissions: {', '.join(error.missing_permissions)}"
    elif isinstance(error, app_commands.CheckFailure): 
        error_message = "You do not meet the requirements to use this command."
    elif isinstance(error, app_commands.CommandOnCooldown):
        error_message = f"This command is on cooldown. Try again in {error.retry_after:.2f} seconds."
    elif isinstance(error, app_commands.TransformerError): 
        error_message = f"Invalid argument: {error.value}. Type expected: {error.type}."
    
    if original_error_is_unknown_interaction: # If defer() itself failed with "Unknown Interaction", we can't respond.
        return

    try:
        if interaction.response.is_done():
            await interaction.followup.send(error_message, ephemeral=True)
        else:
            await interaction.response.send_message(error_message, ephemeral=True)
    except discord.errors.HTTPException as ehttp:
        if ehttp.code == 40060: 
            logger.warning(f"Error handler: Interaction already acknowledged for '{command_name}'. Trying followup for error message. Original error: {error}")
            try:
                await interaction.followup.send(error_message, ephemeral=True)
            except Exception as e_followup:
                logger.error(f"Error handler: Failed to send error followup for '{command_name}': {e_followup}")
        else: 
            logger.error(f"Error handler: HTTPException when sending error for '{command_name}': {ehttp}. Original error: {error}")
    except discord.errors.NotFound: 
        logger.error(f"Error handler: Interaction not found (timed out or deleted) for '{command_name}'. Original error: {error}")
    except Exception as e_generic: 
        logger.error(f"Error handler: Generic error when sending error for '{command_name}': {e_generic}. Original error: {error}")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound):
        pass 
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.reply(f"You're missing an argument for !{ctx.command.name}: {error.param.name}.", silent=True)
    elif isinstance(error, commands.BadArgument):
        await ctx.reply(f"Invalid argument provided for !{ctx.command.name}.", silent=True)
    elif isinstance(error, commands.CheckFailure):
        await ctx.reply("You don't have permissions for that prefix command.", silent=True)
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Error invoking prefix command !{ctx.command.name}: {error.original}", exc_info=error.original)
        await ctx.reply(f"An error occurred with !{ctx.command.name}: {error.original}", silent=True)
    else:
        logger.error(f"Unhandled prefix command error: {error}", exc_info=True)


if __name__ == "__main__":
    if not config.DISCORD_BOT_TOKEN:
        logger.critical("DISCORD_BOT_TOKEN is not set. Bot cannot start.")
    else:
        try:
            bot.run(config.DISCORD_BOT_TOKEN, log_handler=None) 
        except discord.LoginFailure:
            logger.critical("Failed to log in with the provided Discord token. Please check it.")
        except Exception as e:
            logger.critical(f"An unexpected error occurred during bot startup: {e}", exc_info=True)
