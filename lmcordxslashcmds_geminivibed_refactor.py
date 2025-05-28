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
from typing import Union, Optional 

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
        self.SEARX_PREFERENCES = os.getenv("SEARX_PREFERENCES", "eJx1V8uy2zYM_Zp6o4mnaRadLrzqTLftTLvXQCQsISIJhg_bul9f0JIsyrpZRNcEQRA4AA4YBQl7DoTx0qPDAOaX3_50eI_yh5J8oiJ0CssvVgSmsagJTgZcn6HHC-TEJ8MKDF7QncpSsfUGE1565t7giawotj7wY7r8BSbiyWIaWF_--fvf_04RrhgRghouv57SgBYvkYqBU8CYTYotu1Y8ahN0y3HN1MommxuGC4Mszxz603ysjWkyi18KXcLQgqHeWfm9nAd9A4lJt8u9s_RHxjC15NpESQzMQnJXcpTEqgpszOoAReiMGEDXkxPs_uihb9saIQEMnIPYlBvohm17JYMFVvBjYykEDrVMPG_k28TEoVaW040hlx_NnUZq22dSIDzoJve9ctNR6rIaMS0KXdLU95sZpdSXJCdupJGfgsAxBrzWRkR21zcSqNutFib09VLlYAhriUb8EIxbmyOpstZ9o_GJGrGLO1UWF0Mz5G5xE-VG0m3LkvdQFCSG8q_n5lk0cnr--4LIghehfMtpy9_JF7A3C0nSMe38wzRZdgIhblpfH5Xhqw5cnFixugbEJvI13SFgoymgkpxMi8vXQG4kUJUBuSsxL_u9FA-s8SnW2GHo171nPzRRDWwg1NAvO97AVKogbt7UO5YlO7HKYu9Z6xr8AboA5bPcODxPLwtyUJ0lJz-Jc_xctlocSY0QK4cMyRVhagrCkaqNjy_LVqXLXUx4DnGt2w8zBVKVeWECUB7couC8XX9NAJshCwI6rlGxRweSFw0JIgpDVEiuez53hhQsNbjfl12uHPcgIfYU1-bxusP6iKcgLNlBFZhYP2u8rQdyJz2_OzL5tVd_3MGluiCfgqZwbNsuTBsFhJGFza6G7ytWccxddikv65g9hhxfIMzcVSqsKZ9XeYKFkgd8qQl2idRnVSfXhuQLOVfuJR4nThwHHkta1oiFbUlLHwQp9gq7CYZS-hVfFME-ujt1U60io-IBTofSt5u00FvHPMZ34ZutkX5kTviuFTkHdZAKpJHSdBDz9Bb112_ffn9sYems0W18EfHDga31hXoQx1ri4FZyswlC7qYe7Vr7HjGkXCrr1W7PQSFXjIV079hVWyHbztS6d37QyE6qromTYzfZis_id3_298rhEqQVCq79kd687WL20ntmLqOfiI8sLKW1tfOM-ZDTxm8vQydpw92IXOZgQ7sESwFSzaWdFIYC6zeGKKEsJF_FQq5_Xx-dfUp35fOUzJhW2BZOKJWC1VkZdFLqatqPuhtJM4nPlWLxrxDRTvE1xz6XNncseQ9bwq5CVmPYDxWqeVjmyjY3-4XWN9vLmDhAsMh3ICyyAwxkdbd59J2EArblSvt8FbpxvTy_9mXh-tsOAksPZTjrLYKCckwyXJOM73WG-0G4xq2rmRoPQczig78yADSlw7MroidYy_qlLLUqlbX3KN0plffhu4XsoszbOFQuTJzfmvcleT18gIw8NcrEqdRuZJHrPqaxaDzH5Ktb2cy87YfqKaNJpQ92uzRbaXsrj5EmBXDRyFjStZmgHY2VIKVwppUWtlerN1l6Ml4KSI_zsjoPHJOwHMrTWqKfB8ROgaGdX973IC_iw7YMh1YNqMbDjmDQSp5HnOKLFn7iQ0F9XhysLCkVdNX8n5ZJ3u9GHmefaJqrPOSvfPQxQHmCtvKKlThsGbknGXhSmJf_AWSO7BY=&q=%s")


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
    text = re.sub(r'[\*#_~\<\>\[\]\(\)]+', '', text) # Remove common markdown
    text = re.sub(r'http[s]?://\S+', '', text) # Remove URLs
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

    # logger.info(f"Requesting TTS for {filename_suffix}: {cleaned_segment[:100]}...") # Optional: more verbose logging
    tts_audio_data = await tts_request(cleaned_segment)
    
    actual_destination_channel: Optional[discord.abc.Messageable] = None
    if isinstance(destination, discord.Interaction):
        actual_destination_channel = destination.channel
    elif isinstance(destination, discord.Message):
        actual_destination_channel = destination.channel
    elif isinstance(destination, discord.abc.Messageable): # Covers TextChannel, DMChannel, Thread, etc.
        actual_destination_channel = destination
    
    if not actual_destination_channel:
        logger.warning(f"TTS destination channel could not be resolved for type {type(destination)}")
        return

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
        await asyncio.sleep(0.5) 
        await _send_audio_segment(destination, response_text, "main_response", is_thought=False, base_filename=base_filename)
    else:
        logger.info("No <think> tags found. Processing full text for TTS.")
        await _send_audio_segment(destination, text_to_speak, "full", is_thought=False, base_filename=base_filename)

# -------------------------------------------------------------------
# Core LLM Interaction Logic
# -------------------------------------------------------------------

async def get_context_aware_llm_stream(prompt_messages: list[MsgNode], is_vision_request: bool) -> tuple[Optional[AsyncStream], str]:
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
        last_user_message_node
    ]

    generated_context = "Context generation failed or was not applicable."
    try:
        context_response = await llm_client.chat.completions.create(
            model=config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL,
            messages=[msg.to_dict() for msg in context_generation_prompt],
            max_tokens=250,
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

    logger.info("Step 2: Streaming final response with injected context.")
    final_prompt_messages = [MsgNode(m.role, m.content, m.name) for m in prompt_messages]
    final_user_message_node = next((msg for msg in reversed(final_prompt_messages) if msg.role == 'user'), None)
    
    if not final_user_message_node: 
        logger.error("Critical error: final_user_message_node is None in get_context_aware_llm_stream")
        return None, generated_context


    original_question = ""
    if isinstance(final_user_message_node.content, str):
        original_question = final_user_message_node.content
    elif isinstance(final_user_message_node.content, list):
        text_part = next((part['text'] for part in final_user_message_node.content if part['type'] == 'text'), "")
        original_question = text_part

    injected_prompt_text = (
        f"<model_generated_suggested_context>\n{generated_context}\n</model_generated_suggested_context>\n\n"
        f"<user_question>\nWith that context in mind, please respond to the following:\n{original_question}\n</user_question>"
    )

    if isinstance(final_user_message_node.content, str):
        final_user_message_node.content = injected_prompt_text
    elif isinstance(final_user_message_node.content, list):
        text_part_found = False
        for part in final_user_message_node.content:
            if part['type'] == 'text':
                part['text'] = injected_prompt_text
                text_part_found = True
                break
        if not text_part_found:
            final_user_message_node.content.insert(0, {"type": "text", "text": injected_prompt_text})

    current_model = config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL
    logger.info(f"Using model for final streaming: {current_model}")

    try:
        final_stream = await llm_client.chat.completions.create(
            model=current_model,
            messages=[msg.to_dict() for msg in final_prompt_messages],
            max_tokens=config.MAX_COMPLETION_TOKENS,
            stream=True,
            temperature=0.7,
        )
        return final_stream, generated_context
    except Exception as e:
        logger.error(f"Failed to create LLM stream: {e}", exc_info=True)
        return None, generated_context


async def _stream_llm_handler(
    interaction: discord.Interaction, 
    prompt_messages: list[MsgNode],
    title: str,
    initial_message_to_edit: Optional[discord.Message] = None
) -> str:
    sent_messages: list[discord.Message] = []
    full_response_content = ""
    channel = interaction.channel 
    if not channel: 
        logger.error(f"_stream_llm_handler: Interaction channel is None for interaction ID {interaction.id}.")
        return ""


    current_initial_message: Optional[discord.Message] = None
    if initial_message_to_edit:
        current_initial_message = initial_message_to_edit
    else:
        # This path is for when we need to start a new followup chain for the stream
        placeholder_embed = discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"])
        try:
            current_initial_message = await interaction.followup.send(embed=placeholder_embed, wait=True)
        except discord.HTTPException as e:
            logger.error(f"Failed to send initial followup for stream '{title}': {e}")
            return "" # Cannot proceed if initial message fails
    
    if current_initial_message:
        sent_messages.append(current_initial_message)
    else: 
        logger.error(f"Failed to establish an initial message for streaming title '{title}'.")
        return ""

    try:
        is_vision_request = any(isinstance(p.content, list) and any(c.get("type") == "image_url" for c in p.content) for p in prompt_messages)
        stream, generated_context = await get_context_aware_llm_stream(prompt_messages, is_vision_request)

        if stream is None: 
            error_text = f"**Model-Generated Suggested Context:**\n> {generated_context.replace(chr(10), ' ')}\n\n---\n**Response:**\nFailed to get response from LLM."
            error_embed = discord.Embed(title=title, description=error_text, color=config.EMBED_COLOR["error"])
            if sent_messages: await sent_messages[0].edit(embed=error_embed)
            return ""


        response_prefix = f"**Model-Generated Suggested Context:**\n> {generated_context.replace(chr(10), ' ')}\n\n---\n**Response:**\n"
        last_edit_time = asyncio.get_event_loop().time()
        accumulated_delta_for_update = "" 

        # Edit the first message to show context (if it's not a newly created followup which already has a placeholder)
        if initial_message_to_edit and current_initial_message: 
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
                
                logger.info(f"Stream update: Title '{title}'. Chunks: {len(text_chunks)}. Sent msgs: {len(sent_messages)}. AccumDelta: {len(accumulated_delta_for_update)}")
                accumulated_delta_for_update = "" 

                for i, chunk_content in enumerate(text_chunks):
                    embed = discord.Embed(
                        title=title if i == 0 else f"{title} (cont.)",
                        description=chunk_content,
                        color=config.EMBED_COLOR["incomplete"]
                    )
                    try:
                        if i < len(sent_messages):
                            logger.debug(f"Editing sent_messages[{i}] for chunk {i+1}/{len(text_chunks)} for title '{title}'.")
                            await sent_messages[i].edit(embed=embed)
                        else:
                            logger.info(f"Sending NEW message for overflow chunk {i+1}/{len(text_chunks)} for title '{title}'.")
                            if channel: 
                                new_msg = await channel.send(embed=embed)
                                sent_messages.append(new_msg)
                                logger.info(f"Sent new message for chunk {i+1} for '{title}', new sent_messages count: {len(sent_messages)}")
                            else:
                                logger.error(f"Cannot send overflow chunk {i+1} for '{title}': channel is None.")
                                break 
                    except discord.HTTPException as e_edit_send: 
                        logger.warning(f"Failed edit/send embed part {i+1} (stream) for '{title}': {e_edit_send}")
                
                last_edit_time = current_time
        
        # Process any remaining accumulated delta after the loop
        if accumulated_delta_for_update:
            display_text = response_prefix + full_response_content
            text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
            logger.info(f"Final stream accumulation processing: Title '{title}'. Chunks: {len(text_chunks)}. Sent msgs: {len(sent_messages)}.")
            for i, chunk_content in enumerate(text_chunks):
                embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content, color=config.EMBED_COLOR["incomplete"])
                try:
                    if i < len(sent_messages): await sent_messages[i].edit(embed=embed)
                    else:
                        if channel: sent_messages.append(await channel.send(embed=embed))
                except discord.HTTPException as e: logger.warning(f"Failed final accum. edit/send for '{title}': {e}")


        # Final update to set final color and content
        final_display_text = response_prefix + full_response_content
        final_chunks = chunk_text(final_display_text, config.EMBED_MAX_LENGTH)
        logger.info(f"Finalizing stream: Title '{title}'. Final Chunks: {len(final_chunks)}. Current sent msgs: {len(sent_messages)}.")

        if len(final_chunks) < len(sent_messages):
            for k in range(len(final_chunks), len(sent_messages)):
                try: await sent_messages[k].delete()
                except discord.HTTPException: pass 
            sent_messages = sent_messages[:len(final_chunks)]
        
        for i, chunk_content in enumerate(final_chunks):
            embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content, color=config.EMBED_COLOR["complete"])
            if i < len(sent_messages):
                logger.debug(f"Final color edit of sent_messages[{i}] for chunk {i+1}/{len(final_chunks)} for title '{title}'.")
                await sent_messages[i].edit(embed=embed)
            else: 
                logger.info(f"Final color send NEW message for overflow chunk {i+1}/{len(final_chunks)} for title '{title}'.")
                if channel:
                    new_msg = await channel.send(embed=embed)
                    sent_messages.append(new_msg)
                    logger.info(f"Sent new message (finalization color) for chunk {i+1} for '{title}', new sent_messages count: {len(sent_messages)}")
                else:
                    logger.error(f"Cannot send final color overflow chunk {i+1} for '{title}': channel is None.")
                    break

        if not full_response_content.strip() and sent_messages: # Handle empty LLM response
            empty_text = response_prefix + ("\nSam didn't provide a response." if generated_context != "Context generation failed or was not applicable." else "\nSam had an issue and couldn't respond.")
            await sent_messages[0].edit(embed=discord.Embed(title=title, description=empty_text, color=config.EMBED_COLOR["error"]))

    except Exception as e:
        logger.error(f"Error in _stream_llm_handler for '{title}': {e}", exc_info=True)
        error_embed = discord.Embed(title=title, description=f"An error occurred: {str(e)[:1000]}", color=config.EMBED_COLOR["error"])
        if sent_messages: 
            try: await sent_messages[0].edit(embed=error_embed)
            except discord.HTTPException: pass 
        elif interaction: 
            try: await interaction.followup.send(embed=error_embed, ephemeral=True)
            except discord.HTTPException: pass 
            
    return full_response_content


async def stream_llm_response_to_interaction(
    interaction: discord.Interaction, 
    prompt_messages: list, 
    title: str = "Sam's Response", 
    force_new_followup_flow: bool = False
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
            try: await interaction.followup.send("Issue initializing response. Please try again.", ephemeral=True)
            except discord.HTTPException: pass 
            return 
    
    full_response_content = await _stream_llm_handler(
        interaction=interaction,
        prompt_messages=prompt_messages,
        title=title,
        initial_message_to_edit=initial_msg_for_handler
    )

    if full_response_content:
        channel_id = interaction.channel_id
        if channel_id not in message_history: message_history[channel_id] = []
        message_history[channel_id].append(MsgNode(role="assistant", content=full_response_content, name=str(bot.user.id)))
        message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:]
        
        tts_base_id = interaction.id 
        if initial_msg_for_handler: 
            tts_base_id = initial_msg_for_handler.id
        
        await send_tts_audio(interaction, full_response_content, f"interaction_{tts_base_id}")


async def stream_llm_response_to_message(
    target_message: discord.Message, 
    prompt_messages: list, 
    title: str = "Sam's Response"
):
    # For regular messages, we create a new reply message that will be edited.
    initial_embed = discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"])
    reply_message: Optional[discord.Message] = None
    try:
        reply_message = await target_message.reply(embed=initial_embed, silent=True)
    except discord.HTTPException as e:
        logger.error(f"Failed to send initial reply for message stream '{title}': {e}")
        return # Cannot proceed

    sent_messages = [reply_message] if reply_message else []
    full_response_content = ""
    channel = target_message.channel

    try:
        is_vision_request = any(isinstance(p.content, list) and any(c.get("type") == "image_url" for c in p.content) for p in prompt_messages)
        stream, generated_context = await get_context_aware_llm_stream(prompt_messages, is_vision_request)

        if stream is None:
            error_text = f"**Model-Generated Suggested Context:**\n> {generated_context.replace(chr(10), ' ')}\n\n---\n**Response:**\nFailed to get response from LLM."
            if sent_messages: await sent_messages[0].edit(embed=discord.Embed(title=title, description=error_text, color=config.EMBED_COLOR["error"]))
            return

        response_prefix = f"**Model-Generated Suggested Context:**\n> {generated_context.replace(chr(10), ' ')}\n\n---\n**Response:**\n"
        if sent_messages: # Check if reply_message was successfully created
            await sent_messages[0].edit(embed=discord.Embed(title=title, description=response_prefix + "⏳ Thinking...", color=config.EMBED_COLOR["incomplete"]))
        
        last_edit_time = asyncio.get_event_loop().time()
        accumulated_delta_for_update = ""

        async for chunk_data in stream:
            delta_content = chunk_data.choices[0].delta.content or "" if chunk_data.choices and chunk_data.choices[0].delta else ""
            if delta_content: 
                full_response_content += delta_content
                accumulated_delta_for_update += delta_content

            current_time = asyncio.get_event_loop().time()
            if accumulated_delta_for_update and (current_time - last_edit_time >= (1.0 / config.EDITS_PER_SECOND) or len(accumulated_delta_for_update) > 200):
                display_text = response_prefix + full_response_content
                text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
                logger.info(f"Msg Stream update: Title '{title}'. Chunks: {len(text_chunks)}. Sent msgs: {len(sent_messages)}. AccumDelta: {len(accumulated_delta_for_update)}")
                accumulated_delta_for_update = ""

                for i, chunk_content in enumerate(text_chunks):
                    embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content, color=config.EMBED_COLOR["incomplete"])
                    try:
                        if i < len(sent_messages): await sent_messages[i].edit(embed=embed)
                        else: sent_messages.append(await channel.send(embed=embed))
                    except discord.HTTPException as e: logger.warning(f"Msg stream: Edit/send failed for chunk {i+1} of '{title}': {e}")
                last_edit_time = current_time
        
        if accumulated_delta_for_update: # Process remaining
            display_text = response_prefix + full_response_content
            text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
            logger.info(f"Msg Stream final accumulation: Title '{title}'. Chunks: {len(text_chunks)}. Sent msgs: {len(sent_messages)}.")
            for i, chunk_content in enumerate(text_chunks):
                embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content, color=config.EMBED_COLOR["incomplete"])
                try:
                    if i < len(sent_messages): await sent_messages[i].edit(embed=embed)
                    else: sent_messages.append(await channel.send(embed=embed))
                except discord.HTTPException as e: logger.warning(f"Msg stream final accum edit/send for '{title}': {e}")


        final_display_text = response_prefix + full_response_content
        final_chunks = chunk_text(final_display_text, config.EMBED_MAX_LENGTH)
        logger.info(f"Msg Finalizing: Title '{title}'. Chunks: {len(final_chunks)}. Sent: {len(sent_messages)}.")

        if len(final_chunks) < len(sent_messages):
            for k in range(len(final_chunks), len(sent_messages)):
                try: await sent_messages[k].delete()
                except discord.HTTPException: pass
            sent_messages = sent_messages[:len(final_chunks)]

        for i, chunk_content in enumerate(final_chunks):
            embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content, color=config.EMBED_COLOR["complete"])
            if i < len(sent_messages): await sent_messages[i].edit(embed=embed)
            else: sent_messages.append(await channel.send(embed=embed))

        if not full_response_content.strip() and sent_messages:
            empty_text = response_prefix + ("\nSam didn't provide a response." if generated_context != "Context generation failed or was not applicable." else "\nSam had an issue and couldn't respond.")
            await sent_messages[0].edit(embed=discord.Embed(title=title, description=empty_text, color=config.EMBED_COLOR["error"]))
    
    except Exception as e:
        logger.error(f"Error streaming to message '{title}': {e}", exc_info=True)
        if sent_messages: 
            try:
                await sent_messages[0].edit(embed=discord.Embed(title=title, description=f"An error occurred: {str(e)[:1000]}", color=config.EMBED_COLOR["error"]))
            except discord.HTTPException: pass


    if full_response_content:
        channel_id = target_message.channel.id
        if channel_id not in message_history: message_history[channel_id] = []
        message_history[channel_id].append(MsgNode(role="assistant", content=full_response_content, name=str(bot.user.id)))
        message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:]
        await send_tts_audio(target_message, full_response_content, base_filename=f"message_{target_message.id}")

# -------------------------------------------------------------------
# Text-to-Speech (TTS)
# -------------------------------------------------------------------
async def tts_request(text: str, speed: float = 1.3) -> bytes | None:
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
            const articleText = article.textContent || '';
            if (articleText.match(/grok/i)) { return false; } 
            if (b.closest('[role="blockquote"]')) { return false; } 
            return true;
        });
    while (clicks < maxClicks) {
        const buttonsToClick = getButtons();
        if (buttonsToClick.length === 0) break;
        const button = buttonsToClick[0];
        try { button.click(); clicks++; } catch (e) { break; }
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
            const primaryLinkElement = timeTag ? timeTag.closest('a[href*="/status/"]') : null;
            if (primaryLinkElement) { tweetLink = primaryLinkElement.href; }
            else {
                const articleLinks = Array.from(article.querySelectorAll('a[href*="/status/"]'));
                if (articleLinks.length > 0) {
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
                    const hrefParts = userLinkInContext.href.split('/');
                    reposted_by = hrefParts.filter(part => !['analytics', 'likes', 'media', 'status', 'with_replies', 'following', 'followers', ''].includes(part)).pop();
                }
            }
            if (content || article.querySelector('[data-testid="tweetPhoto"], [data-testid="videoPlayer"]')) {
                tweets.push({
                    id: id || `no-id-${Date.now()}-${Math.random()}`, username, content, timestamp: timestamp || new Date().toISOString(),
                    is_repost, reposted_by, tweet_url: tweetLink || (id ? `https://x.com/${username}/status/${id}` : '')
                });
            }
        } catch (e) { /* console.warn('Error extracting tweet:', e); */ }
    });
    return tweets;
}
"""

async def scrape_website(url: str) -> str | None:
    logger.info(f"Scraping website: {url}")
    user_data_dir = os.path.join(os.getcwd(), ".pw-profile") 
    profile_dir_usable = True
    if not os.path.exists(user_data_dir):
        try: os.makedirs(user_data_dir, exist_ok=True)
        except OSError as e: profile_dir_usable = False; logger.error(f"Could not create .pw-profile directory: {e}")
            
    context_manager = None 
    browser_instance_sw = None
    page = None
    try:
        async with async_playwright() as p:
            if profile_dir_usable:
                context = await p.chromium.launch_persistent_context(user_data_dir, headless=False, args=["--disable-blink-features=AutomationControlled", "--no-sandbox"], user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            else: 
                 logger.warning("Using non-persistent context for scrape_website.")
                 browser_instance_sw = await p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled", "--no-sandbox"])
                 context = await browser_instance_sw.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36", java_script_enabled=True, ignore_https_errors=True)
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
                    logger.error(f"Error closing context for {url}: {e_ctx}", exc_info=False) 
        if browser_instance_sw and not profile_dir_usable: 
            try: await browser_instance_sw.close()
            except Exception: pass

async def scrape_latest_tweets(username_queried: str, limit: int = 10) -> list:
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
                logger.info(f"Using persistent profile: {user_data_dir}")
                context = await p.chromium.launch_persistent_context(user_data_dir, headless=False, args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"], user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36", slow_mo=150)
            else:
                logger.warning("Using non-persistent context for tweet scraping.")
                browser_instance_st = await p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"], slow_mo=150)
                context = await browser_instance_st.new_context(user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
            context_manager = context; page = await context_manager.new_page()
            url = f"https://x.com/{username_queried.lstrip('@')}/with_replies" # Always with_replies
            logger.info(f"Navigating to {url}")
            await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            try:
                await page.wait_for_selector("article[data-testid='tweet']", timeout=30000)
                logger.info("Initial tweet articles detected.")
                await asyncio.sleep(1.5); await page.keyboard.press("Escape"); await asyncio.sleep(0.5); await page.keyboard.press("Escape")
            except PlaywrightTimeoutError: 
                logger.warning(f"Timed out waiting for initial tweet articles for @{username_queried}.")
                return [] 
            
            max_scroll_attempts = limit + 15 
            for scroll_attempt in range(max_scroll_attempts):
                if len(tweets_collected) >= limit: break
                logger.debug(f"Tweet scrape attempt {scroll_attempt + 1}/{max_scroll_attempts}. Collected: {len(tweets_collected)}/{limit}")
                try:
                    clicked_count = await page.evaluate(JS_EXPAND_SHOWMORE_TWITTER, 5) 
                    if clicked_count > 0: logger.info(f"Clicked {clicked_count} 'Show more' elements."); await asyncio.sleep(1.5 + random.uniform(0.3, 0.9))
                except Exception as e_sm: logger.warning(f"JS 'Show More' error: {e_sm}")
                
                extracted_this_round = []; newly_added_count = 0
                try: extracted_this_round = await page.evaluate(JS_EXTRACT_TWEETS_TWITTER)
                except Exception as e_js: logger.error(f"JS tweet extraction error: {e_js}")
                
                for data in extracted_this_round: # data is already a dict from JS
                    uid = data.get('id') or (data.get("username","") + (data.get("content") or "")[:30] + data.get("timestamp",""))
                    if uid and uid not in seen_tweet_ids:
                        # Directly append the dictionary 'data' which has the correct structure
                        tweets_collected.append(data) 
                        seen_tweet_ids.add(uid); newly_added_count +=1
                        if len(tweets_collected) >= limit: break
                logger.info(f"Extracted {len(extracted_this_round)}, added {newly_added_count} new unique tweets.")
                
                if newly_added_count == 0 and scroll_attempt > (limit // 2 + 7): 
                    logger.info("No new unique tweets found in several attempts. Assuming end or stuck."); break
                await page.evaluate("window.scrollBy(0, window.innerHeight * 1.5);"); await asyncio.sleep(random.uniform(3.0, 5.0))
    except PlaywrightTimeoutError as e: logger.warning(f"Playwright overall timeout for @{username_queried}: {e}")
    except Exception as e: logger.error(f"Unexpected error scraping tweets for @{username_queried}: {e}", exc_info=True)
    finally:
        if page and not page.is_closed(): 
            try: 
                await page.close() 
            except Exception as e_page_close_final:
                logger.warning(f"Ignoring error closing page (final attempt) for @{username_queried}: {e_page_close_final}")
        if context_manager: 
            try: 
                await context_manager.close()
            except Exception as e_ctx_final:
                 if "Target page, context or browser has been closed" not in str(e_ctx_final):
                     logger.error(f"Error closing context (final attempt) for @{username_queried}: {e_ctx_final}", exc_info=False)
        if browser_instance_st: 
            try: 
                await browser_instance_st.close()
            except Exception as e_browser_final:
                logger.warning(f"Ignoring error closing browser (final attempt) for @{username_queried}: {e_browser_final}")
            
    tweets_collected.sort(key=lambda x: x.get("timestamp", ""), reverse=True) 
    logger.info(f"Finished scraping. Collected {len(tweets_collected)} tweets for @{username_queried}.")
    return tweets_collected[:limit]

async def query_searx(query: str) -> list:
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

async def fetch_youtube_transcript(url: str) -> str | None:
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
                    logger.warning(f"No English transcript for {video_id}. Available: {available_langs}. Trying first.")
                    transcript = transcript_list.find_generated_transcript([available_langs[0]])
        if transcript:
            fetched_data = transcript.fetch()
            full_text = " ".join([entry['text'] for entry in fetched_data])
            # Return full transcript, LLM prompt will truncate if needed
            return f"(Language: {transcript.language}) {full_text}" if transcript.language != 'en' else full_text
        else: logger.warning(f"No transcript found for YouTube video: {url}"); return None
    except Exception as e: logger.error(f"Failed to fetch YouTube transcript for {url}: {e}", exc_info=True); return None

def transcribe_audio_file(file_path: str) -> str | None:
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
            if channel and user:
                embed = discord.Embed(title=f"⏰ Reminder! (Set {original_time_str})", description=message_content, color=discord.Color.blue(), timestamp=reminder_time)
                embed.set_footer(text=f"Reminder for {user.display_name}")
                await channel.send(content=user.mention, embed=embed)
                await send_tts_audio(channel, f"Reminder for {user.display_name}: {message_content}", base_filename=f"reminder_{user_id}_{channel_id}")
            else: logger.warning(f"Could not fetch channel/user for reminder: ChID {channel_id}, UserID {user_id}")
        except discord.errors.NotFound: logger.warning(f"Channel/User not found for reminder: ChID {channel_id}, UserID {user_id}.")
        except Exception as e: logger.error(f"Failed to send reminder (ChID {channel_id}, UserID {user_id}): {e}", exc_info=True)

def parse_time_string_to_delta(time_str: str) -> tuple[Optional[timedelta], Optional[str]]:
    patterns = {'d': r'(\d+)\s*d(?:ay(?:s)?)?', 'h': r'(\d+)\s*h(?:our(?:s)?|r(?:s)?)?', 'm': r'(\d+)\s*m(?:inute(?:s)?|in(?:s)?)?', 's': r'(\d+)\s*s(?:econd(?:s)?|ec(?:s)?)?'}
    delta_args = {'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}; original_parts = []
    time_str_processed = time_str.lower()
    for key, pattern_regex in patterns.items():
        for match in re.finditer(pattern_regex, time_str_processed):
            value = int(match.group(1)); unit_full = {'d': 'days', 'h': 'hours', 'm': 'minutes', 's': 'seconds'}[key]
            delta_args[unit_full] += value; original_parts.append(f"{value} {unit_full.rstrip('s') if value == 1 else unit_full}")
        time_str_processed = re.sub(pattern_regex, "", time_str_processed)
    if not any(val > 0 for val in delta_args.values()): return None, None
    time_delta = timedelta(**delta_args)
    descriptive_str = ", ".join(original_parts) if original_parts else "immediately"
    if not descriptive_str and time_delta.total_seconds() > 0: descriptive_str = "a duration"
    return time_delta, descriptive_str

# -------------------------------------------------------------------
# Slash Commands (Application Commands)
# -------------------------------------------------------------------

@bot.tree.command(name="remindme", description="Sets a reminder. E.g., 1h30m Check the oven.")
@app_commands.describe(time_duration="Duration (e.g., '10m', '2h30m', '1d').", reminder_message="The message for your reminder.")
async def remindme_slash_command(interaction: discord.Interaction, time_duration: str, reminder_message: str):
    time_delta, descriptive_time_str = parse_time_string_to_delta(time_duration)
    if not time_delta or time_delta.total_seconds() <= 0:
        await interaction.response.send_message("Invalid time duration. Format: '10m', '2h30m', '1d'. Min 1s.", ephemeral=True)
        return
    reminder_time = datetime.now() + time_delta
    reminders.append((reminder_time, interaction.channel_id, interaction.user.id, reminder_message, descriptive_time_str or "later"))
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
        prompt_nodes = [get_system_prompt(), MsgNode("user", f"Analyze content from {url} and write a comedy routine:\n{webpage_text[:3000]}")]
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
        try:
            await interaction.response.defer(ephemeral=False)
        except discord.HTTPException as e_defer:
            logger.error(f"Search: Failed to defer interaction: {e_defer}")
            try:
                await interaction.response.send_message("Error starting search. Please try again.",ephemeral=True)
            except discord.HTTPException:
                logger.error("Search: Also failed to send error message after defer failure.")
            return
    
    try:
        search_results = await query_searx(query)
        if not search_results:
            await interaction.followup.send("No search results found.")
            return
            
        snippets = [f"{i+1}. **{discord.utils.escape_markdown(r.get('title','N/A'))}** (<{r.get('url','N/A')}>)\n    {discord.utils.escape_markdown(r.get('content',r.get('description','No snippet'))[:250])}..." for i, r in enumerate(search_results)]
        formatted_results = "\n\n".join(snippets)
        await interaction.followup.send(embed=discord.Embed(title=f"Top Search Results for: {query}", description=formatted_results[:config.EMBED_MAX_LENGTH], color=config.EMBED_COLOR["incomplete"]))

        prompt_nodes = [get_system_prompt(), MsgNode("user", f"Summarize search results for '{query}':\n\n{formatted_results[:3000]}")]
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
        sys_content = ("You are a bot that generates extremely sarcastic, snarky, and troll-like comments "
                       "to mock extremist political views or absurd political statements. Your goal is to be biting and humorous, "
                       "undermining the statement without being directly offensive or vulgar. Focus on wit and irony.") 
        prompt_nodes = [MsgNode("system", sys_content), MsgNode("user", f"Generate sarcastic comeback: \"{statement}\"")]
        await stream_llm_response_to_interaction(interaction, prompt_nodes, title="Sarcastic Political Commentary", force_new_followup_flow=False)
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
        try:
            await interaction.response.defer(ephemeral=False) 
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

        tweet_texts = [f"[{t.get('timestamp', 'N/A')}] @{t.get('original_author', username.lstrip('@'))}: {discord.utils.escape_markdown(t.get('content', 'N/A'))}" for t in tweets]
        raw_tweets_display = "\n\n".join(tweet_texts)
        embed_title = f"Recent Tweets from @{username.lstrip('@')}"
        if not raw_tweets_display: raw_tweets_display = "No tweet content could be displayed."

        raw_tweet_chunks = chunk_text(raw_tweets_display, config.EMBED_MAX_LENGTH)
        for i, chunk in enumerate(raw_tweet_chunks):
            chunk_title = embed_title if i == 0 else f"{embed_title} (cont.)"
            embed = discord.Embed(title=chunk_title, description=chunk, color=config.EMBED_COLOR["incomplete"]) 
            await interaction.followup.send(embed=embed) 

        prompt_nodes_summary = [get_system_prompt(), MsgNode("user", f"Summarize themes, topics, sentiment from @{username.lstrip('@')} tweets:\n\n{raw_tweets_display[:3500]}")]
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
        chosen_celebrity = random.choice(["Keanu Reeves", "Dwayne Johnson", "Zendaya", "Tom Hanks"]) 
        llm_prompt_text = (f"You are an AP photo caption writer. Describe the attached image in a detailed and intricate way, "
                           f"as if for a blind person. However, creatively replace the main subject or character in the image with {chosen_celebrity}. "
                           f"Begin your response with 'AP Photo: {chosen_celebrity}...' "
                           f"If the user provided an additional prompt, consider it: '{user_prompt}'") 
        prompt_nodes = [MsgNode("user", [{"type": "text", "text": llm_prompt_text}, {"type": "image_url", "image_url": {"url": image_url_for_llm}}])]
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
        await interaction.response.send_message("Message history for this channel has been cleared.", ephemeral=True)
    else:
        await interaction.response.send_message("No history to clear for this channel.", ephemeral=True)

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
    await bot.process_commands(message) 
    
    prefixes = await bot.get_prefix(message)
    is_command_attempt = any(message.content.startswith(p) for p in prefixes) if isinstance(prefixes, (list, tuple)) else \
                         (message.content.startswith(prefixes) if isinstance(prefixes, str) else False)
    if is_command_attempt: return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions
    channel_id = message.channel.id # type: ignore
    author_roles = getattr(message.author, 'roles', [])

    allowed_by_channel = not config.ALLOWED_CHANNEL_IDS or channel_id in config.ALLOWED_CHANNEL_IDS or \
                         (isinstance(message.channel, discord.Thread) and message.channel.parent_id in config.ALLOWED_CHANNEL_IDS)
    allowed_by_role = not config.ALLOWED_ROLE_IDS or is_dm or any(role.id in config.ALLOWED_ROLE_IDS for role in author_roles)

    if not (is_dm or is_mentioned or (allowed_by_channel and allowed_by_role)):
        if not (is_dm or is_mentioned) and not allowed_by_channel: return 
        if not (is_dm or is_mentioned) and not allowed_by_role:
             logger.debug(f"Msg from {message.author.name} without allowed role. Ignoring.")
             return
        return 

    logger.info(f"General LLM message from {message.author.name} in {getattr(message.channel, 'name', f'ChID {channel_id}')}: {message.content[:50]}")

    if channel_id not in message_history: message_history[channel_id] = []
    current_message_content_parts = []
    user_message_text = message.content

    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("audio/"):
                try:
                    if not os.path.exists("temp"): os.makedirs("temp")
                    audio_filename = f"temp/temp_audio_{attachment.id}.{attachment.filename.split('.')[-1]}"
                    await attachment.save(audio_filename)
                    transcription = transcribe_audio_file(audio_filename)
                    if os.path.exists(audio_filename): os.remove(audio_filename) 
                    if transcription: user_message_text = (user_message_text + " " + transcription).strip()
                except Exception as e: logger.error(f"Error processing audio attachment: {e}", exc_info=True)
                break 
    if user_message_text: current_message_content_parts.append({"type": "text", "text": user_message_text})

    image_added = False
    if message.attachments:
        for i, attachment in enumerate(message.attachments):
            if i >= config.MAX_IMAGES_PER_MESSAGE: break 
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    img_bytes = await attachment.read()
                    b64_img = base64.b64encode(img_bytes).decode('utf-8')
                    current_message_content_parts.append({"type": "image_url", "image_url": {"url": f"data:{attachment.content_type};base64,{b64_img}"}})
                    image_added = True
                except Exception as e: logger.error(f"Error processing image {attachment.filename}: {e}")
    
    scraped_content_for_llm = "" 
    if detected_urls_in_text := detect_urls(user_message_text): 
        for i, url in enumerate(detected_urls_in_text[:2]): 
            logger.info(f"Processing URL from message: {url}")
            content_piece = None
            youtube_match = re.search(r'(?:v=|\/|embed\/|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})', url)
            if youtube_match: 
                transcript = await fetch_youtube_transcript(url)
                if transcript: content_piece = f"\n\n--- YouTube Transcript for {url} ---\n{transcript}\n--- End Transcript ---" 
            else: 
                scraped_text = await scrape_website(url)
                if scraped_text and "Failed to scrape" not in scraped_text and "Scraping timed out" not in scraped_text:
                     content_piece = f"\n\n--- Webpage Content for {url} ---\n{scraped_text}\n--- End Webpage Content ---" 
            if content_piece: scraped_content_for_llm += content_piece
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
        if image_added: current_message_content_parts.append({"type": "text", "text": "The user sent this image. Please describe it or respond to it if there's an implicit question."})
        else:
            logger.info("Ignoring message with no processable text, audio, or image content after all processing.")
            return

    user_msg_node_content = current_message_content_parts if len(current_message_content_parts) > 1 or image_added else current_message_content_parts[0]["text"]
    message_history[channel_id].append(MsgNode("user", user_msg_node_content, str(message.author.id)))
    message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:]
    llm_conversation_history = [get_system_prompt()] + message_history[channel_id]
    await stream_llm_response_to_message(message, llm_conversation_history)


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == bot.user.id or str(payload.emoji) != '❌': return
    try:
        channel = await bot.fetch_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
    except (discord.NotFound, discord.Forbidden): return
    if message.author.id != bot.user.id: return
    can_delete = True
    if isinstance(channel, discord.TextChannel): # Guild channel
        try:
            member = await channel.guild.fetch_member(payload.user_id)
            if member and member.guild_permissions.manage_messages: can_delete = True
        except discord.HTTPException: pass # Failed to fetch member, proceed
    if not can_delete and message.interaction and message.interaction.user.id == payload.user_id: can_delete = True
    if not can_delete and message.reference and message.reference.message_id:
        try:
            original_message = await channel.fetch_message(message.reference.message_id)
            if original_message.author.id == payload.user_id: can_delete = True
        except discord.NotFound: pass # Original message might be deleted
    if can_delete:
        try: 
            await message.delete()
            logger.info(f"Message {message.id} deleted by reaction from user {payload.user_id}.")
        except Exception as e: logger.error(f"Failed to delete message {message.id} by reaction: {e}")


@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord! ID: {bot.user.id}') # type: ignore
    logger.info(f"discord.py version: {discord.__version__}")
    logger.info(f"Operating in channels: {config.ALLOWED_CHANNEL_IDS if config.ALLOWED_CHANNEL_IDS else 'All permitted by default'}")
    logger.info(f"Restricted to roles: {config.ALLOWED_ROLE_IDS if config.ALLOWED_ROLE_IDS else 'None'}")
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
    except discord.errors.NotFound: logger.error(f"Error handler: Interaction not found for '{command_name}'. OrigErr: {error}")
    except Exception as e_generic: logger.error(f"Error handler: Generic error for '{command_name}': {e_generic}. OrigErr: {error}")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound): pass 
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

