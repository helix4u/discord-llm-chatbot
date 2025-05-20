import asyncio
import logging
from datetime import datetime
from collections import deque # For efficient message history
import io # Added for io.BytesIO

import discord
from discord import File # For sending files like TTS audio

# Import from config.py
from config import (
    EMBED_COLOR, 
    EMBED_MAX_LENGTH, 
    EDITS_PER_SECOND, 
    MAX_MESSAGES,
    MAX_IMAGE_WARNING, 
    MAX_MESSAGE_WARNING,
    LLM, # Needed for some multimodal message construction
    MAX_IMAGES # Needed for MsgNode too_many_images flag logic
)

# Import audio utilities for TTS if streaming function will handle it
try:
    from audio_utils import tts_request
except ImportError:
    logging.warning("audio_utils not found or tts_request missing. TTS in stream_discord_response will be disabled.")
    async def tts_request(*args, **kwargs): return None # Mock function

logger = logging.getLogger(__name__)

class MsgNode:
    """Represents a node in the message history, containing the message content
    and metadata for LLM processing and context management."""
    def __init__(self, discord_message_id: int, role: str, content: list | str, 
                 author_id: int, too_many_images: bool = False, replied_to_id: int | None = None):
        self.discord_message_id = discord_message_id # Original Discord message ID
        self.role = role  # "user" or "assistant"
        self.content = content # For LLM: string or list of content blocks (text, image_url)
        self.author_id = author_id
        self.too_many_images = too_many_images # Flag if message had excessive images
        self.replied_to_id = replied_to_id # Discord ID of the message this node is a reply to (if any)
        self.timestamp = datetime.now()

    def to_llm_format(self) -> dict:
        """Returns the message in the format expected by the LLM."""
        # Name field can be used to identify speaker if needed, or for specific model requirements
        return {"role": self.role, "content": self.content, "name": str(self.author_id)}

class MessageHistoryManager:
    """Manages message history and node creation for multiple channels."""
    def __init__(self, max_history_per_channel: int = MAX_MESSAGES):
        self.message_history_by_channel = {} # Key: channel_id, Value: deque of MsgNode objects
        self.msg_nodes_by_id = {} # Key: discord_message_id, Value: MsgNode object
        self.max_history_per_channel = max_history_per_channel
        logger.info(f"MessageHistoryManager initialized with max history per channel: {max_history_per_channel}")

    def add_message(self, msg: discord.Message, is_bot_response: bool = False, custom_content: str | None = None, llm_content_override: list | str | None = None):
        """
        Adds a Discord message to the history for its channel.
        Creates a MsgNode and stores it.
        `custom_content` can be used if the discord message content isn't what we want to store (e.g. after transcription).
        `llm_content_override` can be used to directly set the content for the LLM (e.g. for multimodal).
        """
        channel_id = msg.channel.id
        if channel_id not in self.message_history_by_channel:
            self.message_history_by_channel[channel_id] = deque(maxlen=self.max_history_per_channel)

        role = "assistant" if is_bot_response or msg.author.bot else "user"
        author_id = msg.author.id
        
        processed_content_for_llm = []
        text_content_to_store = custom_content if custom_content is not None else msg.content
        
        if llm_content_override:
            processed_content_for_llm = llm_content_override
        else:
            # Standard processing: text and images
            if text_content_to_store:
                processed_content_for_llm.append({"type": "text", "text": text_content_to_store})

            num_images_processed = 0
            if msg.attachments:
                for att in msg.attachments:
                    if "image" in att.content_type:
                        if MAX_IMAGES > 0 and num_images_processed >= MAX_IMAGES: # Use MAX_IMAGES from config
                            break 
                        processed_content_for_llm.append({"type": "image_url", "image_url": {"url": att.url, "detail": "low"}})
                        num_images_processed += 1
            
            if LLM == "mistral" and any(item["type"] == "image_url" for item in processed_content_for_llm):
                 pass


        if isinstance(processed_content_for_llm, list) and len(processed_content_for_llm) == 1 and processed_content_for_llm[0]["type"] == "text":
            final_llm_content = processed_content_for_llm[0]["text"]
        else:
            final_llm_content = processed_content_for_llm


        node = MsgNode(
            discord_message_id=msg.id,
            role=role,
            content=final_llm_content, 
            author_id=author_id,
            too_many_images=(MAX_IMAGES > 0 and len([a for a in msg.attachments if "image" in a.content_type]) > MAX_IMAGES), # Use MAX_IMAGES
            replied_to_id=msg.reference.message_id if msg.reference else None
        )
        
        self.message_history_by_channel[channel_id].append(node)
        self.msg_nodes_by_id[msg.id] = node
        logger.debug(f"Added message {msg.id} to history for channel {channel_id}. Role: {role}. Node stored.")
        return node

    def get_reply_chain(self, channel_id: int, current_user_message_node: MsgNode | None = None) -> tuple[list, set]:
        user_warnings = set()
        if channel_id not in self.message_history_by_channel:
            logger.debug(f"No history found for channel {channel_id} when building reply chain.")
            return [current_user_message_node.to_llm_format()] if current_user_message_node else [], user_warnings

        history_nodes = list(self.message_history_by_channel[channel_id])
        
        if len(history_nodes) >= self.max_history_per_channel : 
             user_warnings.add(MAX_MESSAGE_WARNING)

        for node in history_nodes:
            if node.too_many_images:
                user_warnings.add(MAX_IMAGE_WARNING)
        
        llm_chain = [node.to_llm_format() for node in history_nodes]
        
        logger.debug(f"Built reply chain for channel {channel_id} with {len(llm_chain)} messages. Warnings: {user_warnings}")
        return llm_chain, user_warnings

    def clear_history(self, channel_id: int):
        if channel_id in self.message_history_by_channel:
            self.message_history_by_channel[channel_id].clear()
            logger.info(f"Message history cleared for channel {channel_id}.")
        else:
            logger.info(f"No history found to clear for channel {channel_id}.")

    def get_history_size(self, channel_id: int) -> int:
        return len(self.message_history_by_channel.get(channel_id, []))

async def stream_discord_response(
    channel: discord.TextChannel, 
    reply_to_message: discord.Message, 
    llm_response_stream, 
    title: str = "Response", 
    initial_user_warnings: set | None = None,
    do_tts: bool = True 
):
    bot_message_obj = None 
    accumulated_content = ""
    edit_task = None
    last_edit_time = asyncio.get_event_loop().time()
    edit_interval = 1.0 / EDITS_PER_SECOND 

    user_warnings = initial_user_warnings if initial_user_warnings is not None else set()
    is_first_message_part = True # To control where warnings are added

    try:
        async for text_chunk in llm_response_stream: # Removed chunk_index as it wasn't used reliably
            if not text_chunk: continue 

            accumulated_content += text_chunk
            current_time = asyncio.get_event_loop().time()
            
            # Determine if we need to send/edit message
            # Force edit if accumulated content is near max length to avoid overflow on next chunk
            force_edit_due_to_length = (len(accumulated_content) > EMBED_MAX_LENGTH * 0.95)
            time_to_edit = (current_time - last_edit_time >= edit_interval)

            if not bot_message_obj: # First chunk, send new message
                embed = discord.Embed(title=title, description=accumulated_content, color=EMBED_COLOR["incomplete"])
                if is_first_message_part:
                    for warning in sorted(list(user_warnings)): 
                        embed.add_field(name=warning, value="", inline=False)
                bot_message_obj = await reply_to_message.reply(embed=embed, silent=True)
                last_edit_time = current_time
                is_first_message_part = False # Warnings only on the very first message object
            
            elif time_to_edit or force_edit_due_to_length :
                if edit_task and not edit_task.done():
                    await edit_task 

                content_to_display = accumulated_content
                is_final_chunk_for_this_embed = False
                if len(content_to_display) > EMBED_MAX_LENGTH:
                    content_to_display = content_to_display[:EMBED_MAX_LENGTH]
                    is_final_chunk_for_this_embed = True 
                
                embed = discord.Embed(title=title, description=content_to_display, color=EMBED_COLOR["incomplete"])
                # Warnings are only on the first message object, so not added here
                
                edit_task = asyncio.create_task(bot_message_obj.edit(embed=embed))
                last_edit_time = current_time

                if is_final_chunk_for_this_embed: 
                    await edit_task 
                    if do_tts and tts_request: 
                        tts_audio_bytes = await tts_request(content_to_display)
                        if tts_audio_bytes:
                            await channel.send(file=File(io.BytesIO(tts_audio_bytes), filename=f"{title.replace(' ','_')}_part.mp3")) # Simplified filename
                    
                    accumulated_content = accumulated_content[EMBED_MAX_LENGTH:] 
                    embed = discord.Embed(title=f"{title} (continued)", description=accumulated_content or "...", color=EMBED_COLOR["incomplete"])
                    bot_message_obj = await channel.send(embed=embed) 
                    is_first_message_part = False # Subsequent parts don't get initial warnings

        # After the loop, finalize the last message part
        if edit_task and not edit_task.done():
            await edit_task 
        
        if bot_message_obj and accumulated_content: 
            final_embed = discord.Embed(title=title if len(response_msgs_objs) == 1 else f"{title} (final part)", # Logic for title needs `response_msgs_objs`
                                        description=accumulated_content[:EMBED_MAX_LENGTH], 
                                        color=EMBED_COLOR["complete"])
            await bot_message_obj.edit(embed=final_embed)
            if do_tts and tts_request: 
                tts_audio_bytes = await tts_request(accumulated_content[:EMBED_MAX_LENGTH])
                if tts_audio_bytes:
                    await channel.send(file=File(io.BytesIO(tts_audio_bytes), filename=f"{title.replace(' ','_')}_final.mp3"))
        elif not bot_message_obj and accumulated_content : 
             embed = discord.Embed(title=title, description=accumulated_content, color=EMBED_COLOR["complete"])
             if is_first_message_part: # Should only happen if content is very short
                 for warning in sorted(list(user_warnings)): 
                        embed.add_field(name=warning, value="", inline=False)
             bot_message_obj = await reply_to_message.reply(embed=embed, silent=True)
             if do_tts and tts_request:
                tts_audio_bytes = await tts_request(accumulated_content)
                if tts_audio_bytes:
                    await channel.send(file=File(io.BytesIO(tts_audio_bytes), filename=f"{title.replace(' ','_')}_final.mp3"))

        if bot_message_obj: 
            return bot_message_obj

    except Exception as e:
        logger.error(f"Error during response streaming: {e}", exc_info=True)
        # Error reporting logic remains same
        if bot_message_obj: 
            try:
                error_embed = discord.Embed(title=f"{title} - Error", description=f"An error occurred: {e}", color=discord.Color.red())
                await bot_message_obj.edit(embed=error_embed)
            except Exception as e2: logger.error(f"Failed to send error update: {e2}")
        else: 
             try:
                error_embed = discord.Embed(title=f"{title} - Error", description=f"An error occurred: {e}", color=discord.Color.red())
                await channel.send(embed=error_embed)
             except Exception as e3: logger.error(f"Failed to send initial error: {e3}")
    return None
