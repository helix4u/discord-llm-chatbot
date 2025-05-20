import asyncio
# import logging # logger from config is used
import base64 
# import io # Not directly used, handled by File or audio_utils
import re 

import discord
from discord import File, Embed, Intents
import aiohttp # For image/audio downloads

# Configuration
from config import (
    DISCORD_BOT_TOKEN,
    logger, 
    ALLOWED_CHANNEL_IDS,
    ALLOWED_ROLE_IDS,
    IGNORE_COMMANDS,
    EMBED_COLOR,
    MAX_IMAGES, # Used for attachment processing
    MAX_IMAGE_WARNING # Used for attachment processing
)

# Core Discord utilities
from discord_bot_core import MessageHistoryManager, stream_discord_response
# MsgNode is used by MessageHistoryManager

# Command Handlers
from command_handlers import (
    handle_search_command,
    handle_sns_command,
    handle_roast_command,
    handle_remindme_command,
    handle_pol_command,
    handle_clear_history_command,
    handle_show_history_size_command
)

# LLM utilities
from llm_handler import (
    generate_image_description_stream, 
    get_system_prompt, 
    generate_chat_completion_stream
)

# Audio utilities
from audio_utils import transcribe_audio_attachment
# tts_request is called by stream_discord_response

# Web utilities
from web_utils import detect_urls, scrape_website, fetch_youtube_transcript, clean_text

# --- Initialization ---
intents_speak_search = Intents.default()
intents_speak_search.message_content = True
intents_speak_search.reactions = True
discord_client = discord.Client(intents=intents_speak_search)

history_manager = MessageHistoryManager() 

# --- Event Handlers ---

@discord_client.event
async def on_ready():
    logger.info(f'{discord_client.user} (SpeakAndSearch) has connected to Discord!')
    # Config logging is done in config.py

@discord_client.event
async def on_message(msg: discord.Message):
    if msg.author == discord_client.user or msg.author.bot:
        return

    if msg.guild:
        if ALLOWED_CHANNEL_IDS and msg.channel.id not in ALLOWED_CHANNEL_IDS:
            if not (hasattr(msg.channel, 'parent_id') and msg.channel.parent_id in ALLOWED_CHANNEL_IDS):
                return
        if ALLOWED_ROLE_IDS:
            author_roles = [role.id for role in msg.author.roles]
            if not any(role_id in ALLOWED_ROLE_IDS for role_id in author_roles):
                return
    elif ALLOWED_CHANNEL_IDS: 
        return

    command_text_lower = msg.content.lower()
    is_a_known_command = False
    known_command_prefixes = ["!search", "!sns", "!roast", "!remindme", "!pol", "!clear_history", "!show_history_size"]
    for cmd_prefix in known_command_prefixes:
        if command_text_lower.startswith(cmd_prefix):
            is_a_known_command = True
            break

    if not is_a_known_command and any(command_text_lower.startswith(prefix) for prefix in IGNORE_COMMANDS):
        logger.info(f"Ignoring message due to IGNORE_COMMANDS prefix: {msg.content[:30]}")
        return
    
    try:
        # Use tts_enabled=True for command handlers in this file
        if command_text_lower.startswith("!search "):
            query = msg.content[len("!search "):].strip()
            if query: await handle_search_command(query, msg, history_manager, tts_enabled=True)
            else: await msg.channel.send("Usage: `!search <query>`")
            return
        elif command_text_lower.startswith("!sns "):
            query_or_url = msg.content[len("!sns "):].strip()
            if query_or_url: await handle_sns_command(query_or_url, msg, history_manager, tts_enabled=True)
            else: await msg.channel.send("Usage: `!sns <url_or_query>`")
            return
        elif command_text_lower.startswith("!roast "):
            url = msg.content[len("!roast "):].strip()
            if url: await handle_roast_command(url, msg, history_manager, tts_enabled=True)
            else: await msg.channel.send("Usage: `!roast <url>`")
            return
        elif command_text_lower.startswith("!remindme "):
            parts = msg.content[len("!remindme "):].split(maxsplit=1)
            if len(parts) == 2: await handle_remindme_command(parts[0], parts[1], msg) # Reminder TTS handled by schedule_reminder_task
            else: await msg.channel.send("Usage: `!remindme <time> <message>`")
            return
        elif command_text_lower.startswith("!pol "):
            text = msg.content[len("!pol "):].strip()
            if text: await handle_pol_command(text, msg, history_manager, tts_enabled=True)
            else: await msg.channel.send("Usage: `!pol <text>`")
            return
        elif command_text_lower == "!clear_history":
            await handle_clear_history_command(history_manager, msg.channel.id, msg.channel)
            return
        elif command_text_lower == "!show_history_size":
            await handle_show_history_size_command(history_manager, msg.channel.id, msg.channel)
            return

    except Exception as e:
        logger.error(f"Error handling command '{msg.content[:50]}': {e}", exc_info=True)
        await msg.channel.send(f"An error occurred processing command: {e}")
        return

    is_dm_or_mention = msg.channel.type == discord.ChannelType.private or \
                       discord_client.user in msg.mentions
    if not (is_dm_or_mention or (ALLOWED_CHANNEL_IDS and msg.channel.id in ALLOWED_CHANNEL_IDS)):
        return

    async with msg.channel.typing():
        user_text_content = msg.content
        image_contexts_b64 = []
        web_contexts_text = []
        user_warnings = set()
        processed_audio_this_message = False

        if msg.attachments:
            for attachment in msg.attachments:
                if attachment.content_type and ("audio" in attachment.content_type or "video" in attachment.content_type) and not processed_audio_this_message:
                    logger.info(f"Processing audio/video attachment: {attachment.filename}")
                    async with aiohttp.ClientSession() as http_session:
                        transcription = await transcribe_audio_attachment(attachment.url, http_session)
                    processed_audio_this_message = True 
                    if "Error:" not in transcription:
                        # Voice command check
                        if "search for" in transcription.lower():
                            query = transcription.lower().split("search for", 1)[1].strip()
                            if query: await handle_search_command(query, msg, history_manager, tts_enabled=True); return
                        elif "remind me" in transcription.lower():
                            match_vc = re.search(r"remind me in (.+?) to (.+)", transcription, re.IGNORECASE)
                            if match_vc: await handle_remindme_command(match_vc.groups()[0], match_vc.groups()[1], msg); return
                        
                        user_text_content = transcription 
                        await msg.channel.send(embed=Embed(title="🎤 Voice Transcription", description=user_text_content, color=EMBED_COLOR.get("incomplete", discord.Color.orange())))
                    else:
                        await msg.channel.send(f"Transcription failed: {transcription}")
                    break 

                elif attachment.content_type and "image" in attachment.content_type:
                    if MAX_IMAGES > 0 and len(image_contexts_b64) >= MAX_IMAGES:
                        user_warnings.add(MAX_IMAGE_WARNING)
                        break 
                    try:
                        async with aiohttp.ClientSession() as http_session:
                             async with http_session.get(attachment.url) as resp:
                                if resp.status == 200:
                                    image_data = await resp.read()
                                    image_contexts_b64.append(base64.b64encode(image_data).decode("utf-8"))
                                else: logger.warning(f"Failed to download image {attachment.url}")
                    except Exception as e: logger.error(f"Error downloading image {attachment.url}: {e}")
        
        urls = detect_urls(user_text_content)
        if urls:
            url_scrape_tasks = [scrape_website(u) for u in urls if "youtube.com" not in u and "youtu.be" not in u]
            yt_tasks = [fetch_youtube_transcript(u) for u in urls if "youtube.com" in u or "youtu.be" in u]
            
            results = await asyncio.gather(*(url_scrape_tasks + yt_tasks), return_exceptions=True)
            url_idx = 0 # To correctly map results back to original URLs
            for res in results:
                original_url = urls[url_idx]
                url_idx +=1
                if isinstance(res, Exception) or "Failed to scrape" in str(res) or not str(res).strip():
                    logger.warning(f"Scraping/transcript failed for {original_url}: {res}")
                else:
                    prefix = "Transcript from" if "youtube.com" in original_url or "youtu.be" in original_url else "Content from"
                    web_contexts_text.append(f"{prefix} {original_url}:\n{clean_text(str(res)[:1500])}\n")

        llm_content_parts = []
        if user_text_content: llm_content_parts.append({"type": "text", "text": user_text_content})
        for img_b64 in image_contexts_b64:
            llm_content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
        for web_ctx in web_contexts_text:
            if llm_content_parts and llm_content_parts[-1]["type"] == "text": llm_content_parts[-1]["text"] += f"\n\n{web_ctx}"
            else: llm_content_parts.append({"type": "text", "text": web_ctx})

        if not llm_content_parts: logger.debug("No content for LLM."); return
            
        history_manager.add_message(msg, llm_content_override=llm_content_parts)
        llm_messages_context, context_warnings = history_manager.get_reply_chain(msg.channel.id)
        user_warnings.update(context_warnings)
        
        final_llm_payload = get_system_prompt() + llm_messages_context
        
        logger.info(f"Sending request to LLM for channel {msg.channel.id} (SpeakAndSearch). Msgs: {len(final_llm_payload)}")
        
        is_image_focused = bool(image_contexts_b64) and not user_text_content.strip()
        llm_stream = await (generate_image_description_stream(image_contexts_b64[0], user_text_content) 
                            if is_image_focused and image_contexts_b64 
                            else generate_chat_completion_stream(final_llm_payload))
        
        bot_response_msg_obj = await stream_discord_response(
            channel=msg.channel,
            reply_to_message=msg,
            llm_response_stream=llm_stream,
            title="Response" if not is_image_focused else "Image Analysis",
            initial_user_warnings=user_warnings,
            do_tts=True 
        )

        if bot_response_msg_obj:
            response_text_for_history = "".join([e.description for e in bot_response_msg_obj.embeds if e.description])
            history_manager.add_message(bot_response_msg_obj, is_bot_response=True, custom_content=response_text_for_history or "LLM Response")


@discord_client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.emoji.name == '❌' and payload.user_id != discord_client.user.id:
        try:
            channel = await discord_client.fetch_channel(payload.channel_id)
            message = await channel.fetch_message(payload.message_id)
            if message.author == discord_client.user:
                await message.delete()
                logger.info(f"Deleted message {payload.message_id} by reaction from user {payload.user_id}.")
        except Exception as e:
            logger.error(f"Error processing reaction delete: {e}")

async def main():
    if not DISCORD_BOT_TOKEN:
        logger.critical("DISCORD_BOT_TOKEN is not configured. Bot cannot start.")
        return
    try:
        await discord_client.start(DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.critical(f"Error starting Discord client (SpeakAndSearch): {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting LmcordSpeakAndSearch Bot...")
    asyncio.run(main())
