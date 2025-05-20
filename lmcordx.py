import asyncio
import base64 # For image processing
import re # For command parsing and URL detection
import logging # Keep for main script's own logger if needed, though config.logger is primary

import discord
from discord import File, Embed, Intents 
import aiohttp # For image downloads in on_message

# Configuration (from config.py)
from config import (
    DISCORD_BOT_TOKEN,
    logger, 
    ALLOWED_CHANNEL_IDS,
    ALLOWED_ROLE_IDS,
    IGNORE_COMMANDS,
    LLM, 
    EMBED_COLOR,
    MAX_IMAGES, # Used directly in on_message for image processing logic
    MAX_IMAGE_WARNING # Used directly in on_message
)

# Core Discord utilities
from discord_bot_core import MessageHistoryManager, stream_discord_response 
# MsgNode is primarily used within MessageHistoryManager or if type hinting needed here.

# Command Handlers (import all handlers)
from command_handlers import (
    handle_search_command,
    handle_sns_command,
    handle_roast_command,
    handle_gettweets_command,
    handle_remindme_command,
    handle_pol_command,
    handle_ap_command,
    handle_toggle_search_command,
    handle_clear_history_command,
    handle_show_history_size_command
)

# LLM utilities for non-command image processing and chat
from llm_handler import generate_image_description_stream, get_system_prompt, generate_chat_completion_stream

# Audio utilities for voice messages
from audio_utils import transcribe_audio_attachment

# Web utilities for URL processing in general messages
from web_utils import detect_urls, scrape_website, fetch_youtube_transcript, clean_text

# --- Initialization ---
intents = Intents.default()
intents.message_content = True
intents.reactions = True
discord_client = discord.Client(intents=intents) 

history_manager = MessageHistoryManager() 

search_toggled_guilds = {} 

# --- Event Handlers ---

@discord_client.event
async def on_ready():
    logger.info(f'{discord_client.user} has connected to Discord!')
    logger.info(f"Operating in channels: {ALLOWED_CHANNEL_IDS if ALLOWED_CHANNEL_IDS else 'All (mentioned)'}")
    logger.info(f"Role restrictions: {ALLOWED_ROLE_IDS if ALLOWED_ROLE_IDS else 'None'}")
    logger.info(f"Ignoring commands starting with: {IGNORE_COMMANDS}")


@discord_client.event
async def on_message(msg: discord.Message):
    if msg.author == discord_client.user: 
        return
    if msg.author.bot: 
        return

    if msg.guild: 
        if ALLOWED_CHANNEL_IDS and msg.channel.id not in ALLOWED_CHANNEL_IDS:
            if not (hasattr(msg.channel, 'parent_id') and msg.channel.parent_id in ALLOWED_CHANNEL_IDS):
                 logger.debug(f"Ignoring message in non-allowed channel: {msg.channel.id} in guild {msg.guild.id}")
                 return
        
        if ALLOWED_ROLE_IDS:
            author_roles = [role.id for role in msg.author.roles]
            if not any(role_id in ALLOWED_ROLE_IDS for role_id in author_roles):
                logger.debug(f"Ignoring message from user {msg.author.name} due to role restrictions.")
                return
    elif ALLOWED_CHANNEL_IDS: 
        logger.debug(f"Ignoring DM from {msg.author.name} as ALLOWED_CHANNEL_IDS is set.")
        return

    command_text_lower = msg.content.lower()
    # Check IGNORE_COMMANDS from config, but ensure it's not one of this bot's actual commands.
    is_a_primary_command = False
    for cmd_prefix in ["!search", "!sns", "!roast", "!gettweets", "!remindme", "!pol", "!ap", "!toggle_search", "!clear_history", "!show_history_size"]:
        if command_text_lower.startswith(cmd_prefix):
            is_a_primary_command = True
            break
    
    if not is_a_primary_command and any(command_text_lower.startswith(prefix) for prefix in IGNORE_COMMANDS):
        logger.info(f"Ignoring message due to IGNORE_COMMANDS prefix: {msg.content[:30]}")
        return

    try:
        if command_text_lower.startswith("!search "):
            query = msg.content[len("!search "):].strip()
            if query: await handle_search_command(query, msg, history_manager)
            else: await msg.channel.send("Usage: `!search <query>`")
            return
        
        elif command_text_lower.startswith("!sns "):
            query_or_url = msg.content[len("!sns "):].strip()
            if query_or_url: await handle_sns_command(query_or_url, msg, history_manager)
            else: await msg.channel.send("Usage: `!sns <url_or_query>`")
            return

        elif command_text_lower.startswith("!roast "):
            url = msg.content[len("!roast "):].strip()
            if url: await handle_roast_command(url, msg, history_manager)
            else: await msg.channel.send("Usage: `!roast <url>`")
            return

        elif command_text_lower.startswith("!gettweets "):
            parts = msg.content.split()
            username = parts[1].lstrip('@') if len(parts) > 1 else None
            limit = 10
            if len(parts) > 2:
                try: limit = int(parts[2])
                except ValueError: await msg.channel.send("Invalid limit for `!gettweets`. Using default (10).")
            if username: await handle_gettweets_command(username, limit, msg, history_manager)
            else: await msg.channel.send("Usage: `!gettweets <twitter_handle> [limit]`")
            return

        elif command_text_lower.startswith("!remindme "):
            parts = msg.content[len("!remindme "):].split(maxsplit=1)
            if len(parts) == 2: await handle_remindme_command(parts[0], parts[1], msg)
            else: await msg.channel.send("Usage: `!remindme <time_duration> <message>`")
            return

        elif command_text_lower.startswith("!pol "):
            text = msg.content[len("!pol "):].strip()
            if text: await handle_pol_command(text, msg, history_manager)
            else: await msg.channel.send("Usage: `!pol <text>`")
            return

        elif command_text_lower.startswith("!ap"): 
            if msg.attachments: await handle_ap_command(msg, history_manager)
            else: await msg.channel.send("Please attach an image to use the `!ap` command.")
            return

        elif command_text_lower == "!toggle_search":
            guild_id = msg.guild.id if msg.guild else msg.channel.id 
            _, response_msg_text = await handle_toggle_search_command(guild_id, search_toggled_guilds)
            await msg.channel.send(embed=Embed(description=response_msg_text, color=EMBED_COLOR["complete"]))
            return
            
        elif command_text_lower == "!clear_history":
            await handle_clear_history_command(history_manager, msg.channel.id, msg.channel)
            return

        elif command_text_lower == "!show_history_size":
            await handle_show_history_size_command(history_manager, msg.channel.id, msg.channel)
            return

    except Exception as e:
        logger.error(f"Error handling command '{msg.content[:50]}': {e}", exc_info=True)
        await msg.channel.send(f"An error occurred: {e}")
        return

    is_dm_or_mention = msg.channel.type == discord.ChannelType.private or \
                       discord_client.user in msg.mentions
    
    # perform_auto_search logic can be added here if needed
    # guild_settings = search_toggled_guilds.get(msg.guild.id if msg.guild else msg.channel.id, {})
    # if guild_settings.get('search_enabled', False): ...

    if not (is_dm_or_mention or (ALLOWED_CHANNEL_IDS and msg.channel.id in ALLOWED_CHANNEL_IDS)):
        return

    async with msg.channel.typing():
        user_text_content = msg.content
        image_contexts_b64 = [] # Store base64 strings of images
        web_contexts_text = []  # Store text from web/YT
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
                        # Simple voice command check (can be expanded)
                        if "search for" in transcription.lower():
                            query = transcription.lower().split("search for", 1)[1].strip()
                            if query: await handle_search_command(query, msg, history_manager); return
                        elif "remind me" in transcription.lower():
                            match_vc = re.search(r"remind me in (.+?) to (.+)", transcription, re.IGNORECASE)
                            if match_vc: await handle_remindme_command(match_vc.groups()[0], match_vc.groups()[1], msg); return
                        
                        user_text_content = transcription 
                        await msg.channel.send(embed=Embed(title="🎤 Voice Transcription", description=user_text_content, color=EMBED_COLOR.get("incomplete", discord.Color.orange())))
                    else:
                        await msg.channel.send(f"Transcription failed: {transcription}")
                    break 
        
                elif attachment.content_type and "image" in attachment.content_type and not command_text_lower.startswith("!ap"):
                    if MAX_IMAGES > 0 and len(image_contexts_b64) >= MAX_IMAGES:
                        user_warnings.add(MAX_IMAGE_WARNING)
                        break 
                    try:
                        async with aiohttp.ClientSession() as http_session:
                             async with http_session.get(attachment.url) as resp:
                                if resp.status == 200: image_contexts_b64.append(base64.b64encode(await resp.read()).decode("utf-8"))
                                else: logger.warning(f"Failed to download image {attachment.url}")
                    except Exception as e: logger.error(f"Error downloading image {attachment.url}: {e}")
        
        urls = detect_urls(user_text_content)
        if urls:
            logger.info(f"Detected URLs: {urls}")
            url_scrape_tasks = [scrape_website(u) for u in urls if "youtube.com" not in u and "youtu.be" not in u]
            yt_transcript_tasks = [fetch_youtube_transcript(u) for u in urls if "youtube.com" in u or "youtu.be" in u]
            
            scraped_results = await asyncio.gather(*(url_scrape_tasks + yt_transcript_tasks), return_exceptions=True)
            
            url_idx = 0
            for res in scraped_results:
                original_url = urls[url_idx]
                url_idx += 1
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
        
        logger.info(f"Sending request to LLM for channel {msg.channel.id}. Messages: {len(final_llm_payload)}")
        
        # Determine if this is primarily an image interaction for specific handling
        is_image_focused_interaction = bool(image_contexts_b64) and not user_text_content.strip() # e.g. user just sends image

        if is_image_focused_interaction and image_contexts_b64: # Use vision stream if only image
             general_image_system_prompt = "You are an assistant describing an image and responding to related user text."
             llm_stream = await generate_image_description_stream(
                    image_base64_data=image_contexts_b64[0], 
                    text_content=user_text_content, # Will be empty or just spaces
                    system_prompt_override=general_image_system_prompt
                )
        else: # Default text/multimodal chat stream
            llm_stream = await generate_chat_completion_stream(final_llm_payload)
        
        bot_response_msg_obj = await stream_discord_response(
            channel=msg.channel,
            reply_to_message=msg,
            llm_response_stream=llm_stream,
            title="Response" if not is_image_focused_interaction else "Image Analysis",
            initial_user_warnings=user_warnings,
            do_tts=True # lmcordx.py enables TTS
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
        except discord.NotFound:
            logger.warning(f"Message/channel not found for reaction deletion: msg {payload.message_id}, chan {payload.channel_id}")
        except discord.Forbidden:
            logger.error(f"Lacking permissions to delete message {payload.message_id} in {payload.channel_id}")
        except Exception as e:
            logger.error(f"Error processing reaction delete: {e}")

# --- Main Execution ---
async def main():
    if not DISCORD_BOT_TOKEN:
        logger.critical("DISCORD_BOT_TOKEN is not configured. Bot cannot start.")
        return
    try:
        await discord_client.start(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.critical("Failed to log in. Check token in config.py / .env")
    except Exception as e:
        logger.critical(f"Error starting Discord client: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting Lmcordx Bot...")
    asyncio.run(main())
