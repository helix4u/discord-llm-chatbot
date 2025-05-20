import asyncio
# import logging # logger from config is used
import base64 
# import io # Not directly used
import re 
# import os # Not directly used here

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
    LLM, # Used by get_system_prompt, but direct usage here is fine too
    MAX_IMAGES, # Used for attachment processing
    MAX_IMAGE_WARNING # Used for attachment processing
)

# Core Discord utilities
from discord_bot_core import MessageHistoryManager, stream_discord_response
# MsgNode is used by MessageHistoryManager

# Command Handlers (Subset for this script)
from command_handlers import (
    # !search and !sns are handled locally in this file to ensure do_tts=False
    handle_clear_history_command,
    handle_show_history_size_command
)

# LLM utilities
from llm_handler import (
    generate_image_description_stream, 
    get_system_prompt, 
    generate_chat_completion_stream
)

# Audio utilities: Only for transcription
from audio_utils import transcribe_audio_attachment, transcribe_youtube_video 

# Web utilities
from web_utils import detect_urls, scrape_website, fetch_youtube_transcript, clean_text, query_searx

# --- Initialization ---
intents_search_only = Intents.default()
intents_search_only.message_content = True
intents_search_only.reactions = True
discord_client = discord.Client(intents=intents_search_only)

history_manager = MessageHistoryManager()

# --- Event Handlers ---

@discord_client.event
async def on_ready():
    logger.info(f'{discord_client.user} (LLMCordSearch - TTS Disabled) has connected to Discord!')
    # Config logging done in config.py

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
    known_command_prefixes = ["!search", "!sns", "!clear_history", "!show_history_size"]
    for cmd_prefix in known_command_prefixes:
        if command_text_lower.startswith(cmd_prefix):
            is_a_known_command = True
            break
    
    if not is_a_known_command and any(command_text_lower.startswith(prefix) for prefix in IGNORE_COMMANDS):
        logger.info(f"Ignoring message due to IGNORE_COMMANDS prefix: {msg.content[:30]}")
        return
    
    try:
        if command_text_lower.startswith("!search "):
            query = msg.content[len("!search "):].strip()
            if query:
                async with msg.channel.typing():
                    search_results = await query_searx(query) 
                    if not search_results: await msg.channel.send(f"No results for '{query}'."); return
                    
                    search_context = "\n\n".join([f"Title: {r.get('title','N/A')}\nURL: {r.get('url','N/A')}\nSnippet: {r.get('content','N/A')}" for r in search_results[:3]])
                    prompt = f"<think>User searched for '{query}'. Results:\n{search_context}\nSummarize.</think>\nSummary for '{query}':"
                    
                    history_manager.add_message(msg) 
                    llm_messages, user_warnings = history_manager.get_reply_chain(msg.channel.id)
                    final_llm_payload = get_system_prompt() + llm_messages + [{"role": "user", "content": prompt}]
                    
                    llm_stream = await generate_chat_completion_stream(final_llm_payload)
                    bot_response_msg = await stream_discord_response(
                        channel=msg.channel, 
                        reply_to_message=msg, 
                        llm_response_stream=llm_stream, 
                        title=f"Search Summary: {query}", 
                        initial_user_warnings=user_warnings, 
                        do_tts=False 
                    )
                    if bot_response_msg: 
                        desc = bot_response_msg.embeds[0].description if bot_response_msg.embeds else ""
                        history_manager.add_message(bot_response_msg, is_bot_response=True, custom_content=desc)
            else: await msg.channel.send("Usage: `!search <query>`")
            return
            
        elif command_text_lower.startswith("!sns "):
            query_or_url = msg.content[len("!sns "):].strip()
            if query_or_url:
                async with msg.channel.typing():
                    urls = detect_urls(query_or_url)
                    content_to_summarize = ""
                    source_desc = query_or_url
                    if urls:
                        source_desc = urls[0]
                        content_to_summarize = await scrape_website(urls[0])
                    else: 
                        search_results = await query_searx(query_or_url)
                        if search_results and search_results[0].get("url"):
                            source_desc = search_results[0]['url']
                            content_to_summarize = await scrape_website(source_desc)
                        elif not search_results: await msg.channel.send("No search results for SNS."); return
                        else: await msg.channel.send("Search result for SNS had no URL."); return
                    
                    if "Failed to scrape" in content_to_summarize or not content_to_summarize.strip():
                        await msg.channel.send(f"Could not get content from {source_desc} for SNS."); return
                    
                    prompt = f"<think>Content for SNS from {source_desc}:\n{content_to_summarize[:2000]}\nSummarize this.</think>\nSummary of {source_desc}:"
                    history_manager.add_message(msg)
                    llm_messages, user_warnings = history_manager.get_reply_chain(msg.channel.id)
                    final_llm_payload = get_system_prompt() + llm_messages + [{"role": "user", "content": prompt}]
                    llm_stream = await generate_chat_completion_stream(final_llm_payload)
                    bot_response_msg = await stream_discord_response(
                        channel=msg.channel, 
                        reply_to_message=msg, 
                        llm_response_stream=llm_stream, 
                        title=f"SNS Summary: {query_or_url[:30]}...", 
                        initial_user_warnings=user_warnings, 
                        do_tts=False 
                    )
                    if bot_response_msg: 
                        desc = bot_response_msg.embeds[0].description if bot_response_msg.embeds else ""
                        history_manager.add_message(bot_response_msg, is_bot_response=True, custom_content=desc)
            else: await msg.channel.send("Usage: `!sns <url_or_query>`")
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
                    logger.info(f"Processing audio/video attachment for transcription: {attachment.filename}")
                    async with aiohttp.ClientSession() as http_session: 
                        transcription = await transcribe_audio_attachment(attachment.url, http_session)
                    processed_audio_this_message = True
                    if "Error:" not in transcription:
                        user_text_content = transcription 
                        await msg.channel.send(embed=Embed(title="🎤 Voice Transcription", description=user_text_content, color=EMBED_COLOR.get("incomplete", discord.Color.orange())))
                    else:
                        await msg.channel.send(f"Transcription failed: {transcription}")
                    break 
                elif attachment.content_type and "image" in attachment.content_type:
                    if MAX_IMAGES > 0 and len(image_contexts_b64) >= MAX_IMAGES:
                        user_warnings.add(MAX_IMAGE_WARNING); break 
                    try:
                        async with aiohttp.ClientSession() as http_session:
                             async with http_session.get(attachment.url) as resp:
                                if resp.status == 200: image_contexts_b64.append(base64.b64encode(await resp.read()).decode("utf-8"))
                                else: logger.warning(f"Failed to download image {attachment.url}")
                    except Exception as e: logger.error(f"Error downloading image {attachment.url}: {e}")
        
        urls = detect_urls(user_text_content)
        if urls:
            url_scrape_tasks = [scrape_website(u) for u in urls if "youtube.com" not in u and "youtu.be" not in u]
            yt_api_transcript_tasks = [fetch_youtube_transcript(u) for u in urls if "youtube.com" in u or "youtu.be" in u]
            
            scraped_web_results = await asyncio.gather(*url_scrape_tasks, return_exceptions=True)
            youtube_api_results = await asyncio.gather(*yt_api_transcript_tasks, return_exceptions=True)
            
            web_url_idx = 0
            for i, res in enumerate(scraped_web_results):
                current_web_url = next((u for idx, u in enumerate(urls) if ("youtube.com" not in u and "youtu.be" not in u) and idx >= web_url_idx), None)
                if current_web_url: web_url_idx = urls.index(current_web_url) + 1
                if isinstance(res, Exception) or "Failed to scrape" in str(res) or not str(res).strip(): logger.warning(f"Scraping failed for {current_web_url or 'unknown URL'}: {res}")
                else: web_contexts_text.append(f"Content from {current_web_url}:\n{clean_text(str(res)[:1500])}\n")

            yt_api_url_idx = 0
            for i, res in enumerate(youtube_api_results):
                current_yt_url = next((u for idx, u in enumerate(urls) if ("youtube.com" in u or "youtu.be" in u) and idx >= yt_api_url_idx), None)
                if current_yt_url: yt_api_url_idx = urls.index(current_yt_url) + 1

                if isinstance(res, Exception) or not str(res).strip():
                    logger.warning(f"YouTube API transcript failed for {current_yt_url or 'unknown YT URL'}. Attempting yt-dlp transcription.")
                    if current_yt_url:
                        dl_transcript = await transcribe_youtube_video(current_yt_url, None) 
                        if "Error:" not in dl_transcript and dl_transcript.strip():
                            web_contexts_text.append(f"Transcript from {current_yt_url} (yt-dlp):\n{clean_text(dl_transcript[:1500])}\n")
                        else: logger.warning(f"yt-dlp transcription also failed for {current_yt_url}: {dl_transcript}")
                else: web_contexts_text.append(f"Transcript from {current_yt_url} (API):\n{clean_text(str(res)[:1500])}\n")

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
        
        logger.info(f"Sending request to LLM for channel {msg.channel.id} (LLMCordSearch). Message count: {len(final_llm_payload)}")
        
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
            do_tts=False 
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
        logger.critical(f"Error starting Discord client (LLMCordSearch): {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting LLMCordSearch Bot (TTS Disabled)...")
    asyncio.run(main())
