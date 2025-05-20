import asyncio
import logging
import base64
import re
from datetime import datetime, timedelta # For reminders
import io # Added for io.BytesIO

import discord
from discord import File, Embed
import aiohttp # For downloading images in !ap

# Configuration
from config import (
    LLM, EMBED_COLOR, MAX_COMPLETION_TOKENS 
    # Removed IGNORE_COMMANDS, MAX_MESSAGES, MAX_IMAGE_WARNING, MAX_MESSAGE_WARNING, SEARX_URL as not directly used here
    # They are used by other modules imported here, which is fine.
)

# Custom Modules
from llm_handler import (
    generate_chat_completion_stream, 
    generate_sarcastic_response, 
    generate_reminder, 
    get_system_prompt,
    generate_image_description_stream 
)
from web_utils import (
    query_searx, 
    scrape_website, 
    detect_urls, 
    fetch_youtube_transcript,
    clean_text
)
from audio_utils import tts_request 
from twitter_handler import scrape_latest_tweets
from discord_bot_core import MessageHistoryManager, stream_discord_response 
# MsgNode is used by MessageHistoryManager, not directly instantiated here typically

logger = logging.getLogger(__name__)

# --- Time Parsing for Reminders (can be moved to a utility file later) ---
def parse_time_string(time_str: str) -> int | None:
    """Converts a time string (e.g., "1h30m", "2d", "30s") to seconds."""
    total_seconds = 0
    time_units = {
        'd': 86400, 'day': 86400, 'days': 86400,
        'h': 3600, 'hour': 3600, 'hours': 3600,
        'm': 60, 'min': 60, 'minute': 60, 'minutes': 60,
        's': 1, 'sec': 1, 'second': 1, 'seconds': 1
    }
    pattern = re.compile(r'(\d+)\s*([a-zA-Z]+)')
    matches = pattern.findall(time_str)
    if not matches:
        return None
    for value, unit_short in matches:
        unit_full = unit_short.lower()
        # Allow partial matches like "hr" for "hour"
        matched_unit_key = None
        for key in time_units.keys():
            if key.startswith(unit_full):
                matched_unit_key = key
                break
        if not matched_unit_key:
             for key in time_units.keys(): # try to match if unit_full starts with key (e.g. hours from hour)
                if unit_full.startswith(key) and len(unit_full) > len(key): # ensure it's plural like 'hours' from 'hour'
                    matched_unit_key = key
                    break
        if not matched_unit_key:
            return None # Invalid unit
            
        total_seconds += int(value) * time_units[matched_unit_key]
    return total_seconds if total_seconds > 0 else None


# --- Reminder Scheduling Task ---
async def schedule_reminder_task(
    delay: int, 
    channel: discord.TextChannel, 
    original_message_content: str, 
    reminder_text: str,
    author_mention: str
):
    """The actual task that waits and sends the reminder."""
    await asyncio.sleep(delay)
    logger.info(f"Executing reminder: {reminder_text} for {author_mention} (originally: {original_message_content})")
    
    prompt_for_llm = (
        f"<think>The user {author_mention} set a reminder for themselves. "
        f"The original command was approximately: '!remindme {timedelta(seconds=delay)} {original_message_content}'. " 
        f"The specific reminder text they wanted was: '{reminder_text}'. "
        f"It's time to send this reminder. Craft a friendly and clear reminder message based on this text.</think>\n"
        f"Hey {author_mention}, here's your reminder: {reminder_text}" 
    )
    
    reminder_llm_response = await generate_reminder(prompt_for_llm) 

    embed = Embed(title="⏰ Reminder!", description=reminder_llm_response, color=EMBED_COLOR["complete"])
    await channel.send(content=author_mention, embed=embed)
    
    tts_audio = await tts_request(reminder_llm_response)
    if tts_audio:
        await channel.send(file=File(io.BytesIO(tts_audio), filename="reminder.mp3"))

# --- Command Handler Functions ---

async def handle_search_command(query: str, message: discord.Message, message_history_manager: MessageHistoryManager, tts_enabled: bool = True):
    logger.info(f"Handling !search command with query: {query} from {message.author.name}")
    channel = message.channel
    
    search_results = await query_searx(query)
    if not search_results:
        await channel.send(f"No search results found for: '{query}'.")
        return

    search_context = "\n\n".join([
        f"Title: {res.get('title', 'N/A')}\nURL: {res.get('url', 'N/A')}\nSnippet: {res.get('content', 'N/A')}" 
        for res in search_results[:3] 
    ])
    
    prompt = (
        f"<think>User asked to search for '{query}'. I found these top results:\n{search_context}\n\n"
        f"Now, I need to summarize these findings and present them clearly to the user. "
        f"I should mention the source URLs if relevant and provide a concise summary.</think>\n"
        f"Here's a summary of the search results for '{query}':"
    )
    
    message_history_manager.add_message(message) 
    llm_messages, user_warnings = message_history_manager.get_reply_chain(channel.id)
    final_llm_messages = get_system_prompt() + llm_messages + [{"role": "user", "content": prompt}]

    llm_stream = await generate_chat_completion_stream(final_llm_messages)
    
    bot_response_message = await stream_discord_response(
        channel=channel,
        reply_to_message=message, 
        llm_response_stream=llm_stream,
        title=f"Search Summary: {query}",
        initial_user_warnings=user_warnings,
        do_tts=tts_enabled
    )
    if bot_response_message:
         # Ensure content for history is properly extracted if multi-part
         full_response_text = "".join([embed.description for embed in bot_response_message.embeds if embed.description])
         message_history_manager.add_message(bot_response_message, is_bot_response=True, custom_content=full_response_text or "Search summary was generated.")


async def handle_sns_command(query_or_url: str, message: discord.Message, message_history_manager: MessageHistoryManager, tts_enabled: bool = True):
    logger.info(f"Handling !sns command with query/URL: {query_or_url} from {message.author.name}")
    channel = message.channel
    urls = detect_urls(query_or_url)
    scraped_content = ""
    source_description = ""
    final_url_source = query_or_url # Default to query if no URL found or used

    if urls: 
        url = urls[0]
        final_url_source = url
        source_description = f"the webpage: {url}"
        scraped_content = await scrape_website(url)
        if "Failed to scrape" in scraped_content or not scraped_content.strip():
            await channel.send(f"Sorry, I couldn't scrape content from {url}. It might be protected or inaccessible.")
            return
    else: 
        source_description = f"search results for '{query_or_url}'"
        search_results = await query_searx(query_or_url)
        if not search_results:
            await channel.send(f"No search results found for '{query_or_url}' to summarize.")
            return
        if search_results[0].get("url"):
            final_url_source = search_results[0]["url"]
            scraped_content = await scrape_website(final_url_source)
            if "Failed to scrape" in scraped_content or not scraped_content.strip():
                 await channel.send(f"Found search results, but failed to scrape the top one: {final_url_source}. Cannot summarize.")
                 return
            source_description = f"the top search result for '{query_or_url}': {final_url_source}"
        else:
            await channel.send("Search results found, but no valid URL in the top result to scrape.")
            return
            
    if not scraped_content.strip():
        await channel.send(f"The content from {source_description} was empty after cleaning. Nothing to summarize.")
        return

    prompt = (
        f"<think>The user wants a summary of {source_description}. "
        f"The scraped content is:\n```\n{clean_text(scraped_content[:2000])}\n```\n" 
        f"I should provide a concise summary of this content.</think>\n"
        f"Here's a summary of {source_description}:"
    )
    
    message_history_manager.add_message(message)
    llm_messages, user_warnings = message_history_manager.get_reply_chain(channel.id)
    final_llm_messages = get_system_prompt() + llm_messages + [{"role": "user", "content": prompt}]
    
    llm_stream = await generate_chat_completion_stream(final_llm_messages)
    bot_response_message = await stream_discord_response(
        channel=channel,
        reply_to_message=message,
        llm_response_stream=llm_stream,
        title=f"Summary: {final_url_source[:50]}...",
        initial_user_warnings=user_warnings,
        do_tts=tts_enabled
    )
    if bot_response_message:
        full_response_text = "".join([embed.description for embed in bot_response_message.embeds if embed.description])
        message_history_manager.add_message(bot_response_message, is_bot_response=True, custom_content=full_response_text or "SNS summary was generated.")


async def handle_roast_command(url: str, message: discord.Message, message_history_manager: MessageHistoryManager, tts_enabled: bool = True):
    logger.info(f"Handling !roast command for URL: {url} from {message.author.name}")
    channel = message.channel
    
    scraped_content = await scrape_website(url)
    if "Failed to scrape" in scraped_content or not scraped_content.strip():
        await channel.send(f"I tried to find something to roast at {url}, but I couldn't get any material! Maybe the site is protected or empty?")
        return

    prompt = (
        f"<think>The user wants me to roast the content of the webpage: {url}. "
        f"The scraped content is:\n```\n{clean_text(scraped_content[:1500])}\n```\n" 
        f"I need to come up with a witty and humorous comedy routine based on this. "
        f"The tone should be light-hearted and funny, not mean-spirited.</think>\n"
        f"Alright, I've taken a look at {url}. Fasten your seatbelts, here's my take:"
    )
    
    message_history_manager.add_message(message)
    llm_messages, user_warnings = message_history_manager.get_reply_chain(channel.id)
    final_llm_messages = get_system_prompt() + llm_messages + [{"role": "user", "content": prompt}]

    llm_stream = await generate_chat_completion_stream(final_llm_messages)
    bot_response_message = await stream_discord_response(
        channel=channel,
        reply_to_message=message,
        llm_response_stream=llm_stream,
        title=f"Roast of {url[:50]}...",
        initial_user_warnings=user_warnings,
        do_tts=tts_enabled
    )
    if bot_response_message:
        full_response_text = "".join([embed.description for embed in bot_response_message.embeds if embed.description])
        message_history_manager.add_message(bot_response_message, is_bot_response=True, custom_content=full_response_text or "Roast was generated.")

async def handle_gettweets_command(username: str, limit: int, message: discord.Message, message_history_manager: MessageHistoryManager, tts_enabled: bool = True):
    logger.info(f"Handling !gettweets for @{username}, limit {limit}, from {message.author.name}")
    channel = message.channel
    
    await channel.send(embed=Embed(description=f"Fetching last {limit} tweets from **@{username}**...", color=EMBED_COLOR["incomplete"]))
    tweets = await scrape_latest_tweets(username, limit=limit)
    
    if not tweets:
        await channel.send(embed=Embed(description=f"No tweets found for @{username}, or scraping failed.", color=discord.Color.red()))
        return

    raw_tweets_text = ""
    for t in reversed(tweets): 
        raw_tweets_text += f"🐦 **@{t['from_user']}** ({t['timestamp']}):\n{t['content']}\n\n"
        if len(raw_tweets_text) > EMBED_MAX_LENGTH - 500: 
            raw_tweets_text = raw_tweets_text[:EMBED_MAX_LENGTH - 500] + "... (truncated)"
            break
    
    raw_embed = Embed(title=f"Raw Tweets from @{username} (last {len(tweets)})", description=raw_tweets_text, color=EMBED_COLOR["incomplete"])
    await channel.send(embed=raw_embed)

    prompt = (
        f"<think>I have scraped the latest {len(tweets)} tweets from user @{username}. "
        f"The tweets are (oldest to newest):\n{raw_tweets_text}\n\n"
        f"I need to provide a concise summary of these tweets, capturing main themes or sentiments. </think>\n"
        f"Here's a summary of the latest tweets from @{username}:"
    )
    
    message_history_manager.add_message(message) 
    llm_messages, user_warnings = message_history_manager.get_reply_chain(channel.id)
    final_llm_messages = get_system_prompt() + llm_messages + [{"role": "user", "content": prompt}]

    llm_stream = await generate_chat_completion_stream(final_llm_messages)
    bot_response_message = await stream_discord_response(
        channel=channel,
        reply_to_message=message, 
        llm_response_stream=llm_stream,
        title=f"Tweet Summary: @{username}",
        initial_user_warnings=user_warnings,
        do_tts=tts_enabled
    )
    if bot_response_message:
        full_response_text = "".join([embed.description for embed in bot_response_message.embeds if embed.description])
        message_history_manager.add_message(bot_response_message, is_bot_response=True, custom_content=full_response_text or "Tweet summary was generated.")


async def handle_remindme_command(time_str: str, reminder_text: str, message: discord.Message):
    logger.info(f"Handling !remindme: time='{time_str}', text='{reminder_text}' from {message.author.name}")
    channel = message.channel
    delay_seconds = parse_time_string(time_str)

    if delay_seconds is None:
        await channel.send("Invalid time format for reminder. Use formats like '1h30m', '2d', '30s', '1day2hours', etc.")
        return

    if delay_seconds > 60 * 60 * 24 * 30: 
        await channel.send("Sorry, reminders are limited to a maximum of 30 days.")
        return

    reminder_time = datetime.now() + timedelta(seconds=delay_seconds)
    await channel.send(f"Okay, {message.author.mention}! I'll remind you about: \"{reminder_text}\" on {reminder_time.strftime('%Y-%m-%d at %H:%M:%S')}.")
    
    asyncio.create_task(schedule_reminder_task(
        delay=delay_seconds,
        channel=channel,
        original_message_content=message.content, 
        reminder_text=reminder_text,
        author_mention=message.author.mention
    ))


async def handle_pol_command(text: str, message: discord.Message, message_history_manager: MessageHistoryManager, tts_enabled: bool = True):
    logger.info(f"Handling !pol command with text: '{text}' from {message.author.name}")
    channel = message.channel
    
    sarcastic_response_text = await generate_sarcastic_response(text)
    
    embed = Embed(description=sarcastic_response_text, color=EMBED_COLOR["complete"]) 
    bot_msg = await message.reply(embed=embed, silent=True)
    
    if tts_enabled:
        tts_audio = await tts_request(sarcastic_response_text)
        if tts_audio:
            await channel.send(file=File(io.BytesIO(tts_audio), filename="pol_response.mp3"))
    
    message_history_manager.add_message(message)
    message_history_manager.add_message(bot_msg, is_bot_response=True, custom_content=sarcastic_response_text)


async def handle_ap_command(message: discord.Message, message_history_manager: MessageHistoryManager, tts_enabled: bool = True):
    logger.info(f"Handling !ap command from {message.author.name}")
    channel = message.channel
    
    if not message.attachments:
        await channel.send("Please attach an image to use the `!ap` command.")
        return

    attachment = message.attachments[0]
    if not attachment.content_type or "image" not in attachment.content_type:
        await channel.send("The attached file does not seem to be an image.")
        return

    try:
        # Using aiohttp for image download, as it's already a dependency
        async with aiohttp.ClientSession() as http_session:
            async with http_session.get(attachment.url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    base64_image = base64.b64encode(image_data).decode("utf-8")
                else:
                    await channel.send("Could not download the image.")
                    return
    except Exception as e:
        logger.error(f"Error downloading image for !ap: {e}")
        await channel.send("An error occurred while trying to download the image.")
        return

    user_text_content = message.content[len("!ap"):].strip() 

    ap_system_prompt = (
        "You are an AP Photo caption writer. Your task is to describe an image in detail, "
        "as if for an Associated Press photo caption. However, you must replace the main subject "
        "of the photo with a randomly chosen celebrity or well-known fictional character for humorous effect. "
        "Start your response with 'AP Photo, [Celebrity/Character Name], '."
    )

    llm_stream = await generate_image_description_stream(
        image_base64_data=base64_image,
        text_content=user_text_content, 
        system_prompt_override=ap_system_prompt
    )
    
    message_history_manager.add_message(message) 
    bot_response_message = await stream_discord_response(
        channel=channel,
        reply_to_message=message,
        llm_response_stream=llm_stream,
        title="AP Photo Description",
        initial_user_warnings=set(),
        do_tts=tts_enabled 
    )
    if bot_response_message:
        full_response_text = "".join([embed.description for embed in bot_response_message.embeds if embed.description])
        message_history_manager.add_message(bot_response_message, is_bot_response=True, custom_content=full_response_text or "AP Description was generated.")


async def handle_toggle_search_command(guild_id: int, current_search_settings: dict) -> tuple[bool, str]:
    if guild_id not in current_search_settings:
        current_search_settings[guild_id] = {'search_enabled': True} 

    new_state = not current_search_settings[guild_id].get('search_enabled', True)
    current_search_settings[guild_id]['search_enabled'] = new_state
    
    response_message = f"Automatic search functionality is now **{'enabled' if new_state else 'disabled'}** for this server/channel."
    logger.info(f"Toggled search for guild/channel {guild_id} to {new_state}")
    return new_state, response_message


async def handle_clear_history_command(message_history_manager: MessageHistoryManager, channel_id: int, channel: discord.TextChannel):
    logger.info(f"Handling !clear_history command for channel {channel_id}")
    message_history_manager.clear_history(channel_id)
    await channel.send(embed=Embed(description="💬 Message history for this channel has been cleared.", color=EMBED_COLOR["complete"]))


async def handle_show_history_size_command(message_history_manager: MessageHistoryManager, channel_id: int, channel: discord.TextChannel):
    logger.info(f"Handling !show_history_size command for channel {channel_id}")
    size = message_history_manager.get_history_size(channel_id)
    await channel.send(embed=Embed(description=f"📊 Current message history size for this channel: {size} message(s).", color=EMBED_COLOR["complete"]))
