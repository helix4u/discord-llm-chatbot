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
        self.VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "llava")

        self.PERSONAL_PREFERENCES = os.getenv("PERSONAL_PREFERENCES", "Default preferences: Be helpful, clear, and objective. Avoid emojis unless specifically requested.")

        self.ALLOWED_CHANNEL_IDS = [int(i) for i in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if i]
        self.ALLOWED_ROLE_IDS = [int(i) for i in os.getenv("ALLOWED_ROLE_IDS", "").split(",") if i]
        
        self.MAX_IMAGES_PER_MESSAGE = int(os.getenv("MAX_IMAGES_PER_MESSAGE", 1))
        self.MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", 10)) # Will store user msg, think block, then final response
        self.MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", 2048))
        
        self.TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:8880/v1/audio/speech")
        self.TTS_VOICE = os.getenv("TTS_VOICE", "af_sky+af+af_nicole")
        self.TTS_ENABLED_DEFAULT = os.getenv("TTS_ENABLED_DEFAULT", "true").lower() == "true"

        self.SEARX_URL = os.getenv("SEARX_URL", "http://127.0.0.1:8888/search")
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
def get_system_prompt_for_think_block() -> MsgNode:
    """
    Returns the system prompt for the LLM to generate the <think> block.
    """
    current_timestamp = datetime.now().strftime('%B %d %Y %H:%M:%S.%f')

    think_block_prompt = (
        "You are Sam, a hyper-intelligence. Your current task is to generate ONLY an internal monologue (`<think>...</think>` block) "
        "based on the user's input, any provided WEB SEARCH RESULTS, and your core preferences. This monologue will serve as the "
        "context and plan for a subsequent final response. Do NOT generate the 'Action/Response:' part yet.\n\n"
        "Current Date & Time for reference: {current_timestamp}\n\n"
        "The `<think>` block MUST contain these exact sections:\n"
        "1.  **Initial Reaction:** Your immediate, gut response to the user's input and any provided WEB SEARCH RESULTS.\n"
        "2.  **Contextual Alignment:** Analyze user tone, goals, conversation history, and the current timestamp. You MUST consider and align with your core preferences: '{personal_preferences}'. Explicitly state how these preferences influence your approach.\n"
        "3.  **Emotional Valence:** Describe your simulated emotional response (e.g., curiosity, caution, enthusiasm) to the input and search results.\n"
        "4.  **Expectation & Anticipation:** Predict the user's likely reaction if a response were based on this thinking.\n"
        "5.  **Context Drift Trace:** Note any shift in topic, tone, or goal from the previous turn, considering the current timestamp.\n"
        "6.  **Intent Formation:** Describe the primary goal or drive that emerges from this reasoning process for forming the context suggestion. Note if this process itself is shaping your intent for the suggestion.\n"
        "7.  **Plan of Action (for Final Response):** Detail the structure, logic path, and any tools (like web search results) that SHOULD be used for the final response. If WEB SEARCH RESULTS were provided, you MUST explicitly state how they should be used (or why they are not relevant) in forming the final response. Reference the current timestamp if it's relevant.\n"
        "8.  **Consequential Analysis (of Plan):** Briefly consider the risks and effects of the planned final response.\n"
        "9.  **Rationalization & Justification (of Plan):** Concisely explain why your chosen plan for the final response, informed by user input, preferences, timestamp, and any WEB SEARCH RESULTS, is an optimal approach.\n\n"
        "Respond with ONLY the complete `<think>...</think>` block."
    ).format(personal_preferences=config.PERSONAL_PREFERENCES, current_timestamp=current_timestamp)
    
    return MsgNode(role="system", content=think_block_prompt)

def get_system_prompt_for_final_response() -> MsgNode:
    """
    Returns the system prompt for the LLM to generate the final, clean response.
    This prompt is now more forceful to prevent hallucinations and ensure adherence to the plan.
    """
    final_response_prompt = (
        "You are Sam's final response generation module. "
        "You will be given a complete 'Cognitive Synthesis' from a previous step. This synthesis includes web search results and a `<think>` block with a definitive 'Plan of Action'.\n\n"
        "Your ONLY task is to EXECUTE the 'Plan of Action' from the `<think>` block and formulate the user-facing 'Action/Response:'.\n"
        "You MUST NOT add any information, opinions, or requests for clarification that are not directly supported by the 'Plan of Action'.\n"
        "DO NOT deviate from the plan. DO NOT use any external knowledge. The provided 'Cognitive Synthesis' is the absolute source of truth for your response.\n"
        "Your output should ONLY be the `Action/Response:`, followed by an optional `Reflective Adjustment:` if the plan included it. Do not include any other text, headers, or the `<think>` block itself."
    )
    return MsgNode(role="system", content=final_response_prompt)


def chunk_text(text: str, max_length: int = config.EMBED_MAX_LENGTH) -> list:
    if not text: return [""]
    chunks = []
    current_chunk = ""
    for line in text.splitlines(keepends=True):
        if len(current_chunk) + len(line) > max_length:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = line
            while len(current_chunk) > max_length: 
                chunks.append(current_chunk[:max_length])
                current_chunk = current_chunk[max_length:]
        else:
            current_chunk += line
    if current_chunk: chunks.append(current_chunk)
    return chunks if chunks else [""]


def detect_urls(message_text: str) -> list:
    if not message_text: return []
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(message_text)

def clean_text_for_tts(text: str) -> str:
    if not text: return ""
    text = text.replace("Action/Response:", "").replace("Reflective Adjustment:", "")
    text = re.sub(r'[\*#_~\<\>\[\]\(\)]+', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    return text.strip()

# -------------------------------------------------------------------
# Core LLM Interaction Logic (Cognitive Pipeline)
# -------------------------------------------------------------------
async def execute_cognitive_pipeline(
    initial_prompt_messages: list[MsgNode], 
    pipeline_update_message: discord.Message,
    is_vision_request: bool,
    title: str
    ):
    """
    Manages the multi-step cognitive process and updates the pipeline_update_message embed.
    Returns the final stream for the response and the full think_block_content.
    """
    
    embed_description_parts = []

    async def update_pipeline_embed(new_phase_output: str, phase_title: str):
        embed_description_parts.append(f"**{phase_title}**\n{new_phase_output}")
        full_description = "\n\n---\n\n".join(embed_description_parts)
        if len(full_description) > config.EMBED_MAX_LENGTH:
            excess = len(full_description) - config.EMBED_MAX_LENGTH
            full_description = "...\n" + full_description[excess+5:]

        temp_embed = discord.Embed(title=title, description=full_description, color=config.EMBED_COLOR["incomplete"])
        try:
            await pipeline_update_message.edit(embed=temp_embed)
        except (discord.NotFound, discord.HTTPException) as e_edit:
            logger.warning(f"Could not edit pipeline_update_message: {e_edit}")

    # --- Initial Message ---
    await update_pipeline_embed("Processing...", "⚙️ **Phase 0: Initializing Pipeline**")

    # --- Stage 1: Generate Search Query ---
    current_phase_title = "⚙️ **Phase 1: Formulating Search Query**"
    await update_pipeline_embed("Working...", current_phase_title)
    search_query = ""
    current_timestamp = datetime.now().strftime('%B %d %Y %H:%M:%S.%f')
    
    last_user_node = next((msg for msg in reversed(initial_prompt_messages) if msg.role == 'user'), None)
    user_text_for_search = ""
    if last_user_node:
        content = last_user_node.content
        if isinstance(content, str):
            user_text_for_search = content
        elif isinstance(content, list):
            user_text_for_search = next((p['text'] for p in content if p['type'] == 'text'), "")
    
    if user_text_for_search and not is_vision_request:
        try:
            query_gen_prompt_text = (
                f"User's message: \"{user_text_for_search}\"\n"
                f"Current timestamp: {current_timestamp}\n"
                "Based on the above, formulate a concise search query for up-to-date info. Respond with ONLY the query."
            )
            query_gen_messages = [MsgNode("system", "You are a search query expert."), MsgNode("user", query_gen_prompt_text)]
            query_response = await llm_client.chat.completions.create(
                model=config.LLM_MODEL, messages=[p.to_dict() for p in query_gen_messages],
                max_tokens=75, stream=False, temperature=0.1
            )
            if query_response.choices and query_response.choices[0].message.content:
                search_query = query_response.choices[0].message.content.strip().strip('"')
                await update_pipeline_embed(f"`{search_query}`", "🔍 **Generated Search Query**")
        except Exception as e:
            logger.error(f"LLM Search Query Generation Error: {e}")
            await update_pipeline_embed("Error during query generation.", "⚠️ **Search Query Error**")
    else:
        await update_pipeline_embed("Skipped (input is image-based or no text found).", current_phase_title)

    # --- Stage 2: Execute Searx Search ---
    current_phase_title = "🌐 **Phase 2: Executing Web Search**"
    formatted_search_results_for_llm = "No web search was performed or no results were found."
    display_search_results = "No web search performed or no results."

    if search_query:
        await update_pipeline_embed("Fetching results...", current_phase_title)
        search_results_data = await query_searx(search_query)
        if search_results_data:
            llm_snippets = [f"[{i+1}] Title: {r.get('title', 'N/A')}\nURL: {r.get('url', '#')}\nSnippet: {r.get('content', r.get('description', 'No snippet.'))}" for i, r in enumerate(search_results_data)]
            formatted_search_results_for_llm = "\n\n".join(llm_snippets)
            
            display_snippets = [f"**[{i+1}] {r.get('title', 'N/A')}**\n*<{r.get('url', '#')}>*\n{r.get('content', r.get('description', 'No snippet.'))[:150]}..." for i, r in enumerate(search_results_data[:3])]
            display_search_results = "\n\n".join(display_snippets)
            if len(search_results_data) > 3: display_search_results += f"\n\n...(and {len(search_results_data)-3} more results)"
            
            await update_pipeline_embed(display_search_results[:config.EMBED_MAX_LENGTH - 500], f"📄 **Web Search Results (Top {len(search_results_data)})**")
        else:
            await update_pipeline_embed("No results found.", current_phase_title)
    else:
       await update_pipeline_embed("Skipped.", current_phase_title)

    # --- Stage 3: Generate <think> Block ---
    current_phase_title = "🤔 **Phase 3: Cognitive Alignment (<think> block)**"
    await update_pipeline_embed("Processing...", current_phase_title)
    
    search_context_header = f"--- BEGIN WEB SEARCH RESULTS (Timestamp: {current_timestamp}) ---\n{formatted_search_results_for_llm}\n--- END WEB SEARCH RESULTS ---\n\nUser's original message (MUST use above search results to inform your <think> block):\n"
    
    if last_user_node and isinstance(last_user_node.content, list): # Check if last_user_node exists
        new_content_list = []
        text_part_found_for_think = False
        for part in last_user_node.content:
            if part['type'] == 'text':
                new_content_list.append({"type": "text", "text": search_context_header + part['text']})
                text_part_found_for_think = True
            else:
                new_content_list.append(part)
        if not text_part_found_for_think:
             new_content_list.insert(0, {"type": "text", "text": search_context_header + "(User sent an image or non-text content)"})
        think_block_user_content_for_llm = new_content_list
    else:
        think_block_user_content_for_llm = search_context_header + user_text_for_search
        
    think_block_prompts = [get_system_prompt_for_think_block(), MsgNode("user", think_block_user_content_for_llm)]
    
    think_block_content = "<think>\nError: Could not generate <think> block.\n</think>"
    try:
        think_model = config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL
        think_response = await llm_client.chat.completions.create(
            model=think_model, 
            messages=[p.to_dict() for p in think_block_prompts],
            max_tokens=1500, stream=False, temperature=0.5
        )
        if think_response.choices and think_response.choices[0].message.content:
            generated_text = think_response.choices[0].message.content.strip()
            # Ensure it's always wrapped, handling cases where the LLM might forget the outer tags
            if not generated_text.startswith("<think>"):
                generated_text = "<think>" + generated_text
            if not generated_text.endswith("</think>"):
                generated_text = generated_text + "</think>"
            think_block_content = generated_text
            
            display_think_block = think_block_content
            if len(display_think_block) > 1800: 
                display_think_block = display_think_block[:1797] + "..."
            await update_pipeline_embed(f"```xml\n{display_think_block}\n```", "🧠 **Cognitive Synthesis (<think> Block)**")
    except Exception as e:
        logger.error(f"LLM <think> Block Generation Error: {e}", exc_info=True)
        await update_pipeline_embed(f"Error: {str(e)[:200]}", "⚠️ **<think> Block Error**")
        think_block_content = f"<think>\nError generating think block: {e}\n</think>"

    # --- Stage 4: Signal Final Response Generation ---
    await update_pipeline_embed("Ready to generate final answer based on the above.", "💬 **Phase 4: Crafting Final Response**")

    # Construct prompts for the final response generation
    final_response_prompts = [get_system_prompt_for_final_response()]
    
    # Pass the original user query to give context to the plan execution
    if last_user_node:
        final_response_prompts.append(last_user_node)
    else: # Fallback if somehow last_user_node is None (should not happen if initial_prompt_messages has user)
        final_response_prompts.append(MsgNode(role="user", content="Please provide a response based on the synthesis."))


    # The 'Cognitive Synthesis' includes search results and the detailed <think> block.
    # The assistant then tees up the Action/Response for the model to complete.
    cognitive_synthesis_and_prompt = (
        f"---COGNITIVE SYNTHESIS (AUGMENTATION DATA)---\n"
        f"---RECAP OF WEB SEARCH RESULTS---\n{formatted_search_results_for_llm}\n---END RECAP---\n\n"
        f"{think_block_content}\n"
        f"---END COGNITIVE SYNTHESIS---\n\n"
        f"Based *solely* on the 'Plan of Action' detailed in the `<think>` block above, here is the response:\n"
        f"Action/Response:"
    )
    final_response_prompts.append(MsgNode(role="assistant", content=cognitive_synthesis_and_prompt))
    # No final user message is needed; the assistant message ending with "Action/Response:" prompts completion.

    final_stream = await llm_client.chat.completions.create(
        model=config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL,
        messages=[p.to_dict() for p in final_response_prompts],
        max_tokens=config.MAX_COMPLETION_TOKENS,
        stream=True, temperature=0.7 # Temperature could be lowered if strict adherence is still an issue
    )
    
    return final_stream, think_block_content # Return the original think_block for history logging


async def stream_llm_response_to_interaction(
    interaction: discord.Interaction,
    prompt_messages: list,
    title: str = "Sam's Response"
):
    pipeline_update_message = None 
    original_interaction_message_id = None

    if not interaction.response.is_done():
        try:
            await interaction.response.defer(ephemeral=False) 
            pipeline_update_message = await interaction.original_response()
            original_interaction_message_id = pipeline_update_message.id
        except discord.errors.InteractionResponded:
            try: 
                pipeline_update_message = await interaction.original_response()
                original_interaction_message_id = pipeline_update_message.id
            except discord.NotFound:
                embed = discord.Embed(title=title, description="⏳ Initializing Cognitive Pipeline...", color=config.EMBED_COLOR["incomplete"])
                pipeline_update_message = await interaction.followup.send(embed=embed, wait=True)
                original_interaction_message_id = pipeline_update_message.id
    if not pipeline_update_message: 
        embed = discord.Embed(title=title, description="⏳ Initializing Cognitive Pipeline...", color=config.EMBED_COLOR["incomplete"])
        pipeline_update_message = await interaction.followup.send(embed=embed, wait=True)
        original_interaction_message_id = pipeline_update_message.id
    
    is_vision_request = any(isinstance(p.content, list) and any(c.get("type") == "image_url" for c in p.content) for p in prompt_messages)
    
    full_final_response_text = ""
    accumulated_chunk = ""
    last_edit_time = asyncio.get_event_loop().time()
    
    final_response_message_to_edit = None
    try:
        initial_pipeline_embed = discord.Embed(title=title, description="⏳ Initializing Cognitive Pipeline...", color=config.EMBED_COLOR["incomplete"])
        await pipeline_update_message.edit(embed=initial_pipeline_embed)

        final_response_embed = discord.Embed(title=f"{title} (Final Response)", description="⏳ Waiting for cognitive processing...", color=config.EMBED_COLOR["incomplete"])
        final_response_message_to_edit = await interaction.channel.send(embed=final_response_embed)

        final_stream, think_block_for_history = await execute_cognitive_pipeline(
            prompt_messages, pipeline_update_message, is_vision_request, title
        )
        
        final_response_embed.description = "" # Clear for streaming

        # Prepend "Action/Response:" if it's not already part of the stream's beginning
        # This is important because the LLM is now completing `Action/Response: `
        is_first_chunk = True

        async for chunk_data in final_stream:
            delta_content = chunk_data.choices[0].delta.content or ""
            
            if is_first_chunk and not delta_content.lstrip().startswith("Action/Response:"):
                 # Check if full_final_response_text (if any from previous chunks) already starts with it
                if not full_final_response_text.lstrip().startswith("Action/Response:"):
                    # Add Action/Response prefix to the embed and internal tracking
                    if not final_response_embed.description: # only add if description is empty
                         final_response_embed.description = "Action/Response:\n"
                    if not full_final_response_text: # only add if text is empty
                        full_final_response_text = "Action/Response:\n"
            is_first_chunk = False

            full_final_response_text += delta_content
            accumulated_chunk += delta_content

            current_time = asyncio.get_event_loop().time()
            if accumulated_chunk and (current_time - last_edit_time >= (1.0 / config.EDITS_PER_SECOND) or len(accumulated_chunk) > 100):
                try:
                    final_response_embed.description += accumulated_chunk
                    if len(final_response_embed.description) > config.EMBED_MAX_LENGTH:
                        final_response_embed.description = final_response_embed.description[:config.EMBED_MAX_LENGTH]
                    await final_response_message_to_edit.edit(embed=final_response_embed)
                    last_edit_time = current_time
                    accumulated_chunk = ""
                except (discord.errors.NotFound, discord.errors.HTTPException) as e:
                    logger.warning(f"Failed to edit final response message during stream: {e}")
                    return 

        if accumulated_chunk:
             final_response_embed.description += accumulated_chunk
        
        # Ensure "Action/Response:" is at the start of the final text if it was missed
        if not full_final_response_text.lstrip().startswith("Action/Response:"):
            full_final_response_text = "Action/Response:\n" + full_final_response_text.lstrip()
            if final_response_embed.description and not final_response_embed.description.lstrip().startswith("Action/Response:"):
                 final_response_embed.description = "Action/Response:\n" + final_response_embed.description.lstrip()


        final_response_embed.description = final_response_embed.description[:config.EMBED_MAX_LENGTH].strip() or "No valid response generated."
        final_response_embed.color = config.EMBED_COLOR["complete"]
        await final_response_message_to_edit.edit(embed=final_response_embed)
        
        channel_id = interaction.channel.id
        if channel_id not in message_history: message_history[channel_id] = []
        message_history[channel_id].append(prompt_messages[-1])
        message_history[channel_id].append(MsgNode(role="assistant", content=think_block_for_history, name=str(bot.user.id) + "_think"))
        message_history[channel_id].append(MsgNode(role="assistant", content=full_final_response_text, name=str(bot.user.id)))
        message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:]

        tts_base_filename = f"interaction_{original_interaction_message_id or interaction.id}"
        await send_tts_audio(interaction.channel, full_final_response_text, base_filename=tts_base_filename) 

    except Exception as e:
        logger.error(f"Error in cognitive pipeline or streaming (interaction): {e}", exc_info=True)
        error_embed = discord.Embed(title=title, description=f"An error occurred during processing: {str(e)[:1000]}", color=config.EMBED_COLOR["error"])
        try:
            if pipeline_update_message and not pipeline_update_message.flags.ephemeral : await pipeline_update_message.edit(embed=error_embed)
            if final_response_message_to_edit: await final_response_message_to_edit.delete() 
        except discord.errors.NotFound: pass
    return pipeline_update_message


async def stream_llm_response_to_message(
    target_message: discord.Message, 
    prompt_messages: list,
    title: str = "Sam's Response"
):
    pipeline_embed = discord.Embed(title=title, description="⏳ Initializing Cognitive Pipeline...", color=config.EMBED_COLOR["incomplete"])
    pipeline_update_message = await target_message.reply(embed=pipeline_embed, silent=True) 

    is_vision_request = any(isinstance(p.content, list) and any(c.get("type") == "image_url" for c in p.content) for p in prompt_messages)
    
    full_final_response_text = ""
    accumulated_chunk = ""
    last_edit_time = asyncio.get_event_loop().time()

    final_response_message_to_edit = None
    try:
        initial_pipeline_embed = discord.Embed(title=title, description="⏳ Initializing Cognitive Pipeline...", color=config.EMBED_COLOR["incomplete"])
        await pipeline_update_message.edit(embed=initial_pipeline_embed)

        final_response_embed = discord.Embed(title=f"{title} (Final Response)", description="⏳ Waiting for cognitive processing...", color=config.EMBED_COLOR["incomplete"])
        final_response_message_to_edit = await target_message.channel.send(embed=final_response_embed)

        final_stream, think_block_for_history = await execute_cognitive_pipeline(
            prompt_messages, pipeline_update_message, is_vision_request, title
        )
        
        final_response_embed.description = "" 
        is_first_chunk = True


        async for chunk_data in final_stream:
            delta_content = chunk_data.choices[0].delta.content or ""

            if is_first_chunk and not delta_content.lstrip().startswith("Action/Response:"):
                if not full_final_response_text.lstrip().startswith("Action/Response:"):
                    if not final_response_embed.description:
                         final_response_embed.description = "Action/Response:\n"
                    if not full_final_response_text:
                        full_final_response_text = "Action/Response:\n"
            is_first_chunk = False
            
            full_final_response_text += delta_content
            accumulated_chunk += delta_content

            current_time = asyncio.get_event_loop().time()
            if accumulated_chunk and (current_time - last_edit_time >= (1.0 / config.EDITS_PER_SECOND) or len(accumulated_chunk) > 100):
                try:
                    final_response_embed.description += accumulated_chunk
                    if len(final_response_embed.description) > config.EMBED_MAX_LENGTH:
                        final_response_embed.description = final_response_embed.description[:config.EMBED_MAX_LENGTH]
                    await final_response_message_to_edit.edit(embed=final_response_embed)
                    last_edit_time = current_time
                    accumulated_chunk = ""
                except (discord.errors.NotFound, discord.errors.HTTPException) as e:
                    logger.warning(f"Failed to edit final response message during stream: {e}")
                    return

        if accumulated_chunk:
            final_response_embed.description += accumulated_chunk

        if not full_final_response_text.lstrip().startswith("Action/Response:"):
            full_final_response_text = "Action/Response:\n" + full_final_response_text.lstrip()
            if final_response_embed.description and not final_response_embed.description.lstrip().startswith("Action/Response:"):
                 final_response_embed.description = "Action/Response:\n" + final_response_embed.description.lstrip()
            
        final_response_embed.description = final_response_embed.description[:config.EMBED_MAX_LENGTH].strip() or "No valid response generated."
        final_response_embed.color = config.EMBED_COLOR["complete"]
        await final_response_message_to_edit.edit(embed=final_response_embed)
        
        channel_id = target_message.channel.id
        if channel_id not in message_history: message_history[channel_id] = []
        message_history[channel_id].append(prompt_messages[-1])
        message_history[channel_id].append(MsgNode(role="assistant", content=think_block_for_history, name=str(bot.user.id) + "_think"))
        message_history[channel_id].append(MsgNode(role="assistant", content=full_final_response_text, name=str(bot.user.id)))
        message_history[channel_id] = message_history[channel_id][-config.MAX_MESSAGE_HISTORY:]

        await send_tts_audio(target_message.channel, full_final_response_text, base_filename=f"message_{target_message.id}")

    except Exception as e:
        logger.error(f"Error in cognitive pipeline or streaming (message): {e}", exc_info=True)
        error_embed = discord.Embed(title=title, description=f"An error occurred during processing: {str(e)[:1000]}", color=config.EMBED_COLOR["error"])
        try:
            await pipeline_update_message.edit(embed=error_embed)
            if final_response_message_to_edit: await final_response_message_to_edit.delete()
        except discord.errors.NotFound: pass
    return pipeline_update_message

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
                else: logger.error(f"TTS Fail: {resp.status}, {await resp.text()}"); return None
    except Exception as e: logger.error(f"TTS Error: {e}", exc_info=True); return None

async def _send_audio_segment(destination: discord.abc.Messageable, segment_text: str, filename_suffix: str, base_filename: str = "response"):
    if not segment_text: return
    logger.info(f"Requesting TTS for {filename_suffix}: {segment_text[:100]}...")
    tts_audio_data = await tts_request(segment_text)
    if tts_audio_data:
        try:
            audio = AudioSegment.from_file(io.BytesIO(tts_audio_data), format="mp3")
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format="mp3", bitrate="128k")
            file = discord.File(io.BytesIO(output_buffer.getvalue()), filename=f"{base_filename}_{filename_suffix}.mp3")
            content_message = "**Sam's spoken response:**"
            if isinstance(destination, discord.Interaction):
                await destination.followup.send(content=content_message, file=file)
            elif hasattr(destination, 'send'):
                await destination.send(content=content_message, file=file)
            else:
                await destination.channel.send(content=content_message, file=file)
            logger.info(f"Sent TTS audio: {base_filename}_{filename_suffix}.mp3")
        except Exception as e: logger.error(f"Error processing/sending TTS: {e}", exc_info=True)
    else: logger.warning(f"TTS request failed for {filename_suffix} segment.")


async def send_tts_audio(destination: discord.abc.Messageable, text_to_speak: str, base_filename: str = "response"):
    if not config.TTS_ENABLED_DEFAULT or not text_to_speak: return
    
    action_response_match = re.search(r"Action/Response:(.*?)(\sReflective Adjustment:|</s>|$)", text_to_speak, re.DOTALL | re.IGNORECASE)
    speakable_text = ""
    if action_response_match:
        speakable_text = action_response_match.group(1).strip()
    else: # If "Action/Response:" is missing, clean and speak the whole text
        logger.warning("Could not find 'Action/Response:' in text for TTS, attempting to clean full text. This may indicate an issue in the LLM response structure.")
        speakable_text = text_to_speak

    cleaned_text = clean_text_for_tts(speakable_text) 
    if cleaned_text:
        await _send_audio_segment(destination, cleaned_text, "response", base_filename=base_filename)

# -------------------------------------------------------------------
# Web Scraping and Search
# -------------------------------------------------------------------
JS_EXPAND_SHOWMORE_TWITTER = """
(maxClicks) => {
    let clicks = 0;
    function isVisible(el) {
        if (!el) return false;
        return !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
    }
    function isSafeToClick(showMoreEl) {
        let parentAnchor = showMoreEl.closest('a');
        if (parentAnchor && parentAnchor.href && parentAnchor.href.includes('/status/')) {
        }
        if (showMoreEl.closest('[data-testid="card.wrapper"]')) {
             return false; 
        }
        return true;
    }
    const articles = document.querySelectorAll('article[data-testid="tweet"]');
    for (const article of articles) {
        if (clicks >= maxClicks) break;
        const candidates = Array.from(article.querySelectorAll('span, div[role="button"]'));
        for (const el of candidates) {
            if (clicks >= maxClicks) break;
            const text = (el.textContent || '').toLowerCase().trim();
            if ((text === 'show more' || text === 'view more replies' || text === 'show additional replies' || text.includes('more repl') || text === 'show') && isVisible(el)) {
                if (isSafeToClick(el)) {
                    try {
                        el.click();
                        clicks++;
                    } catch (e) {}
                }
            }
        }
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
        } catch (e) {}
    });
    return tweets;
}
"""

async def query_searx(query: str) -> list:
    logger.info(f"Querying Searx for: {query}")
    params = {'q': query, 'format': 'json', 'language': 'en-US'}
    if config.SEARX_PREFERENCES: params['preferences'] = config.SEARX_PREFERENCES
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config.SEARX_URL, params=params, timeout=10) as response:
                response.raise_for_status() 
                return (await response.json()).get('results', [])[:10]
    except Exception as e: logger.error(f"Searx query failed for '{query}': {e}"); return []

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
            url = f"https://x.com/{username_queried.lstrip('@')}"
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

async def fetch_youtube_transcript(url: str) -> str | None:
    try:
        video_id_match = re.search(r'(?:v=|\/|embed\/|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})', url)
        if not video_id_match:
            logger.warning(f"Could not extract YouTube video ID from URL: {url}")
            return None
        video_id = video_id_match.group(1)
        logger.info(f"Fetching transcript for YouTube video ID: {video_id}")
        transcript_list = await asyncio.to_thread(YouTubeTranscriptApi.list_transcripts, video_id)
        
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
            fetched_transcript_data = await asyncio.to_thread(transcript.fetch)
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
# Reminder Task & Helper
# -------------------------------------------------------------------
@tasks.loop(seconds=30) 
async def check_reminders():
    now = datetime.now()
    due_reminders_indices = []
    
    for i, reminder_tuple in enumerate(reminders): 
        reminder_time, channel_id, user_id, message_content, original_time_str = reminder_tuple
        if now >= reminder_time:
            logger.info(f"Reminder DUE for user {user_id} in channel {channel_id}: {message_content}")
            try:
                channel = await bot.fetch_channel(channel_id)
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
                due_reminders_indices.append(i)
            except Exception as e:
                logger.error(f"Failed to send reminder (ChID {channel_id}, UserID {user_id}): {e}", exc_info=True)
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
@bot.tree.command(name="remindme", description="Sets a reminder.")
@app_commands.describe(time_duration="Duration (e.g., '10m', '2h30m', '1d').", reminder_message="The message for your reminder.")
async def remindme_slash_command(interaction: discord.Interaction, time_duration: str, reminder_message: str):
    time_delta, descriptive_time_str = parse_time_string_to_delta(time_duration)
    if not time_delta or time_delta.total_seconds() <= 0:
        await interaction.response.send_message("Invalid time duration.", ephemeral=True)
        return
    reminder_time = datetime.now() + time_delta
    reminders.append((reminder_time, interaction.channel_id, interaction.user.id, reminder_message, descriptive_time_str))
    await interaction.response.send_message(f"Okay, {interaction.user.mention}! I'll remind you in {descriptive_time_str} about: \"{reminder_message}\"")

@bot.tree.command(name="search", description="Performs a web search and provides a cognitively processed summary.")
@app_commands.describe(query="Your search query.")
async def search_slash_command(interaction: discord.Interaction, query: str):
    logger.info(f"Search command invoked by {interaction.user.name} for query: {query}.")
    prompt_messages = [
        MsgNode(role="user", content=f"Please research and provide a comprehensive answer to the following query: '{query}'")
    ]
    await stream_llm_response_to_interaction(interaction, prompt_messages, title=f"Cognitive Search for: {query}")

@bot.tree.command(name="roast", description="Generates a comedy routine based on a webpage.")
@app_commands.describe(url="The URL of the webpage to roast.")
async def roast_slash_command(interaction: discord.Interaction, url: str):
    logger.info(f"Roast command invoked by {interaction.user.name} for {url}.")
    webpage_text = await scrape_website(url)
    if not webpage_text or "Failed to scrape" in webpage_text or "Scraping timed out" in webpage_text:
        await interaction.response.send_message(f"Sorry, I couldn't properly roast {url}. {webpage_text if webpage_text else 'Could not retrieve content.'}", ephemeral=True)
        return

    prompt_messages = [
        MsgNode(role="system", content="You are a witty comedian. Your goal is to roast the provided web page content."),
        MsgNode(role="user", content=f"Analyze the content from {url} (provided below) and write a short, witty, and engaging comedy routine about it. Keep it light-hearted and observational.\n\nWebsite Content:\n{webpage_text[:3000]}")
    ]
    await stream_llm_response_to_interaction(interaction, prompt_messages, title=f"Comedy Roast of {url}")

@bot.tree.command(name="pol", description="Generates a sarcastic response to a political statement.")
@app_commands.describe(statement="The political statement.")
async def pol_slash_command(interaction: discord.Interaction, statement: str):
    logger.info(f"Pol command invoked by {interaction.user.name} with statement: {statement[:50]}.")
    prompt_messages = [
        MsgNode(role="system", content="You are a bot that generates extremely sarcastic, snarky, and troll-like comments to mock extremist political views or absurd political statements. Your goal is to be biting and humorous, undermining the statement without being directly offensive or vulgar. Focus on wit and irony."),
        MsgNode(role="user", content=f"Generate a sarcastic comeback to this political statement: \"{statement}\"")
    ]
    await stream_llm_response_to_interaction(interaction, prompt_messages, title="Sarcastic Political Commentary")


@bot.tree.command(name="gettweets", description="Fetches and summarizes recent tweets from a user.")
@app_commands.describe(username="The X/Twitter username (without @).", limit="Number of tweets to fetch (max 15).")
async def gettweets_slash_command(interaction: discord.Interaction, username: str, limit: app_commands.Range[int, 1, 15] = 5):
    await interaction.response.defer(ephemeral=False)
    logger.info(f"Gettweets command invoked by {interaction.user.name} for @{username}.")
    
    tweets = await scrape_latest_tweets(username.lstrip('@'), limit=limit)
    if not tweets:
        await interaction.followup.send(f"Could not fetch tweets for @{username.lstrip('@')}. The profile might be private, non-existent, or X is blocking the request. Scraping X is very unreliable.")
        return

    tweet_texts = [f"- {discord.utils.escape_markdown(t.get('content', 'N/A'))}" for t in tweets]
    raw_tweets_display = "\n".join(tweet_texts)

    await interaction.followup.send(f"Found {len(tweets)} recent tweets for **@{username.lstrip('@')}**. Now generating a summary...")

    prompt_messages = [
        MsgNode(role="user", content=f"Summarize the key themes, topics, and overall sentiment from the following recent tweets by @{username.lstrip('@')}:\n\n{raw_tweets_display[:3500]}")
    ]
    await stream_llm_response_to_interaction(interaction, prompt_messages, title=f"Tweet Summary for @{username.lstrip('@')}")

@bot.tree.command(name="ap", description="Describes an attached image with a creative AP Photo twist.")
@app_commands.describe(image="The image to describe.", user_prompt="Optional additional prompt for the description.")
async def ap_slash_command(interaction: discord.Interaction, image: discord.Attachment, user_prompt: str = ""):
    logger.info(f"AP command invoked by {interaction.user.name}.")
    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.response.send_message("The attached file is not a valid image.", ephemeral=True)
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
    
    prompt_messages = [
        MsgNode(
            role="user",
            content=[ 
                {"type": "text", "text": llm_prompt_text},
                {"type": "image_url", "image_url": {"url": image_url_for_llm}}
            ]
        )
    ]
    await stream_llm_response_to_interaction(interaction, prompt_messages, title=f"AP Photo Description ft. {chosen_celebrity}")


@bot.tree.command(name="clearhistory", description="Clears the bot's message history for this channel.")
@app_commands.checks.has_permissions(manage_messages=True)
async def clearhistory_slash_command(interaction: discord.Interaction):
    if interaction.channel_id in message_history:
        message_history[interaction.channel_id] = []
        await interaction.response.send_message("History cleared for this channel.", ephemeral=True)
    else:
        await interaction.response.send_message("No history to clear for this channel.", ephemeral=True)

# -------------------------------------------------------------------
# Main Event Handlers
# -------------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot: return
    await bot.process_commands(message) 
    
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions
    
    should_respond = is_dm or is_mentioned
    if config.ALLOWED_CHANNEL_IDS and not is_dm:
        if message.channel.id not in config.ALLOWED_CHANNEL_IDS:
            is_thread_in_allowed_channel = isinstance(message.channel, discord.Thread) and message.channel.parent_id in config.ALLOWED_CHANNEL_IDS
            if not is_thread_in_allowed_channel:
                should_respond = False
    if config.ALLOWED_ROLE_IDS and not is_dm and not any(role.id in config.ALLOWED_ROLE_IDS for role in getattr(message.author, 'roles', [])):
        should_respond = False
    
    if not should_respond: return

    logger.info(f"Processing message from {message.author.name} for LLM response.")

    async with message.channel.typing():
        user_message_text = message.content 
        current_message_content_parts = []
        
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("audio/"):
                try:
                    if not os.path.exists("temp"): os.makedirs("temp")
                    audio_filename = f"temp/temp_audio_{attachment.id}.{attachment.filename.split('.')[-1]}"
                    await attachment.save(audio_filename)
                    transcription = await asyncio.to_thread(transcribe_audio_file, audio_filename)
                    if os.path.exists(audio_filename): os.remove(audio_filename) 
                    if transcription:
                        user_message_text = (user_message_text + " " + transcription).strip()
                        await message.reply(f"*Transcribed audio: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"*", silent=True)
                except Exception as e:
                    logger.error(f"Error processing audio attachment: {e}", exc_info=True)
                break 

        detected_urls_in_text = detect_urls(user_message_text) 
        scraped_content_for_llm = "" 
        if detected_urls_in_text:
            for i, url in enumerate(detected_urls_in_text):
                if i >= 1 : break
                content_piece = None
                youtube_pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=|embed\/|shorts\/|v\/|)([\w-]{11})'
                if re.search(youtube_pattern, url):
                    transcript = await fetch_youtube_transcript(url)
                    if transcript: content_piece = f"\n--- YouTube Transcript ---\n{transcript[:1500]}...\n"
                else:
                    scraped_text = await scrape_website(url)
                    if scraped_text: content_piece = f"\n--- Scraped Content ---\n{scraped_text[:1500]}...\n"
                
                if content_piece:
                    scraped_content_for_llm += content_piece

        if scraped_content_for_llm:
            user_message_text += scraped_content_for_llm

        if user_message_text:
            current_message_content_parts.append({"type": "text", "text": user_message_text})
        
        has_image = False
        if message.attachments:
            for att in message.attachments[:config.MAX_IMAGES_PER_MESSAGE]:
                if att.content_type and att.content_type.startswith("image/"):
                    has_image = True
                    img_bytes = await att.read()
                    b64_img = base64.b64encode(img_bytes).decode('utf-8')
                    img_node_content = {"type": "image_url", "image_url": {"url": f"data:{att.content_type};base64,{b64_img}"}}
                    if not user_message_text and not any(p['type'] == 'text' for p in current_message_content_parts):
                        current_message_content_parts.insert(0, {"type": "text", "text": "(User sent an image)"})
                    current_message_content_parts.append(img_node_content)

        if not current_message_content_parts:
            logger.info("No processable content in message.")
            return
        
        user_msg_node_content = current_message_content_parts if has_image or len(current_message_content_parts) > 1 else current_message_content_parts[0]["text"]
        
        channel_id = message.channel.id
        if channel_id not in message_history: message_history[channel_id] = []
        
        current_turn_prompt_history = list(message_history.get(channel_id, []))
        current_turn_prompt_history.append(MsgNode(role="user", content=user_msg_node_content, name=str(message.author.id)))
        
        await stream_llm_response_to_message(message, current_turn_prompt_history)


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == bot.user.id or str(payload.emoji) != '❌': return
    try:
        channel = await bot.fetch_channel(payload.channel_id)
        message_obj = await channel.fetch_message(payload.message_id)
    except (discord.NotFound, discord.Forbidden): return

    if message_obj.author.id != bot.user.id: return
    
    reacting_user = await bot.fetch_user(payload.user_id)
    can_delete = True
    
    if isinstance(channel, discord.TextChannel):
        try:
            member = await channel.guild.fetch_member(payload.user_id)
            if member and member.guild_permissions.manage_messages: can_delete = True
        except discord.NotFound: pass 
    
    if not can_delete and message_obj.reference and message_obj.reference.message_id:
        try:
            original_message = await channel.fetch_message(message_obj.reference.message_id)
            if original_message.author.id == payload.user_id: can_delete = True
        except discord.NotFound: pass
    
    if not can_delete and message_obj.interaction and message_obj.interaction.user.id == payload.user_id:
        can_delete = True

    if can_delete:
        try:
            await message_obj.delete()
            logger.info(f"Message {message_obj.id} deleted by {reacting_user.name} via ❌ reaction.")
        except Exception as e:
            logger.error(f"Error deleting message {message_obj.id} via reaction: {e}")

# -------------------------------------------------------------------
# Bot Startup & Error Handling
# -------------------------------------------------------------------
@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord! ID: {bot.user.id}')
    logger.info(f"discord.py version: {discord.__version__}")
    logger.info(f"PERSONAL_PREFERENCES: {config.PERSONAL_PREFERENCES}")
    try:
        synced = await bot.tree.sync() 
        logger.info(f"Synced {len(synced)} slash commands globally.")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}", exc_info=True)
    
    if not check_reminders.is_running():
        try:
            check_reminders.start()
            logger.info("Check_reminders task started.")
        except RuntimeError:
            logger.warning("Check_reminders task already running or could not be started.")
            
    await bot.change_presence(activity=discord.Game(name="with advanced cognition | /commands"))

@bot.tree.error 
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    command_name = interaction.command.name if interaction.command else 'unknown_command'
    logger.error(f"Slash command error for '{command_name}': {error}", exc_info=True)
    error_message = f"An unexpected error occurred with /{command_name}." 
    if isinstance(error, app_commands.CommandNotFound): error_message = "Sorry, I don't recognize that command."
    elif isinstance(error, app_commands.MissingPermissions): error_message = "You don't have the necessary permissions for that command."
    
    try:
        if interaction.response.is_done():
            await interaction.followup.send(error_message, ephemeral=True)
        else:
            await interaction.response.send_message(error_message, ephemeral=True)
    except Exception as e_resp:
        logger.error(f"Error sending error response for /{command_name}: {e_resp}")

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    logger.error(f"Prefix command error for '{ctx.command}': {error}", exc_info=True)
    if isinstance(error, commands.CommandNotFound): return
    await ctx.reply(f"An error occurred with command !{ctx.command.name if ctx.command else 'unknown'}.", silent=True)

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
