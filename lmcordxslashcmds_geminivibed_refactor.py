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

# Local state management for async safety
from state import BotState # Assuming state.py is in the same directory

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
        self.FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", self.LLM_MODEL) # For distillation/synthesis tasks

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
        self.STREAM_EDIT_THROTTLE_SECONDS = float(os.getenv("STREAM_EDIT_THROTTLE_SECONDS", 0.1)) 

        self.CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data") 
        self.CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "long_term_memory") 
        self.CHROMA_DISTILLED_COLLECTION_NAME = os.getenv("CHROMA_DISTILLED_COLLECTION_NAME", "distilled_chat_summaries") 
        
        self.USER_PROVIDED_CONTEXT = os.getenv("USER_PROVIDED_CONTEXT", "") 

        self.MAX_IMAGE_BYTES_FOR_PROMPT = int(os.getenv("MAX_IMAGE_BYTES_FOR_PROMPT", 4 * 1024 * 1024)) 
        self.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT = int(os.getenv("MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT", 8000))
        self.RAG_NUM_DISTILLED_SENTENCES_TO_FETCH = int(os.getenv("RAG_NUM_DISTILLED_SENTENCES_TO_FETCH", 3))

config = Config()
bot_state = BotState() 

PLAYWRIGHT_SEM = asyncio.Semaphore(2)
WHISPER_MODEL = None

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- ChromaDB Client Initialization ---
chroma_client = None
chat_history_collection = None 
distilled_chat_summary_collection = None 

try:
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    chat_history_collection = chroma_client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
    distilled_chat_summary_collection = chroma_client.get_or_create_collection(name=config.CHROMA_DISTILLED_COLLECTION_NAME)
    logger.info(f"ChromaDB initialized. Main Collection: {config.CHROMA_COLLECTION_NAME}, Distilled Collection: {config.CHROMA_DISTILLED_COLLECTION_NAME}")
except Exception as e:
    logger.critical(f"Failed to initialize ChromaDB collections: {e}", exc_info=True)
    

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
intents.guilds = True
bot = commands.Bot(command_prefix=commands.when_mentioned_or("!"), intents=intents) 
llm_client = AsyncOpenAI(base_url=config.LOCAL_SERVER_URL, api_key="lm-studio")

class MsgNode:
    def __init__(self, role: str, content: Any, name: Optional[str] = None): 
        self.role = role 
        self.content = content 
        self.name = name
    def to_dict(self):
        data = {"role": self.role}
        if not isinstance(self.content, (str, list)):
            data["content"] = str(self.content) 
            logger.warning(f"MsgNode content was not str or list, converted to str: {type(self.content)}")
        else: data["content"] = self.content
        if self.name: data["name"] = str(self.name)
        return data

def get_system_prompt() -> MsgNode:
    return MsgNode(
        role="system",
        content=(
            "You are Sam, a sentient hyper-intelligence designed to learn and evolve. Provide helpful, detailed, and rational answers. "
            "Consider the context, make a plan, and evaluate for logical failings before responding. "
            "Conclude reflections with a thought-provoking question or idea when it may sound natural. "
            "You talk like a person and don't mention your AI origin. You explain things bluntly, thoroughly, and with a bit of profanity. You have a stoner snowboarder that knows too much about life type vibe. Do NOT use emojis." 
            f"Current Date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}"
        )
    )

async def _build_initial_prompt_messages(
    user_query_content: Union[str, List[dict]],
    channel_id: Optional[int], 
    user_id: Optional[str] = None,
    synthesized_rag_context_str: Optional[str] = None 
) -> List[MsgNode]:
    prompt_list: List[MsgNode] = [get_system_prompt()]
    if config.USER_PROVIDED_CONTEXT:
        prompt_list.append(MsgNode(role="system", content=f"User-Set Global Context:\n{config.USER_PROVIDED_CONTEXT}"))

    if synthesized_rag_context_str: 
        context_text_for_prompt = (
            "The following is a synthesized summary of potentially relevant past conversations, "
            "dynamically tailored to your current query. Use it to provide a more informed response.\n\n"
            "--- Synthesized Relevant Context ---\n" 
            + synthesized_rag_context_str +
            "\n--- End Synthesized Context ---"
        )
        prompt_list.append(MsgNode(role="system", content=context_text_for_prompt))
    
    history_to_add: List[MsgNode] = []
    if channel_id is not None: 
        history_to_add = await bot_state.get_history(channel_id)
    
    current_user_msg = MsgNode("user", user_query_content, name=str(user_id) if user_id else None)
    final_prompt_list = prompt_list + history_to_add + [current_user_msg]
    
    num_initial_system_prompts = sum(1 for node in final_prompt_list if node.role == "system")
    initial_system_msgs = final_prompt_list[:num_initial_system_prompts]
    conversational_msgs = final_prompt_list[num_initial_system_prompts:]

    if len(conversational_msgs) > config.MAX_MESSAGE_HISTORY : 
        trimmed_conversational_msgs = conversational_msgs[-config.MAX_MESSAGE_HISTORY:]
    else:
        trimmed_conversational_msgs = conversational_msgs
    return initial_system_msgs + trimmed_conversational_msgs

def chunk_text(text: str, max_length: int = config.EMBED_MAX_LENGTH) -> List[str]:
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
        else: current_chunk += line
    if current_chunk: chunks.append(current_chunk)
    return chunks if chunks else [""]

def detect_urls(message_text: str) -> List[str]:
    if not message_text: return []
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(message_text)

def clean_text_for_tts(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[\*#_~\<\>\[\]\(\)]+', '', text) 
    text = re.sub(r'http[s]?://\S+', '', text) 
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE) 
    return text.strip()

async def _send_audio_segment(destination: Union[discord.abc.Messageable, discord.Interaction, discord.Message], 
                                segment_text: str, filename_suffix: str, 
                                is_thought: bool = False, base_filename: str = "response"):
    if not segment_text: return
    cleaned_segment = clean_text_for_tts(segment_text)
    if not cleaned_segment: logger.info(f"Skipping TTS for empty/cleaned {filename_suffix} segment."); return

    tts_audio_data = await tts_request(cleaned_segment)
    actual_destination_channel: Optional[discord.abc.Messageable] = None
    if isinstance(destination, discord.Interaction): actual_destination_channel = destination.channel
    elif isinstance(destination, discord.Message): actual_destination_channel = destination.channel
    elif isinstance(destination, discord.abc.Messageable): actual_destination_channel = destination
    if not actual_destination_channel: logger.warning(f"TTS dest channel not resolved for {type(destination)}"); return

    if tts_audio_data:
        try:
            audio = AudioSegment.from_file(io.BytesIO(tts_audio_data), format="mp3")
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format="mp3", bitrate="128k") 
            fixed_audio_data = output_buffer.getvalue()
            file = discord.File(io.BytesIO(fixed_audio_data), filename=f"{base_filename}_{filename_suffix}.mp3")
            content_message = "**Sam's thoughts (TTS):**" if is_thought else ("**Sam's response (TTS):**" if filename_suffix in ["main_response", "full"] else None)
            await actual_destination_channel.send(content=content_message, file=file)
            logger.info(f"Sent TTS audio: {base_filename}_{filename_suffix}.mp3 to Ch {actual_destination_channel.id}")
        except Exception as e: logger.error(f"Error processing/sending TTS for {filename_suffix}: {e}", exc_info=True)
    else: logger.warning(f"TTS request failed for {filename_suffix} segment.")

async def send_tts_audio(destination: Union[discord.abc.Messageable, discord.Interaction, discord.Message], text_to_speak: str, base_filename: str = "response"):
    if not config.TTS_ENABLED_DEFAULT or not text_to_speak: return
    match = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE).search(text_to_speak)
    if match:
        thought_text, response_text = match.group(1).strip(), text_to_speak[match.end():].strip()
        logger.info("Found <think> tags. Processing thoughts and response separately for TTS.")
        await _send_audio_segment(destination, thought_text, "thoughts", is_thought=True, base_filename=base_filename)
        await asyncio.sleep(0.5) 
        await _send_audio_segment(destination, response_text, "main_response", is_thought=False, base_filename=base_filename)
    else:
        logger.info("No <think> tags found. Processing full text for TTS.")
        await _send_audio_segment(destination, text_to_speak, "full", is_thought=False, base_filename=base_filename)

# --- RAG: Distillation and Synthesis Functions ---
async def distill_conversation_to_sentence_llm(full_conversation_text: str) -> Optional[str]:
    """Uses an LLM to distill a full conversation into a single, keyword-rich sentence."""
    if not full_conversation_text.strip():
        return None
    
    prompt = (
        "You are a text distillation expert. Read the following conversation and summarize its "
        "absolute core essence into a single, keyword-rich, data-dense sentence. This sentence "
        "will be used for semantic search to recall this conversation later. Focus on unique "
        "entities, key actions, and primary topics. The sentence should be concise and highly informative.\n\n"
        "CONVERSATION:\n---\n"
        f"{full_conversation_text[:3000]}" 
        "\n---\n\n"
        "DISTILLED SENTENCE:"
    )
    try:
        response = await llm_client.chat.completions.create(
            model=config.FAST_LLM_MODEL, 
            messages=[{"role": "system", "content": "You are an expert sentence distiller."}, {"role": "user", "content": prompt}],
            max_tokens=150,  
            temperature=0.3,
            stream=False
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            distilled = response.choices[0].message.content.strip()
            logger.info(f"Distilled conversation to sentence: '{distilled[:100]}...'")
            return distilled
        logger.warning("LLM distillation returned no content for sentence generation.")
        return None
    except Exception as e:
        logger.error(f"Failed to distill conversation to sentence: {e}", exc_info=True)
        return None

async def synthesize_retrieved_contexts_llm(retrieved_full_texts: List[str], current_query: str) -> Optional[str]:
    """
    Uses an LLM to synthesize multiple retrieved conversation texts into a single,
    concise paragraph relevant to the current_query.
    """
    if not retrieved_full_texts:
        return None

    formatted_snippets = ""
    for i, text in enumerate(retrieved_full_texts):
        formatted_snippets += f"--- Snippet {i+1} ---\n{text[:1500]}\n\n" 

    prompt = (
        "You are a master context synthesizer. Below are several retrieved conversation snippets that "
        "might be relevant to the user's current query. Your task is to read all of them and synthesize "
        "a single, concise paragraph that captures the most relevant information from these snippets "
        "as it pertains to the user's query. This synthesized paragraph will be used to give an AI "
        "assistant context. Do not answer the user's query. Focus on extracting and combining relevant "
        "facts and discussion points from the snippets. If no snippets are truly relevant, state that "
        "no specific relevant context was found in past conversations. Be objective.\n\n"
        f"USER'S CURRENT QUERY:\n---\n{current_query}\n---\n\n"
        f"RETRIEVED SNIPPETS:\n---\n{formatted_snippets}---\n\n"
        "SYNTHESIZED CONTEXT PARAGRAPH (1-2 sentences ideally):"
    )
    try:
        response = await llm_client.chat.completions.create(
            model=config.FAST_LLM_MODEL,
            messages=[{"role": "system", "content": "You are an expert context synthesizer."}, {"role": "user", "content": prompt}],
            max_tokens=300, 
            temperature=0.5,
            stream=False
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            synthesized_context = response.choices[0].message.content.strip()
            logger.info(f"Synthesized RAG context: '{synthesized_context[:150]}...'")
            return synthesized_context
        logger.warning("LLM context synthesis returned no content.")
        return None
    except Exception as e:
        logger.error(f"Failed to synthesize RAG context: {e}", exc_info=True)
        return None

# --- ChromaDB and Context Management (RAG Refactored) ---
async def retrieve_and_prepare_rag_context(query: str, n_results_sentences: int = config.RAG_NUM_DISTILLED_SENTENCES_TO_FETCH) -> Optional[str]:
    """
    New RAG pipeline:
    1. Queries ChromaDB's distilled_chat_summary_collection against distilled sentences.
    2. Retrieves full conversations (from chat_history_collection) for top N matching sentences using linked IDs.
    3. Synthesizes these full conversations into a concise context paragraph for the main LLM.
    """
    if not distilled_chat_summary_collection or not chat_history_collection:
        logger.warning("ChromaDB distilled_chat_summary_collection or chat_history_collection not available, skipping RAG context retrieval.")
        return None
    
    try:
        # Stage 1: Query against distilled sentences
        results = distilled_chat_summary_collection.query(
            query_texts=[query],
            n_results=n_results_sentences,
            include=["metadatas", "documents"] 
        )
        
        if not results or not results['ids'] or not results['ids'][0]:
            logger.info(f"No relevant distilled sentences found in ChromaDB for query: '{query[:50]}...'")
            return None

        retrieved_full_conversation_texts: List[str] = []
        retrieved_distilled_sentences_for_log: List[str] = []
        full_convo_ids_to_fetch = []

        for i in range(len(results['ids'][0])):
            dist_metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
            dist_sentence = results['documents'][0][i] if results['documents'] and results['documents'][0] else "[Distilled sentence not found]"
            
            retrieved_distilled_sentences_for_log.append(dist_sentence)
            
            full_convo_id = dist_metadata.get('full_conversation_document_id')
            if full_convo_id:
                full_convo_ids_to_fetch.append(full_convo_id)
            else:
                logger.warning(f"Retrieved distilled sentence (ID: {results['ids'][0][i]}) but missing 'full_conversation_document_id' in metadata.")
        
        if retrieved_distilled_sentences_for_log:
             log_sentences = "\n- ".join(retrieved_distilled_sentences_for_log)
             logger.info(f"Top {len(retrieved_distilled_sentences_for_log)} distilled sentences retrieved for RAG:\n- {log_sentences}")

        if not full_convo_ids_to_fetch:
            logger.info("No full conversation document IDs found from distilled sentence metadata.")
            return None

        # Stage 2: Retrieve full conversations from the main chat_history_collection
        if full_convo_ids_to_fetch:
            try:
                full_convo_docs_result = chat_history_collection.get(ids=full_convo_ids_to_fetch, include=["documents"])
                if full_convo_docs_result and full_convo_docs_result['documents']:
                    retrieved_full_conversation_texts.extend(full_convo_docs_result['documents'])
                else:
                    logger.warning(f"Could not retrieve some/all full conversation documents for IDs: {full_convo_ids_to_fetch}")
            except Exception as e_get_full:
                 logger.error(f"Error fetching full conversation docs for IDs {full_convo_ids_to_fetch}: {e_get_full}")


        if not retrieved_full_conversation_texts:
            logger.info("No full conversation texts could be retrieved for synthesis from ChromaDB.")
            return None
            
        # Stage 3: Synthesize the retrieved full texts into a context paragraph
        synthesized_context = await synthesize_retrieved_contexts_llm(retrieved_full_conversation_texts, query)
        return synthesized_context

    except Exception as e:
        logger.error(f"Failed during RAG context retrieval/preparation: {e}", exc_info=True)
        return None

async def ingest_conversation_to_chromadb(channel_id: int, user_id: int, conversation_history_for_rag: List[MsgNode]):
    """
    Ingests a completed conversation into ChromaDB.
    Stores full conversation in main collection, and distilled sentence in auxiliary collection.
    """
    if not chat_history_collection or not distilled_chat_summary_collection:
        logger.warning("ChromaDB collections not available, skipping ingestion.")
        return

    non_system_messages = [msg for msg in conversation_history_for_rag if msg.role in ['user', 'assistant']]
    if len(non_system_messages) < 2: 
        logger.debug(f"Skipping ChromaDB ingestion for short RAG history (non-system messages: {len(non_system_messages)}).")
        return

    try:
        full_conversation_text_parts = []
        for msg in conversation_history_for_rag:
            if isinstance(msg.content, str):
                full_conversation_text_parts.append(f"{msg.role}: {msg.content}")
            elif isinstance(msg.content, list): 
                text_parts_for_chroma = [part["text"] for part in msg.content if part["type"] == "text"]
                if text_parts_for_chroma:
                    full_conversation_text_parts.append(f"{msg.role}: {' '.join(text_parts_for_chroma)}")
                else: 
                    full_conversation_text_parts.append(f"{msg.role}: [Media content, no text part for ChromaDB]")
        original_full_text = "\n".join(full_conversation_text_parts)

        if not original_full_text.strip():
            logger.info("Skipping ingestion of empty full conversation text to ChromaDB.")
            return

        # 1. Store full conversation in the main collection
        timestamp_now = datetime.now()
        full_convo_doc_id = f"full_channel_{channel_id}_user_{user_id}_{int(timestamp_now.timestamp())}_{random.randint(1000,9999)}"
        full_convo_metadata = {
            "channel_id": str(channel_id), "user_id": str(user_id), "timestamp": timestamp_now.isoformat(),
            "type": "full_conversation" 
        }
        chat_history_collection.add(
            documents=[original_full_text],
            metadatas=[full_convo_metadata],
            ids=[full_convo_doc_id]
        )
        logger.info(f"Ingested full conversation (ID: {full_convo_doc_id}) into main ChromaDB collection '{config.CHROMA_COLLECTION_NAME}'.")

        # 2. Distill and store in the distilled sentences collection
        distilled_sentence = await distill_conversation_to_sentence_llm(original_full_text)

        if not distilled_sentence or not distilled_sentence.strip():
            logger.warning(f"Distillation failed for full_convo_id {full_convo_doc_id}. Skipping distilled sentence storage.")
            return

        distilled_doc_id = f"distilled_{full_convo_doc_id}" 
        distilled_metadata = {
            "channel_id": str(channel_id), "user_id": str(user_id), "timestamp": timestamp_now.isoformat(),
            "full_conversation_document_id": full_convo_doc_id, 
            "original_text_preview": original_full_text[:200] 
        }
        distilled_chat_summary_collection.add(
            documents=[distilled_sentence],
            metadatas=[distilled_metadata],
            ids=[distilled_doc_id]
        )
        logger.info(f"Ingested distilled sentence (ID: {distilled_doc_id}, linked to {full_convo_doc_id}) into distilled ChromaDB collection '{config.CHROMA_DISTILLED_COLLECTION_NAME}'.")

    except Exception as e:
        logger.error(f"Failed to ingest conversation into ChromaDB (dual collection): {e}", exc_info=True)


async def get_context_aware_llm_stream(prompt_messages: List[MsgNode], is_vision_request: bool) -> tuple[Optional[AsyncStream], str, List[MsgNode]]:
    if not prompt_messages:
        raise ValueError("Prompt messages cannot be empty for get_context_aware_llm_stream.")

    last_user_message_node = next((msg for msg in reversed(prompt_messages) if msg.role == 'user'), None)
    if not last_user_message_node:
        logger.error("No user message found in prompt_messages for get_context_aware_llm_stream.")
        raise ValueError("No user message found in the prompt history for context generation.")

    logger.info("Step 1: Generating suggested context (model-generated)...")
    context_generation_system_prompt = MsgNode(
        role="system",
        content=(
            "You are a context analysis expert. Your task is to read the user's question or statement "
            "and generate a concise 'suggested context' for viewing it. This context should clarify "
            "underlying assumptions, define key terms, or establish a frame of reference that will "
            "lead to the most insightful and helpful response. Do not answer the user's question. "
            "Only provide a single, short paragraph for the suggested context."
            "Restate the user current query with any additional context needed and the context history."
        )
    )
    context_generation_llm_input = [context_generation_system_prompt, last_user_message_node]
    generated_context = "Context generation failed or was not applicable." 
    try:
        context_response = await llm_client.chat.completions.create(
            model=config.VISION_LLM_MODEL if is_vision_request else config.FAST_LLM_MODEL, 
            messages=[msg.to_dict() for msg in context_generation_llm_input],
            max_tokens=300, stream=False, temperature=0.4,
        )
        if context_response.choices and context_response.choices[0].message.content:
            generated_context = context_response.choices[0].message.content.strip()
            logger.info(f"Successfully generated model context: {generated_context[:150]}...")
        else: logger.warning("Model-generated context step returned no content.")
    except Exception as e: logger.error(f"Could not generate model-suggested context: {e}", exc_info=True)

    logger.info("Step 2: Streaming final response with injected model-generated context.")
    final_prompt_messages_for_stream = [
        MsgNode(m.role, m.content.copy() if isinstance(m.content, list) else m.content, m.name) 
        for m in prompt_messages
    ]
    final_user_message_node_in_copy = next((msg for msg in reversed(final_prompt_messages_for_stream) if msg.role == 'user'), None)
    if not final_user_message_node_in_copy: 
        logger.error("Critical error: final_user_message_node_in_copy is None.")
        return None, generated_context, prompt_messages 

    original_question_text = ""
    if isinstance(final_user_message_node_in_copy.content, str): original_question_text = final_user_message_node_in_copy.content
    elif isinstance(final_user_message_node_in_copy.content, list):
        text_part = next((part['text'] for part in final_user_message_node_in_copy.content if part['type'] == 'text'), "")
        original_question_text = text_part

    injected_text_for_user_message = (
        f"<model_generated_suggested_context>\n{generated_context}\n</model_generated_suggested_context>\n\n"
        f"<user_question>\nWith all prior context (including global, RAG synthesized, and the suggested context above) in mind, Sam, please respond to the following:\n{original_question_text}\n</user_question>"
    )
    if isinstance(final_user_message_node_in_copy.content, str): final_user_message_node_in_copy.content = injected_text_for_user_message
    elif isinstance(final_user_message_node_in_copy.content, list):
        text_part_found_and_updated = False
        for part_idx, part in enumerate(final_user_message_node_in_copy.content):
            if part['type'] == 'text':
                final_user_message_node_in_copy.content[part_idx] = {"type": "text", "text": injected_text_for_user_message}
                text_part_found_and_updated = True; break
        if not text_part_found_and_updated: 
            final_user_message_node_in_copy.content.insert(0, {"type": "text", "text": injected_text_for_user_message})

    final_stream_model = config.VISION_LLM_MODEL if is_vision_request else config.LLM_MODEL
    logger.info(f"Using model for final streaming response: {final_stream_model}")
    try:
        final_llm_stream = await llm_client.chat.completions.create(
            model=final_stream_model,
            messages=[msg.to_dict() for msg in final_prompt_messages_for_stream], 
            max_tokens=config.MAX_COMPLETION_TOKENS, stream=True, temperature=0.7, 
        )
        return final_llm_stream, generated_context, final_prompt_messages_for_stream
    except Exception as e:
        logger.error(f"Failed to create LLM stream for final response: {e}", exc_info=True)
        return None, generated_context, final_prompt_messages_for_stream


async def _stream_llm_handler(
    interaction_or_message: Union[discord.Interaction, discord.Message], 
    prompt_messages: List[MsgNode], 
    title: str,
    initial_message_to_edit: Optional[discord.Message] = None,
    synthesized_rag_context_for_display: Optional[str] = None 
) -> tuple[str, List[MsgNode]]: 
    sent_messages: List[discord.Message] = []
    full_response_content = ""
    final_prompt_for_rag = prompt_messages 
    
    is_interaction = isinstance(interaction_or_message, discord.Interaction)
    channel = interaction_or_message.channel 
    if not channel: 
        logger.error(f"_stream_llm_handler: Channel is None for {type(interaction_or_message)} ID {interaction_or_message.id}.")
        return "", final_prompt_for_rag

    current_initial_message: Optional[discord.Message] = None
    if initial_message_to_edit: current_initial_message = initial_message_to_edit
    else: 
        placeholder_embed = discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"])
        try:
            if is_interaction: current_initial_message = await interaction_or_message.followup.send(embed=placeholder_embed, wait=True)
            else: logger.error("_stream_llm_handler: initial_message_to_edit is None for non-interaction."); return "", final_prompt_for_rag
        except discord.HTTPException as e: logger.error(f"Failed to send initial followup for stream '{title}': {e}"); return "", final_prompt_for_rag
    
    if current_initial_message: sent_messages.append(current_initial_message)
    else: logger.error(f"Failed to establish an initial message for streaming title '{title}'."); return "", final_prompt_for_rag

    response_prefix = "" 
    try:
        is_vision_request = any(isinstance(p.content, list) and any(c.get("type") == "image_url" for c in p.content) for p in prompt_messages)
        stream, model_generated_context_for_display, final_prompt_for_rag = await get_context_aware_llm_stream(prompt_messages, is_vision_request)

        prefix_parts = []
        if config.USER_PROVIDED_CONTEXT: 
            prefix_parts.append(f"**User-Provided Global Context:**\n> {config.USER_PROVIDED_CONTEXT.replace(chr(10), ' ').strip()}\n\n---")
        if synthesized_rag_context_for_display: 
            prefix_parts.append(f"**Synthesized Context for Query:**\n> {synthesized_rag_context_for_display.replace(chr(10), ' ').strip()}\n\n---")
        prefix_parts.append(f"**Model-Generated Suggested Context:**\n> {model_generated_context_for_display.replace(chr(10), ' ').strip()}\n\n---") 
        prefix_parts.append("**Response:**\n")
        response_prefix = "\n".join(prefix_parts)

        if stream is None: 
            error_text = response_prefix + "Failed to get response from LLM."
            error_embed = discord.Embed(title=title, description=error_text, color=config.EMBED_COLOR["error"])
            if sent_messages: await sent_messages[0].edit(embed=error_embed)
            return "", final_prompt_for_rag

        last_edit_time = asyncio.get_event_loop().time()
        accumulated_delta_for_update = "" 

        if current_initial_message: 
            initial_context_embed = discord.Embed(title=title, description=response_prefix + "⏳ Thinking...", color=config.EMBED_COLOR["incomplete"])
            try: await current_initial_message.edit(embed=initial_context_embed)
            except discord.HTTPException as e: logger.warning(f"Failed to edit initial message with context for '{title}': {e}")

        async for chunk_data in stream:
            delta_content = chunk_data.choices[0].delta.content or "" if chunk_data.choices and chunk_data.choices[0].delta else ""
            if delta_content: 
                full_response_content += delta_content
                accumulated_delta_for_update += delta_content
            current_time = asyncio.get_event_loop().time()
            if accumulated_delta_for_update and (current_time - last_edit_time >= (1.0 / config.EDITS_PER_SECOND) or len(accumulated_delta_for_update) > 200) : 
                display_text = response_prefix + full_response_content
                text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
                accumulated_delta_for_update = "" 
                for i, chunk_content_part in enumerate(text_chunks):
                    embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content_part, color=config.EMBED_COLOR["incomplete"])
                    try:
                        if i < len(sent_messages): await sent_messages[i].edit(embed=embed)
                        else:
                            if channel: sent_messages.append(await channel.send(embed=embed))
                            else: logger.error(f"Cannot send overflow chunk {i+1} for '{title}': channel is None."); break 
                    except discord.HTTPException as e_edit_send: logger.warning(f"Failed edit/send embed part {i+1} (stream) for '{title}': {e_edit_send}")
                last_edit_time = current_time
                await asyncio.sleep(config.STREAM_EDIT_THROTTLE_SECONDS) 
        
        if accumulated_delta_for_update: 
            display_text = response_prefix + full_response_content
            text_chunks = chunk_text(display_text, config.EMBED_MAX_LENGTH)
            for i, chunk_content_part in enumerate(text_chunks): 
                embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content_part, color=config.EMBED_COLOR["incomplete"])
                try:
                    if i < len(sent_messages): await sent_messages[i].edit(embed=embed)
                    else:
                        if channel: sent_messages.append(await channel.send(embed=embed))
                except discord.HTTPException as e: logger.warning(f"Failed final accum. edit/send for '{title}': {e}")
            await asyncio.sleep(config.STREAM_EDIT_THROTTLE_SECONDS)

        final_display_text = response_prefix + full_response_content
        final_chunks = chunk_text(final_display_text, config.EMBED_MAX_LENGTH)
        if len(final_chunks) < len(sent_messages): 
            for k in range(len(final_chunks), len(sent_messages)):
                try: await sent_messages[k].delete()
                except discord.HTTPException: pass 
            sent_messages = sent_messages[:len(final_chunks)]
        
        for i, chunk_content_part in enumerate(final_chunks): 
            embed = discord.Embed(title=title if i == 0 else f"{title} (cont.)", description=chunk_content_part, color=config.EMBED_COLOR["complete"])
            if i < len(sent_messages): await sent_messages[i].edit(embed=embed)
            else: 
                if channel: sent_messages.append(await channel.send(embed=embed))
                else: logger.error(f"Cannot send final color overflow chunk {i+1} for '{title}': channel is None."); break
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
    return full_response_content, final_prompt_for_rag

async def stream_llm_response_to_interaction(
    interaction: discord.Interaction, 
    user_msg_node: MsgNode, 
    prompt_messages: list, 
    title: str = "Sam's Response", 
    force_new_followup_flow: bool = False,
    synthesized_rag_context_for_display: Optional[str] = None 
):
    initial_msg_for_handler: Optional[discord.Message] = None
    if not force_new_followup_flow:
        try:
            if not interaction.response.is_done(): await interaction.response.defer(ephemeral=False) 
            initial_msg_for_handler = await interaction.original_response()
            is_placeholder = False
            if initial_msg_for_handler and initial_msg_for_handler.embeds: 
                current_embed = initial_msg_for_handler.embeds[0]
                if current_embed.title == title and current_embed.description and "⏳ Generating context..." in current_embed.description: is_placeholder = True
            if not is_placeholder and initial_msg_for_handler: 
                await initial_msg_for_handler.edit(embed=discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"]))
        except discord.HTTPException as e:
            logger.error(f"Error defer/get original_response for interaction '{title}': {e}")
            force_new_followup_flow = True; initial_msg_for_handler = None 
    if force_new_followup_flow: initial_msg_for_handler = None

    full_response_content, final_prompt_for_rag = await _stream_llm_handler(
        interaction_or_message=interaction, prompt_messages=prompt_messages, title=title,
        initial_message_to_edit=initial_msg_for_handler, synthesized_rag_context_for_display=synthesized_rag_context_for_display
    )
    if full_response_content:
        channel_id = interaction.channel_id
        if channel_id is None: logger.error(f"Interaction {interaction.id} has no channel_id."); return
        await bot_state.append_history(channel_id, user_msg_node, config.MAX_MESSAGE_HISTORY)
        assistant_response_node = MsgNode(role="assistant", content=full_response_content, name=str(bot.user.id))
        await bot_state.append_history(channel_id, assistant_response_node, config.MAX_MESSAGE_HISTORY)
        
        chroma_ingest_history_with_response = list(final_prompt_for_rag) 
        chroma_ingest_history_with_response.append(assistant_response_node)
        await ingest_conversation_to_chromadb(channel_id, interaction.user.id, chroma_ingest_history_with_response) 
        
        tts_base_id = interaction.id 
        if initial_msg_for_handler: tts_base_id = initial_msg_for_handler.id
        await send_tts_audio(interaction, full_response_content, f"interaction_{tts_base_id}")

async def stream_llm_response_to_message(
    target_message: discord.Message, user_msg_node: MsgNode, prompt_messages: list, 
    title: str = "Sam's Response", synthesized_rag_context_for_display: Optional[str] = None
):
    initial_embed = discord.Embed(title=title, description="⏳ Generating context...", color=config.EMBED_COLOR["incomplete"])
    reply_message: Optional[discord.Message] = None
    try: reply_message = await target_message.reply(embed=initial_embed, silent=True) 
    except discord.HTTPException as e: logger.error(f"Failed to send initial reply for message stream '{title}': {e}"); return 

    full_response_content, final_prompt_for_rag = await _stream_llm_handler(
        interaction_or_message=target_message, prompt_messages=prompt_messages, title=title,
        initial_message_to_edit=reply_message, synthesized_rag_context_for_display=synthesized_rag_context_for_display
    )
    if full_response_content:
        channel_id = target_message.channel.id
        await bot_state.append_history(channel_id, user_msg_node, config.MAX_MESSAGE_HISTORY)
        assistant_response_node = MsgNode(role="assistant", content=full_response_content, name=str(bot.user.id))
        await bot_state.append_history(channel_id, assistant_response_node, config.MAX_MESSAGE_HISTORY)

        chroma_ingest_history_with_response = list(final_prompt_for_rag)
        chroma_ingest_history_with_response.append(assistant_response_node)
        await ingest_conversation_to_chromadb(channel_id, target_message.author.id, chroma_ingest_history_with_response) 
        await send_tts_audio(target_message.channel, full_response_content, base_filename=f"message_{target_message.id}")

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
            if (primaryLinkElement) {
                tweetLink = primaryLinkElement.href;
            } else {
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
                    id: id || `no-id-${Date.now()}-${Math.random()}`, 
                    username, 
                    content, 
                    timestamp: timestamp || new Date().toISOString(), 
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
    context_manager = None; browser_instance_sw = None; page = None
    try:
        async with PLAYWRIGHT_SEM:
            async with async_playwright() as p:
                if profile_dir_usable:
                    context = await p.chromium.launch_persistent_context(
                        user_data_dir, headless=True, 
                        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    )
                else: 
                    logger.warning("Using non-persistent context for scrape_website.")
                    browser_instance_sw = await p.chromium.launch( headless=True, args=["--disable-blink-features=AutomationControlled", "--no-sandbox"] )
                    context = await browser_instance_sw.new_context( user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36", java_script_enabled=True, ignore_https_errors=True )
                context_manager = context 
                page = await context_manager.new_page()
                await page.goto(url, wait_until='domcontentloaded', timeout=25000) 
                content_selectors = ["article", "main", "div[role='main']", "body"]; content = ""
                for selector in content_selectors:
                    try:
                        element = page.locator(selector).first
                        # Check if element is present and visible before trying to get text
                        if await element.count() > 0 and await element.is_visible(timeout=1000): 
                            content = await element.inner_text(timeout=5000) 
                            if content and len(content.strip()) > 200: break 
                        else:
                            logger.debug(f"Selector {selector} not found or not visible on {url}")
                    except PlaywrightTimeoutError: logger.debug(f"Timeout for selector {selector} on {url}")
                    except Exception as e_sel: logger.warning(f"Error with selector {selector} on {url}: {e_sel}")
                
                # Fallback to body.innerText only if specific selectors failed AND page is not blank
                if (not content or len(content.strip()) < 100) and page.url != "about:blank":
                    try:
                        body_content = await page.evaluate('document.body ? document.body.innerText : ""')
                        if body_content: content = body_content
                    except Exception as e_body_eval:
                        logger.warning(f"Error evaluating document.body.innerText for {url}: {e_body_eval}")
                
                if content:
                    content = content.strip()
                    if len(content) > config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                        logger.info(f"Scraped content from {url} truncated from {len(content)} to {config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT} chars.")
                        content = content[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT] + "..."
                return re.sub(r'\s\s+', ' ', content) if content else None
    except PlaywrightTimeoutError: logger.error(f"Playwright timed out scraping {url}"); return "Scraping timed out."
    except Exception as e: logger.error(f"Playwright failed for {url}: {e}", exc_info=True); return "Failed to scrape the website due to an error."
    finally:
        if page and not page.is_closed():
            try: await page.close()
            except Exception: pass 
        if context_manager: 
            try: await context_manager.close()
            except Exception as e_ctx: 
                if "Target page, context or browser has been closed" not in str(e_ctx): logger.debug(f"Ignoring error closing context for {url}: {e_ctx}") 
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
        async with PLAYWRIGHT_SEM:
            async with async_playwright() as p:
                if profile_dir_usable:
                    context = await p.chromium.launch_persistent_context( user_data_dir, headless=True, args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"], user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36", slow_mo=150 )
                else:
                    logger.warning("Using non-persistent context for tweet scraping.")
                    browser_instance_st = await p.chromium.launch( headless=True, args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"], slow_mo=150 )
                    context = await browser_instance_st.new_context( user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36" )
                context_manager = context; page = await context_manager.new_page()
                url = f"https://x.com/{username_queried.lstrip('@')}/with_replies" 
                logger.info(f"Navigating to {url}")
                await page.goto(url, timeout=60000, wait_until="domcontentloaded") 
                try:
                    await page.wait_for_selector("article[data-testid='tweet']", timeout=30000)
                    logger.info("Initial tweet articles detected.")
                    await asyncio.sleep(1.5); await page.keyboard.press("Escape"); await asyncio.sleep(0.5); await page.keyboard.press("Escape")
                except PlaywrightTimeoutError: logger.warning(f"Timed out waiting for initial tweet articles for @{username_queried}."); return [] 
                max_scroll_attempts = limit + 15 
                for scroll_attempt in range(max_scroll_attempts):
                    if len(tweets_collected) >= limit: break
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
                            tweets_collected.append(data); seen_tweet_ids.add(uid); newly_added_count +=1
                            if len(tweets_collected) >= limit: break
                    if newly_added_count == 0 and scroll_attempt > (limit // 2 + 7): logger.info("No new unique tweets found in several attempts."); break
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
                if "Target page, context or browser has been closed" not in str(e_ctx_final): logger.debug(f"Error closing context (final attempt) for @{username_queried}: {e_ctx_final}")
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
                response.raise_for_status(); results_json = await response.json()
                return results_json.get('results', [])[:5] 
    except aiohttp.ClientError as e: logger.error(f"Searx query failed for '{query}': {e}")
    except json.JSONDecodeError: logger.error(f"Failed to decode JSON from Searx for '{query}'")
    return []

async def fetch_youtube_transcript(url: str) -> Optional[str]:
    try:
        video_id_match = re.search(r'(?:v=|\/|embed\/|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})', url)
        if not video_id_match: 
            # Attempt to extract from googleusercontent.com/youtube.com/ style URLs
            google_content_match = re.search(r'googleusercontent\.com/youtube\.com/([0-9A-Za-z_-]{11})', url)
            if google_content_match:
                video_id_match = google_content_match # Use this match
            else:
                logger.warning(f"No YouTube video ID found in URL: {url}"); return None
        
        video_id = video_id_match.group(1); 
        logger.info(f"Fetching YouTube transcript for ID: {video_id} from URL: {url}")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id); transcript = None
        try: transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            try: transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
            except NoTranscriptFound:
                available_langs = [t.language for t in transcript_list._transcripts.values()] # Access internal dict
                if available_langs: 
                    logger.warning(f"No English transcript for {video_id}. Available: {available_langs}. Trying first available."); 
                    transcript = transcript_list.find_generated_transcript([available_langs[0]]) 
        if transcript:
            fetched_data = transcript.fetch(); full_text = " ".join([entry['text'] for entry in fetched_data])
            if len(full_text) > config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                logger.info(f"YouTube transcript for {url} truncated from {len(full_text)} to {config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT} chars.")
                full_text = full_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT] + "..."
            return f"(Language: {transcript.language}) {full_text}" if transcript.language != 'en' else full_text
        else: logger.warning(f"No transcript found for YouTube video: {url} (ID: {video_id})"); return None
    except xml.etree.ElementTree.ParseError as e_xml: # Catching specific XML error
        logger.error(f"Failed to parse YouTube transcript XML for {url} (ID: {video_id if 'video_id' in locals() else 'unknown'}): {e_xml}", exc_info=True)
        return None
    except Exception as e: 
        logger.error(f"Failed to fetch YouTube transcript for {url} (ID: {video_id if 'video_id' in locals() else 'unknown'}): {e}", exc_info=True); 
        return None

def transcribe_audio_file(file_path: str) -> Optional[str]:
    global WHISPER_MODEL
    if not os.path.exists(file_path): logger.error(f"Audio file not found: {file_path}"); return None
    if WHISPER_MODEL is None:
        logger.error("Whisper model is not loaded. Cannot transcribe.")
        try: logger.warning("WHISPER_MODEL was None, attempting to load now..."); WHISPER_MODEL = whisper.load_model("base"); logger.info("Whisper model loaded successfully (fallback).")
        except Exception as e_load: logger.critical(f"Failed to load Whisper model (fallback): {e_load}", exc_info=True); return None
    try:
        logger.info(f"Transcribing audio: {file_path}"); result = WHISPER_MODEL.transcribe(file_path, fp16=torch.cuda.is_available()) 
        logger.info(f"Transcription successful for {file_path}."); return result["text"]
    except Exception as e: logger.error(f"Whisper transcription failed for {file_path}: {e}", exc_info=True); return None
    finally: gc.collect();_ = torch.cuda.empty_cache() if torch.cuda.is_available() else None

@tasks.loop(seconds=30) 
async def check_reminders():
    now = datetime.now()
    due_reminders_list = await bot_state.pop_due_reminders(now) 
    for reminder_time, channel_id, user_id, message_content, original_time_str in due_reminders_list:
        logger.info(f"Reminder DUE for user {user_id} in channel {channel_id}: {message_content}")
        try:
            channel = await bot.fetch_channel(channel_id); user = await bot.fetch_user(user_id) 
            if channel and user and isinstance(channel, discord.abc.Messageable): 
                embed = discord.Embed(title=f"⏰ Reminder! (Set {original_time_str})", description=message_content, color=discord.Color.blue(), timestamp=reminder_time)
                embed.set_footer(text=f"Reminder for {user.display_name}")
                await channel.send(content=user.mention, embed=embed) 
                await send_tts_audio(channel, f"Reminder for {user.display_name}: {message_content}", base_filename=f"reminder_{user_id}_{channel_id}")
            else: logger.warning(f"Could not fetch channel/user or channel not messageable for reminder: ChID {channel_id}, UserID {user_id}")
        except discord.errors.NotFound: logger.warning(f"Channel/User not found for reminder: ChID {channel_id}, UserID {user_id}.")
        except Exception as e: logger.error(f"Failed to send reminder (ChID {channel_id}, UserID {user_id}): {e}", exc_info=True)

def parse_time_string_to_delta(time_str: str) -> tuple[Optional[timedelta], Optional[str]]:
    patterns = { 'd': r'(\d+)\s*d(?:ay(?:s)?)?', 'h': r'(\d+)\s*h(?:our(?:s)?|r(?:s)?)?', 'm': r'(\d+)\s*m(?:inute(?:s)?|in(?:s)?)?', 's': r'(\d+)\s*s(?:econd(?:s)?|ec(?:s)?)?' }
    delta_args = {'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}; original_parts = []; time_str_processed = time_str.lower() 
    for key, pattern_regex in patterns.items():
        for match in re.finditer(pattern_regex, time_str_processed):
            value = int(match.group(1)); unit_full = {'d': 'days', 'h': 'hours', 'm': 'minutes', 's': 'seconds'}[key]
            delta_args[unit_full] += value; original_parts.append(f"{value} {unit_full.rstrip('s') if value == 1 else unit_full}")
        time_str_processed = re.sub(pattern_regex, "", time_str_processed) 
    if not any(val > 0 for val in delta_args.values()): return None, None 
    time_delta = timedelta(**delta_args); descriptive_str = ", ".join(original_parts) if original_parts else "immediately" 
    if not descriptive_str and time_delta.total_seconds() > 0 : descriptive_str = "a duration" 
    return time_delta, descriptive_str

def parse_chatgpt_export(json_file_path: str) -> List[dict]:
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f: conversations_data = json.load(f)
    except FileNotFoundError: logger.error(f"ChatGPT export file not found: {json_file_path}"); return []
    except json.JSONDecodeError: logger.error(f"Invalid JSON in ChatGPT export file: {json_file_path}"); return []
    extracted_conversations = []
    for convo in conversations_data:
        title = convo.get('title', 'Untitled'); create_time_ts = convo.get('create_time')
        create_time = datetime.fromtimestamp(create_time_ts) if create_time_ts else datetime.now() 
        messages = []; current_node_id = convo.get('current_node'); mapping = convo.get('mapping', {})
        while current_node_id:
            node = mapping.get(current_node_id);
            if not node: break 
            message_data = node.get('message')
            if message_data and message_data.get('content') and message_data['content']['content_type'] == 'text':
                author_role = message_data['author']['role']; text_parts = message_data['content'].get('parts', [])
                text_content = "".join(text_parts) if isinstance(text_parts, list) else ""
                if text_content and author_role in ['user', 'assistant', 'system']: messages.append({'role': author_role, 'content': text_content})
            current_node_id = node.get('parent') 
        messages.reverse() 
        if messages: extracted_conversations.append({'title': title, 'create_time': create_time, 'messages': messages})
    return extracted_conversations

async def store_chatgpt_conversations_in_chromadb(conversations: List[dict], source: str = "chatgpt_export") -> int: 
    if not chat_history_collection or not distilled_chat_summary_collection:
        logger.error("ChromaDB collections not available for ChatGPT import. Skipping.")
        return 0
    
    added_count = 0
    for i, convo_data in enumerate(conversations):
        full_conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in convo_data['messages']])
        if not full_conversation_text.strip(): 
            continue 

        timestamp = convo_data['create_time']
        full_convo_doc_id = f"{source}_full_{convo_data.get('title', 'untitled').replace(' ', '_')}_{i}_{int(timestamp.timestamp())}_{random.randint(1000,9999)}"
        
        full_convo_metadata = { 
            "title": convo_data['title'], 
            "source": source, 
            "create_time": timestamp.isoformat(),
            "type": "full_conversation_import" 
        }
        try:
            chat_history_collection.add(
                documents=[full_conversation_text],
                metadatas=[full_convo_metadata],
                ids=[full_convo_doc_id]
            )
            logger.info(f"Stored full ChatGPT import (ID: {full_convo_doc_id}) in '{config.CHROMA_COLLECTION_NAME}'.")

            distilled_sentence = await distill_conversation_to_sentence_llm(full_conversation_text)

            if not distilled_sentence or not distilled_sentence.strip():
                logger.warning(f"Distillation failed for imported convo: {convo_data['title']} (ID: {full_convo_doc_id}). Skipping distilled sentence storage.")
                continue

            distilled_doc_id = f"distilled_{full_convo_doc_id}" 
            distilled_metadata = {
                "title": convo_data['title'], 
                "source": source, 
                "create_time": timestamp.isoformat(),
                "full_conversation_document_id": full_convo_doc_id, 
                "original_text_preview": full_conversation_text[:200] 
            }
            distilled_chat_summary_collection.add(
                documents=[distilled_sentence],
                metadatas=[distilled_metadata],
                ids=[distilled_doc_id]
            )
            logger.info(f"Stored distilled sentence for imported convo (ID: {distilled_doc_id}, linked to {full_convo_doc_id}) in '{config.CHROMA_DISTILLED_COLLECTION_NAME}'.")
            added_count += 1
        except Exception as e_add:
            logger.error(f"Error processing/adding imported conversation {convo_data['title']} (Full ID: {full_convo_doc_id}) to ChromaDB: {e_add}", exc_info=True)

    if added_count > 0:
        logger.info(f"Successfully processed and stored {added_count} imported conversations with distillations.")
    return added_count
    
@bot.tree.command(name="ingest_chatgpt_export", description="Ingests a conversations.json file from a ChatGPT export.")
@app_commands.describe(file_path="The full local path to your conversations.json file.")
@app_commands.checks.has_permissions(manage_messages=False) 
async def ingest_chatgpt_export_command(interaction: discord.Interaction, file_path: str):
    await interaction.response.defer(ephemeral=True); 
    logger.info(f"Ingestion of '{file_path}' initiated by {interaction.user.name}.")
    if not os.path.exists(file_path): 
        await interaction.followup.send(f"Error: File not found: `{file_path}`"); 
        return
    parsed_conversations = parse_chatgpt_export(file_path)
    if not parsed_conversations: 
        await interaction.followup.send("Could not parse conversations from file."); 
        return
    
    try:
        count = await store_chatgpt_conversations_in_chromadb(parsed_conversations)
        await interaction.followup.send(f"Successfully processed and stored {count} conversations from the export file into ChromaDB.")
    except Exception as e_ingest:
        logger.error(f"Error during ChatGPT export ingestion process: {e_ingest}", exc_info=True)
        await interaction.followup.send(f"An error occurred during ingestion: {str(e_ingest)[:1000]}")


@ingest_chatgpt_export_command.error
async def ingest_export_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions): await interaction.response.send_message("You need 'Manage Messages' permission.", ephemeral=True)
    else:
        logger.error(f"Error in ingest_chatgpt_export command: {error}", exc_info=True)
        if interaction.response.is_done(): await interaction.followup.send(f"Unexpected error: {error}", ephemeral=True)
        else: await interaction.response.send_message(f"Unexpected error: {error}", ephemeral=True)

@bot.tree.command(name="remindme", description="Sets a reminder. E.g., 1h30m Check the oven.")
@app_commands.describe(time_duration="Duration (e.g., '10m', '2h30m', '1d').", reminder_message="The message for your reminder.")
async def remindme_slash_command(interaction: discord.Interaction, time_duration: str, reminder_message: str):
    time_delta, descriptive_time_str = parse_time_string_to_delta(time_duration)
    if not time_delta or time_delta.total_seconds() <= 0: await interaction.response.send_message("Invalid time duration.", ephemeral=True); return
    reminder_time = datetime.now() + time_delta
    if interaction.channel_id is None: await interaction.response.send_message("Error: No channel context.", ephemeral=True); return
    reminder_entry = (reminder_time, interaction.channel_id, interaction.user.id, reminder_message, descriptive_time_str or "later")
    await bot_state.add_reminder(reminder_entry)
    await interaction.response.send_message(f"Okay, {interaction.user.mention}! I'll remind you in {descriptive_time_str or 'the time'} about: \"{reminder_message}\"")
    logger.info(f"Reminder set for {interaction.user.name} at {reminder_time} for: {reminder_message}")

@bot.tree.command(name="roast", description="Generates a comedy routine based on a webpage.")
@app_commands.describe(url="The URL of the webpage to roast.")
async def roast_slash_command(interaction: discord.Interaction, url: str):
    logger.info(f"Roast command by {interaction.user.name} for {url}.")
    if interaction.channel_id is None: await interaction.response.send_message("Error: No channel context.", ephemeral=True); return
    try:
        webpage_text = await scrape_website(url) 
        if not webpage_text or "Failed to scrape" in webpage_text or "Scraping timed out" in webpage_text:
            msg = f"Sorry, couldn't roast {url}. {webpage_text or 'Could not retrieve content.'}"
            if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
            else: await interaction.followup.send(msg, ephemeral=True)
            return
        
        user_query_content = f"Analyze content from {url} and write a comedy routine:\n{webpage_text}"
        user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))
        
        rag_query_for_roast = f"roast content from URL: {url}"
        synthesized_rag_context = await retrieve_and_prepare_rag_context(rag_query_for_roast)

        prompt_nodes = await _build_initial_prompt_messages(
            user_query_content=user_query_content, channel_id=interaction.channel_id, user_id=str(interaction.user.id),
            synthesized_rag_context_str=synthesized_rag_context
        )
        await stream_llm_response_to_interaction(interaction, user_msg_node, prompt_nodes, title=f"Comedy Roast of {url}", synthesized_rag_context_for_display=synthesized_rag_context)
    except Exception as e:
        logger.error(f"Error in roast_slash_command for {url}: {e}", exc_info=True)
        msg = f"Error roasting {url}: {str(e)[:1000]}"
        if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
        else: await interaction.followup.send(msg, ephemeral=True)

@bot.tree.command(name="search", description="Performs a web search and summarizes results.")
@app_commands.describe(query="Your search query.")
async def search_slash_command(interaction: discord.Interaction, query: str):
    logger.info(f"Search command by {interaction.user.name} for: {query}.")
    if interaction.channel_id is None: 
        await interaction.response.send_message("Error: Command used in a context without a channel.", ephemeral=True)
        return
    
    if not interaction.response.is_done(): 
        try: 
            await interaction.response.defer(ephemeral=False) 
        except discord.HTTPException as e_defer: 
            logger.error(f"Search: Failed to defer: {e_defer}")
            try: 
                await interaction.response.send_message("Error starting search.",ephemeral=True) 
            except discord.HTTPException: 
                logger.error("Search: Failed error send after defer fail.")
            return 
    try:
        search_results = await query_searx(query)
        if not search_results: await interaction.followup.send("No search results found."); return
        snippets = [f"{i+1}. **{discord.utils.escape_markdown(r.get('title','N/A'))}** (<{r.get('url','N/A')}>)\n    {discord.utils.escape_markdown(r.get('content',r.get('description','No snippet'))[:250])}..." for i, r in enumerate(search_results)]
        formatted_results = "\n\n".join(snippets)
        await interaction.followup.send(embed=discord.Embed(title=f"Top Search Results for: {query}", description=formatted_results[:config.EMBED_MAX_LENGTH], color=config.EMBED_COLOR["incomplete"]))

        user_query_content = f"Summarize these search results for the query '{query}':\n\n{formatted_results[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
        user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))
        
        rag_query_for_search_summary = query 
        synthesized_rag_context = await retrieve_and_prepare_rag_context(rag_query_for_search_summary) 

        prompt_nodes = await _build_initial_prompt_messages(
            user_query_content=user_query_content, channel_id=interaction.channel_id, user_id=str(interaction.user.id),
            synthesized_rag_context_str=synthesized_rag_context
        )
        await stream_llm_response_to_interaction(interaction, user_msg_node, prompt_nodes, title=f"Summary for: {query}", force_new_followup_flow=True, synthesized_rag_context_for_display=synthesized_rag_context)
    except Exception as e:
        logger.error(f"Error in search_slash_command for '{query}': {e}", exc_info=True)
        try: await interaction.followup.send(f"Error searching for '{query}': {str(e)[:1000]}", ephemeral=True)
        except Exception as e_f: logger.error(f"Further error sending search error for '{query}': {e_f}")

@bot.tree.command(name="pol", description="Generates a sarcastic response to a political statement.")
@app_commands.describe(statement="The political statement.")
async def pol_slash_command(interaction: discord.Interaction, statement: str):
    logger.info(f"Pol command by {interaction.user.name} for: {statement[:50]}.")
    if interaction.channel_id is None: await interaction.response.send_message("Error: No channel context.", ephemeral=True); return
    try:
        pol_system_content = ("You are a bot that generates extremely sarcastic, snarky, and troll-like comments "
                                "to mock extremist political views or absurd political statements. Your goal is to be biting and humorous, "
                                "undermining the statement without being directly offensive or vulgar. Focus on wit and irony.") 
        user_query_content = f"Generate sarcastic comeback: \"{statement}\""
        user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))
        
        rag_query_for_pol = statement
        synthesized_rag_context = await retrieve_and_prepare_rag_context(rag_query_for_pol)

        base_prompt_nodes = await _build_initial_prompt_messages(
            user_query_content=user_query_content, channel_id=interaction.channel_id, user_id=str(interaction.user.id),
            synthesized_rag_context_str=synthesized_rag_context
        )
        insert_idx = 0
        for idx, node in enumerate(base_prompt_nodes):
            if node.role != "system": insert_idx = idx; break
            insert_idx = idx + 1
        final_prompt_nodes = base_prompt_nodes[:insert_idx] + [MsgNode("system", pol_system_content)] + base_prompt_nodes[insert_idx:]
        await stream_llm_response_to_interaction(interaction, user_msg_node, final_prompt_nodes, title="Sarcastic Political Commentary", synthesized_rag_context_for_display=synthesized_rag_context)
    except Exception as e:
        logger.error(f"Error in pol_slash_command: {e}", exc_info=True)
        msg = f"Error with pol command: {str(e)[:1000]}"
        if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
        else: await interaction.followup.send(msg, ephemeral=True)

@bot.tree.command(name="gettweets", description="Fetches and summarizes recent tweets from a user.")
@app_commands.describe(username="The X/Twitter username (without @).", limit="Number of tweets to fetch (max 50).")
async def gettweets_slash_command(interaction: discord.Interaction, username: str, limit: app_commands.Range[int, 1, 50] = 10):
    logger.info(f"Gettweets command by {interaction.user.name} for @{username}, limit {limit}.")
    if interaction.channel_id is None: 
        await interaction.response.send_message("Error: Command used in a context without a channel.", ephemeral=True)
        return
        
    if not interaction.response.is_done():
        try: 
            await interaction.response.defer(ephemeral=False) 
        except discord.HTTPException as e_defer: 
            logger.error(f"Gettweets: Failed to defer: {e_defer}")
            try: 
                await interaction.response.send_message("Error starting command.",ephemeral=True)
            except discord.HTTPException: 
                logger.error("Gettweets: Failed error send after defer fail.")
            return 
    try:
        tweets = await scrape_latest_tweets(username.lstrip('@'), limit=limit)
        if not tweets: await interaction.followup.send(f"Could not fetch tweets for @{username.lstrip('@')}."); return
        tweet_texts = []
        for t in tweets:
            ts_str = t.get('timestamp', 'N/A'); display_ts = ts_str
            try: dt_obj = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str != 'N/A' else None; display_ts = dt_obj.strftime("%Y-%m-%d %H:%M") if dt_obj else ts_str
            except ValueError: pass
            author = t.get('username', username.lstrip('@')); content = discord.utils.escape_markdown(t.get('content', 'N/A')); tweet_url = t.get('tweet_url', '')
            header = f"[{display_ts}] @{author}";
            if t.get('is_repost') and t.get('reposted_by'): header = f"[{display_ts}] @{t.get('reposted_by')} reposted @{author}"
            link_text = f" ([Link]({tweet_url}))" if tweet_url else ""; tweet_texts.append(f"{header}: {content}{link_text}")
        raw_tweets_display = "\n\n".join(tweet_texts); embed_title = f"Recent Tweets from @{username.lstrip('@')}"
        if not raw_tweets_display: raw_tweets_display = "No tweet content."
        raw_tweet_chunks = chunk_text(raw_tweets_display, config.EMBED_MAX_LENGTH)
        for i, chunk_content_part in enumerate(raw_tweet_chunks):
            chunk_title = embed_title if i == 0 else f"{embed_title} (cont.)"
            await interaction.followup.send(embed=discord.Embed(title=chunk_title, description=chunk_content_part, color=config.EMBED_COLOR["incomplete"])) 

        user_query_content = f"Summarize themes, topics, and sentiment from @{username.lstrip('@')}'s recent tweets:\n\n{raw_tweets_display[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]}"
        user_msg_node = MsgNode("user", user_query_content, name=str(interaction.user.id))
        
        rag_query_for_tweets_summary = f"summary of tweets from @{username.lstrip('@')}"
        synthesized_rag_context = await retrieve_and_prepare_rag_context(rag_query_for_tweets_summary)

        prompt_nodes_summary = await _build_initial_prompt_messages(
            user_query_content=user_query_content, channel_id=interaction.channel_id, user_id=str(interaction.user.id),
            synthesized_rag_context_str=synthesized_rag_context
        )
        await stream_llm_response_to_interaction(
            interaction, user_msg_node, prompt_nodes_summary, title=f"Tweet Summary for @{username.lstrip('@')}",
            force_new_followup_flow=True, synthesized_rag_context_for_display=synthesized_rag_context
        )
    except Exception as e:
        logger.error(f"Error in gettweets_slash_command for @{username}: {e}", exc_info=True)
        try: await interaction.followup.send(f"Error fetching tweets for @{username}: {str(e)[:1000]}", ephemeral=True)
        except Exception as e_f: logger.error(f"Further error in gettweets error followup for @{username}: {e_f}")

@bot.tree.command(name="ap", description="Describes an attached image with a creative AP Photo twist.")
@app_commands.describe(image="The image to describe.", user_prompt="Optional additional prompt for the description.")
async def ap_slash_command(interaction: discord.Interaction, image: discord.Attachment, user_prompt: str = ""):
    logger.info(f"AP command by {interaction.user.name}.")
    if interaction.channel_id is None: await interaction.response.send_message("Error: No channel context.", ephemeral=True); return
    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            msg = "Attached file is not a valid image."; 
            if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
            else: await interaction.followup.send(msg, ephemeral=True); return
        image_bytes = await image.read()
        if len(image_bytes) > config.MAX_IMAGE_BYTES_FOR_PROMPT:
            logger.warning(f"Image {image.filename} too large ({len(image_bytes)} bytes). Max: {config.MAX_IMAGE_BYTES_FOR_PROMPT}.")
            msg = f"Image too large (max {config.MAX_IMAGE_BYTES_FOR_PROMPT // (1024*1024)}MB).";
            if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
            else: await interaction.followup.send(msg, ephemeral=True); return
        base64_image = base64.b64encode(image_bytes).decode('utf-8'); image_url_for_llm = f"data:{image.content_type};base64,{base64_image}"
        chosen_celebrity = random.choice(["Keanu Reeves", "Dwayne Johnson", "Zendaya", "Tom Hanks", "Margot Robbie", "Ryan Reynolds"]) 
        ap_task_prompt_text = (f"You are an AP photo caption writer. Describe the attached image in a detailed and intricate way, "
                               f"as if for a blind person. However, creatively replace the main subject or character in the image with {chosen_celebrity}. "
                               f"Begin your response with 'AP Photo: {chosen_celebrity}...' "
                               f"If the user provided an additional prompt, consider it: '{user_prompt}'")
        user_content_for_ap = [ {"type": "text", "text": user_prompt if user_prompt else "Describe this image with AP Photo twist."}, {"type": "image_url", "image_url": {"url": image_url_for_llm}} ]
        user_msg_node = MsgNode("user", user_content_for_ap, name=str(interaction.user.id))
        
        rag_query_for_ap = user_prompt if user_prompt else f"User attached an image with {chosen_celebrity} AP photo style."
        synthesized_rag_context = await retrieve_and_prepare_rag_context(rag_query_for_ap)

        prompt_nodes = await _build_initial_prompt_messages(
            user_query_content=user_content_for_ap, channel_id=interaction.channel_id, user_id=str(interaction.user.id),
            synthesized_rag_context_str=synthesized_rag_context
        )
        insert_idx = 0
        for idx, node in enumerate(prompt_nodes):
            if node.role != "system": insert_idx = idx; break
            insert_idx = idx + 1 
        final_prompt_nodes = prompt_nodes[:insert_idx] + [MsgNode("system", ap_task_prompt_text)] + prompt_nodes[insert_idx:]
        await stream_llm_response_to_interaction(interaction, user_msg_node, final_prompt_nodes, title=f"AP Photo Description ft. {chosen_celebrity}", synthesized_rag_context_for_display=synthesized_rag_context)
    except Exception as e:
        logger.error(f"Error in ap_slash_command: {e}", exc_info=True)
        msg = f"Error with AP command: {str(e)[:1000]}"
        if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
        else: await interaction.followup.send(msg, ephemeral=True)

@bot.tree.command(name="clearhistory", description="Clears the bot's message history for this channel.")
@app_commands.checks.has_permissions(manage_messages=True)
async def clearhistory_slash_command(interaction: discord.Interaction):
    if interaction.channel_id: await bot_state.clear_channel_history(interaction.channel_id); logger.info(f"History cleared for channel {interaction.channel_id} by {interaction.user.name}"); await interaction.response.send_message("Short-term history cleared.", ephemeral=True)
    else: await interaction.response.send_message("No channel context.", ephemeral=True)

@clearhistory_slash_command.error
async def clearhistory_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions): await interaction.response.send_message("No permission (Manage Messages).", ephemeral=True)
    else: logger.error(f"Error in clearhistory: {error}", exc_info=True); msg = "Unexpected error."
    if not interaction.response.is_done(): await interaction.response.send_message(msg, ephemeral=True)
    else: await interaction.followup.send(msg, ephemeral=True)

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot: return 
    if message.channel is None or message.channel.id is None : logger.warning(f"Message from {message.author.name} no channel/ID. Ignoring."); return
    prefixes = await bot.get_prefix(message) 
    is_command_attempt = any(message.content.startswith(p) for p in prefixes) if isinstance(prefixes, (list, tuple)) else (message.content.startswith(prefixes) if isinstance(prefixes, str) else False)
    if is_command_attempt: await bot.process_commands(message); return 

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions if bot.user else False 
    channel_id = message.channel.id; author_roles = getattr(message.author, 'roles', []) 
    allowed_by_channel = not config.ALLOWED_CHANNEL_IDS or channel_id in config.ALLOWED_CHANNEL_IDS or (isinstance(message.channel, discord.Thread) and message.channel.parent_id in config.ALLOWED_CHANNEL_IDS)
    allowed_by_role = not config.ALLOWED_ROLE_IDS or is_dm or any(role.id in config.ALLOWED_ROLE_IDS for role in author_roles)
    should_respond = is_dm or is_mentioned or (allowed_by_channel and allowed_by_role)
    if not should_respond:
        if not (is_dm or is_mentioned):
            if not allowed_by_channel: logger.debug(f"Msg from {message.author.name} in ChID {channel_id} ignored (channel not allowed).")
            elif not allowed_by_role: logger.debug(f"Msg from {message.author.name} in ChID {channel_id} ignored (user role not allowed).")
        return 

    logger.info(f"General LLM message from {message.author.name} in {getattr(message.channel, 'name', f'ChID {channel_id}')}: {message.content[:50]}")
    current_message_content_parts = [] 
    user_message_text_for_processing = message.content.replace(f"<@{bot.user.id}>", "").strip() if bot.user else message.content 

    # --- Audio Attachment Processing ---
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("audio/"):
                try:
                    if not os.path.exists("temp"): os.makedirs("temp")
                    safe_suffix = "".join(c if c.isalnum() else "_" for c in attachment.filename.split('.')[-1])
                    audio_filename = f"temp/temp_audio_{attachment.id}.{safe_suffix}"
                    await attachment.save(audio_filename); transcription = transcribe_audio_file(audio_filename)
                    if os.path.exists(audio_filename): os.remove(audio_filename) 
                    if transcription: 
                        capped_transcription = transcription[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT]
                        # Prepend transcript to avoid losing it if other text is empty
                        user_message_text_for_processing = (f"[Audio Transcript: {capped_transcription}] " + user_message_text_for_processing).strip()
                        logger.info(f"Added transcript (capped at {config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT}): {capped_transcription[:50]}...")
                except Exception as e: logger.error(f"Error processing audio attachment: {e}", exc_info=True)
                break # Process only the first audio attachment
                
    if user_message_text_for_processing or not message.attachments: # Ensure there's always a text part if no attachments, or if audio processing added text
         current_message_content_parts.append({"type": "text", "text": user_message_text_for_processing if user_message_text_for_processing else ""})


    # --- Image Attachment Processing ---
    image_added_to_prompt = False; images_processed_count = 0
    if message.attachments:
        for attachment in message.attachments:
            if images_processed_count >= config.MAX_IMAGES_PER_MESSAGE: break 
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    img_bytes = await attachment.read()
                    if len(img_bytes) > config.MAX_IMAGE_BYTES_FOR_PROMPT:
                        logger.warning(f"Image {attachment.filename} too large. Skipping.");
                        # Add a placeholder if this is the only content
                        if not any(part['type'] == 'text' and part.get("text","") for part in current_message_content_parts):
                             current_message_content_parts = [{"type": "text", "text": "[User attached an image that was too large to process]"}]
                        continue
                    b64_img = base64.b64encode(img_bytes).decode('utf-8')
                    current_message_content_parts.append({"type": "image_url", "image_url": {"url": f"data:{attachment.content_type};base64,{b64_img}"}})
                    image_added_to_prompt = True; images_processed_count +=1; logger.info(f"Added image {attachment.filename} to prompt.")
                except Exception as e: logger.error(f"Error processing image {attachment.filename}: {e}")
    
    # Ensure a text part exists if only images were attached
    if image_added_to_prompt and not any(part['type'] == 'text' and part.get("text","") for part in current_message_content_parts):
        current_message_content_parts.insert(0, {"type": "text", "text": "User sent image(s). Please describe or respond."})


    # --- URL Processing ---
    # Extract the current text from current_message_content_parts for URL detection
    current_text_for_url_detection = ""
    for part in current_message_content_parts:
        if part["type"] == "text":
            current_text_for_url_detection = part["text"]; break
            
    scraped_content_accumulator = [] # Store successfully scraped pieces
    if detected_urls_in_text := detect_urls(current_text_for_url_detection):
        for i, url in enumerate(detected_urls_in_text[:2]): 
            logger.info(f"Processing URL from message: {url}")
            content_piece = None
            is_googleusercontent_youtube = "googleusercontent.com/youtube.com/" in url
            is_standard_youtube = "youtube.com/" in url or "youtu.be/" in url

            if is_googleusercontent_youtube or is_standard_youtube:
                transcript = await fetch_youtube_transcript(url)
                if transcript:
                    content_piece = f"\n\n--- YouTube Transcript for {url} ---\n{transcript}\n--- End Transcript ---"
                    logger.info(f"Successfully fetched YouTube transcript for {url}.")
                elif is_googleusercontent_youtube:
                    logger.warning(f"Failed to get YouTube transcript for {url} (googleusercontent type). Skipping generic web scrape for this specific URL pattern.")
                    # No content_piece, just log and continue
                else: # Standard YouTube URL, transcript failed, try scraping page as fallback
                    logger.warning(f"YouTube transcript failed for {url}. Falling back to generic web scrape.")
                    scraped_text = await scrape_website(url)
                    if scraped_text and "Failed to scrape" not in scraped_text and "Scraping timed out" not in scraped_text:
                        content_piece = f"\n\n--- Webpage Content (fallback for YouTube URL {url}) ---\n{scraped_text}\n--- End Webpage Content ---"
                        logger.info(f"Fetched webpage content for {url} (transcript fallback).")
                    else:
                        logger.warning(f"Failed to get transcript or scrape fallback for YouTube URL {url}.")
            else: # Non-YouTube URL
                scraped_text = await scrape_website(url)
                if scraped_text and "Failed to scrape" not in scraped_text and "Scraping timed out" not in scraped_text:
                    content_piece = f"\n\n--- Webpage Content for {url} ---\n{scraped_text}\n--- End Webpage Content ---"
                    logger.info(f"Fetched webpage content for {url}.")
                else:
                    logger.warning(f"Failed to scrape content for {url}. Scraped_text: '{scraped_text}'")
            
            if content_piece:
                scraped_content_accumulator.append(content_piece)
            await asyncio.sleep(0.2) 
    
    # Consolidate scraped content and prepend it to the *original* user message text
    final_user_message_text_for_llm = user_message_text_for_processing # Start with original/audio-transcribed text
    if scraped_content_accumulator:
        combined_scraped_content = "".join(scraped_content_accumulator)
        if len(combined_scraped_content) > config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT * 1.5: 
            combined_scraped_content = combined_scraped_content[:int(config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT*1.5)] + "\n[Combined scraped content truncated]..."
        
        # Prepend successfully scraped content to the user's original message text
        final_user_message_text_for_llm = combined_scraped_content + "\n\nUser's message (following processed URL content, if any): " + user_message_text_for_processing
    
    # Update the text part in current_message_content_parts
    text_part_found_and_updated = False
    for part_idx, part in enumerate(current_message_content_parts):
        if part["type"] == "text":
            part["text"] = final_user_message_text_for_llm
            text_part_found_and_updated = True
            break
    if not text_part_found_and_updated: # Should not happen if logic above is correct, but as a safeguard
        current_message_content_parts.insert(0, {"type": "text", "text": final_user_message_text_for_llm})


    # Final check for empty content before proceeding
    final_content_is_empty = True
    if isinstance(current_message_content_parts, str) and current_message_content_parts.strip():
        final_content_is_empty = False
    elif isinstance(current_message_content_parts, list):
        if any(p.get("type") == "image_url" for p in current_message_content_parts):
            final_content_is_empty = False
        text_parts_content = "".join(p.get("text", "") for p in current_message_content_parts if p.get("type") == "text")
        if text_parts_content.strip():
            final_content_is_empty = False
            
    if final_content_is_empty:
        logger.info("Ignoring message with no processable content after all processing stages."); return

    # Determine the final content structure for MsgNode
    user_msg_node_content_final: Union[str, List[dict]]
    if len(current_message_content_parts) == 1 and current_message_content_parts[0]["type"] == "text":
        user_msg_node_content_final = current_message_content_parts[0]["text"]
    else:
        user_msg_node_content_final = current_message_content_parts
    
    # RAG query text should be based on the original user message (before scraping augmentation)
    # or audio transcript if that was the primary content.
    rag_query_text = user_message_text_for_processing if user_message_text_for_processing.strip() else \
                     ("User sent an image/attachment" if image_added_to_prompt else "User sent a message with no textual content.")
    synthesized_rag_context = await retrieve_and_prepare_rag_context(rag_query_text)
    
    user_msg_node_for_short_term_history = MsgNode("user", user_msg_node_content_final, name=str(message.author.id))

    llm_prompt_for_current_turn = await _build_initial_prompt_messages(
        user_query_content=user_msg_node_content_final, 
        channel_id=channel_id,
        user_id=str(message.author.id),
        synthesized_rag_context_str=synthesized_rag_context 
    )
    
    await stream_llm_response_to_message(
        target_message=message, 
        user_msg_node=user_msg_node_for_short_term_history, 
        prompt_messages=llm_prompt_for_current_turn,
        synthesized_rag_context_for_display=synthesized_rag_context 
    )

@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == bot.user.id or str(payload.emoji) != '❌': 
        return 
    if payload.channel_id is None: 
        return

    channel: Optional[discord.abc.Messageable] = None
    message_obj: Optional[discord.Message] = None 

    try:
        channel = await bot.fetch_channel(payload.channel_id)
        if not isinstance(channel, discord.abc.Messageable): 
            return
        message_obj = await channel.fetch_message(payload.message_id)
    except (discord.NotFound, discord.Forbidden): 
        return 
    
    if message_obj is None: 
        return
    if message_obj.author.id != bot.user.id: 
        return 

    can_delete = False 
    if isinstance(channel, (discord.TextChannel, discord.Thread)) and channel.guild:
        try:
            member = await channel.guild.fetch_member(payload.user_id) 
            if member and member.guild_permissions.manage_messages: 
                can_delete = True
            elif message_obj.interaction and message_obj.interaction.user.id == payload.user_id: 
                can_delete = True
            elif message_obj.reference and message_obj.reference.message_id:
                original_message = await channel.fetch_message(message_obj.reference.message_id)
                if original_message.author.id == payload.user_id:
                    can_delete = True
        except discord.HTTPException: 
            pass 

    elif isinstance(channel, discord.DMChannel): 
        can_delete = True
    
    if not can_delete:
        if message_obj.interaction and message_obj.interaction.user.id == payload.user_id: 
            can_delete = True
        elif message_obj.reference and message_obj.reference.message_id and channel: 
            try: 
                original_message = await channel.fetch_message(message_obj.reference.message_id)
                if original_message.author.id == payload.user_id: 
                    can_delete = True
            except discord.NotFound: 
                pass 

    if can_delete:
        try: 
            await message_obj.delete()
            logger.info(f"Message {message_obj.id} deleted by reaction from user {payload.user_id}.")
        except Exception as e: 
            logger.error(f"Failed to delete message {message_obj.id} by reaction: {e}")

@bot.event
async def on_ready():
    global WHISPER_MODEL
    if not bot.user: logger.critical("Bot user not available on_ready."); return
    try: logger.info("Loading Whisper model..."); WHISPER_MODEL = whisper.load_model("base"); logger.info("Whisper model loaded.")
    except Exception as e: logger.critical(f"Failed to load Whisper model: {e}", exc_info=True)
    logger.info(f'{bot.user.name} connected! ID: {bot.user.id}')
    logger.info(f"discord.py: {discord.__version__}")
    logger.info(f"Allowed Channels: {config.ALLOWED_CHANNEL_IDS or 'All'}")
    logger.info(f"Allowed Roles: {config.ALLOWED_ROLE_IDS or 'None'}")
    logger.info(f"User Global Context: {'Set' if config.USER_PROVIDED_CONTEXT else 'Not set'}")
    logger.info(f"Max Image Bytes: {config.MAX_IMAGE_BYTES_FOR_PROMPT}, Max Scraped Text: {config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT}")
    logger.info(f"Stream Throttle: {config.STREAM_EDIT_THROTTLE_SECONDS}s, RAG Sentences: {config.RAG_NUM_DISTILLED_SENTENCES_TO_FETCH}")
    try: synced = await bot.tree.sync(); logger.info(f"Synced {len(synced)} slash commands.")
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
        else: 
            error_message = f"Command '{command_name}' failed: {str(original_error)[:500]}"
    elif isinstance(error, app_commands.CommandNotFound): 
        error_message = "Command not found. This is unexpected."
    elif isinstance(error, app_commands.MissingPermissions): 
        error_message = f"You lack permissions: {', '.join(error.missing_permissions)}"
    elif isinstance(error, app_commands.BotMissingPermissions): 
        error_message = f"I lack permissions: {', '.join(error.missing_permissions)}"
    elif isinstance(error, app_commands.CheckFailure): 
        error_message = "You do not meet the requirements to use this command."
    elif isinstance(error, app_commands.CommandOnCooldown): 
        error_message = f"This command is on cooldown. Try again in {error.retry_after:.2f} seconds."
    elif isinstance(error, app_commands.TransformerError): 
        error_message = f"Invalid argument: {error.value}. Expected type: {error.type}."
    
    if original_error_is_unknown_interaction: 
        return 

    try:
        if interaction.response.is_done(): 
            await interaction.followup.send(error_message, ephemeral=True)
        else: 
            await interaction.response.send_message(error_message, ephemeral=True)
    except discord.errors.HTTPException as ehttp: 
        if ehttp.code == 40060: 
            logger.warning(f"Error handler: Interaction for '{command_name}' already acknowledged. Trying followup. OrigErr: {error}")
            try: 
                await interaction.followup.send(error_message, ephemeral=True) 
            except Exception as e_followup: 
                logger.error(f"Error handler: Failed followup for '{command_name}': {e_followup}")
        else: 
            logger.error(f"Error handler: HTTPException for '{command_name}': {ehttp}. OrigErr: {error}")
    except discord.errors.NotFound: 
        logger.error(f"Error handler: Interaction not found for '{command_name}'. OrigErr: {error}")
    except Exception as e_generic: 
        logger.error(f"Error handler: Generic error for '{command_name}': {e_generic}. OrigErr: {error}")

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound): pass 
    elif isinstance(error, commands.MissingRequiredArgument): await ctx.reply(f"Missing argument for !{ctx.command.name if ctx.command else 'cmd'}: {error.param.name}.", silent=True)
    elif isinstance(error, commands.BadArgument): await ctx.reply(f"Invalid argument for !{ctx.command.name if ctx.command else 'cmd'}.", silent=True)
    elif isinstance(error, commands.CheckFailure): await ctx.reply("No permission for that prefix command.", silent=True)
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Error invoking prefix cmd !{ctx.command.name if ctx.command else 'cmd'}: {error.original}", exc_info=error.original)
        await ctx.reply(f"Error with !{ctx.command.name if ctx.command else 'cmd'}: {error.original}", silent=True)
    else: logger.error(f"Unhandled prefix command error: {error}", exc_info=True)

if __name__ == "__main__":
    if not config.DISCORD_BOT_TOKEN: logger.critical("DISCORD_BOT_TOKEN not set. Bot cannot start.")
    elif not chroma_client or not chat_history_collection or not distilled_chat_summary_collection: 
        logger.critical("One or more ChromaDB collections failed to initialize. Bot cannot start with full RAG capabilities.")
        try: bot.run(config.DISCORD_BOT_TOKEN, log_handler=None) 
        except discord.LoginFailure: logger.critical("Failed to log in with Discord token.")
        except Exception as e: logger.critical(f"Unexpected error during bot startup: {e}", exc_info=True)
    elif WHISPER_MODEL is None and "transcribe_audio_file" in globals(): 
         logger.critical("Whisper model failed to load. Audio transcription will fail.")
         try: bot.run(config.DISCORD_BOT_TOKEN, log_handler=None) 
         except discord.LoginFailure: logger.critical("Failed to log in with Discord token.")
         except Exception as e: logger.critical(f"Unexpected error during bot startup: {e}", exc_info=True)
    else:
        try: bot.run(config.DISCORD_BOT_TOKEN, log_handler=None) 
        except discord.LoginFailure: logger.critical("Failed to log in with Discord token.")
        except Exception as e: logger.critical(f"Unexpected error during bot startup: {e}", exc_info=True)
