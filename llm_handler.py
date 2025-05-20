import asyncio
import logging
from datetime import datetime
# import base64 # Removed, as it's not directly used here. Base64 operations are expected to happen outside.

# Assuming config.py contains llm_client, LLM model name, MAX_COMPLETION_TOKENS, etc.
from config import llm_client, LLM, MAX_COMPLETION_TOKENS # Removed EMBED_MAX_LENGTH, EMBED_COLOR

# Configure logger for this module
logger = logging.getLogger(__name__)

def get_system_prompt() -> list:
    # Uses config.LLM to determine the appropriate system prompt
    # This function was previously in lmcordx.py
    # Ensure config.LLM is accessible and correctly configured.
    # The datetime import is used here.
    if LLM in ["gpt-4-vision-preview", "mistral", "local"]: # Check against known model types if needed
        return [
            {
                "role": "system",
                "content": (
                    f"A chat between a curious user and a hyper-intelligent assistance entity named Sam. "
                    f"Sam gives helpful, detailed, and rational answers to the user's questions. "
                    f"Sam creates an initial reaction, considers the context, makes a plan of action and a rationale "
                    f"and evaluates it for logical failings before responding. Provocative Endings: Each reflection "
                    f"concludes with a question or thought provoking further exploration. "
                    f"Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
                ),
            }
        ]
    # Default system prompt or other specific prompts can be added here
    return [
        {
            "role": "system",
            "content": (
                f"A chat between a curious user and a hyper-intelligent assistance entity named Sam. "
                f"Sam gives helpful, detailed, and rational answers to the user's questions. "
                f"Sam creates an initial reaction, considers the context, makes a plan of action and a rationale "
                f"and evaluates it for logical failings before responding. Provocative Endings: Each reflection "
                f"concludes with a question or thought provoking further exploration. "
                f"Chat Date Timestamped: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n "
            ),
        }
    ]

async def generate_sarcastic_response(user_message: str) -> str:
    # Uses config.llm_client and config.LLM
    prompt = (
        "The user is fed up with extremist political views and wants to push back using sarcasm. "
        "You are here to make a single reply to mock these alt-right weirdos. "
        "The bot should respond to any political discussion or keyword with the most sarcastic, "
        "snarky, and troll-like comments possible. The goal is to mock and undermine these extremist views "
        "in a way that’s both biting and humorous.\n\n"
        f"User: {user_message}\nBot:"
    )
    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=LLM, # Use LLM from config
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6, # Could be a config variable
                max_tokens=4096, # This was hardcoded in original, consider config
                top_p=1.0, # Could be config
                frequency_penalty=0.0, # Could be config
                presence_penalty=0.0 # Could be config
            ),
            timeout=30 # Could be config
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate sarcastic response: {e}")
        return "Sorry, an error occurred while generating the sarcastic response."

async def generate_completion(prompt: str, max_tokens_override: int = None) -> str:
    # Uses config.llm_client, config.LLM, and config.MAX_COMPLETION_TOKENS
    # Allows overriding max_tokens if needed for specific calls
    current_max_tokens = max_tokens_override if max_tokens_override is not None else MAX_COMPLETION_TOKENS

    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=LLM, # Use LLM from config
                messages=[{"role": "user", "content": prompt}], # Adjust role if necessary based on usage
                temperature=0.6, # Could be config
                max_tokens=current_max_tokens, # Use from config or override
                top_p=1.0, # Could be config
                frequency_penalty=0.0, # Could be config
                presence_penalty=0.0 # Could be config
            ),
            timeout=30 # Could be config
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate completion: {e}")
        return "Sorry, an error occurred while generating the completion."

async def generate_reminder(prompt: str) -> str:
    # Uses config.llm_client and config.LLM
    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=LLM, # Use LLM from config
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6, # Could be config
                max_tokens=256,  # Specific for reminders, could be config
                top_p=1.0, # Could be config
                frequency_penalty=0.0, # Could be config
                presence_penalty=0.0 # Could be config
            ),
            timeout=30 # Could be config
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate reminder: {e}")
        return "Sorry, an error occurred while generating the reminder."

async def generate_ap_image_description(image_base64_data: str, text_content: str):
    # Logic extracted from the !ap command in lmcordx.py
    # Uses config.llm_client, config.LLM
    # image_base64_data should be the raw base64 string, without "data:image/jpeg;base64,"
    prompt_template = (
        "Describe this image in a very detailed and intricate way, "
        "as if you were describing it to a blind person for reasons of accessibility. "
        "Replace the main character or element in the description with a random celebrity "
        "or popular well-known character. Use the {{name}} variable for this. " 
        "Begin your response with \"AP Photo, {{name}}, \" followed by the description.\n " 
        f"User's additional context: {text_content if text_content else 'N/A'}"
    )
    
    reply_chain = [
        {"role": "system", "content": "You are an AP Photo caption writer with a humorous twist."}, 
        {"role": "user", "content": [
            {"type": "text", "text": prompt_template},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_data}"}}
        ]}
    ]
    
    logger.info(f"Generating AP image description. User text: {text_content[:50]}...")
    
    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=LLM,
                messages=reply_chain,
                max_tokens=1024, 
                stream=False 
            ),
            timeout=60 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate AP image description: {e}")
        return "Sorry, an error occurred while generating the AP image description."

async def generate_image_description(image_base64_data: str, text_content: str):
    # Uses config.llm_client, config.LLM
    system_prompt_content = (
        "A chat between a curious user and an intelligent assistance system. "
        "The system is equipped with a vision model that analyzes the image "
        "information that the user provides. The system gives helpful, detailed, "
        "and rational answers.\n</s> " + 
        f"Today's date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n " +
        "</s>......"
    )
    
    user_prompt_text = (
        "Base Instruction: \"Describe the image in a very detailed and intricate way, "
        "as if you were describing it to a blind person for accessibility. "
        "Begin your response with: \"'Image Description':, \". "
        "Extended Instruction: \"Below is a user comment or request. Write a response "
        "that appropriately completes the request.\". " +
        f"User's prompt: {text_content if text_content else 'Describe the image.'}\n </s>......" 
    )

    reply_chain = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_data}"}}
        ]}
    ]
    
    logger.info(f"Generating general image description. User text: {text_content[:50]}...")

    try:
        response = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model=LLM,
                messages=reply_chain,
                max_tokens=1024, 
                stream=False 
            ),
            timeout=60 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate general image description: {e}")
        return "Sorry, an error occurred while generating the image description."

async def generate_image_description_stream(image_base64_data: str, text_content: str, system_prompt_override: str = None):
    system_prompt = system_prompt_override
    if not system_prompt: 
        system_prompt = (
            "A chat between a curious user and an intelligent assistance system. "
            "The system is equipped with a vision model that analyzes the image "
            "information that the user provides. The system gives helpful, detailed, "
            "and rational answers.\n</s> " +
            f"Today's date: {datetime.now().strftime('%B %d %Y %H:%M:%S.%f')}\n " +
            "</s>......"
        )
    
    user_prompt = (
        "Base Instruction: \"Describe the image in a very detailed and intricate way, "
        "as if you were describing it to a blind person for accessibility. "
        # "Begin your response with: \"'Image Description':, \". " # Removed for more natural streaming start
        "Extended Instruction: \"Below is a user comment or request. Write a response "
        "that appropriately completes the request.\". " +
        f"User's prompt: {text_content if text_content else 'Describe the image.'}\n </s>......"
    )

    reply_chain = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_data}"}}
        ]}
    ]
    
    logger.info(f"Generating image description (stream). User text: {text_content[:50]}...")
    
    try:
        stream = await llm_client.chat.completions.create(
            model=LLM,
            messages=reply_chain,
            max_tokens=1024, 
            stream=True,
        )
        return stream 
    except Exception as e:
        logger.error(f"Failed to initiate stream for image description: {e}")
        raise

async def generate_chat_completion_stream(messages: list):
    """
    Generic function to stream chat completions.
    'messages' should be the complete list of message dicts for the LLM.
    """
    logger.info(f"Generating chat completion stream. Message count: {len(messages)}")
    try:
        stream = await llm_client.chat.completions.create(
            model=LLM, 
            messages=messages, 
            max_tokens=MAX_COMPLETION_TOKENS, 
            stream=True,
        )
        return stream
    except Exception as e:
        logger.error(f"Failed to initiate stream for chat completion: {e}")
        raise
