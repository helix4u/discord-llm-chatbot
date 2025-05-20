# LLMCord-X: Intelligent Discord Bot with Search, Scrape, Speech, and X/Twitter Integration

LLMCord-X is a versatile Discord bot designed to bring powerful AI and web capabilities to your Discord server. It integrates with local Large Language Models (LLMs), performs web searches via Searx, scrapes websites and YouTube transcripts, interacts with X/Twitter, generates speech from text (TTS), and understands voice messages.

Weird personal side project fork. lmcordspeakandsearch.py

SearxNG integration, Whisper integration, reminders, key-phrase enabled voice reminders, key-phrase enabled voice search, URL/webpage summarization, youtube transcript summarization, multimodal image description, junk like that. A place to try some junk out.
regular bugs. feel free to clean up.
requirements are not up to date

## Features

*   **Large Language Model (LLM) Integration:** Connects to a local LLM (e.g., via LM Studio, Ollama) to understand and generate human-like text for conversations, summarizations, and more.
*   **Web Search:** Queries a local Searx/SearxNG instance to fetch search results from the internet.
*   **Web Scraping:**
    *   Extracts text content from general webpages using Playwright and BeautifulSoup.
    *   Fetches transcripts from YouTube videos.
*   **X/Twitter Integration:** Scrapes the latest tweets from a specified X/Twitter user handle using Playwright.
*   **Text-to-Speech (TTS):** Converts bot text responses into audible speech using a local TTS server (e.g., Kokoro-FastAPI), and can send them as audio files in Discord.
*   **Voice Message Transcription:** Transcribes user voice messages (audio attachments) into text using Whisper.
*   **Image Understanding:**
    *   Can process and describe images attached to messages using a vision-capable LLM.
    *   Special `!ap` command to generate AP-style photo descriptions with a humorous twist.
*   **Interactive Chat:**
    *   Responds to direct mentions and configured channels.
    *   Maintains message history for contextual conversations.
    *   Streams responses by progressively editing messages to show ongoing generation.
    *   Allows users to delete bot messages via '❌' reaction.
*   **Rich Command Set:** Offers a variety of commands for search, summarization, reminders, and more.

## Commands

The bot primarily responds to mentions, but also supports the following commands:

*   **`!search <query>`**: Performs a Searx search for the given query, then uses the LLM to summarize the findings. The summary is streamed into an embed, and a TTS audio file of the summary is provided.
*   **`!sns <url_or_query>`**: (Search and Summarize)
    *   If a URL is provided, scrapes the webpage and provides an LLM-generated summary.
    *   If a query is provided, performs a Searx search, scrapes the top results, and then summarizes them.
    *   Streams the summary and provides TTS audio.
*   **`!roast <url>`**: Scrapes the content of the given URL and uses the LLM to generate a "comedy routine" based on it. Streams the output and provides TTS audio.
*   **`!gettweets <twitter_handle> [limit]`**: Scrapes the latest tweets (default 10, or specified limit) from the given X/Twitter user. It then provides both the raw tweets and an LLM-generated summary, along with TTS for the summary. (Requires `lmcordx.py`)
*   **`!remindme <time_duration> <message>`**: Sets a reminder.
    *   Example: `!remindme 1h30m Check on the experiment`
    *   The bot will send a message after the specified duration with an LLM-generated reminder prompt and TTS audio.
*   **`!pol <message>`**: Generates a sarcastic, LLM-powered response, typically aimed at mocking extremist political views.
*   **`!ap <image_attachment>`**: When an image is attached with this command, the bot uses a vision model to describe it in the style of an "Associated Press" photo caption, often replacing the main subject with a randomly chosen celebrity for humorous effect.
*   **`!toggle_search`**: Toggles the automatic Searx search functionality for general messages on or off. (Note: Behavior might vary slightly between scripts).
*   **`!clear_history`**: Clears the bot's message history for the current channel, allowing for a fresh conversation context.
*   **`!show_history_size`**: Displays the number of messages currently stored in the history for that channel.

**Voice Commands (via transcribed audio messages):**
The bot transcribes audio messages sent as attachments. If the transcription matches certain patterns, it can trigger commands:
*   *"Remind me in [time] to [message]"*: Sets a reminder.
*   *"Search for [query]"*: Performs a search and summarizes results, similar to the `!search` command, including TTS.

**General Interaction:**
*   **Mentioning the bot (`@BotName ...`)**: Initiates a conversation or asks a question. The bot uses message history for context.
*   **Image Attachments (without `!ap`)**: If an image is attached to a message (especially when mentioning the bot), the vision model will describe it or use it as context for the query.
*   **URL Pasting**: Pasting URLs in a message (especially when mentioning the bot) will often cause the bot to scrape the URL and use its content for context or summarization.
*   **Message Deletion**: Reacting to a bot's message with '❌' will delete it.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Python Environment:**
    *   Ensure you have Python 3.8+ installed.
    *   It's recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    *   Install required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
    *   **Playwright Browsers:** Playwright requires browser binaries. Install them by running:
        ```bash
        playwright install
        ```
        This is especially important for the X/Twitter scraping feature in `lmcordx.py`.

4.  **Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file with your specific configurations:
        *   `DISCORD_BOT_TOKEN`: **Required.** Your Discord bot token.
        *   `LOCAL_SERVER_URL`: **Required.** The URL of your local LLM server (e.g., LM Studio, Ollama compatible with OpenAI API). Default: `http://localhost:1234/v1`.
        *   `LLM`: The model identifier to be used by the LLM client (e.g., `gpt-4-vision-preview`, `local-model`, `mistral`). This should match a model available on your `LOCAL_SERVER_URL`.
        *   `TTS_API_URL`: **Required for TTS.** The URL for your Text-to-Speech server (e.g., Kokoro-FastAPI). Default: `http://localhost:8880/v1/audio/speech`.
        *   `TTS_VOICE`: The voice model to be used by the TTS server. Default: `af_sky+af+af_nicole`.
        *   `ALLOWED_CHANNEL_IDS`: Comma-separated list of Discord channel IDs where the bot is allowed to operate without a direct mention.
        *   `ALLOWED_ROLE_IDS`: Comma-separated list of Discord role IDs that are allowed to interact with the bot.
        *   `MAX_IMAGES`: Maximum number of images to process per message. Default: `0` (no limit, though practical limits may apply).
        *   `MAX_MESSAGES`: Maximum number of messages to keep in history for context. Default: `10`.
        *   `SEARX_URL`: The URL of your Searx/SearxNG instance (e.g., `http://192.168.1.3:9092`). This is needed for the `!search` and `!sns` (with query) commands. The example preferences string in the code is very long and specific to a particular Searx setup; you might need to adjust it if your Searx instance has different settings.

5.  **External Services:**
    *   **LLM Server:** You need a running LLM server that's compatible with the OpenAI API format (e.g., LM Studio, Ollama with an OpenAI-compatible serving endpoint). Ensure the model specified in `LLM` is loaded and accessible.
    *   **TTS Server:** For text-to-speech functionality, you need a running TTS server like [Kokoro-FastAPI](https://github.com/AUTOMATIC1111/kokoro-fastapi) (or any other TTS server that matches the API structure used in the scripts, specifically the `/v1/audio/speech` endpoint).
    *   **Searx/SearxNG Instance:** For web search capabilities (`!search`, `!sns <query>`), you need a running Searx or SearxNG instance accessible at the URL specified in `SEARX_URL`.
    *   **(For `lmcordx.py` Twitter/X Scraping):** The `lmcordx.py` script uses Playwright with a persistent context (`user_data_dir=".pw-profile"`). This means you might need to log in to X/Twitter manually in the browser window that Playwright opens the first time you run `!gettweets` to establish a session within that profile directory. Subsequent runs should then use the saved session. Make sure the machine running the bot is not IP-blocked by X/Twitter.

6.  **Whisper Model:**
    *   The first time audio transcription is used, Whisper will download its model files (e.g., "tiny"). Ensure you have internet access for this.

## Usage

1.  **Activate Virtual Environment:**
    ```bash
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Run the Bot:**
    *   The project contains a few main scripts:
        *   `llmcord_search.py`: Core LLM and search functionalities.
        *   `lmcordspeakandsearch.py`: Adds Text-to-Speech (TTS) to `llmcord_search.py`.
        *   `lmcordx.py`: The most comprehensive version, including TTS and X/Twitter scraping.
    *   It's generally recommended to run `lmcordx.py` for the full feature set:
        ```bash
        python lmcordx.py
        ```
    *   If you prefer a version without TTS or X/Twitter features, you can run `llmcord_search.py` or `lmcordspeakandsearch.py`.

3.  **Interacting with the Bot in Discord:**
    *   **Mentions:** Mention the bot (e.g., `@YourBotName How does photosynthesis work?`) to get a response.
    *   **Commands:** Use the commands listed in the "Commands" section (e.g., `!search "latest AI news"`).
    *   **Allowed Channels:** If `ALLOWED_CHANNEL_IDS` is set, the bot will respond to messages in those channels even without a direct mention.
    *   **Voice Messages:** Simply attach an audio file (e.g., a voice recording from your phone) to a message. The bot will transcribe it and respond accordingly. If the transcription matches a voice command pattern (like "search for..."), it will execute it.
    *   **Image Interactions:**
        *   Attach an image to a message when you mention the bot to have it described or used as context.
        *   Use the `!ap` command with an image attachment for the "AP Photo" style description.

4.  **Stopping the Bot:**
    *   Press `Ctrl+C` in the terminal where the bot script is running.

## Dependencies

### Python Packages

The core Python dependencies are listed in `requirements.txt`. Key libraries include:

*   **`discord.py`**: For Discord API interaction.
*   **`openai`**: OpenAI API client library (used for interfacing with local LLMs).
*   **`aiohttp`**: Asynchronous HTTP client/server.
*   **`python-dotenv`**: For managing environment variables.
*   **`beautifulsoup4`**: For HTML parsing (web scraping).
*   **`playwright`**: For advanced web scraping and browser automation (used for general scraping and X/Twitter).
*   **`playwright-stealth`**: To help Playwright avoid detection.
*   **`youtube-transcript-api`**: For fetching YouTube video transcripts.
*   **`yt-dlp`**: For downloading YouTube audio if transcriptions are unavailable (used as a fallback in `llmcord_search.py`).
*   **`openai-whisper`**: For audio transcription.
*   **`torch`**: Required by Whisper.
*   **`pydub`**: For audio manipulation (specifically for fixing MP3 length metadata for TTS).

Install all Python dependencies using:
```bash
pip install -r requirements.txt
```
And ensure Playwright's browser binaries are installed:
```bash
playwright install
```

### External Services & Software

*   **Local LLM Server:** An OpenAI API-compatible server (e.g., LM Studio, Ollama with an appropriate serving configuration) running an LLM.
*   **Local TTS Server:** A Text-to-Speech server compatible with the expected API (e.g., Kokoro-FastAPI at `http://localhost:8880/v1/audio/speech`).
*   **Searx/SearxNG Instance:** A running instance of the Searx or SearxNG metasearch engine.
*   **FFmpeg (Optional but Recommended):** While `yt-dlp` can download audio, having FFmpeg available can sometimes improve audio extraction or conversion if `llmcord_search.py`'s YouTube audio fallback or other audio processing is needed. `pydub` also relies on FFmpeg (or libav) for MP3 processing.

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for more details.

## Troubleshooting

*   **Bot Not Responding:**
    *   Check that the `DISCORD_BOT_TOKEN` is correct in your `.env` file.
    *   Ensure the bot script is running without errors in your terminal.
    *   Verify the bot has the necessary permissions in your Discord server/channel.
    *   Check `ALLOWED_CHANNEL_IDS` and `ALLOWED_ROLE_IDS` in your `.env` file if the bot is not responding in specific channels or to specific users.
*   **LLM Errors / No LLM Response:**
    *   Ensure your local LLM server (e.g., LM Studio) is running and the correct model (matching `LLM` in `.env`) is loaded and accessible at `LOCAL_SERVER_URL`.
    *   Check the LLM server logs for any errors.
*   **TTS Not Working:**
    *   Verify your TTS server (e.g., Kokoro-FastAPI) is running and accessible at `TTS_API_URL`.
    *   Check the TTS server logs.
    *   Ensure `ffmpeg` is installed and accessible if `pydub` requires it for MP3 processing.
*   **Search Not Working (`!search`, `!sns <query>`):**
    *   Ensure your Searx/SearxNG instance is running and accessible at `SEARX_URL`.
    *   The preferences string for Searx in the code is very specific. If you encounter issues, you might need to generate your own preferences string from your Searx instance and update it in the script(s) (`llmcord_search.py`, `lmcordspeakandsearch.py`, `lmcordx.py` in the `query_searx` function).
*   **Web Scraping Failures (`!sns <url>`, `!roast <url>`, `!gettweets`):**
    *   Websites change frequently. Scrapers may break.
    *   For `!gettweets`, ensure you have run `playwright install` and that you may need to manually log in to X/Twitter via the Playwright browser the first time to establish a session in the `.pw-profile` directory.
    *   Ensure the machine running the bot isn't IP-blocked or facing CAPTCHAs from the target sites.
    *   Playwright's headless mode is set to `False` in `lmcordx.py` for `scrape_latest_tweets` and `scrape_website`. This is for debugging; you might want to set it to `True` for production, but it could increase detection by websites.
*   **Audio Transcription Not Working:**
    *   Whisper will download model files on its first run. Ensure internet connectivity.
    *   Check for any errors in the console related to Whisper or PyTorch.
*   **`playwright._impl._api_types.TimeoutError` during scraping (especially `!gettweets`):**
    *   This often means the page took too long to load or a specific element wasn't found.
    *   Could be due to network issues, site changes, or anti-scraping measures.
    *   The `scrape_latest_tweets` function in `lmcordx.py` has timeouts and retry logic, but persistent errors might require code adjustments.
