# main.py
import os
import sqlite3
import logging
import re
from datetime import datetime
from functools import wraps
import json
import asyncio


import google.generativeai as genai
try:
    from google.ai import generativelanguage as glm
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("CRITICAL: Failed to import google.ai.generativelanguage (glm). Ensure 'google-ai-generativelanguage' is installed.")
    glm = None

# Imports for Official YouTube Data API & OAuth2 (Placeholder usage in this script)
_google_oauth_libs_imported_successfully = False
build = None
google_oauth2_credentials = None
GoogleAuthRequest = None
GoogleAuthRefreshError = None
HttpError = None

try:
    from googleapiclient.discovery import build as imported_build
    from googleapiclient.errors import HttpError as ImportedHttpError
    import google.oauth2.credentials as imported_google_oauth2_credentials
    from google.auth.transport.requests import Request as Imported_GoogleAuthRequest
    from google.auth.exceptions import RefreshError as Imported_GoogleAuthRefreshError

    build = imported_build
    HttpError = ImportedHttpError
    google_oauth2_credentials = imported_google_oauth2_credentials
    GoogleAuthRequest = Imported_GoogleAuthRequest
    GoogleAuthRefreshError = Imported_GoogleAuthRefreshError
    _google_oauth_libs_imported_successfully = True
except ImportError:
    print(
        "WARNING: Failed to import one or more Google API/Auth libraries (googleapiclient, google-auth, google-auth-oauthlib). "
        "Please ensure these are installed in your virtual environment if you plan to develop that feature: "
        "pip install google-api-python-client google-auth google-auth-oauthlib"
    )

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler,
)

# --- I. Critical Foundations & Security ---

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info("Logging configured at script start.")

load_dotenv()
logger.info("dotenv loaded.")

AI_COUNCIL_TELEGRAM_BOT_TOKEN = os.getenv("AI_COUNCIL_TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AUTHORIZED_USER_ID = int(os.getenv("AUTHORIZED_USER_ID", "0"))

GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION")

YOUTUBE_OAUTH_CLIENT_SECRETS_PATH = os.getenv("YOUTUBE_OAUTH_CLIENT_SECRETS_PATH")
YOUTUBE_OAUTH_REFRESH_TOKEN = os.getenv("YOUTUBE_OAUTH_REFRESH_TOKEN")

if not all([AI_COUNCIL_TELEGRAM_BOT_TOKEN, AUTHORIZED_USER_ID != 0]):
    logger.critical("CRITICAL ERROR: Missing AI_COUNCIL_TELEGRAM_BOT_TOKEN or AUTHORIZED_USER_ID is not set correctly. Exiting.")
    exit()

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Successfully configured google-generativeai with GEMINI_API_KEY.")
    except Exception as e:
        logger.error(f"Failed to configure google-generativeai with GEMINI_API_KEY: {e}.")
else:
    logger.warning("GEMINI_API_KEY not set. Direct Gemini API and Transcript+Gemini summarization methods will be impacted.")

if not all([GOOGLE_PROJECT_ID, GOOGLE_LOCATION]):
    logger.warning("GOOGLE_PROJECT_ID or GOOGLE_LOCATION not set. Vertex AI summarization method may fail.")
logger.info("Environment variable checks complete.")

# --- Models & Constants ---
DB_NAME = "telegram_bot.db"
REQ_TYPE_GENERAL = "general_query"
REQ_TYPE_YOUTUBE_GEMINI_API = "youtube_summary_gemini_api"
REQ_TYPE_YOUTUBE_VERTEX_AI = "youtube_summary_vertex_ai"
REQ_TYPE_YOUTUBE_SUPER_TRANSCRIPT = "youtube_super_transcript"
REQ_TYPE_CUSTOM_PROMPT_PLACEHOLDER = "custom_prompt_placeholder" ### NEW ###
REQ_TYPE_REPLY_TO_SUMMARY = "reply_to_summary"
STATUS_PENDING = "pending"
STATUS_SUCCESS = "success"
STATUS_ERROR_LLM = "error_llm"
STATUS_ERROR_YOUTUBE = "error_youtube"
STATUS_ERROR_GENERAL = "error_general"
STATUS_PLACEHOLDER = "placeholder_acknowledged"

CHAT_MODEL_NAME = os.getenv("GEMINI_MODEL_CHAT", "gemini-1.5-flash")
GEMINI_API_VIDEO_MODEL = os.getenv("GEMINI_MODEL_VIDEO_API", "gemini-1.5-flash")
VERTEX_AI_VIDEO_MODEL = os.getenv("GEMINI_MODEL_VIDEO_VERTEX", "gemini-1.5-flash")
TEXT_SUMMARY_MODEL = os.getenv("GEMINI_MODEL_TEXT_SUMMARY", "gemini-1.5-flash")

# ### MODIFIED ###: Added new state for the submenu
CHOOSING_SUMMARY_METHOD, CHOOSING_LANGUAGE, CHOOSING_CUSTOM_PROMPT, PROCESSING_SUMMARY = range(4)

LANGUAGE_NAME_MAP = {
    "ru": "русском",
    "en": "английском"
}

# --- Helper function to load prompts ---
def load_prompt(filename: str) -> str | None:
    """Loads a prompt from a file in the 'prompts' directory."""
    try:
        # Get the absolute path of the directory where the script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, 'prompts', filename)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"CRITICAL: Prompt file not found at {prompt_path}")
        return None

# --- Load Prompts from Files ---
logger.info("Loading prompts from files...")
DETAILED_VIDEO_SUMMARY_PROMPT_TEMPLATE = load_prompt("detailed_video_summary.txt")
SUPER_TRANSCRIPT_PROMPT_TEMPLATE = load_prompt("super_transcript.txt")

# Critical check to ensure prompts were loaded
if not all([DETAILED_VIDEO_SUMMARY_PROMPT_TEMPLATE, SUPER_TRANSCRIPT_PROMPT_TEMPLATE]):
    logger.critical("One or more essential prompt files could not be loaded from the 'prompts' folder. Exiting.")
    exit()

logger.info("All essential prompts loaded successfully.")


# --- Authorization Decorator & DB Functions ---
def authorized(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if update.effective_user.id != AUTHORIZED_USER_ID:
            logger.warning(f"Unauthorized access from user_id: {update.effective_user.id}")
            await update.message.reply_text("Sorry, you do not have access to this bot.")
            if isinstance(context, ConversationHandler):
                return ConversationHandler.END
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

def init_db():
    logger.info("Attempting to initialize database...")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    reply_to_message_id INTEGER,
                    request_timestamp TEXT NOT NULL,
                    request_text TEXT,
                    response_text TEXT,
                    request_type TEXT,
                    youtube_url TEXT,
                    processing_status TEXT,
                    error_message TEXT
                )
            ''')
        logger.info(f"Database '{DB_NAME}' initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing DB: {e}", exc_info=True)
        raise

def add_interaction(user_id, chat_id, message_id, request_text, request_type, reply_to_id=None) -> int | None:
    timestamp = datetime.now().isoformat()
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions (user_id, chat_id, message_id, reply_to_message_id, request_timestamp, request_text, request_type, processing_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, chat_id, message_id, reply_to_id, timestamp, request_text, request_type, STATUS_PENDING))
            conn.commit()
            return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Error adding request to DB: {e}", exc_info=True)
        return None

def update_interaction(interaction_id, response_text=None, status=STATUS_SUCCESS, error_message=None, youtube_url=None):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            query = "UPDATE interactions SET response_text = ?, processing_status = ?, error_message = ?"
            params = [response_text, status, error_message]
            if youtube_url:
                query += ", youtube_url = ?"
                params.append(youtube_url)
            query += " WHERE id = ?"
            params.append(interaction_id)
            cursor.execute(query, tuple(params))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error updating record (ID: {interaction_id}) in DB: {e}", exc_info=True)

def get_conversation_history(user_id, limit=5):
    history = []
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT request_text, response_text FROM interactions
                WHERE user_id = ? AND processing_status = ? AND request_type NOT IN (?, ?, ?, ?, ?)
                ORDER BY id DESC LIMIT ?
            ''', (user_id, STATUS_SUCCESS,
                  REQ_TYPE_YOUTUBE_GEMINI_API, REQ_TYPE_YOUTUBE_VERTEX_AI,
                  REQ_TYPE_YOUTUBE_SUPER_TRANSCRIPT, REQ_TYPE_CUSTOM_PROMPT_PLACEHOLDER, REQ_TYPE_REPLY_TO_SUMMARY,
                  limit))
            rows = cursor.fetchall()
            for row in reversed(rows):
                if row['request_text']:
                    history.append({'role': 'user', 'parts': [row['request_text']]})
                if row['response_text']:
                    history.append({'role': 'model', 'parts': [row['response_text']]})
        return history
    except sqlite3.Error as e:
        logger.error(f"Error getting conversation history for user_id {user_id}: {e}", exc_info=True)
        return []

def get_summary_text_from_bot_message(bot_message_id: int, chat_id: int, bot_id: int) -> str | None:
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT request_text FROM interactions
                WHERE message_id = ? AND chat_id = ? AND user_id = ? AND request_type IN (?, ?, ?, ?)
            ''', (bot_message_id, chat_id, bot_id,
                  REQ_TYPE_YOUTUBE_GEMINI_API, REQ_TYPE_YOUTUBE_VERTEX_AI,
                  REQ_TYPE_YOUTUBE_SUPER_TRANSCRIPT, REQ_TYPE_CUSTOM_PROMPT_PLACEHOLDER))
            result = cursor.fetchone()
            return result[0] if result else None
    except sqlite3.Error as e:
        logger.error(f"Error getting summary text from bot message: {e}", exc_info=True)
        return None
logger.info("DB functions defined.")


# --- AI Integration & Features ---
async def generate_chat_response(prompt, history=None):
    try:
        model = genai.GenerativeModel(CHAT_MODEL_NAME)
        chat = model.start_chat(history=history or [])
        response = await chat.send_message_async(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini for chat: {e}", exc_info=True)
        raise

def extract_video_id(url):
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

async def summarize_youtube_via_gemini_api(youtube_link: str, video_id: str, prompt_text: str):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be set for this summarization method.")
    if not glm:
        raise ImportError("google.ai.generativelanguage (glm) module is not available.")
    try:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        model = genai.GenerativeModel(GEMINI_API_VIDEO_MODEL)
        logger.info(f"Using Direct Gemini API model for video: {GEMINI_API_VIDEO_MODEL}")
        video_part = glm.Part(file_data=glm.FileData(mime_type="video/mp4", file_uri=youtube_link))

        contents = [video_part, glm.Part(text=prompt_text)]
        logger.info(f"Sending content to Direct Gemini API for summarization. URI: {youtube_link}")

        response = await model.generate_content_async(
            contents,
            safety_settings=safety_settings
        )

        try:
            summary = response.text
        except ValueError:
            logger.warning(f"response.text failed. Raw response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason == 'SAFETY':
                if response.candidates and response.candidates[0].finish_reason == 4:
                     raise ValueError("Content blocked by API safety filters due to potential recitation of copyrighted material. Please try a different video.")
                raise ValueError("The API response was blocked by safety filters for reasons other than recitation.")
            raise ValueError("The API returned an invalid or empty response.")

        if not summary:
            raise ValueError("Failed to get summary: Empty content in response from Direct Gemini API.")
        return summary

    except Exception as e:
        logger.error(f"Error during YouTube summarization with Direct Gemini API: {e}", exc_info=True)
        raise e

async def summarize_youtube_via_vertex_ai(youtube_link: str, video_id: str, prompt_text: str):
    if not GOOGLE_PROJECT_ID or not GOOGLE_LOCATION:
        raise ValueError("GOOGLE_PROJECT_ID and GOOGLE_LOCATION must be set for Vertex AI.")
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS env var must be set for Vertex AI.")
    if not glm:
        raise ImportError("google.ai.generativelanguage (glm) module is not available.")
    try:
        model = genai.GenerativeModel(VERTEX_AI_VIDEO_MODEL)
        logger.info(f"Using Vertex AI model for video: {VERTEX_AI_VIDEO_MODEL}")
        video_part = glm.Part(file_data=glm.FileData(mime_type="video/youtube", file_uri=youtube_link))

        contents = [video_part, glm.Part(text=prompt_text)]
        logger.info(f"Sending content to Vertex AI for summarization. URI: {youtube_link}")
        generation_config = genai.types.GenerationConfig(temperature=0.7, top_p=0.95, max_output_tokens=8192)
        response = await model.generate_content_async(contents=contents, generation_config=generation_config)

        summary = ""
        try:
            summary = response.text
        except ValueError:
            logger.warning(f"response.text accessor failed. Checking candidates. Raw Vertex response: {response}")
            if response.candidates and response.candidates[0].content.parts:
                summary = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text)
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name
                raise ValueError(f"Content blocked by Vertex AI. Reason: {block_reason_msg}")
        if not summary.strip():
            raise ValueError("Failed to get summary from Vertex AI: Empty or no text content in response.")
        return summary
    except Exception as e:
        logger.error(f"Error during YouTube summarization with Vertex AI: {e}", exc_info=True)
        raise ValueError(f"Vertex AI error: {e}")

logger.info("AI and summarization functions defined.")


async def send_long_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, reply_to_message_id: int | None = None):
    MAX_LENGTH = 4096
    if len(text) <= MAX_LENGTH:
        message = await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_to_message_id=reply_to_message_id
        )
        return message
    parts = []
    while len(text) > 0:
        if len(text) > MAX_LENGTH:
            part = text[:MAX_LENGTH]
            last_newline = part.rfind('\n')
            if last_newline > 0:
                part = text[:last_newline]
                text = text[last_newline + 1:]
            else:
                part = text[:MAX_LENGTH]
                text = text[MAX_LENGTH:]
            parts.append(part)
        else:
            parts.append(text)
            break
    sent_message = None
    for i, part in enumerate(parts):
        part_text_with_indicator = f"[{i+1}/{len(parts)}]\n{part}"
        if i == 0:
            message = await context.bot.send_message(
                chat_id=chat_id,
                text=part_text_with_indicator,
                reply_to_message_id=reply_to_message_id
            )
            sent_message = message
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=part_text_with_indicator
            )
    return sent_message

# --- Telegram Handlers ---
@authorized
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(rf"Hi, {user.mention_html()}! I'm ready to work.")

### MODIFIED ###
# This helper function is now used in multiple places to build the main menu
async def _build_main_menu_keyboard(query: CallbackQueryHandler | None = None, update: Update | None = None):
    """Builds and sends/edits the main menu keyboard."""
    keyboard = [
        [InlineKeyboardButton("Gemini API (Video)", callback_data='summarize_gemini_api')],
        [InlineKeyboardButton("Vertex AI (Video)", callback_data='summarize_vertex_ai')],
        [InlineKeyboardButton("Super Transcript (File)", callback_data='summarize_super_transcript')],
        [InlineKeyboardButton("Custom Prompts >>", callback_data='custom_prompts_menu')],
        [InlineKeyboardButton("Cancel", callback_data='cancel_summary')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message_text = 'Choose summarization method for this YouTube video:'
    if query:
        await query.edit_message_text(message_text, reply_markup=reply_markup)
    elif update:
        await update.message.reply_text(message_text, reply_markup=reply_markup)

@authorized
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int | None:
    text = update.message.text
    video_id = extract_video_id(text)

    if video_id:
        logger.info(f"YouTube link detected: {text}")
        context.user_data['youtube_link'] = text
        context.user_data['video_id'] = video_id
        context.user_data['original_message_id'] = update.message.message_id
        
        await _build_main_menu_keyboard(update=update)
        return CHOOSING_SUMMARY_METHOD
    else:
        await _process_general_query(update, context)
        return ConversationHandler.END

@authorized
async def choose_summary_method_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    choice = query.data
    context.user_data['summary_method_choice'] = choice

    ### MODIFIED ###
    if choice == 'custom_prompts_menu':
        keyboard = [
            [InlineKeyboardButton("Magazine Article", callback_data='prompt_magazine')],
            [InlineKeyboardButton("Textbook Chapter", callback_data='prompt_textbook')],
            [InlineKeyboardButton("Learning Path", callback_data='prompt_learning_path')],
            [InlineKeyboardButton("Custom Script", callback_data='prompt_custom_script')],
            [InlineKeyboardButton("<< Back to Main Menu", callback_data='back_to_main_menu')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text="Choose a custom prompt format:", reply_markup=reply_markup)
        return CHOOSING_CUSTOM_PROMPT

    elif choice in ['summarize_gemini_api', 'summarize_vertex_ai', 'summarize_super_transcript']:
        if choice in ['summarize_gemini_api', 'summarize_super_transcript'] and not GEMINI_API_KEY:
            await query.edit_message_text(text="Error: GEMINI_API_KEY is not configured for this method.")
            context.user_data.clear()
            return ConversationHandler.END
        if choice == 'summarize_vertex_ai' and not (GOOGLE_PROJECT_ID and GOOGLE_LOCATION and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
            await query.edit_message_text(text="Error: Vertex AI is not fully configured for this method.")
            context.user_data.clear()
            return ConversationHandler.END

        language_keyboard = [
            [InlineKeyboardButton("Русский", callback_data='lang_ru')],
            [InlineKeyboardButton("English", callback_data='lang_en')],
            [InlineKeyboardButton("<< Back to Main Menu", callback_data='back_to_main_menu')],
        ]
        reply_markup = InlineKeyboardMarkup(language_keyboard)
        await query.edit_message_text(text="Choose summary language:", reply_markup=reply_markup)
        return CHOOSING_LANGUAGE

    elif choice == 'cancel_summary':
        await query.edit_message_text(text="Summarization cancelled.")
        context.user_data.clear()
        return ConversationHandler.END
    else:
        await query.edit_message_text(text="Invalid choice.")
        context.user_data.clear()
        return ConversationHandler.END

### NEW ###
@authorized
async def choose_custom_prompt_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles placeholder actions for the custom prompts submenu."""
    query = update.callback_query
    await query.answer()
    choice = query.data
    
    prompt_name_map = {
        'prompt_magazine': 'Magazine Article',
        'prompt_textbook': 'Textbook Chapter',
        'prompt_learning_path': 'Learning Path',
        'prompt_custom_script': 'Custom Script'
    }
    prompt_name = prompt_name_map.get(choice, 'Selected custom prompt')

    await query.edit_message_text(text=f"Acknowledged: '{prompt_name}' method.")
    placeholder_message = f"The '{prompt_name}' feature is not yet implemented. No action taken."
    original_message_id = context.user_data.get('original_message_id')
    youtube_link = context.user_data.get('youtube_link')

    if original_message_id and youtube_link:
        await context.bot.send_message(chat_id=query.message.chat_id, text=placeholder_message, reply_to_message_id=original_message_id)
        interaction_id = add_interaction(query.from_user.id, query.message.chat_id, original_message_id, youtube_link, REQ_TYPE_CUSTOM_PROMPT_PLACEHOLDER)
        if interaction_id:
            update_interaction(interaction_id, response_text=placeholder_message, status=STATUS_PLACEHOLDER, youtube_url=youtube_link)
    
    context.user_data.clear()
    return ConversationHandler.END

@authorized
async def choose_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    chosen_language_code = query.data.split('_')[1]

    youtube_link = context.user_data.get('youtube_link')
    video_id = context.user_data.get('video_id')
    original_message_id = context.user_data.get('original_message_id')
    summary_method_choice = context.user_data.get('summary_method_choice')

    if not all([youtube_link, video_id, original_message_id, summary_method_choice]):
        await query.edit_message_text(text="Error: Session data lost. Please try sending the link again.")
        context.user_data.clear()
        return ConversationHandler.END

    language_name_locative = LANGUAGE_NAME_MAP.get(chosen_language_code, "выбранном")
    
    prompt_text = ""
    request_type = ""
    processing_message = ""
    summarization_function_to_call = None

    if summary_method_choice == 'summarize_super_transcript':
        prompt_text = SUPER_TRANSCRIPT_PROMPT_TEMPLATE.format(language_name_locative=language_name_locative)
        request_type = REQ_TYPE_YOUTUBE_SUPER_TRANSCRIPT
        summarization_function_to_call = summarize_youtube_via_gemini_api
        processing_message = f"Generating Super Transcript file with Gemini API ({language_name_locative}). This may take a moment..."
    else:
        prompt_text = DETAILED_VIDEO_SUMMARY_PROMPT_TEMPLATE.format(language_name_locative=language_name_locative)
        if summary_method_choice == 'summarize_gemini_api':
            processing_message = f"Processing with Gemini API (Video) for {language_name_locative} summary..."
            summarization_function_to_call = summarize_youtube_via_gemini_api
            request_type = REQ_TYPE_YOUTUBE_GEMINI_API
        elif summary_method_choice == 'summarize_vertex_ai':
            processing_message = f"Processing with Vertex AI (Video) for {language_name_locative} summary..."
            summarization_function_to_call = summarize_youtube_via_vertex_ai
            request_type = REQ_TYPE_YOUTUBE_VERTEX_AI

    if not all([prompt_text, request_type, processing_message, summarization_function_to_call]):
        await query.edit_message_text(text="Error: Could not determine processing action. Invalid method stored.")
        context.user_data.clear()
        return ConversationHandler.END

    await query.edit_message_text(text=processing_message)
    interaction_id_user_request = add_interaction(query.from_user.id, query.message.chat_id, original_message_id, youtube_link, request_type)
    if interaction_id_user_request is None:
        await context.bot.send_message(chat_id=query.message.chat_id, text="Error: Could not save your request before processing.")
        context.user_data.clear()
        return ConversationHandler.END

    try:
        ai_content = await summarization_function_to_call(youtube_link, video_id, prompt_text)

        if summary_method_choice == 'summarize_super_transcript':
            filename = f"{video_id}_super_transcript.md"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(ai_content)
                
                with open(filename, 'rb') as f:
                    await context.bot.send_document(
                        chat_id=query.message.chat_id,
                        document=f,
                        filename=filename,
                        caption=f"Here is the Super Transcript for '{video_id}'.",
                        reply_to_message_id=original_message_id
                    )
                update_interaction(interaction_id_user_request, response_text=f"Successfully generated and sent {filename}", status=STATUS_SUCCESS, youtube_url=youtube_link)
            finally:
                if os.path.exists(filename):
                    os.remove(filename)
                    logger.info(f"Cleaned up temporary file: {filename}")
        else:
            bot_summary_message = await send_long_message(context, chat_id=query.message.chat_id, text=ai_content, reply_to_message_id=original_message_id)
            update_interaction(interaction_id_user_request, response_text=ai_content, status=STATUS_SUCCESS, youtube_url=youtube_link)
            if bot_summary_message:
                 add_interaction(context.bot.id, query.message.chat_id, bot_summary_message.message_id, ai_content, request_type)

    except Exception as e:
        logger.error(f"Error during YouTube processing ({summary_method_choice}, lang: {chosen_language_code}): {e}", exc_info=True)
        await context.bot.send_message(chat_id=query.message.chat_id, text=f"Could not complete request. Reason: {e}", reply_to_message_id=original_message_id)
        if interaction_id_user_request:
            update_interaction(interaction_id_user_request, status=STATUS_ERROR_YOUTUBE, error_message=str(e))

    context.user_data.clear()
    return ConversationHandler.END

### MODIFIED ###
async def back_to_main_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Navigates back to the main method selection menu."""
    query = update.callback_query
    await query.answer()
    await _build_main_menu_keyboard(query=query)
    return CHOOSING_SUMMARY_METHOD


@authorized
async def handle_reply_to_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    reply_to = update.message.reply_to_message

    if reply_to and reply_to.from_user.is_bot:
        summary_context_text = get_summary_text_from_bot_message(reply_to.message.chat_id, context.bot.id)
        if summary_context_text:
            interaction_id_user_request = add_interaction(update.effective_user.id, update.message.chat_id, update.message.message_id, text, REQ_TYPE_REPLY_TO_SUMMARY, reply_to_message_id=reply_to.message_id)
            if interaction_id_user_request is None:
                await update.message.reply_text("Error: Could not save your question about the summary.")
                return
            prompt = (f"Previous context (video summary):\n---START OF CONTEXT---\n{summary_context_text}\n---END OF CONTEXT---\n\n"
                      f"User's question about this context: {text}\n\nAnswer the user's question.")
            response = await generate_chat_response(prompt)
            await update.message.reply_text(response, reply_to_message_id=update.message.message_id)
            update_interaction(interaction_id_user_request, response_text=response, status=STATUS_SUCCESS)
            return
    await _process_general_query(update, context)


async def _process_general_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.message.chat_id
    message_id = update.message.message_id
    text = update.message.text
    interaction_id = add_interaction(user_id, chat_id, message_id, text, REQ_TYPE_GENERAL)
    if interaction_id is None:
        await update.message.reply_text("Could not process your request due to a database issue.")
        return
    history = get_conversation_history(user_id)
    try:
        response = await generate_chat_response(text, history)
        await update.message.reply_text(response)
        update_interaction(interaction_id, response_text=response, status=STATUS_SUCCESS)
    except Exception as e:
        logger.error(f"LLM error processing general query (ID: {interaction_id}): {e}", exc_info=True)
        await update.message.reply_text("Sorry, could not get a response from the AI.")
        update_interaction(interaction_id, status=STATUS_ERROR_LLM, error_message=str(e))

async def cancel_summary_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if query:
        await query.answer()
        await query.edit_message_text(text="Summarization cancelled.")
    else:
        await update.message.reply_text("Summarization cancelled.")
    context.user_data.clear()
    return ConversationHandler.END
logger.info("Telegram handler functions defined.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Sorry, an unexpected error occurred. The issue has been logged."
            )
        except Exception as e:
            logger.error(f"Failed to send error message to user: {e}")

def main() -> None:
    logger.info("Entered main function.")
    try:
        init_db()
        logger.info("Database initialization completed.")

        if not AI_COUNCIL_TELEGRAM_BOT_TOKEN:
            logger.critical("AI_COUNCIL_TELEGRAM_BOT_TOKEN is missing after initial checks. Exiting.")
            return

        logger.info("Building application...")
        application = Application.builder().token(AI_COUNCIL_TELEGRAM_BOT_TOKEN).build()
        logger.info("Application built.")

        application.add_error_handler(error_handler)
        logger.info("Error handler added to application.")

        ### MODIFIED ###
        summary_conv_handler = ConversationHandler(
            entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.REPLY, handle_text_message)],
            states={
                CHOOSING_SUMMARY_METHOD: [
                    CallbackQueryHandler(choose_summary_method_callback, pattern='^summarize_gemini_api$'),
                    CallbackQueryHandler(choose_summary_method_callback, pattern='^summarize_vertex_ai$'),
                    CallbackQueryHandler(choose_summary_method_callback, pattern='^summarize_super_transcript$'),
                    CallbackQueryHandler(choose_summary_method_callback, pattern='^custom_prompts_menu$'),
                    CallbackQueryHandler(cancel_summary_conversation, pattern='^cancel_summary$'),
                ],
                CHOOSING_LANGUAGE: [
                    CallbackQueryHandler(choose_language_callback, pattern='^lang_ru$'),
                    CallbackQueryHandler(choose_language_callback, pattern='^lang_en$'),
                    CallbackQueryHandler(back_to_main_menu_callback, pattern='^back_to_main_menu$'),
                ],
                CHOOSING_CUSTOM_PROMPT: [
                    CallbackQueryHandler(choose_custom_prompt_callback, pattern='^prompt_'),
                    CallbackQueryHandler(back_to_main_menu_callback, pattern='^back_to_main_menu$'),
                ],
            },
            fallbacks=[CommandHandler('cancel', cancel_summary_conversation)],
        )
        logger.info("ConversationHandler created.")

        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(summary_conv_handler)
        application.add_handler(MessageHandler(filters.REPLY & filters.TEXT & ~filters.COMMAND, handle_reply_to_summary))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.REPLY, _process_general_query))
        logger.info("All handlers added.")

        logger.info("Starting bot polling...")
        application.run_polling()
        logger.info("Application.run_polling() has exited.")
    except Exception as e:
        logger.critical(f"Critical error in main function: {e}", exc_info=True)

if __name__ == '__main__':
    logger.info("Script execution started in __main__ block.")
    main()
    logger.info("main() function has completed and script is exiting.")