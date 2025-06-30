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

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
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

### MODIFIED ###
# Load a comma-separated string of user IDs and parse it into a list of integers.
AUTHORIZED_USER_IDS_STR = os.getenv("AUTHORIZED_USER_IDS", "")
AUTHORIZED_USER_IDS = []
if AUTHORIZED_USER_IDS_STR:
    try:
        AUTHORIZED_USER_IDS = [int(user_id.strip()) for user_id in AUTHORIZED_USER_IDS_STR.split(',')]
        logger.info(f"Loaded {len(AUTHORIZED_USER_IDS)} authorized user IDs.")
    except ValueError:
        logger.critical("CRITICAL ERROR: AUTHORIZED_USER_IDS in .env file contains non-integer values. Please fix it. Exiting.")
        exit()
else:
    logger.warning("WARNING: AUTHORIZED_USER_IDS is not set in the .env file. The bot will not respond to any user.")


if not all([AI_COUNCIL_TELEGRAM_BOT_TOKEN, AUTHORIZED_USER_IDS]):
    logger.critical("CRITICAL ERROR: Missing AI_COUNCIL_TELEGRAM_BOT_TOKEN or AUTHORIZED_USER_IDS is not set correctly. Exiting.")
    exit()

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Successfully configured google-generativeai with GEMINI_API_KEY.")
    except Exception as e:
        logger.error(f"Failed to configure google-generativeai with GEMINI_API_KEY: {e}.")
else:
    logger.warning("GEMINI_API_KEY not set. Bot functionality will be severely limited.")

logger.info("Environment variable checks complete.")

# --- Models & Constants ---
DB_NAME = "telegram_bot.db"
REQ_TYPE_GENERAL = "general_query"
REQ_TYPE_YOUTUBE_ANALYSIS = "youtube_analysis"
REQ_TYPE_REPLY_TO_SUMMARY = "reply_to_summary"
STATUS_PENDING = "pending"
STATUS_SUCCESS = "success"
STATUS_ERROR_LLM = "error_llm"
STATUS_ERROR_YOUTUBE = "error_youtube"

CHAT_MODEL_NAME = os.getenv("GEMINI_MODEL_CHAT", "gemini-2.5-flash")
GEMINI_API_VIDEO_MODEL = os.getenv("GEMINI_MODEL_VIDEO_API", "gemini-2.5-flash")

# Conversation states
CHOOSING_METHOD, CHOOSING_LANGUAGE = range(2)

LANGUAGE_NAME_MAP = {
    "ru": "русском",
    "en": "английском"
}

# --- Helper function to load prompts ---
def load_prompt(filename: str) -> str | None:
    """Loads a prompt from a file in the 'prompts' directory."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, 'prompts', filename)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"CRITICAL: Prompt file not found at {prompt_path}")
        return None

# --- Load Prompts from Files ---
logger.info("Loading prompts from files...")
PROMPTS = {
    "detailed_summary": load_prompt("detailed_video_summary.txt"),
    "super_transcript": load_prompt("super_transcript.txt"),
    "magazine_article": load_prompt("magazine_article.txt"),
    "textbook_chapter": load_prompt("textbook_chapter.txt"),
    "step_by_step_guide": load_prompt("step_by_step_guide.txt"),
    "obsidian_note": load_prompt("obsidian_note.txt"),
}

# Critical check to ensure all prompts were loaded
if not all(PROMPTS.values()):
    logger.critical("One or more essential prompt files could not be loaded from the 'prompts' folder. Exiting.")
    exit()

logger.info("All essential prompts loaded successfully.")


# --- Authorization Decorator & DB Functions ---
def authorized(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        ### MODIFIED ###
        # Check if the user's ID is in the list of authorized IDs
        if update.effective_user.id not in AUTHORIZED_USER_IDS:
            logger.warning(f"Unauthorized access from user_id: {update.effective_user.id}")
            await update.message.reply_text("Sorry, you do not have access to this bot.")
            if isinstance(context, ConversationHandler):
                return ConversationHandler.END
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

def init_db():
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
                VALUES ?, ?, ?, ?, ?, ?, ?, ?
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


# --- AI Integration & Features ---
def extract_video_id(url):
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

async def process_video_with_gemini(youtube_link: str, video_id: str, prompt_text: str):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be set for this method.")
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
        video_part = glm.Part(file_data=glm.FileData(mime_type="video/mp4", file_uri=youtube_link))
        contents = [video_part, glm.Part(text=prompt_text)]
        logger.info(f"Sending content to Gemini API. URI: {youtube_link}")
        response = await model.generate_content_async(contents, safety_settings=safety_settings)
        try:
            return response.text
        except ValueError:
            logger.warning(f"response.text failed. Raw response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason == 'SAFETY':
                if response.candidates and response.candidates[0].finish_reason == 4:
                     raise ValueError("Content blocked by API safety filters due to potential recitation of copyrighted material.")
                raise ValueError("The API response was blocked by safety filters.")
            raise ValueError("The API returned an invalid or empty response.")
    except Exception as e:
        logger.error(f"Error during video processing with Gemini API: {e}", exc_info=True)
        raise e


async def send_long_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, reply_to_message_id: int | None = None):
    MAX_LENGTH = 4096
    if len(text) <= MAX_LENGTH:
        await context.bot.send_message(chat_id=chat_id, text=text, reply_to_message_id=reply_to_message_id)
        return
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
    for i, part in enumerate(parts):
        part_text_with_indicator = f"[{i+1}/{len(parts)}]\n{part}"
        reply_id = reply_to_message_id if i == 0 else None
        await context.bot.send_message(chat_id=chat_id, text=part_text_with_indicator, reply_to_message_id=reply_id)


# --- Telegram Handlers ---
@authorized
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(rf"Hi, {update.effective_user.mention_html()}! I'm ready to work.")

async def _build_main_menu_keyboard(query: CallbackQueryHandler | None = None, update: Update | None = None):
    keyboard = [
        [InlineKeyboardButton("Detailed Summary", callback_data='method_detailed_summary')],
        [InlineKeyboardButton("Super Transcript (File)", callback_data='method_super_transcript')],
        [InlineKeyboardButton("Custom Formats >>", callback_data='custom_prompts_menu')],
        [InlineKeyboardButton("Cancel", callback_data='cancel_summary')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_text = 'Choose analysis method for this YouTube video:'
    if query:
        await query.edit_message_text(message_text, reply_markup=reply_markup)
    elif update:
        await update.message.reply_text(message_text, reply_markup=reply_markup)

@authorized
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int | None:
    video_id = extract_video_id(update.message.text)
    if video_id:
        context.user_data['youtube_link'] = update.message.text
        context.user_data['video_id'] = video_id
        context.user_data['original_message_id'] = update.message.message_id
        await _build_main_menu_keyboard(update=update)
        return CHOOSING_METHOD
    await update.message.reply_text("Please send me a valid YouTube link.")
    return ConversationHandler.END

@authorized
async def choose_method_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    choice = query.data

    if choice == 'custom_prompts_menu':
        keyboard = [
            [InlineKeyboardButton("Magazine Article", callback_data='method_magazine_article')],
            [InlineKeyboardButton("Textbook Chapter", callback_data='method_textbook_chapter')],
            [InlineKeyboardButton("Step-by-Step Guide", callback_data='method_step_by_step_guide')],
            [InlineKeyboardButton("Obsidian MD Note (File)", callback_data='method_obsidian_note')],
            [InlineKeyboardButton("<< Back to Main Menu", callback_data='back_to_main_menu')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text="Choose a custom prompt format:", reply_markup=reply_markup)
        return CHOOSING_METHOD

    context.user_data['method_choice'] = choice.replace('method_', '')
    language_keyboard = [
        [InlineKeyboardButton("Русский", callback_data='lang_ru')],
        [InlineKeyboardButton("English", callback_data='lang_en')],
        [InlineKeyboardButton("<< Back to Main Menu", callback_data='back_to_main_menu')],
    ]
    reply_markup = InlineKeyboardMarkup(language_keyboard)
    await query.edit_message_text(text="Choose summary language:", reply_markup=reply_markup)
    return CHOOSING_LANGUAGE

@authorized
async def process_request_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    chosen_language_code = query.data.split('_')[1]

    user_data = context.user_data
    method = user_data.get('method_choice')
    
    prompt_template = PROMPTS.get(method)
    if not prompt_template:
        await query.edit_message_text("Error: Could not find a valid prompt for the selected action.")
        return ConversationHandler.END

    language_name = LANGUAGE_NAME_MAP.get(chosen_language_code, "selected")
    prompt_text = prompt_template.format(language_name_locative=language_name)
    
    await query.edit_message_text(f"Processing '{method.replace('_', ' ').title()}' in {language_name}... This may take a moment.")

    interaction_id = add_interaction(
        query.from_user.id, query.message.chat_id, user_data['original_message_id'],
        user_data['youtube_link'], f"youtube_{method}"
    )

    try:
        ai_content = await process_video_with_gemini(user_data['youtube_link'], user_data['video_id'], prompt_text)
        
        # Always generate a file for the response
        filename = f"{user_data['video_id']}_{method}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(ai_content)
            with open(filename, 'rb') as f:
                await context.bot.send_document(
                    chat_id=query.message.chat_id, 
                    document=f, 
                    filename=filename,
                    caption=f"Here is your '{method.replace('_', ' ').title()}' file.",
                    reply_to_message_id=user_data['original_message_id']
                )
            # For non-file-output methods, also send the response as a text message
            if method not in ['super_transcript', 'obsidian_note']:
                await send_long_message(context, query.message.chat_id, ai_content, user_data['original_message_id'])
            update_interaction(interaction_id, response_text=ai_content, status=STATUS_SUCCESS)
        finally:
            if os.path.exists(filename):
                os.remove(filename)
                logger.info(f"Temporary file {filename} deleted.")

    except Exception as e:
        logger.error(f"Error during processing for method '{method}': {e}", exc_info=True)
        await context.bot.send_message(query.message.chat_id, f"Could not complete request. Reason: {e}", reply_to_message_id=user_data['original_message_id'])
        if interaction_id:
            update_interaction(interaction_id, status=STATUS_ERROR_YOUTUBE, error_message=str(e))

    user_data.clear()
    return ConversationHandler.END

async def back_to_main_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await _build_main_menu_keyboard(query=query)
    return CHOOSING_METHOD

async def cancel_summary_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if query:
        await query.answer()
        await query.edit_message_text(text="Summarization cancelled.")
    context.user_data.clear()
    return ConversationHandler.END

def main() -> None:
    application = Application.builder().token(AI_COUNCIL_TELEGRAM_BOT_TOKEN).build()

    summary_conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)],
        states={
            CHOOSING_METHOD: [
                CallbackQueryHandler(choose_method_callback, pattern='^method_'),
                CallbackQueryHandler(choose_method_callback, pattern='^custom_prompts_menu$'),
                CallbackQueryHandler(back_to_main_menu_callback, pattern='^back_to_main_menu$'),
            ],
            CHOOSING_LANGUAGE: [
                CallbackQueryHandler(process_request_callback, pattern='^lang_'),
                CallbackQueryHandler(back_to_main_menu_callback, pattern='^back_to_main_menu$'),
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel_summary_conversation), CallbackQueryHandler(cancel_summary_conversation, pattern='^cancel_summary$')],
        per_message=False
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(summary_conv_handler)
    
    logger.info("Starting bot polling...")
    application.run_polling()

if __name__ == '__main__':
    init_db()
    main()