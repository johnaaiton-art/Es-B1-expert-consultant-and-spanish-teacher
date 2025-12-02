import os
import json
import hashlib
import re
import zipfile
import time
import random
from datetime import datetime
from collections import defaultdict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from openai import OpenAI
from google.cloud import texttospeech
from google.oauth2 import service_account
import asyncio
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment variables")
if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in environment variables")
if not GOOGLE_CREDENTIALS_JSON:
    raise ValueError("Missing GOOGLE_CREDENTIALS_JSON in environment variables")

# Conversation states
TOPIC, LEVEL = range(2)

class Config:
    MAX_TOPIC_LENGTH = 100
    MAX_VOCAB_ITEMS = 15
    TTS_TIMEOUT = 30
    API_RETRY_ATTEMPTS = 3
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW = 3600
    MAX_FILE_SIZE = 50 * 1024 * 1024
    TRACKING_SHEET_ID = os.getenv("TRACKING_SHEET_ID")

config = Config()

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


class RateLimiter:
    def __init__(self, max_requests=5, window=3600):
        self.requests = defaultdict(list)
        self.max_requests = max_requests
        self.window = window
    
    def is_allowed(self, user_id):
        now = time.time()
        user_requests = self.requests[user_id]
        user_requests[:] = [req_time for req_time in user_requests if now - req_time < self.window]
        if len(user_requests) >= self.max_requests:
            return False
        user_requests.append(now)
        return True
    
    def get_reset_time(self, user_id):
        if not self.requests[user_id]:
            return 0
        oldest_request = min(self.requests[user_id])
        reset_time = oldest_request + self.window - time.time()
        return max(0, int(reset_time))

rate_limiter = RateLimiter(
    max_requests=config.RATE_LIMIT_REQUESTS,
    window=config.RATE_LIMIT_WINDOW
)

# Level configurations
LEVEL_CONFIGS = {
    "B1": {
        "description": "Intermediate",
        "speaking_rate": 0.85,
        "wavenet_voice": "es-ES-Wavenet-B",
        "chirp_voices": [
            "es-ES-Chirp-HD-F",
            "es-ES-Chirp-HD-O",
            "es-ES-Chirp3-HD-Gacrux",
            "es-US-Chirp3-HD-Leda",
            "es-ES-Chirp3-HD-Algenib",
            "es-ES-Chirp3-HD-Charon",
            "es-US-Chirp3-HD-Algieba"
        ],
        "prompt_modifier": "SOLID B1 level. Use vocabulary that is clearly B1, avoiding anything that might be borderline A2/B1. "
    },
    "B2": {
        "description": "Upper Intermediate",
        "speaking_rate": 0.80,
        "wavenet_voice": "es-ES-Wavenet-C",
        "chirp_voices": [
            "es-ES-Chirp-HD-F",
            "es-ES-Chirp-HD-O",
            "es-ES-Chirp3-HD-Gacrux",
            "es-US-Chirp3-HD-Leda",
            "es-ES-Chirp3-HD-Algenib",
            "es-ES-Chirp3-HD-Charon",
            "es-US-Chirp3-HD-Algieba"
        ],
        "prompt_modifier": "B2 level. Use vocabulary appropriate for upper-intermediate learners."
    },
    "C1": {
        "description": "Advanced",
        "speaking_rate": 0.75,
        "wavenet_voice": "es-ES-Wavenet-D",
        "chirp_voices": [
            "es-ES-Chirp-HD-F",
            "es-ES-Chirp-HD-O",
            "es-ES-Chirp3-HD-Gacrux",
            "es-US-Chirp3-HD-Leda",
            "es-ES-Chirp3-HD-Algenib",
            "es-ES-Chirp3-HD-Charon",
            "es-US-Chirp3-HD-Algieba"
        ],
        "prompt_modifier": "C1 level. Use advanced vocabulary and complex sentence structures."
    },
    "C2": {
        "description": "Proficient",
        "speaking_rate": 0.75,
        "wavenet_voice": "es-ES-Wavenet-E",
        "chirp_voices": [
            "es-ES-Chirp-HD-F",
            "es-ES-Chirp-HD-O",
            "es-ES-Chirp3-HD-Gacrux",
            "es-US-Chirp3-HD-Leda",
            "es-ES-Chirp3-HD-Algenib",
            "es-ES-Chirp3-HD-Charon",
            "es-US-Chirp3-HD-Algieba"
        ],
        "prompt_modifier": "C2 (near-native) level. Use sophisticated vocabulary and nuanced expressions."
    }
}

def get_google_tts_client():
    credentials_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return texttospeech.TextToSpeechClient(credentials=credentials)
    
def get_sheets_client():
    """Initialize Google Sheets client"""
    credentials_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build('sheets', 'v4', credentials=credentials)

async def track_usage_google_sheets(user_id, username, first_name, last_name, topic, level):
    """Track student usage in Google Sheets"""
    try:
        if not config.TRACKING_SHEET_ID:
            logger.warning("[Tracking] No TRACKING_SHEET_ID configured, skipping")
            return
        
        sheets_client = get_sheets_client()
        
        # Prepare data row
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_name = f"{first_name or ''} {last_name or ''}".strip() or "Unknown"
        
        row_data = [[
            timestamp,
            user_id,
            username or "No username",
            full_name,
            topic[:50],  # Truncate long topics
            level
        ]]
        
        # Append to sheet
        sheets_client.spreadsheets().values().append(
            spreadsheetId=config.TRACKING_SHEET_ID,
            range="A:F",  # Now includes level column
            valueInputOption="RAW",
            body={"values": row_data}
        ).execute()
        
        logger.info(f"[Tracking] ✅ Logged to Google Sheets: {full_name} ({username}) - '{topic[:30]}' - Level: {level}")
    except Exception as e:
        logger.error(f"[Tracking] ❌ Failed to log to Google Sheets: {e}")

def validate_topic(topic):
    topic = re.sub(r'\s+', ' ', topic.strip())
    if re.search(r'[<>"|&;`$()]', topic):
        raise ValueError("Topic contains invalid characters")
    inappropriate_patterns = [r'\b(porn|sex|violence|hate|kill|death)\b']
    for pattern in inappropriate_patterns:
        if re.search(pattern, topic, re.IGNORECASE):
            raise ValueError("Topic contains inappropriate content")
    if len(topic) > config.MAX_TOPIC_LENGTH:
        topic = topic[:config.MAX_TOPIC_LENGTH]
    if not topic:
        raise ValueError("Topic cannot be empty")
    return topic

def validate_level(level_text):
    level_text = level_text.upper().strip()
    if level_text in LEVEL_CONFIGS:
        return level_text
    else:
        raise ValueError(f"Invalid level. Please choose from: {', '.join(LEVEL_CONFIGS.keys())}")

def split_text_into_sentences(text, max_length=200):
    # Remove asterisks from text before TTS generation
    text = text.replace('*', '')
    sentences = re.split(r'([.!?])\s+', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    final_result = []
    for sentence in result:
        if len(sentence) > max_length:
            parts = re.split(r'([,;])\s+', sentence)
            temp = ""
            for part in parts:
                if len(temp + part) > max_length and temp:
                    final_result.append(temp)
                    temp = part
                else:
                    temp += part
            if temp:
                final_result.append(temp)
        else:
            final_result.append(sentence)
    return [s.strip() for s in final_result if s.strip()]

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(Exception)
)
def generate_tts_chirp3_sync(text, voice_name, speaking_rate=0.75):
    """Generate TTS using Chirp3 HD voices for main texts - SPANISH VERSION"""
    try:
        # Remove asterisks from text
        text = text.replace('*', '')
        logger.info(f"[Chirp3 TTS Español] Generating for voice '{voice_name}', speed: {speaking_rate}, text length: {len(text)}")
        client = get_google_tts_client()
        sentences = split_text_into_sentences(text, max_length=200)
        logger.info(f"[Chirp3 TTS Español] Split into {len(sentences)} sentences")
        
        all_audio = b""
        for idx, sentence in enumerate(sentences):
            synthesis_input = texttospeech.SynthesisInput(text=sentence)
            voice = texttospeech.VoiceSelectionParams(
                language_code="es-ES",
                name=voice_name
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate
            )
            response = client.synthesize_speech(
                input=synthesis_input, 
                voice=voice, 
                audio_config=audio_config
            )
            all_audio += response.audio_content
            logger.info(f"[Chirp3 TTS Español] Sentence {idx+1}/{len(sentences)} completed")
        
        logger.info(f"[Chirp3 TTS Español] ✅ Success: {len(all_audio)} bytes")
        return all_audio
    except Exception as e:
        logger.error(f"[Chirp3 TTS Español] ❌ Error: {type(e).__name__}: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(Exception)
)
def generate_tts_wavenet_sync(text, voice_name="es-ES-Wavenet-B", speaking_rate=0.95):
    """Generate TTS using Wavenet voices for Anki cards - SPANISH VERSION"""
    try:
        # Remove asterisks from Anki card text too
        text = text.replace('*', '')
        logger.info(f"[Wavenet TTS Español] Generating for '{text[:50]}...' with voice '{voice_name}'")
        client = get_google_tts_client()
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES",
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        audio_size = len(response.audio_content)
        logger.info(f"[Wavenet TTS Español] ✅ Success: {audio_size} bytes for '{text[:30]}'")
        return response.audio_content
    except Exception as e:
        logger.error(f"[Wavenet TTS Español] ❌ Failed for '{text[:50]}': {type(e).__name__}: {str(e)}")
        raise

async def generate_tts_chirp3_async(text, voice_name, speaking_rate=0.75):
    """Async wrapper for Chirp3 TTS"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_tts_chirp3_sync, text, voice_name, speaking_rate)

async def generate_tts_wavenet_async(text, voice_name="es-ES-Wavenet-B", speaking_rate=0.95):
    """Async wrapper for Wavenet TTS"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_tts_wavenet_sync, text, voice_name, speaking_rate)

def safe_filename(filename):
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    filename = os.path.basename(filename)
    filename = filename[:100]
    return filename.strip('_')

def validate_deepseek_response(content):
    required_keys = ["main_text", "collocations", "opinion_texts", "discussion_questions"]
    if not all(k in content for k in required_keys):
        missing = [k for k in required_keys if k not in content]
        raise ValueError(f"Missing required keys: {missing}")
    if not isinstance(content['collocations'], list):
        raise ValueError("collocations must be a list")
    if len(content['collocations']) > config.MAX_VOCAB_ITEMS:
        content['collocations'] = content['collocations'][:config.MAX_VOCAB_ITEMS]
    for item in content['collocations']:
        if not all(k in item for k in ['spanish', 'english']):
            raise ValueError("Each collocation must have 'spanish', 'english'")
    if not all(k in content['opinion_texts'] for k in ['positive', 'negative', 'mixed']):
        raise ValueError("opinion_texts must have 'positive', 'negative', 'mixed'")
    if not isinstance(content['discussion_questions'], list):
        raise ValueError("discussion_questions must be a list")
    return True

def get_fallback_content(topic, level):
    """Provide fallback content if DeepSeek API fails"""
    logger.info(f"[Fallback] Generating fallback content for level {level}")
    
    level_config = LEVEL_CONFIGS[level]
    
    # Simple fallback content that matches the structure
    return {
        "main_text": f"Este texto trata sobre el tema '{topic}' a nivel {level}. Es un ejemplo de contenido educativo en español que incluye vocabulario útil y expresiones apropiadas para el nivel {level}. El objetivo es proporcionar materiales de aprendizaje significativos para estudiantes que buscan mejorar sus habilidades lingüísticas en contextos prácticos.",
        "collocations": [
            {"spanish": "tratar sobre", "english": "to be about"},
            {"spanish": "nivel apropiado", "english": "appropriate level"},
            {"spanish": "vocabulario útil", "english": "useful vocabulary"},
            {"spanish": "materiales de aprendizaje", "english": "learning materials"},
            {"spanish": "expresiones comunes", "english": "common expressions"},
            {"spanish": "contenido educativo", "english": "educational content"},
            {"spanish": "objetivo principal", "english": "main objective"},
            {"spanish": "hablar de", "english": "to talk about"},
            {"spanish": "aprender español", "english": "to learn Spanish"},
            {"spanish": "mejorar habilidades", "english": "to improve skills"},
            {"spanish": "practicar regularmente", "english": "to practice regularly"},
            {"spanish": "entender conceptos", "english": "to understand concepts"},
            {"spanish": "comunicarse efectivamente", "english": "to communicate effectively"},
            {"spanish": "desarrollar confianza", "english": "to develop confidence"},
            {"spanish": "lograr metas", "english": "to achieve goals"}
        ],
        "opinion_texts": {
            "positive": f"Este tema es muy relevante para estudiantes de español a nivel {level}. Proporciona oportunidades excelentes para practicar vocabulario y expresiones útiles en contextos reales, lo que facilita la adquisición natural del idioma.",
            "negative": f"Aunque el tema puede ser interesante, algunos estudiantes de nivel {level} podrían encontrar desafíos al discutir conceptos complejos sin suficiente preparación previa o vocabulario especializado.",
            "mixed": f"El tema ofrece tanto oportunidades como desafíos para estudiantes de nivel {level}. Con la guía adecuada y recursos apropiados, puede ser una experiencia de aprendizaje muy valiosa que desarrolla múltiples competencias lingüísticas."
        },
        "discussion_questions": [
            f"¿Qué aspectos de este tema te parecen más interesantes para practicar español a nivel {level}?",
            f"¿Has tenido experiencias personales relacionadas con este tema que quisieras compartir en español?",
            f"¿Qué diferencias notas en cómo se discute este tema en español comparado con otros idiomas que conoces?",
            f"¿Cómo crees que este tema podría evolucionar o cambiar en el futuro según tu perspectiva?",
            f"¿Qué argumentos a favor y en contra podrías presentar sobre este tema en una discusión en español?"
        ]
    }

@retry(
    stop=stop_after_attempt(config.API_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(f"Retry {retry_state.attempt_number}: {retry_state.outcome.exception()}")
)
def generate_content_with_deepseek(topic, level):
    logger.info(f"[DeepSeek Español] Generating content for: '{topic}' at level {level}")
    
    # Truncate topic if it's too long for the prompt
    max_topic_length = 80  # Keep topic short for API
    if len(topic) > max_topic_length:
        truncated_topic = topic[:max_topic_length] + "..."
        logger.warning(f"[DeepSeek Español] Truncating topic from {len(topic)} to {max_topic_length} chars")
    else:
        truncated_topic = topic
    
    level_config = LEVEL_CONFIGS[level]
    prompt_modifier = level_config["prompt_modifier"]
    
    prompt = f"""You are both an expert consultant on the topic given and a Spanish language teaching assistant. Create informationally highly insightful and nuanced learning materials with expert advice about the topic: "{truncated_topic}"

Please generate a JSON response with the following structure:
{{
  "main_text": "An engaging and expertly insightful Spanish text at CEFR {level} level ({level_config['description']}) about {truncated_topic}. Should be 200-250 words long, must contain expert-level insights / information, should be natural. MUST contain relevant terminology and 3 Spanish verb phrases or useful expressions that are either typical for this context OR generically useful for {level} learners. Use {prompt_modifier} Include the complete phrases as they would appear in the text.",
  "collocations": [
    {{"spanish": "Spanish phrase/expression from text", "english": "English translation"}},
    // Exactly 15 items total
    // MUST include all 3 verb phrases/expressions (as they appear in the text)
    // should include terminology IF RELEVANT
    // Remaining items should be useful collocations, expressions, verb+noun, or adjective+noun pairs from the text
    // ALL collocations must come directly from the main_text
    // Use ONLY {level}-appropriate vocabulary
  ],
  "opinion_texts": {{
    "positive": "A natural Spanish response ({level} level, 80-120 words) adding extra expert insights to the main topic. Must incorporate some vocabulary from the collocations list naturally.",
    "negative": "A natural Spanish response ({level} level, 80-120 words) giving contrasting expert insights to the main topic. Must incorporate some vocabulary from the collocations list naturally.",
    "mixed": "A natural Spanish response ({level} level, 80-120 words) giving a balanced/mixed reaction to the main topic. Must incorporate some vocabulary from the collocations list naturally."
  }},
  "discussion_questions": [
    "Question 1 in Spanish ({level} level) - should ask about personal reaction to presented insights",
    "Question 2 in Spanish ({level} level) - should ask about personal experience in reference to the topic",
    "Question 3 in Spanish ({level} level) - should encourage reflection nuances in the differences in the insights",
    "Question 4 in Spanish ({level} level) - should encourage a prediction",
    "Question 5 in Spanish ({level} level) - should stimulate debate"
  ]
}}

CRITICAL REQUIREMENTS:
1. Main text MUST contain 3 Spanish verb phrases or useful expressions appropriate for {level} level
2. ALL collocations must come from the main_text
3. The first 3 collocations MUST be the verb phrases/expressions
4. Should, if relevant and only if relevant, include some terminology
5. Use {prompt_modifier}
6. Reaction texts must naturally use some collocations but sound expert and insightful
7. Discussion questions should be thought-provoking but use {level}-level language
8. Return ONLY valid JSON, no additional text"""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"You are an expert Spanish language teacher who creates engaging, natural content at CEFR {level} level with a focus on useful expressions and verb phrases. {prompt_modifier} Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            timeout=60.0  # Increased timeout from 45 to 60 seconds
        )
        
        content_text = response.choices[0].message.content
        logger.info(f"[DeepSeek Español] Received response, parsing...")
        
        json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
        if json_match:
            content_text = json_match.group()
        
        content = json.loads(content_text)
        validate_deepseek_response(content)
        logger.info(f"[DeepSeek Español] ✅ Content validated successfully for level {level}")
        return content
        
    except Exception as e:
        logger.error(f"[DeepSeek Español] ❌ Error generating content: {type(e).__name__}: {str(e)}")
        # Return fallback content if DeepSeek fails
        logger.warning(f"[DeepSeek Español] Using fallback content for topic: {truncated_topic}")
        return get_fallback_content(truncated_topic, level)

async def create_vocabulary_file_with_tts(collocations, topic, level_config, progress_callback=None):
    """Create Anki vocabulary file with Wavenet TTS - SPANISH VERSION"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic_name = safe_filename(topic)
    filename = f"{safe_topic_name}_{timestamp}_collocations.txt"
    
    content = ""
    audio_files = {}
    total_items = len(collocations)
    
    logger.info(f"[Anki TTS Español] Starting generation for {total_items} collocations using {level_config['wavenet_voice']}")
    
    # Generate TTS for all collocations using the appropriate Wavenet voice for the level
    tts_tasks = []
    for item in collocations:
        tts_tasks.append(generate_tts_wavenet_async(
            item['spanish'], 
            voice_name=level_config['wavenet_voice'],
            speaking_rate=0.95
        ))
    
    logger.info(f"[Anki TTS Español] Awaiting {len(tts_tasks)} concurrent TTS generations...")
    audio_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
    logger.info(f"[Anki TTS Español] All TTS generations completed")
    
    success_count = 0
    failed_count = 0
    
    for idx, (item, audio_data) in enumerate(zip(collocations, audio_results)):
        spanish_text = item['spanish']
        
        if progress_callback:
            await progress_callback(idx + 1, total_items)
        
        # Check if audio generation succeeded
        if isinstance(audio_data, Exception):
            logger.error(f"[Anki TTS Español] ❌ Exception for '{spanish_text}': {type(audio_data).__name__}: {audio_data}")
            failed_count += 1
            # Add row without audio: English | Spanish
            content += f"{item['english']}\t{item['spanish']}\n"
        elif not audio_data:
            logger.error(f"[Anki TTS Español] ❌ Empty data for '{spanish_text}'")
            failed_count += 1
            # Add row without audio: English | Spanish
            content += f"{item['english']}\t{item['spanish']}\n"
        else:
            # Success - create filename using MD5 hash
            hash_object = hashlib.md5(spanish_text.encode())
            audio_filename = f"tts_{hash_object.hexdigest()}.mp3"
            audio_filename = safe_filename(audio_filename)
            
            # Store audio data
            audio_files[audio_filename] = audio_data
            
            # Create Anki sound tag
            anki_tag = f"[sound:{audio_filename}]"
            
            # Add row with 3 columns: English | Spanish | Audio
            content += f"{item['english']}\t{item['spanish']}\t{anki_tag}\n"
            success_count += 1
            logger.info(f"[Anki TTS Español] ✅ {idx+1}/{total_items}: '{spanish_text[:30]}' -> {audio_filename}")
    
    logger.info(f"[Anki TTS Español] SUMMARY: ✅ {success_count} succeeded, ❌ {failed_count} failed out of {total_items} total")
    
    if failed_count > 0:
        logger.warning(f"[Anki TTS Español] ⚠️ WARNING: {failed_count}/{total_items} TTS generations failed")
    
    return filename, content, audio_files

def create_zip_package(vocab_filename, vocab_content, audio_files, html_filename, html_content, topic, timestamp):
    """Create ZIP with all files"""
    safe_topic_name = safe_filename(topic)
    zip_filename = f"{safe_topic_name}_{timestamp}_complete_package.zip"
    zip_buffer = BytesIO()
    
    logger.info(f"[ZIP Español] Creating package with {len(audio_files)} audio files")
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add vocabulary text file
        safe_vocab = safe_filename(vocab_filename)
        zip_file.writestr(safe_vocab, vocab_content.encode('utf-8'))
        logger.info(f"[ZIP Español] Added vocabulary file: {safe_vocab}")
        
        # Add all Anki TTS audio files
        for audio_filename, audio_data in audio_files.items():
            safe_audio = safe_filename(audio_filename)
            zip_file.writestr(safe_audio, audio_data)
        logger.info(f"[ZIP Español] Added {len(audio_files)} Anki TTS audio files")
        
        # Add HTML document
        safe_html = safe_filename(html_filename)
        zip_file.writestr(safe_html, html_content.encode('utf-8'))
        logger.info(f"[ZIP Español] Added HTML file: {safe_html}")
    
    zip_buffer.seek(0)
    file_size = zip_buffer.getbuffer().nbytes
    logger.info(f"[ZIP Español] Package size: {file_size / 1024 / 1024:.2f}MB")
    
    if file_size > config.MAX_FILE_SIZE:
        raise ValueError(f"ZIP too large: {file_size / 1024 / 1024:.1f}MB")
    
    return zip_filename, zip_buffer

def create_html_document(topic, content, timestamp, level):
    """Create HTML document - SPANISH VERSION"""
    safe_topic = safe_filename(topic)
    html_filename = f"{safe_topic}_{timestamp}_materials.html"
    
    level_config = LEVEL_CONFIGS[level]
    
    # Remove asterisks from all text content
    def remove_asterisks(text):
        return text.replace('*', '')
    
    clean_main_text = remove_asterisks(content['main_text'])
    clean_positive = remove_asterisks(content['opinion_texts']['positive'])
    clean_negative = remove_asterisks(content['opinion_texts']['negative'])
    clean_mixed = remove_asterisks(content['opinion_texts']['mixed'])
    
    vocab_rows = ""
    for i, item in enumerate(content['collocations'], 1):
        # Also clean collocation text if it has asterisks
        clean_spanish = remove_asterisks(item['spanish'])
        vocab_rows += f"""
        <tr>
            <td>{i}</td>
            <td class="spanish">{clean_spanish}</td>
            <td class="english">{item['english']}</td>
        </tr>
        """
    
    questions_html = ""
    for i, question in enumerate(content['discussion_questions'], 1):
        # Remove asterisks from questions too
        clean_question = remove_asterisks(question)
        questions_html += f"""
        <div class="question">
            <span class="question-number">{i}</span>
            <span class="question-text">{clean_question}</span>
        </div>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Materiales de Aprendizaje de Español: {topic}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.8;
            color: #333;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .header .subtitle {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .level-badge {{
            display: inline-block;
            background: white;
            color: #f5576c;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
            font-size: 0.9em;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #f5576c;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #f5576c;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-icon {{
            font-size: 1.2em;
        }}
        .main-text {{
            background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
            padding: 30px;
            border-radius: 15px;
            font-size: 1.15em;
            line-height: 1.9;
            color: #2c3e50;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .spanish {{
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
        }}
        .english {{
            color: #7f8c8d;
            font-style: italic;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        thead {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }}
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        tbody tr:hover {{
            background: #e9ecef;
            transition: background 0.3s;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        .opinion-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 5px solid;
        }}
        .opinion-positive {{
            border-left-color: #4cd137;
        }}
        .opinion-negative {{
            border-left-color: #e84118;
        }}
        .opinion-mixed {{
            border-left-color: #fbc531;
        }}
        .opinion-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            font-size: 1.3em;
            font-weight: 600;
        }}
        .opinion-text {{
            font-size: 1.05em;
            line-height: 1.8;
            color: #2c3e50;
        }}
        .question {{
            background: #f8f9fa;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 10px;
            display: flex;
            gap: 15px;
            align-items: start;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }}
        .question-number {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }}
        .question-text {{
            font-size: 1.05em;
            line-height: 1.7;
            color: #2c3e50;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
        @media (max-width: 768px) {{
            .content {{
                padding: 20px;
            }}
            .header {{
                padding: 30px 20px;
            }}
            .main-text {{
                font-size: 1em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Materiales de Aprendizaje de Español</h1>
            <div class="subtitle">Tema: {topic}</div>
            <div class="subtitle">Nivel: CEFR {level} ({level_config['description']})</div>
            <div class="level-badge">NIVEL {level}</div>
            <div class="subtitle">Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
        </div>
        <div class="content">
            <!-- Expresiones y Frases -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">📚</span>
                    Expresiones y Frases Útiles
                </h2>
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Español</th>
                            <th>Inglés (English)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {vocab_rows}
                    </tbody>
                </table>
            </div>
            <!-- Texto Principal -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">📖</span>
                    Texto Principal
                </h2>
                <div class="main-text">{clean_main_text}</div>
            </div>
            <!-- Textos de Opinión -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">💭</span>
                    Diferentes Reacciones
                </h2>
                <div class="opinion-card opinion-positive">
                    <div class="opinion-header">
                        <span>😊</span>
                        <span>Reacción Positiva</span>
                    </div>
                    <div class="opinion-text">{clean_positive}</div>
                </div>
                <div class="opinion-card opinion-negative">
                    <div class="opinion-header">
                        <span>🤔</span>
                        <span>Reacción Crítica</span>
                    </div>
                    <div class="opinion-text">{clean_negative}</div>
                </div>
                <div class="opinion-card opinion-mixed">
                    <div class="opinion-header">
                        <span>⚖️</span>
                        <span>Reacción Equilibrada</span>
                    </div>
                    <div class="opinion-text">{clean_mixed}</div>
                </div>
            </div>
            <!-- Preguntas de Discusión -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">💬</span>
                    Preguntas de Discusión
                </h2>
                {questions_html}
            </div>
        </div>
        <div class="footer">
            <p>Generado por Spanish Learning Bot 🤖</p>
            <p>Materiales de Nivel CEFR {level} ({level_config['description']})</p>
            <p>Velocidad de audio: {int(level_config['speaking_rate'] * 100)}%</p>
        </div>
    </div>
</body>
</html>"""
    
    logger.info(f"[HTML Español] Created document for level {level}: {html_filename}")
    return html_filename, html_content

async def handle_topic_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle topic input - first step"""
    user_id = update.effective_user.id
    topic_raw = update.message.text.strip()
    
    logger.info(f"[Bot Español] User {user_id} entered topic: '{topic_raw}'")
    
    # Validate topic
    try:
        topic = validate_topic(topic_raw)
        logger.info(f"[Bot Español] Topic validated: '{topic}'")
    except ValueError as e:
        logger.error(f"[Bot Español] Invalid topic from user {user_id}: {str(e)}")
        await update.message.reply_text(f"❌ Tema inválido: {str(e)}\n\nPor favor intenta con un tema diferente.")
        return ConversationHandler.END
    
    # Store topic in user context
    context.user_data['topic'] = topic
    context.user_data['user_id'] = user_id
    
    # Ask for level
    level_options = "\n".join([f"• {level} - {config['description']}" for level, config in LEVEL_CONFIGS.items()])
    await update.message.reply_text(
        f"✅ Tema válido: '{topic}'\n\n"
        f"📊 Ahora selecciona el nivel de español:\n\n"
        f"{level_options}\n\n"
        f"Por favor responde con: B1, B2, C1 o C2"
    )
    
    return LEVEL

async def handle_level_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle level input - second step"""
    user_id = update.effective_user.id
    level_text = update.message.text.strip()
    
    # Validate level
    try:
        level = validate_level(level_text)
    except ValueError as e:
        logger.error(f"[Bot Español] Invalid level from user {user_id}: {level_text}")
        await update.message.reply_text(f"❌ {str(e)}")
        return LEVEL
    
    topic = context.user_data.get('topic')
    if not topic:
        logger.error(f"[Bot Español] No topic found for user {user_id}")
        await update.message.reply_text("❌ Error: No se encontró el tema. Por favor comienza de nuevo con /start")
        return ConversationHandler.END
    
    logger.info(f"[Bot Español] User {user_id} selected level {level} for topic: '{topic}'")
    
    # Store level in user context
    context.user_data['level'] = level
    
    # Check rate limit
    if not rate_limiter.is_allowed(user_id):
        reset_time = rate_limiter.get_reset_time(user_id)
        logger.warning(f"[Bot Español] User {user_id} rate limited, reset in {reset_time}s")
        await update.message.reply_text(
            f"⏱️ ¡Límite de tasa alcanzado!\n\n"
            f"Has usado tus 5 solicitudes para esta hora.\n"
            f"Por favor, intenta de nuevo en {reset_time // 60} minutos."
        )
        return ConversationHandler.END
    
    # Track usage (but don't fail if tracking fails)
    user = update.effective_user
    try:
        await track_usage_google_sheets(
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            topic=topic,
            level=level
        )
    except Exception as e:
        logger.error(f"[Bot Español] Failed to track usage, continuing anyway: {e}")
    
    await update.message.chat.send_action(action="typing")
    
    # Get level configuration
    level_config = LEVEL_CONFIGS[level]
    
    progress_msg = await update.message.reply_text(
        f"📚 Materiales para tu tema '{topic[:20]}...'...\n"
        f"🏆 Nivel: {level} ({level_config['description']})\n\n"
        f"⏳ Progreso: 0/5\n"
        f"⬜⬜⬜⬜⬜\n"
        f"Inicializando..."
    )
    
    # Progress tracking
    async def update_progress(step, message):
        progress_bar = "🟩" * step + "⬜" * (5 - step)
        try:
            await progress_msg.edit_text(
                f"📚 Materiales para tu tema '{topic[:20]}...'...\n"
                f"🏆 Nivel: {level} ({level_config['description']})\n\n"
                f"⏳ Progreso: {step}/5\n"
                f"{progress_bar}\n"
                f"{message}"
            )
        except:
            pass
    
    try:
        # Step 1: Generate content with DeepSeek
        await update_progress(1, "🤖 Generando contenido con IA...")
        await update.message.chat.send_action(action="typing")
        
        logger.info(f"[Bot Español] Starting content generation for user {user_id}, level {level}")
        content = generate_content_with_deepseek(topic, level)
        
        if not content:
            logger.error(f"[Bot Español] Empty content returned")
            await update.message.reply_text("❌ Error al generar contenido. Por favor intenta de nuevo.")
            return ConversationHandler.END
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = safe_filename(topic)
        
        # Step 2: Create HTML document
        await update_progress(2, "📄 Creando documento HTML...")
        html_filename, html_content = create_html_document(topic, content, timestamp, level)
        logger.info(f"[Bot Español] HTML document created: {html_filename}")
        
        # Step 3: Generate TTS for main text and opinion texts using Spanish Chirp3 HD voices
        await update_progress(3, f"🎧 Generando audio de narración (Chirp3 HD Español, {int(level_config['speaking_rate']*100)}% velocidad)...")
        await update.message.chat.send_action(action="record_voice")

        text_mapping = {
            "Texto_Principal.mp3": content['main_text'],
            "Reacción_Positiva.mp3": content['opinion_texts']['positive'],
            "Reacción_Crítica.mp3": content['opinion_texts']['negative'],
            "Reacción_Equilibrada.mp3": content['opinion_texts']['mixed']
        }

        # Select 4 random Spanish Chirp3 voices for this level
        selected_voices = random.sample(level_config['chirp_voices'], min(4, len(level_config['chirp_voices'])))
        logger.info(f"[Bot Español] Selected Spanish Chirp3 voices for level {level}: {selected_voices}")
        
        audio_tasks = []
        for i, (filename, text) in enumerate(text_mapping.items()):
            voice = selected_voices[i % len(selected_voices)]
            audio_tasks.append(generate_tts_chirp3_async(
                text, 
                voice, 
                speaking_rate=level_config['speaking_rate']
            ))

        logger.info(f"[Bot Español] Generating {len(audio_tasks)} Spanish Chirp3 narration files for level {level}...")
        audio_results = await asyncio.gather(*audio_tasks, return_exceptions=True)
        
        narration_files = []
        for i, (filename, _) in enumerate(text_mapping.items()):
            audio_data = audio_results[i]
            if not isinstance(audio_data, Exception) and audio_data:
                audio_buffer = BytesIO(audio_data)
                audio_buffer.name = filename
                narration_files.append((filename, audio_buffer))
                logger.info(f"[Bot Español] ✅ Chirp3 audio generated for level {level}: {filename}")
            else:
                logger.error(f"[Bot Español] ❌ Chirp3 TTS failed for {filename}: {audio_data}")

        # Step 4: Generate Anki vocabulary file with Spanish Wavenet TTS
        await update_progress(4, f"🎵 Generando TTS para expresiones de Anki ({level_config['wavenet_voice']})...")
        await update.message.chat.send_action(action="record_voice")

        async def vocab_progress(current, total):
            if current % 3 == 0:
                await update_progress(4, f"🎵 Generando TTS para Anki... ({current}/{total})")

        vocab_filename, vocab_content, audio_files = await create_vocabulary_file_with_tts(
            content['collocations'], 
            safe_topic, 
            level_config,
            progress_callback=vocab_progress
        )
        
        if not audio_files:
            logger.error(f"[Bot Español] No Anki audio files generated!")
            await update.message.reply_text("⚠️ Advertencia: No se pudo generar TTS para las tarjetas de Anki.")
        else:
            logger.info(f"[Bot Español] ✅ Generated {len(audio_files)} Anki TTS files for level {level}")

        # Step 5: Create ZIP package
        await update_progress(5, "📦 Creando paquete ZIP...")
        zip_filename, zip_buffer = create_zip_package(
            vocab_filename, vocab_content, audio_files, html_filename, html_content, topic, timestamp
        )
        logger.info(f"[Bot Español] ZIP package created: {zip_filename}")

        # === Send files in order ===
        
        # 1. Send HTML document
        html_file = BytesIO(html_content.encode('utf-8'))
        html_file.name = html_filename
        await update.message.reply_document(
            document=html_file,
            filename=html_filename,
            caption=f"📄 Abre este documento para ver tus textos del tema y lista de vocabulario (Nivel {level})"
        )
        logger.info(f"[Bot Español] Sent HTML document")

        # 2. Instructional message
        await update.message.reply_text(
            f"👆 Puedes escuchar los textos del documento reproduciendo el audio a continuación 👇\n"
            f"🎧 Velocidad: {int(level_config['speaking_rate']*100)}%"
        )

        # 3. Send narration audio files (Spanish Chirp3)
        if narration_files:
            for filename, audio_buffer in narration_files:
                await update.message.reply_audio(audio=audio_buffer, filename=filename)
                logger.info(f"[Bot Español] Sent audio: {filename}")
        else:
            await update.message.reply_text("⚠️ No se pudo generar el audio de narración.")

        # 4. Emoji separator
        await update.message.reply_text("••• 💭 •••")

        # 5. Anki instructions in Spanish
        await update.message.reply_text(
            "📇 Si usas Anki, importa el documento de texto a continuación en Anki, "
            "y coloca los archivos de audio de la carpeta ZIP en tu carpeta `collection.media` de Anki."
        )

        # 6. Send Anki .txt file
        anki_file = BytesIO(vocab_content.encode('utf-8'))
        anki_file.name = f"importar_anki_{level}.txt"
        await update.message.reply_document(
            document=anki_file,
            filename=f"importar_anki_{level}.txt"
        )
        logger.info(f"[Bot Español] Sent Anki import file")

        # 7. Send ZIP package
        zip_file_obj = BytesIO(zip_buffer.getvalue())
        zip_file_obj.name = zip_filename
        await update.message.reply_document(
            document=zip_file_obj,
            filename=zip_filename
        )
        logger.info(f"[Bot Español] Sent ZIP package")
        
        # Final summary in Spanish
        file_size = zip_buffer.getbuffer().nbytes
        logger.info(f"[Bot Español] ✅ Successfully completed request for user {user_id}, level {level}")
        await update.message.reply_text(
            f"✅ ¡Todos los materiales generados!\n\n"
            f"📊 Resumen:\n"
            f"• Nivel: {level} ({level_config['description']})\n"
            f"• Expresiones: {len(content['collocations'])}\n"
            f"• Archivos TTS para Anki: {len(audio_files)}\n"
            f"• Audios de narración: {len(narration_files)}\n"
            f"• Velocidad de audio: {int(level_config['speaking_rate']*100)}%\n"
            f"• Tamaño ZIP: {file_size / 1024 / 1024:.2f}MB"
        )
        
    except Exception as e:
        error_msg = f"❌ Error inesperado: {str(e)[:200]}"
        logger.error(f"[Bot Español] ERROR for user {user_id}, level {level}: {type(e).__name__}: {str(e)}", exc_info=True)
        await update.message.reply_text(error_msg)
    
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the conversation"""
    await update.message.reply_text("Operación cancelada.")
    return ConversationHandler.END

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - SPANISH VERSION"""
    await update.message.reply_text(
        """¡Bienvenido al Spanish Learning Bot! 🎯

Por favor, dame un tema sobre el que quieras hablar:

Sé específico, por ejemplo:

NO - Cómo podemos usar la IA en los negocios ( = demasiado general)

BUENO = ¿Cómo pueden los no programadores que trabajan en una empresa de TI usar la IA?

Algunos ejemplos de temas:
- "Cómo ha estado cambiando X"
- "Qué está pasando a finales de 2025 con ..."
- "¿Es X mejor que Y?"
- "Predicciones para X en 2026"
- "Cómo ..."
- "¿Por qué la gente...?"

¡Escribe tu tema ahora! Luego te preguntaré por el nivel (B1, B2, C1 o C2). 🇪🇸
"""
    )
    return TOPIC

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command - SPANISH VERSION"""
    user_id = update.effective_user.id
    reset_time = rate_limiter.get_reset_time(user_id)
    
    level_descriptions = "\n".join([f"• {level} - {config['description']}" for level, config in LEVEL_CONFIGS.items()])
    
    help_text = (
        "📖 **Cómo Usar:**\n\n"
        "1. Envíame un tema (máx 100 caracteres)\n"
        "2. Selecciona el nivel (B1, B2, C1 o C2)\n"
        "3. Recibirás:\n"
        "   • Documento HTML con todos los materiales\n"
        "   • 4 archivos de audio de narración (voces Chirp3 HD en español)\n"
        "   • Archivo .txt para importar en Anki\n"
        "   • Paquete ZIP con archivos TTS para Anki\n\n"
        "📊 **Niveles Disponibles:**\n"
        f"{level_descriptions}\n\n"
        "📦 **Para Anki:**\n"
        "   • Extrae los archivos MP3 del ZIP a la carpeta collection.media\n"
        "   • Importa el archivo .txt en Anki\n\n"
        "⚡ **Límite de Tasa:** 5 solicitudes/hora\n"
        "🇪🇸 **Idioma:** Español"
    )
    
    if reset_time > 0:
        help_text += f"\n⏱️ Se restablece en {reset_time // 60} min"
    
    await update.message.reply_text(help_text, parse_mode='Markdown')
    return ConversationHandler.END

async def handle_direct_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle direct messages (fallback)"""
    await update.message.reply_text(
        "Por favor usa /start para comenzar o /help para ayuda.\n\n"
        "Primero escribe un tema, luego seleccionarás el nivel (B1, B2, C1 o C2)."
    )

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🤖 Iniciando Spanish Learning Telegram Bot con selección de nivel")
    logger.info("=" * 60)
    logger.info(f"Niveles disponibles:")
    for level, level_config in LEVEL_CONFIGS.items():
        logger.info(f"  - {level}: {level_config['description']} (velocidad: {int(level_config['speaking_rate']*100)}%, voz: {level_config['wavenet_voice']})")
    logger.info(f"Límite de tasa: {config.RATE_LIMIT_REQUESTS} solicitudes por {config.RATE_LIMIT_WINDOW}s")
    logger.info("=" * 60)
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_topic_input)],
            LEVEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_level_input)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_direct_message))
    
    logger.info("✅ El bot está ejecutándose y listo para aceptar mensajes...")
    application.run_polling()
