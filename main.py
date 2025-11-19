from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import langcodes
from gradio_client import Client
import os
import asyncio
from functools import lru_cache
import logging


# Initialize Gradio client (lazy)
translator = None
GRADIO_URL = os.getenv("GRADIO_URL", "KMayanja/testTranslate")
HF_TOKEN = os.getenv("HF_TOKEN")

@lru_cache(maxsize=1)
def _create_translator():
    if HF_TOKEN:
        return Client(GRADIO_URL, hf_token=HF_TOKEN)
    return Client(GRADIO_URL)

def get_translator():
    try:
        client = _create_translator()
        logger.info("Gradio client initialized: %s", GRADIO_URL)
        return client
    except Exception as e:
        logger.exception("Failed to initialize Gradio client")
        raise RuntimeError(f"Failed to initialize Gradio client: {e}")

# Input schema for translation endpoint
class TranslationInput(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    max_length: Optional[int] = 512

# Validate language codes
def validate_language(lang: str) -> bool:
    try:
        langcodes.Language.get(lang)
        return True
    except langcodes.LanguageTagError:
        return False

# Initialize FastAPI app
app = FastAPI(title="Translation API", description="API for translations using Gradio client")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://translate-app-web-ui.vercel.app", "http://localhost:3000"],  # remove trailing slash
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# explicit OPTIONS handler for serverless preflight
@app.options("/translate")
async def translate_options():
    return Response(status_code=200)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "client_loaded": translator is not None}

# Translation endpoint
@app.post("/translate")
async def translate(input: TranslationInput):
    try:
        client = get_translator()
    except RuntimeError as e:
        # surface the error in logs (don't expose stack in prod)
        logger.error("Translator init error: %s", e)
        raise HTTPException(status_code=503, detail="Translation service unavailable")

    # Validate language codes
    if not validate_language(input.source_lang):
        raise HTTPException(status_code=400, detail=f"Invalid source language code: {input.source_lang}")
    if not validate_language(input.target_lang):
        raise HTTPException(status_code=400, detail=f"Invalid target language code: {input.target_lang}")

    try:
        result = await asyncio.to_thread(
            client.predict,
            text=input.text,
            source_language=input.source_lang,
            target_language=input.target_lang,
            api_name="/predict"
        )
        return {
            "translated_text": result,
            "source_lang": input.source_lang,
            "target_lang": input.target_lang
        }
    except Exception as e:
        logger.exception("predict() failed")
        raise HTTPException(status_code=502, detail="Upstream translation failed")

# Add a warm endpoint you can call on deploy to pre-initialize the client
@app.post("/warm")
async def warm():
    try:
        get_translator()
        return {"status": "warmed"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Translation API. Use /translate for translations or /health to check status."}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("translation-api")