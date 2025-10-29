from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import langcodes
from gradio_client import Client
import os
import asyncio
from functools import lru_cache


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
        return _create_translator()
    except Exception as e:
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
    # allow_origins=["https://your-nextjs-app.vercel.app", "http://localhost:3000"],  # Add your Next.js app’s domains
    # allow_credentials=True,
    # allow_methods=["*"],
    # allow_headers=["*"],
)

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
        raise HTTPException(status_code=503, detail=str(e))

    # Validate language codes
    if not validate_language(input.source_lang):
        raise HTTPException(status_code=400, detail=f"Invalid source language code: {input.source_lang}")
    if not validate_language(input.target_lang):
        raise HTTPException(status_code=400, detail=f"Invalid target language code: {input.target_lang}")

    try:
        # Run blocking network call off the event loop to reduce async blocking
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
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

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