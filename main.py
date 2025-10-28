from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import langcodes
from gradio_client import Client
from contextlib import asynccontextmanager


# Initialize Gradio client
translator = None
GRADIO_URL = "KMayanja/testTranslate"

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

# Initialize client on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global translator
    try:
        translator = Client(GRADIO_URL)
        yield  # must yield control for an async context manager
    finally:
        translator = None

# Initialize FastAPI app
app = FastAPI(title="Translation API", description="API for translations using Gradio client", lifespan=lifespan)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "client_loaded": translator is not None}

# Translation endpoint
@app.post("/translate")
async def translate(input: TranslationInput):
    if not translator:
        raise HTTPException(status_code=503, detail="Gradio client not initialized")

    # Validate language codes
    if not validate_language(input.source_lang):
        raise HTTPException(status_code=400, detail=f"Invalid source language code: {input.source_lang}")
    if not validate_language(input.target_lang):
        raise HTTPException(status_code=400, detail=f"Invalid target language code: {input.target_lang}")

    try:
        result = translator.predict(
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

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Translation API. Use /translate for translations or /health to check status."}