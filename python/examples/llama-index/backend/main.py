from dotenv import load_dotenv

load_dotenv()

import logging
import os

import uvicorn
from app.api.routers.chat import chat_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from instrument import instrument

do_not_instrument = os.getenv("INSTRUMENT_LLAMA_INDEX", "true") == "false"
if not do_not_instrument:
    instrument()

app = FastAPI()
environment = os.getenv("ENVIRONMENT", "dev")  # Default to 'development' if not set


if environment == "dev":
    logger = logging.getLogger("uvicorn")
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(chat_router, prefix="/api/chat")


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)
