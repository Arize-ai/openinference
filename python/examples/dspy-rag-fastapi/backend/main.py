from dotenv import load_dotenv

load_dotenv()

import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.rag import rag_router
from instrument import instrument


do_not_instrument = os.getenv("INSTRUMENT_DSPY", "true") == "false"
if not do_not_instrument:
    instrument()

app = FastAPI(title="DSPy x FastAPI")


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

app.include_router(rag_router, prefix="/api/rag", tags=["RAG"])

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", reload=True)
