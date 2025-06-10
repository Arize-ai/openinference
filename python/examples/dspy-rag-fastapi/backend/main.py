from dotenv import load_dotenv

load_dotenv()

import logging  # noqa
import os  # noqa

import uvicorn  # noqa
from app.api.routers.rag import rag_router  # noqa
from fastapi import FastAPI  # noqa
from fastapi.middleware.cors import CORSMiddleware  # noqa
from instrument import instrument  # noqa

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
