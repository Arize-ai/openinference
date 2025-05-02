from dotenv import load_dotenv

load_dotenv()

import logging  # noqa

from llama_index.core.indices import (  # noqa
    VectorStoreIndex,
)

from app.engine.constants import STORAGE_DIR  # noqa
from app.engine.loaders import get_documents  # noqa
from app.settings import init_settings  # noqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_datasource():
    logger.info("Creating new index")
    # load the documents and create the index
    documents = get_documents()
    index = VectorStoreIndex.from_documents(
        documents,
    )
    # store it for later
    index.storage_context.persist(STORAGE_DIR)
    logger.info(f"Finished creating new index. Stored in {STORAGE_DIR}")


if __name__ == "__main__":
    init_settings()
    generate_datasource()
