import logging

import yaml

from app.engine.loaders.file import FileLoaderConfig, get_file_documents

logger = logging.getLogger(__name__)


def load_configs():
    with open("config/loaders.yaml") as f:
        configs = yaml.safe_load(f)
    return configs


def get_documents():
    documents = []
    config = load_configs()
    for loader_type, loader_config in config.items():
        logger.info(f"Loading documents from loader: {loader_type}, config: {loader_config}")
        match loader_type:
            case "file":
                document = get_file_documents(FileLoaderConfig(**loader_config))
            case _:
                raise ValueError(f"Invalid loader type: {loader_type}")
        documents.extend(document)

    return documents
