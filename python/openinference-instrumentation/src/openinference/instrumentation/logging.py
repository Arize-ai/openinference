import logging


class CustomFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelname

        return f"{level}: {record.getMessage()}"


logger = logging.getLogger(__name__)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set the custom formatter to the handler
    console_handler.setFormatter(CustomFormatter())

    logger.addHandler(console_handler)


logger.propagate = False
