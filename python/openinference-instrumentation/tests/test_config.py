import os
import random

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.config import (
    DEFAULT_BASE64_IMAGE_MAX_LENGTH,
    DEFAULT_HIDE_INPUT_IMAGES,
    DEFAULT_HIDE_INPUT_MESSAGES,
    DEFAULT_HIDE_INPUT_TEXT,
    DEFAULT_HIDE_INPUTS,
    DEFAULT_HIDE_OUTPUT_MESSAGES,
    DEFAULT_HIDE_OUTPUT_TEXT,
    DEFAULT_HIDE_OUTPUTS,
    OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
    OPENINFERENCE_HIDE_INPUT_IMAGES,
    OPENINFERENCE_HIDE_INPUT_MESSAGES,
    OPENINFERENCE_HIDE_INPUT_TEXT,
    OPENINFERENCE_HIDE_INPUTS,
    OPENINFERENCE_HIDE_OUTPUT_MESSAGES,
    OPENINFERENCE_HIDE_OUTPUT_TEXT,
    OPENINFERENCE_HIDE_OUTPUTS,
)


def test_default_settings() -> None:
    config = TraceConfig()
    assert config.hide_inputs == DEFAULT_HIDE_INPUTS
    assert config.hide_outputs == DEFAULT_HIDE_OUTPUTS
    assert config.hide_input_messages == DEFAULT_HIDE_INPUT_MESSAGES
    assert config.hide_output_messages == DEFAULT_HIDE_OUTPUT_MESSAGES
    assert config.hide_input_images == DEFAULT_HIDE_INPUT_IMAGES
    assert config.hide_input_text == DEFAULT_HIDE_INPUT_TEXT
    assert config.hide_output_text == DEFAULT_HIDE_OUTPUT_TEXT
    assert config.base64_image_max_length == DEFAULT_BASE64_IMAGE_MAX_LENGTH


def test_settings_from_env_vars() -> None:
    os.environ[OPENINFERENCE_HIDE_INPUTS] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_OUTPUTS] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_INPUT_MESSAGES] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_OUTPUT_MESSAGES] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_INPUT_IMAGES] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_INPUT_TEXT] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_OUTPUT_TEXT] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH] = str(10_000)

    config = TraceConfig()
    assert config.hide_inputs == bool(os.getenv(OPENINFERENCE_HIDE_INPUTS))
    assert config.hide_outputs == bool(os.getenv(OPENINFERENCE_HIDE_OUTPUTS))
    assert config.hide_input_messages == bool(os.getenv(OPENINFERENCE_HIDE_INPUT_MESSAGES))
    assert config.hide_output_messages == bool(os.getenv(OPENINFERENCE_HIDE_OUTPUT_MESSAGES))
    assert config.hide_input_images == bool(os.getenv(OPENINFERENCE_HIDE_INPUT_IMAGES))
    assert config.hide_input_text == bool(os.getenv(OPENINFERENCE_HIDE_INPUT_TEXT))
    assert config.hide_output_text == bool(os.getenv(OPENINFERENCE_HIDE_OUTPUT_TEXT))
    assert config.base64_image_max_length == int(
        os.getenv(OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH, default=0)
    )


def test_settings_from_code() -> None:
    os.environ[OPENINFERENCE_HIDE_INPUTS] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_OUTPUTS] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_INPUT_MESSAGES] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_OUTPUT_MESSAGES] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_INPUT_IMAGES] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_INPUT_TEXT] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_HIDE_OUTPUT_TEXT] = str(random.choice([True, False]))
    os.environ[OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH] = str(10_000)

    # code specified values must ignore the environment variables
    hide_inputs = random.choice([True, False])
    hide_outputs = random.choice([True, False])
    hide_input_messages = random.choice([True, False])
    hide_output_messages = random.choice([True, False])
    hide_input_images = random.choice([True, False])
    hide_input_text = random.choice([True, False])
    hide_output_text = random.choice([True, False])
    base64_image_max_length = 10_000
    config = TraceConfig(
        hide_inputs=hide_inputs,
        hide_outputs=hide_outputs,
        hide_input_messages=hide_input_messages,
        hide_output_messages=hide_output_messages,
        hide_input_images=hide_input_images,
        hide_input_text=hide_input_text,
        hide_output_text=hide_output_text,
        base64_image_max_length=base64_image_max_length,
    )
    assert config.hide_inputs == hide_inputs
    assert config.hide_outputs == hide_outputs
    assert config.hide_input_messages == hide_input_messages
    assert config.hide_output_messages == hide_output_messages
    assert config.hide_input_images == hide_input_images
    assert config.hide_input_text == hide_input_text
    assert config.hide_output_text == hide_output_text
    assert config.base64_image_max_length == base64_image_max_length
