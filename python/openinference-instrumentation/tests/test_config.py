import os
from typing import Optional

import pytest
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


@pytest.mark.parametrize("hide_inputs", [False, True])
@pytest.mark.parametrize("hide_outputs", [False, True])
@pytest.mark.parametrize("hide_input_messages", [False, True])
@pytest.mark.parametrize("hide_output_messages", [False, True])
@pytest.mark.parametrize("hide_input_images", [False, True])
@pytest.mark.parametrize("hide_input_text", [False, True])
@pytest.mark.parametrize("hide_output_text", [False, True])
@pytest.mark.parametrize("base64_image_max_length", [10_000])
def test_settings_from_env_vars_and_code(
    hide_inputs: bool,
    hide_outputs: bool,
    hide_input_messages: bool,
    hide_output_messages: bool,
    hide_input_images: bool,
    hide_input_text: bool,
    hide_output_text: bool,
    base64_image_max_length: int,
) -> None:
    # First part of the test verifies that environment variables are read correctly
    os.environ[OPENINFERENCE_HIDE_INPUTS] = str(hide_inputs)
    os.environ[OPENINFERENCE_HIDE_OUTPUTS] = str(hide_outputs)
    os.environ[OPENINFERENCE_HIDE_INPUT_MESSAGES] = str(hide_input_messages)
    os.environ[OPENINFERENCE_HIDE_OUTPUT_MESSAGES] = str(hide_output_messages)
    os.environ[OPENINFERENCE_HIDE_INPUT_IMAGES] = str(hide_input_images)
    os.environ[OPENINFERENCE_HIDE_INPUT_TEXT] = str(hide_input_text)
    os.environ[OPENINFERENCE_HIDE_OUTPUT_TEXT] = str(hide_output_text)
    os.environ[OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH] = str(base64_image_max_length)

    config = TraceConfig()
    assert config.hide_inputs is parse_bool_from_env(OPENINFERENCE_HIDE_INPUTS)
    assert config.hide_outputs is parse_bool_from_env(OPENINFERENCE_HIDE_OUTPUTS)
    assert config.hide_input_messages is parse_bool_from_env(OPENINFERENCE_HIDE_INPUT_MESSAGES)
    assert config.hide_output_messages is parse_bool_from_env(OPENINFERENCE_HIDE_OUTPUT_MESSAGES)
    assert config.hide_input_images is parse_bool_from_env(OPENINFERENCE_HIDE_INPUT_IMAGES)
    assert config.hide_input_text is parse_bool_from_env(OPENINFERENCE_HIDE_INPUT_TEXT)
    assert config.hide_output_text is parse_bool_from_env(OPENINFERENCE_HIDE_OUTPUT_TEXT)
    assert config.base64_image_max_length == int(
        os.getenv(OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH, default=-1)
    )

    # This next part of the text verifies that the code specified values overwrite
    # the configuration from the environment variables
    new_base64_image_max_length = base64_image_max_length + 500
    new_hide_inputs = not hide_inputs
    new_hide_outputs = not hide_outputs
    new_hide_input_messages = not hide_input_messages
    new_hide_output_messages = not hide_output_messages
    new_hide_input_images = not hide_input_images
    new_hide_input_text = not hide_input_text
    new_hide_output_text = not hide_output_text
    config = TraceConfig(
        hide_inputs=new_hide_inputs,
        hide_outputs=new_hide_outputs,
        hide_input_messages=new_hide_input_messages,
        hide_output_messages=new_hide_output_messages,
        hide_input_images=new_hide_input_images,
        hide_input_text=new_hide_input_text,
        hide_output_text=new_hide_output_text,
        base64_image_max_length=new_base64_image_max_length,
    )
    assert config.hide_inputs is new_hide_inputs
    assert config.hide_outputs is new_hide_outputs
    assert config.hide_input_messages is new_hide_input_messages
    assert config.hide_output_messages is new_hide_output_messages
    assert config.hide_input_images is new_hide_input_images
    assert config.hide_input_text is new_hide_input_text
    assert config.hide_output_text is new_hide_output_text
    assert config.base64_image_max_length == new_base64_image_max_length


def parse_bool_from_env(env_var: str) -> Optional[bool]:
    env_value = os.getenv(env_var)
    if isinstance(env_value, str) and env_value.lower() == "true":
        return True
    elif isinstance(env_value, str) and env_value.lower() == "false":
        return False
    else:
        return None
