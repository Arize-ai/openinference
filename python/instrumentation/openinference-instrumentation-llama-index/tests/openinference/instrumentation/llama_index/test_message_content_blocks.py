import base64

from llama_index.core.base.llms.types import ImageBlock, TextBlock

from openinference.instrumentation.llama_index._handler import (
    _get_attributes_from_image_block,
    _get_attributes_from_text_block,
)
from openinference.semconv.trace import ImageAttributes, MessageContentAttributes


def test_get_attributes_from_text_block() -> None:
    """Test attributes extraction from TextBlock."""
    text_block = TextBlock(text="Hello, world!")
    attributes = dict(_get_attributes_from_text_block(text_block))

    assert len(attributes) == 2
    assert attributes[f"{MESSAGE_CONTENT_TYPE}"] == "text"
    assert attributes[f"{MESSAGE_CONTENT_TEXT}"] == "Hello, world!"

    # Test with prefix
    attributes_with_prefix = dict(_get_attributes_from_text_block(text_block, prefix="test."))
    assert attributes_with_prefix[f"test.{MESSAGE_CONTENT_TYPE}"] == "text"
    assert attributes_with_prefix[f"test.{MESSAGE_CONTENT_TEXT}"] == "Hello, world!"


def test_get_attributes_from_image_block_with_binary() -> None:
    """Test attributes extraction from ImageBlock with binary data."""

    # Create a simple image block with binary data
    image_data = b"fake_image_data"
    image_block = ImageBlock(image=image_data, image_mimetype="image/jpeg")

    attributes = dict(_get_attributes_from_image_block(image_block))

    assert len(attributes) == 2
    assert (
        attributes[f"{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}"]
        == f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}"
    )
    assert attributes[f"{MESSAGE_CONTENT_TYPE}"] == "image"


def test_get_attributes_from_image_block_with_url() -> None:
    """Test attributes extraction from ImageBlock with URL."""
    image_block = ImageBlock(url="https://example.com/image.jpg")

    attributes = dict(_get_attributes_from_image_block(image_block))

    assert len(attributes) == 2
    assert attributes[f"{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}"] == "https://example.com/image.jpg"
    assert attributes[f"{MESSAGE_CONTENT_TYPE}"] == "image"


MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE

IMAGE_URL = ImageAttributes.IMAGE_URL
