"""Base64 validation and decoding utilities for embeddings."""

import base64
import binascii
import re
import struct
from typing import List, Optional

# Regex to match valid base64 format: full groups of 4 valid chars,
# optional padded ending (2 chars + == or 3 chars + =), no stray '=' elsewhere.
_BASE64_PATTERN = re.compile(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$")


def decode_base64_float32(value: str) -> Optional[List[float]]:
    """
    Decodes a base64-encoded float32 vector.
    Returns a list of floats, or None if invalid or cannot be decoded.
    """
    if not value:
        return None
    if not _BASE64_PATTERN.match(value):
        return None
    # Compute padding count (0-2, enforced by regex)
    if value.endswith("=="):
        padding_count = 2
    elif value.endswith("="):
        padding_count = 1
    else:
        padding_count = 0
    # Decoded length: (groups * 3 bytes) minus padding adjustment
    val_len = len(value)
    decoded_len = (val_len // 4 * 3) - padding_count
    # Only proceed if >0 and divisible by 4 (float32 compatible)
    if decoded_len <= 0 or decoded_len % 4 != 0:
        return None
    try:
        decoded = base64.b64decode(value)
        # Unpack as float32 values using pre-computed length
        num_floats = decoded_len // 4
        return list(struct.unpack(f"{num_floats}f", decoded))
    except (binascii.Error, struct.error):
        return None
