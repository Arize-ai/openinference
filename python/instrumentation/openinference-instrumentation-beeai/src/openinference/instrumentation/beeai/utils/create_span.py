# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Any, Dict, Optional, TypedDict

from opentelemetry.trace import StatusCode


class SpanContext(TypedDict):
    span_id: str


class SpanStatus(TypedDict):
    code: StatusCode
    message: str


class SpanAttributes(TypedDict, total=False):
    target: str
    metadata: Optional[Dict[str, Any]]
    data: Optional[Dict[str, Any]]
    exception_message: Optional[str]
    exception_type: Optional[str]


class FrameworkSpan(TypedDict):
    name: str
    attributes: SpanAttributes
    context: SpanContext
    parent_id: Optional[str]
    status: SpanStatus
    start_time: int
    end_time: int


def create_span(
    *,
    id: str,
    name: str,
    target: str,
    started_at: int,
    ctx: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    parent: Optional[Dict[str, str]] = None,
) -> FrameworkSpan:
    attributes: SpanAttributes = {
        "target": target,
        "metadata": ctx if ctx else None,
        "data": data if data else None,
    }

    if error:
        attributes["exception_message"] = error
        attributes["exception_type"] = "FrameworkError"

    return {
        "name": name,
        "attributes": attributes,
        "context": {"span_id": id},
        "parent_id": parent["id"] if parent else None,
        "status": {
            "code": StatusCode.ERROR if error else StatusCode.OK,
            "message": error or "",
        },
        "start_time": started_at,
        "end_time": time.time_ns(),
    }
