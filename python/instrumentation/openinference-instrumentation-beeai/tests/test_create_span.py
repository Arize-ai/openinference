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

from opentelemetry.trace import StatusCode

from openinference.instrumentation.beeai.utils.create_span import create_span


def test_create_span_basic() -> None:
    start_time = time.time_ns()
    result = create_span(id="span-1", name="test-span", target="test-target", started_at=start_time)

    assert result["name"] == "test-span"
    assert result["context"]["span_id"] == "span-1"
    assert result["attributes"]["target"] == "test-target"
    assert result["attributes"]["metadata"] is None
    assert result["attributes"]["data"] is None
    assert result["parent_id"] is None
    assert result["status"]["code"] == StatusCode.OK
    assert result["status"]["message"] == ""
    assert result["start_time"] == start_time
    assert result["end_time"] >= start_time


def test_create_span_with_error() -> None:
    start_time = time.time_ns()
    result = create_span(
        id="span-2",
        name="error-span",
        target="failing-task",
        started_at=start_time,
        error="Something went wrong",
    )

    assert result["status"]["code"] == StatusCode.ERROR
    assert result["status"]["message"] == "Something went wrong"
    assert result["attributes"]["exception_message"] == "Something went wrong"
    assert result["attributes"]["exception_type"] == "FrameworkError"


def test_create_span_with_metadata_and_data() -> None:
    ctx = {"user": "tester"}
    data = {"result": "42"}
    start_time = time.time_ns()

    result = create_span(
        id="span-3",
        name="span-with-data",
        target="analytics",
        started_at=start_time,
        ctx=ctx,
        data=data,
    )

    assert result["attributes"]["metadata"] == ctx
    assert result["attributes"]["data"] == data


def test_create_span_with_parent() -> None:
    start_time = time.time_ns()

    result = create_span(
        id="child-span",
        name="child",
        target="child-process",
        started_at=start_time,
        parent={"id": "parent-span"},
    )

    assert result["parent_id"] == "parent-span"
