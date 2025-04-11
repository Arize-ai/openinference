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

from openinference.instrumentation.beeai.utils.id_name_manager import IdNameManager


def test_should_pair_events_by_run_id_structure() -> None:
    id_name_manager = IdNameManager()

    # set 3 events for run level 1
    id_name_manager.get_ids(id="event-1", path="react.run.start", run_id="run-1")
    id_name_manager.get_ids(
        id="event-2",
        path="react.run.something",
        run_id="run-1",
    )
    id_name_manager.get_ids(
        id="event-3",
        path="react.run.something",
        run_id="run-1",
    )

    # add event from nested run
    result = id_name_manager.get_ids(
        id="event-nested-1",
        path="ll.nested",
        run_id="run-2",
        parent_run_id="run-1",
    )

    assert result["spanId"] == "ll.nested-1"
    assert result["parentSpanId"] == "react.run.something-2"


def test_should_use_group_id() -> None:
    id_name_manager = IdNameManager()

    # Top level events grouped by groupId
    result_1 = id_name_manager.get_ids(
        id="event-1",
        path="react.run.start",
        run_id="run-1",
        group_id="iteration-1",
    )
    assert result_1 == {
        "spanId": "react.run.start-1",
        "parentSpanId": "iteration-1",
    }

    result_2 = id_name_manager.get_ids(
        id="event-2",
        path="react.run.something",
        run_id="run-1",
        group_id="iteration-1",
    )
    assert result_2 == {
        "spanId": "react.run.something-1",
        "parentSpanId": "iteration-1",
    }

    # nested event with standard parentId
    result_3 = id_name_manager.get_ids(
        id="tool-1",
        path="react.tool.start",
        run_id="run-2",
        parent_run_id="run-1",
        group_id="iteration-1",
    )
    assert result_3 == {
        "spanId": "react.tool.start-1",
        "parentSpanId": "react.run.something-1",
    }
