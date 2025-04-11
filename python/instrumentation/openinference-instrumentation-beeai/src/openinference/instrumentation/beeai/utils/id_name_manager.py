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

from typing import Dict, Optional, TypedDict


class GetIdsResult(TypedDict):
    spanId: str
    parentSpanId: Optional[str]


class IdNameManager:
    def __init__(self) -> None:
        self._id_names_counter: Dict[str, int] = {}
        self._id_name_map: Dict[str, str] = {}
        self._run_id_map: Dict[str, str] = {}

    def _span_id_generator(self, name: str) -> str:
        count = self._id_names_counter.get(name, 0)
        self._id_names_counter[name] = count + 1
        return f"{name}-{self._id_names_counter[name]}"

    def get_ids(
        self,
        *,
        path: str,
        id: str,
        run_id: str,
        parent_run_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> GetIdsResult:
        self._run_id_map[run_id] = id

        span_id = self._span_id_generator(path)
        self._id_name_map[id] = span_id

        parent_span_id = None
        if parent_run_id:
            parent_event_id = self._run_id_map.get(parent_run_id, "")
            parent_span_id = self._id_name_map.get(parent_event_id)
        elif group_id:
            parent_span_id = group_id

        return {
            "spanId": span_id,
            "parentSpanId": parent_span_id,
        }
