/**
 * Copyright 2025 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { SpanStatusCode, TimeInput } from "@opentelemetry/api";
import { FrameworkSpan } from "../types";
import { isEmpty } from "remeda";

interface CreateSpanProps {
  id: string;
  name: string;
  target: string;
  startedAt: TimeInput;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ctx?: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data?: any;
  error?: string;
  parent?: { id: string };
}

export function createSpan({
  target,
  name,
  data,
  error,
  ctx,
  parent,
  id,
  startedAt,
}: CreateSpanProps): FrameworkSpan {
  return {
    name: name,
    attributes: {
      target,
      metadata: ctx && !isEmpty(ctx) ? { ...ctx } : undefined,
      ...(error && {
        "exception.message": error,
        "exception.type": "FrameworkError",
      }),
      data: data && !isEmpty(data) ? { ...data } : null,
    },
    context: {
      span_id: id,
    },
    parent_id: parent?.id,
    status: {
      code: error ? SpanStatusCode.ERROR : SpanStatusCode.OK,
      message: error ? error : "",
    },
    start_time: startedAt,
    end_time: performance.now(),
  };
}
