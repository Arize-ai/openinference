/**
 * Real OpenAI API responses captured from live traffic against `gpt-5.6-luna` and
 * `gpt-5.6-terra`, used to test prompt cache token extraction without hitting the network.
 *
 * Each entry holds two consecutive calls that share a large cacheable prefix:
 *   - `cacheWrite`: the first (cold) call, which writes the prefix to the cache
 *   - `cacheRead`:  the second call, which reads that prefix back from the cache
 *
 * The payloads are verbatim except for the echoed `instructions` field, which was
 * truncated because it contained the multi-thousand-token prefix used to trigger
 * caching. The `usage` blocks are unmodified.
 *
 * Recorded with the same requests as the Python VCR cassettes in
 * python/instrumentation/openinference-instrumentation-openai/tests/openinference/
 * instrumentation/openai/cassettes/test_cache_token_counts/
 *
 * `cache_write_tokens` is not yet present in the OpenAI SDK types, so these payloads
 * are typed structurally rather than with the SDK response types.
 */

type CacheTokenDetails = {
  cached_tokens: number;
  cache_write_tokens: number;
  [key: string]: unknown;
};

type RecordedChatCompletion = {
  usage: {
    prompt_tokens: number;
    prompt_tokens_details: CacheTokenDetails;
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

type RecordedResponse = {
  usage: {
    input_tokens: number;
    input_tokens_details: CacheTokenDetails;
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

type RecordedPair<T> = {
  /** First call with a cold cache: the prefix is written to the cache. */
  cacheWrite: T;
  /** Second call with the same prefix: it is read back from the cache. */
  cacheRead: T;
};

export const realCacheTokenResponses: {
  chatCompletionsLuna: RecordedPair<RecordedChatCompletion>;
  chatCompletionsTerra: RecordedPair<RecordedChatCompletion>;
  responsesLuna: RecordedPair<RecordedResponse>;
  responsesTerra: RecordedPair<RecordedResponse>;
} = {
  chatCompletionsLuna: {
    cacheWrite: {
      id: "chatcmpl-EAfWnybNpH7B822ycaXxUO206Lzgz",
      object: "chat.completion",
      created: 1786211621,
      model: "gpt-5.6-luna",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content:
              "Moonlight fills the pond  \nQuiet reeds bow to the breeze  \nNight holds silver breath",
            refusal: null,
            annotations: [],
          },
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 5395,
        completion_tokens: 131,
        total_tokens: 5526,
        prompt_tokens_details: {
          cached_tokens: 0,
          cache_write_tokens: 5392,
          audio_tokens: 0,
        },
        completion_tokens_details: {
          reasoning_tokens: 105,
          audio_tokens: 0,
          accepted_prediction_tokens: 0,
          rejected_prediction_tokens: 0,
        },
      },
      service_tier: "default",
      system_fingerprint: null,
    },
    cacheRead: {
      id: "chatcmpl-EAfWp2NtziT2z9S1UqkeyBMeUhRXl",
      object: "chat.completion",
      created: 1786211623,
      model: "gpt-5.6-luna",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content:
              "When dusk unfolds its violet-veined delight,  \nThe waking stars ascend the quiet air;  \nThe moon lays silver pathways through the night,  \nAnd whispers old, forgotten dreams are there.  \n\nThe restless day retreats beyond the hill,  \nIts golden footsteps fading into blue;  \nYet in the hush, the world grows strangely still,  \nAs if the earth were listening for you.  \n\nOne tender thought can warm the coldest stone,  \nOne gentle word can calm the troubled sea;  \nAnd hearts that wander far from being home  \nMay find their way through simple sympathy.  \n\nSo let the darkness gather where it must\u2014  \nLove leaves a light no shadow can distrust.",
            refusal: null,
            annotations: [],
          },
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 5395,
        completion_tokens: 177,
        total_tokens: 5572,
        prompt_tokens_details: {
          cached_tokens: 5382,
          cache_write_tokens: 10,
          audio_tokens: 0,
        },
        completion_tokens_details: {
          reasoning_tokens: 32,
          audio_tokens: 0,
          accepted_prediction_tokens: 0,
          rejected_prediction_tokens: 0,
        },
      },
      service_tier: "default",
      system_fingerprint: null,
    },
  },
  chatCompletionsTerra: {
    cacheWrite: {
      id: "chatcmpl-EAfWr1nIj9OrWH0RedwxRuWOeZgwQ",
      object: "chat.completion",
      created: 1786211625,
      model: "gpt-5.6-terra",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content:
              "Moonlight fills the pond  \nTall reeds whisper to the wind  \nNight folds into dawn",
            refusal: null,
            annotations: [],
          },
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 5428,
        completion_tokens: 130,
        total_tokens: 5558,
        prompt_tokens_details: {
          cached_tokens: 0,
          cache_write_tokens: 5425,
          audio_tokens: 0,
        },
        completion_tokens_details: {
          reasoning_tokens: 104,
          audio_tokens: 0,
          accepted_prediction_tokens: 0,
          rejected_prediction_tokens: 0,
        },
      },
      service_tier: "default",
      system_fingerprint: null,
    },
    cacheRead: {
      id: "chatcmpl-EAfWtGwpt1DsJTi7v9JY3oK3ObeLZ",
      object: "chat.completion",
      created: 1786211627,
      model: "gpt-5.6-terra",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content:
              "At dusk, the amber light unthreads the day,  \nAnd folds its warmth within the quiet air;  \nThe restless birds take wing, then drift away,  \nWhile evening gathers softly everywhere.  \n\nA single star awakens in the blue,  \nIts patient fire a promise, small and bright;  \nIt tells the heart that loss may still renew,  \nAnd tender hopes can live within the night.  \n\nSo let the dark descend without dismay;  \nNo shadow wholly silences the dawn.  \nThe moon will keep its watch above the gray,  \nTill gold returns upon the waking lawn.  \n\nFor every fading flame leaves seeds of light,  \nTo bloom again beyond the reach of night.",
            refusal: null,
            annotations: [],
          },
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 5428,
        completion_tokens: 163,
        total_tokens: 5591,
        prompt_tokens_details: {
          cached_tokens: 5415,
          cache_write_tokens: 10,
          audio_tokens: 0,
        },
        completion_tokens_details: {
          reasoning_tokens: 14,
          audio_tokens: 0,
          accepted_prediction_tokens: 0,
          rejected_prediction_tokens: 0,
        },
      },
      service_tier: "default",
      system_fingerprint: null,
    },
  },
  responsesLuna: {
    cacheWrite: {
      id: "resp_0829a999c343a2ab006a776cd33c3481979718739c915a8ce1",
      object: "response",
      created_at: 1786211539,
      status: "completed",
      background: false,
      billing: {
        payer: "developer",
      },
      completed_at: 1786211540,
      error: null,
      frequency_penalty: 0.0,
      incomplete_details: null,
      instructions: "GqpaY1r8Uh9Y1bm5yloT6Mk80fiBwTxpDdTi8vDQ...<truncated cacheable prefix>",
      max_output_tokens: null,
      max_tool_calls: null,
      model: "gpt-5.6-luna",
      moderation: null,
      output: [
        {
          id: "msg_0829a999c343a2ab006a776cd3d53c8197aacf534a2c67d249",
          type: "message",
          status: "completed",
          content: [
            {
              type: "output_text",
              annotations: [],
              logprobs: [],
              text: "Morning mist rises  \nA sparrow greets the new light  \nDay unfolds softly",
            },
          ],
          phase: "final_answer",
          role: "assistant",
        },
      ],
      parallel_tool_calls: true,
      presence_penalty: 0.0,
      previous_response_id: null,
      prompt_cache_key: null,
      prompt_cache_retention: "24h",
      reasoning: {
        context: "all_turns",
        effort: "medium",
        mode: "standard",
        summary: null,
      },
      safety_identifier: null,
      service_tier: "default",
      store: true,
      temperature: 1.0,
      text: {
        format: {
          type: "text",
        },
        verbosity: "medium",
      },
      tool_choice: "auto",
      tool_usage: {
        image_gen: {
          input_tokens: 0,
          input_tokens_details: {
            image_tokens: 0,
            text_tokens: 0,
          },
          output_tokens: 0,
          output_tokens_details: {
            image_tokens: 0,
            text_tokens: 0,
          },
          total_tokens: 0,
        },
        web_search: {
          num_requests: 0,
        },
      },
      tools: [],
      top_logprobs: 0,
      top_p: 0.98,
      truncation: "disabled",
      usage: {
        input_tokens: 5405,
        input_tokens_details: {
          cache_write_tokens: 5402,
          cached_tokens: 0,
        },
        output_tokens: 20,
        output_tokens_details: {
          reasoning_tokens: 0,
        },
        total_tokens: 5425,
      },
      user: null,
      metadata: {},
    },
    cacheRead: {
      id: "resp_0dd0436f00ef863f006a776cd494508195878bb8b04b9384ff",
      object: "response",
      created_at: 1786211540,
      status: "completed",
      background: false,
      billing: {
        payer: "developer",
      },
      completed_at: 1786211542,
      error: null,
      frequency_penalty: 0.0,
      incomplete_details: null,
      instructions: "GqpaY1r8Uh9Y1bm5yloT6Mk80fiBwTxpDdTi8vDQ...<truncated cacheable prefix>",
      max_output_tokens: null,
      max_tool_calls: null,
      model: "gpt-5.6-luna",
      moderation: null,
      output: [
        {
          id: "rs_0dd0436f00ef863f006a776cd51520819586a9ddb8a84a7fa1",
          type: "reasoning",
          content: [],
          encrypted_content:
            "gAAAAABqd2zXsREBF6SKUM9adyGyawjYBGDxV4MV7BXLFZ-NMRw5H41ZSo_uTMKlpZcd9ymK6dA6lnJAN7oSqPNV5If57A58YhfWlkBnf11hc7OrebUk8I3YOnntoVGrRaB2ITIseDtrGmdgxj6VUP2ZWJUJ-aoETr0_Ai4MlBYb6tlBiaUSHI84tV7mesqLz9-frPKDq_2oWgHf39_telsNJGZ4Wfgha-fhQjZSTR8xeZ2jDOXPb8Ctfj87QQJRhDlNLdphoKRWdPQdCnNbj5SHtZyzZovkkPw-DCdiPIxZB-Lk8cRGuIUD2_PR0-eqP3j5mbpn6OMmyV0XYnLQh--JwtFAcVE2ijXVRTAX77ZNlEje1KF0OvxKTFmv370Gd9nz2DjekU8ST2eEd3DIT4nEIIGOI-19igq4Uq9QIZnVbhlF5Iqi7pxqjg4MeUMJwKzQt4VtGVK8zPtYSmEyLc_aOQSpdqKCWd_wVQvCREVGYYIobHlSTdj0BCdnZOCD-ecy5jRh0-DGDnxxcQ7xz1d0JUPQ7fcbm9jbyGT4jOnpsNMogdUwLQONLG0RKmnI_Zbflh27laAgpX4UkQnGY5W82Xs9vxSQtTzFsidOi51XH6fyuj8_QHTWLHfjMHhTmgIioAUEeu3HPcXbKPtlYUHfsNpN0k60pVmDXySQperrZ4356RkH6lNmfUM5fMHAccAl-NzFOFRh6TrpiqW6o9uxkmC8yeDlblOuLonbsp2UbrDPDPifYr-78QlSwqnbjhKYE-8Jwuggl7DvOUmWmL55VcpHM15Jlt9VDlz4Pmu6Sjl2yQku1-EVvu9gxvK-HEb23M7h_05rbKcnq_B8ttpmFcjz5UW7-NoxIYYhgptxO58iiviVa4r6fkE_frxefqRTKqDSWDWUTJwq7XhM4y9AN4IGUJaWkr6xhMdpyMjxayUutJsupkvhORz79XWx2HpuoN2gsvBJ5CoKZd7kb-4uxhWHbQzHZqRQT16ZBoRu7uUb4ZSi0xCaTxftZFKSOYoK8Zv8w3ksNJiF25_NMHIzOsFhGi4Ddlt0SysSKaLe2RAUud6TfFc=",
          summary: [],
        },
        {
          id: "msg_0dd0436f00ef863f006a776cd564e08195ac48150f83f5dd8a",
          type: "message",
          status: "completed",
          content: [
            {
              type: "output_text",
              annotations: [],
              logprobs: [],
              text: "When dusk unthreads the gold along the sky,  \nAnd shadows fold the gardens into sleep,  \nA quiet star ascends the fields nearby,  \nTo guard the dreams the waking hours keep.  \n\nThe wind moves softly through the waiting trees,  \nAnd carries scents of rain and rose away;  \nThe moon lays silver pathways on the seas,  \nWhile night grows deep beneath the fading day.  \n\nYet in my heart, one brighter flame remains,  \nA steady light no darkness can undo;  \nIt sings above the silence and the rains,  \nAnd finds its way through every thought of you.  \n\nSo let the world grow still, its burdens cease\u2014  \nYour memory turns the midnight into peace.",
            },
          ],
          phase: "final_answer",
          role: "assistant",
        },
      ],
      parallel_tool_calls: true,
      presence_penalty: 0.0,
      previous_response_id: null,
      prompt_cache_key: null,
      prompt_cache_retention: "24h",
      reasoning: {
        context: "all_turns",
        effort: "medium",
        mode: "standard",
        summary: null,
      },
      safety_identifier: null,
      service_tier: "default",
      store: true,
      temperature: 1.0,
      text: {
        format: {
          type: "text",
        },
        verbosity: "medium",
      },
      tool_choice: "auto",
      tool_usage: {
        image_gen: {
          input_tokens: 0,
          input_tokens_details: {
            image_tokens: 0,
            text_tokens: 0,
          },
          output_tokens: 0,
          output_tokens_details: {
            image_tokens: 0,
            text_tokens: 0,
          },
          total_tokens: 0,
        },
        web_search: {
          num_requests: 0,
        },
      },
      tools: [],
      top_logprobs: 0,
      top_p: 0.98,
      truncation: "disabled",
      usage: {
        input_tokens: 5405,
        input_tokens_details: {
          cache_write_tokens: 10,
          cached_tokens: 5392,
        },
        output_tokens: 179,
        output_tokens_details: {
          reasoning_tokens: 32,
        },
        total_tokens: 5584,
      },
      user: null,
      metadata: {},
    },
  },
  responsesTerra: {
    cacheWrite: {
      id: "resp_080cf077f8d107d9006a776cd74be08194af227e5f58eace7b",
      object: "response",
      created_at: 1786211543,
      status: "completed",
      background: false,
      billing: {
        payer: "developer",
      },
      completed_at: 1786211544,
      error: null,
      frequency_penalty: 0.0,
      incomplete_details: null,
      instructions: "GBqXvywzhtvRMAn6cc1rWCZfQfQVpMz9NGwlnELM...<truncated cacheable prefix>",
      max_output_tokens: null,
      max_tool_calls: null,
      model: "gpt-5.6-terra",
      moderation: null,
      output: [
        {
          id: "msg_080cf077f8d107d9006a776cd7d42081948ca08de13822dd26",
          type: "message",
          status: "completed",
          content: [
            {
              type: "output_text",
              annotations: [],
              logprobs: [],
              text: "Soft rain taps the leaves  \nMoonlight gathers in still pools  \nNight breathes silver dreams",
            },
          ],
          phase: "final_answer",
          role: "assistant",
        },
      ],
      parallel_tool_calls: true,
      presence_penalty: 0.0,
      previous_response_id: null,
      prompt_cache_key: null,
      prompt_cache_retention: "24h",
      reasoning: {
        context: "all_turns",
        effort: "medium",
        mode: "standard",
        summary: null,
      },
      safety_identifier: null,
      service_tier: "default",
      store: true,
      temperature: 1.0,
      text: {
        format: {
          type: "text",
        },
        verbosity: "medium",
      },
      tool_choice: "auto",
      tool_usage: {
        image_gen: {
          input_tokens: 0,
          input_tokens_details: {
            image_tokens: 0,
            text_tokens: 0,
          },
          output_tokens: 0,
          output_tokens_details: {
            image_tokens: 0,
            text_tokens: 0,
          },
          total_tokens: 0,
        },
        web_search: {
          num_requests: 0,
        },
      },
      tools: [],
      top_logprobs: 0,
      top_p: 0.98,
      truncation: "disabled",
      usage: {
        input_tokens: 5448,
        input_tokens_details: {
          cache_write_tokens: 5445,
          cached_tokens: 0,
        },
        output_tokens: 22,
        output_tokens_details: {
          reasoning_tokens: 0,
        },
        total_tokens: 5470,
      },
      user: null,
      metadata: {},
    },
    cacheRead: {
      id: "resp_09ddaaa05f8dee28006a776cd8afa08193b2ee1aae48dbc430",
      object: "response",
      created_at: 1786211544,
      status: "completed",
      background: false,
      billing: {
        payer: "developer",
      },
      completed_at: 1786211548,
      error: null,
      frequency_penalty: 0.0,
      incomplete_details: null,
      instructions: "GBqXvywzhtvRMAn6cc1rWCZfQfQVpMz9NGwlnELM...<truncated cacheable prefix>",
      max_output_tokens: null,
      max_tool_calls: null,
      model: "gpt-5.6-terra",
      moderation: null,
      output: [
        {
          id: "rs_09ddaaa05f8dee28006a776cd944188193885eb00be84c86f1",
          type: "reasoning",
          content: [],
          encrypted_content:
            "gAAAAABqd2zcugda9DOwx3p2sGh-xQKOc0iKYBoHKdrRdYuTs91GGUFo2d7F8618ybaID3A5hXEW3JMNYY6Vvabb1guzSmt9eAkgkS8Jul8nr_xbxaKRWR8H8UVE-b1TPatMeYBl-ssbNLVmkqRP0CM8U15ouLyKGjqjPm5GXPH8I_1vU4gQeHXnX7ZwnNfDJngczYL8NkSplxJDsw3dxYXyTgLPV3Fw0bJ7_DW1Oz84JGEWW9PAHLdBa0f3-j7M7N_IhtVEpljxOqpCCK2grmeSdv3yXQouNe4HNd0l00S3-2Xq5rx-tLJQBN9U0tviUWIaMmpcK3olcGWZmdDn6BYloCeMUyQ2CBT7ieo4ILxUA12jmoIptYarLJZ1tWuva0o4fi_KjyCSoHqqKhPO4H9KaX6EEbWu5terLdq-p_-iN-WyRAA5QG9O_dmuwHmaPF42pu_uPzNmIWVrpjCR_nzLc1Bgqdtwbt2MsJIEIiJNNW-CU_WPBm8F6okzHo3LoJ5KNt7sklpPm4cF2dUFce0m4L9eeLclWfhO8AwEr6xw5U44NadLGN-QLLEQEr9ig4vgcbOdfvrQK32YtkNLp_LcMEOVzPz8v55RYxrG-zS1gXmfo3LTxKZwnFg7JfFSS_EtQff7R-KvcgNZaJ06I-6O9-eA4nh8QMxIwCVBmj96mUWq6oIdqKcJ8N5HGTCBVd_FWWkTup9R4E6Kz_ehqXzXuo-OJ8XBGQQ9O9QmMJtA-cyI47z7PUvXF1l68FvafnoRsi9sFkaKqZaZ415Jzu4ZHHWjajvYG_a_f3GEVn3HnbfaEKZwM78vnavI1G1rQYc_HEqh6TILTLA6v3N21awd8sfd0YAjuJHcL91z2OM8srZbo7JZXe4ZKNuw9nwADIEGfJvdMi17FrzCzqBz09rThkgqqqA05ctGp8m1ba07TTIncgeKF28VoXKAUmg3_AmQscOsdnIJXIXFMmCcjXFAEKPnpGtnWwVkJQYDmBu3GQO61WUIhHY=",
          summary: [],
        },
        {
          id: "msg_09ddaaa05f8dee28006a776cd9685c8193b5a935a732b814b4",
          type: "message",
          status: "completed",
          content: [
            {
              type: "output_text",
              annotations: [],
              logprobs: [],
              text: "At dusk, the sky unbuttons into blue,  \nAnd evening pours its gold on field and stream;  \nThe day relinquishes its brightened view,  \nTo let the stars inherit every dream.  \n\nA quiet wind moves gently through the trees,  \nIts whispered hymns make restless sparrows still;  \nThe moon lays silver on the darkened seas,  \nAnd softens every shadow on the hill.  \n\nSo may your heart, when daylight turns to night,  \nRemember all the warmth the morning gave;  \nFor even loss can teach the dark to light,  \nAnd hope may bloom where sorrow made its grave.  \n\nThough time\u2019s cold tide will draw all things away,  \nLove leaves a dawn within the deepest gray.",
            },
          ],
          phase: "final_answer",
          role: "assistant",
        },
      ],
      parallel_tool_calls: true,
      presence_penalty: 0.0,
      previous_response_id: null,
      prompt_cache_key: null,
      prompt_cache_retention: "24h",
      reasoning: {
        context: "all_turns",
        effort: "medium",
        mode: "standard",
        summary: null,
      },
      safety_identifier: null,
      service_tier: "default",
      store: true,
      temperature: 1.0,
      text: {
        format: {
          type: "text",
        },
        verbosity: "medium",
      },
      tool_choice: "auto",
      tool_usage: {
        image_gen: {
          input_tokens: 0,
          input_tokens_details: {
            image_tokens: 0,
            text_tokens: 0,
          },
          output_tokens: 0,
          output_tokens_details: {
            image_tokens: 0,
            text_tokens: 0,
          },
          total_tokens: 0,
        },
        web_search: {
          num_requests: 0,
        },
      },
      tools: [],
      top_logprobs: 0,
      top_p: 0.98,
      truncation: "disabled",
      usage: {
        input_tokens: 5448,
        input_tokens_details: {
          cache_write_tokens: 10,
          cached_tokens: 5435,
        },
        output_tokens: 170,
        output_tokens_details: {
          reasoning_tokens: 17,
        },
        total_tokens: 5618,
      },
      user: null,
      metadata: {},
    },
  },
};
