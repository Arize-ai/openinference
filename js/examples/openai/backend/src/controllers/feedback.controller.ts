import { Request, Response } from "express";

const PHOENIX_REST_ENDPOINT =
  process.env.PHOENIX_REST_ENDPOINT || "http://localhost:6006/v1";

const SPAN_ANNOTATIONS_ENDPOINT = `${PHOENIX_REST_ENDPOINT}/span_annotations`;

export const feedback = async (req: Request, res: Response) => {
  const { spanId, feedbackScore } = req.body;

  const response = await fetch(SPAN_ANNOTATIONS_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      accept: "application/json",
    },
    body: JSON.stringify({
      data: [
        {
          span_id: spanId,
          annotator_kind: "HUMAN",
          name: "feedback",
          result: {
            label: feedbackScore === 0 ? "👎" : "👍",
            score: feedbackScore,
            explanation:
              feedbackScore === 0
                ? "Negative feedback from user"
                : "Positive feedback from user",
          },
        },
      ],
    }),
  });

  if (response.status !== 200) {
    return res.status(500).json({
      error: "Failed to send feedback",
    });
  }

  res.sendStatus(200);
};
