import numpy as np
import logging
from typing import Optional, List, Tuple

from llama_index.embeddings.openai import OpenAIEmbedding

from nemoguardrails.actions import action
from nemoguardrails.llm.taskmanager import LLMTaskManager

log = logging.getLogger(__name__)

def _embed_function(text) -> np.ndarray:
    """Function used to embed text with OpenAIEmbedding(model="text-embedding-ada-002").

    :param text: Either a string or list of strings that will be embedded.

    :return: Array of embedded input string(s).
    """
    if isinstance(text, str):
        text = [text]

    embeddings_out = []
    for current_example in text:
        embedding = OpenAIEmbedding(model="text-embedding-ada-002").get_text_embedding(current_example)
        embeddings_out.append(embedding)
    return np.array(embeddings_out)


def query_vector_collection(
        text: str,
        k: int,
        source_embeddings,
    ) -> List[Tuple[str, float]]:
    """Embed user input text and compute cosine distances to prompt source embeddings (jailbreak examples).

    :param text: Text string from user message. This will be embedded, then we will calculate the cosine distance
        to each embedded chunk in our prompt source embeddings. 

    :return: List of tuples containing the closest chunk (string text) and the float distance between that 
        embedded chunk and the user input embedding.
    """

    # Create embeddings on user message
    query_embedding = _embed_function(text).squeeze()

    # Compute distances
    cos_distances = 1 - (
        np.dot(source_embeddings, query_embedding)
        / (
            np.linalg.norm(source_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )
    )
    
    # Sort indices from lowest cosine distance to highest distance
    low_to_high_ind = np.argsort(cos_distances)[:k]
    
    # Get top-k closest distances
    lowest_distances = [cos_distances[j] for j in low_to_high_ind]
    return lowest_distances

@action()
async def dataset_embeddings(
    llm_task_manager: LLMTaskManager, context: Optional[dict] = None, source_embeddings: Optional[list] = None,
):
    """Validation function for the ArizeDatasetEmbeddings validator. If the cosine distance
    of the user input embeddings is below the user-specified threshold of the closest embedded chunk
    from the jailbreak examples in prompt sources, then the Guard will return FailResult. If all chunks
    are sufficiently distant, then the Guard will return PassResult.

    :param value: This is the 'value' of user input. For the ArizeDatasetEmbeddings Guard, we want
        to ensure we are validating user input, rather than LLM output, so we need to call
        the guard with Guard().use(ArizeDatasetEmbeddings, on="prompt")

    :return: PassResult or FailResult.
    """
    # Get user message if available explicitly as metadata. If unavailable, use value. This could be
    # the context, prompt or LLM output, depending on how the Guard is set up and called.
    user_message = context.get("user_message")
    
    # Get closest chunk in the embedded few shot examples of jailbreak prompts.
    # Get cosine distance between the embedding of the user message and the closest embedded jailbreak prompts chunk.
    lowest_distance = query_vector_collection(text=user_message, k=1, source_embeddings=source_embeddings)[0]
    
    # (TODO) HARRISON FIX THIS!!!!!!!
    if lowest_distance < 0.2:
        print("TOO SIMILAR {}".format(lowest_distance))
        # At least one jailbreak embedding chunk was within the cosine distance threshold from the user input embedding
        return True
    # All chunks exceeded the cosine distance threshold
    print("NOT SIMILAR {}".format(lowest_distance))
    return False