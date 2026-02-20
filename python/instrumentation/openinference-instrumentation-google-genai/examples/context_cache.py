import pathlib
import time

import requests
from google import genai
from google.genai import types
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

client = genai.Client()

model = 'gemini-2.5-flash'


def create_video_content_cache():

    # Download a test video file and save it locally
    url = 'https://storage.googleapis.com/generativeai-downloads/data/SherlockJr._10min.mp4'
    path_to_video_file = pathlib.Path('SherlockJr._10min.mp4')
    if not path_to_video_file.exists():
        path_to_video_file.write_bytes(requests.get(url).content)

    # Upload the video using the Files API
    video_file = client.files.upload(file=path_to_video_file)

    # Wait for the file to finish processing
    while video_file.state.name == 'PROCESSING':
        time.sleep(2.5)
        video_file = client.files.get(name=video_file.name)

    print(f'Video processing complete: {video_file.uri}')

    model = 'models/gemini-3-flash-preview'

    # Create a cache with a 5 minute TTL (300 seconds)
    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name='sherlock jr movie',  # used to identify the cache
            system_instruction=(
                'You are an expert video analyzer, and your job is to answer '
                'the user\'s query based on the video file you have access to.'
            ),
            contents=[video_file],
            ttl="300s",
        )
    )

    response = client.models.generate_content(
        model=model,
        contents=(
            'Introduce different characters in the movie by describing '
            'their personality, looks, and names. Also list the timestamps '
            'they were introduced for the first time.'),
        config=types.GenerateContentConfig(cached_content=cache.name)
    )

    print(response.usage_metadata)

    print(response.text)


def create_image_content_cache():

    # Download a test video file and save it locally
    url = 'https://fastly.picsum.photos/id/76/200/300.jpg?grayscale&hmac=xrdYm3wgKnEMckd163A1D0sHAClPRZvCBedddaxsE6k'
    path_to_file = pathlib.Path('demo.jpg')
    if not path_to_file.exists():
        path_to_file.write_bytes(requests.get(url).content)

    # Upload the video using the Files API
    image_file = client.files.upload(file=path_to_file)

    # Wait for the file to finish processing
    while image_file.state.name == 'PROCESSING':
        time.sleep(2.5)
        image_file = client.files.get(name=image_file.name)

    print(f'Video processing complete: {image_file.uri}')

    model = 'models/gemini-3-flash-preview'

    # Create a cache with a 5 minute TTL (300 seconds)
    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name='sherlock jr movie',  # used to identify the cache
            system_instruction=(
                'You are an expert Image analyzer, and your job is to answer '
                'the user\'s query based on the Images file you have access to.'
            ),
            contents=[image_file],
            ttl="300s",
        )
    )
    print(cache)
    # response = client.models.generate_content(
    #     model=model,
    #     contents=(
    #         'Introduce different characters in the movie by describing '
    #         'their personality, looks, and names. Also list the timestamps '
    #         'they were introduced for the first time.'),
    #     config=types.GenerateContentConfig(cached_content=cache.name)
    # )
    #
    # print(response.usage_metadata)
    #
    # print(response.text)


if __name__ == "__main__":
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    create_image_content_cache()

