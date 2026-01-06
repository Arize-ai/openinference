import os
from datetime import datetime
from logging import getLogger

from dotenv import load_dotenv
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from openinference.instrumentation.pipecat import PipecatInstrumentor

load_dotenv(override=True)

# Conversation ID for the conversation (used for session tracking in Arize or Phoenix)
conversation_id = f"test-conversation-001_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Debug log filename for the conversation (used for viewing the frames in the conversation)
debug_log_filename = os.path.join(os.getcwd(), f"pipecat_frames_{conversation_id}.log")

# Logger for the conversation (used for logging)
logger = getLogger(__name__)


def setup_tracer_provider():
    """
    Setup the tracer provider.
    """
    project_name = os.getenv("PROJECT_NAME", "pipecat-voice-agent")
    space_id = os.getenv("ARIZE_SPACE_ID", None)
    api_key = os.getenv("ARIZE_API_KEY", None)
    if space_id and api_key:
        # Register the Arize tracer provider
        from arize.otel import register as register_arize

        return register_arize(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
        )
    else:
        # Register the Phoenix tracer provider
        from phoenix.otel import register as register_phoenix

        return register_phoenix(project_name=project_name)


# Auto-instrument the Pipecat pipeline
tracer_provider = setup_tracer_provider()
PipecatInstrumentor().instrument(
    tracer_provider=tracer_provider,
    debug_log_filename=debug_log_filename,
)

# Transport parameters for the Pipecat pipeline
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """
    Run the Pipecat pipeline.
    """
    logger.info("Starting bot")

    ### STT ###
    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-transcribe",
        prompt="Expect normal helpful conversation.",
    )
    ### alternative stt - cartesia ###
    # stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY"))

    ### LLM ###
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    ### TTS ###
    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="ballad",
        params=OpenAITTSService.InputParams(
            instructions="Please speak clearly and at a moderate pace."
        ),
    )

    # LLM prompt for the conversation
    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. "
            + "Your goal is to demonstrate your capabilities in a succinct way. "
            + "Your output will be converted to audio so don't "
            + "include special characters in your answers. "
            + "Respond to what the user said in a creative and helpful way.",
        }
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    ### PIPELINE ###
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    ### TASK ###
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        conversation_id=conversation_id,  # Use dynamic conversation ID for session tracking
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """
        Handle the client connected event.
        """
        logger.info("Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        """
        Handle the client disconnected event.
        """
        logger.info("Client disconnected")
        await task.cancel()

    # Create the Pipecat pipeline runner
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    # Run the Pipecat pipeline
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
