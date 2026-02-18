import os
from datetime import datetime
from logging import getLogger

from dotenv import load_dotenv
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from openinference.instrumentation.pipecat import PipecatInstrumentor

load_dotenv(override=True)

# Conversation ID for the conversation (used for session tracking in Arize or Phoenix)
conversation_id = f"test-conversation-001_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Debug log filename for the conversation (used for viewing the frames in the conversation)
debug_log_filename = os.path.join(os.getcwd(), f"pipecat_frames_{conversation_id}.log")
print(f"🪲debug_log_filename: {debug_log_filename}")

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
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# Tool call handler functions
async def get_weather(params: FunctionCallParams):
    """Get the current weather for a location."""
    location = params.arguments["location"]
    format = params.arguments.get("format", "fahrenheit")
    # In a real application, you would call a weather API here
    if format == "celsius":
        await params.result_callback(
            f"The weather in {location} is currently 22 degrees Celsius and sunny."
        )
    else:
        await params.result_callback(
            f"The weather in {location} is currently 72 degrees Fahrenheit and sunny."
        )


async def get_current_time(params: FunctionCallParams):
    """Get the current time for a timezone."""
    timezone = params.arguments.get("timezone", "UTC")
    # In a real application, you would use pytz or similar for proper timezone handling
    current_time = datetime.now().strftime("%I:%M %p")
    await params.result_callback(f"The current time in {timezone} is {current_time}.")


# Function schemas for tool calling
weather_function = FunctionSchema(
    name="get_weather",
    description="Get the current weather for a location",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use. Infer this from the user's location.",
        },
    },
    required=["location"],
)

time_function = FunctionSchema(
    name="get_current_time",
    description="Get the current time for a specific timezone",
    properties={
        "timezone": {
            "type": "string",
            "description": "The timezone, e.g. America/New_York, Europe/London, Asia/Tokyo",
        },
    },
    required=["timezone"],
)

tools = ToolsSchema(standard_tools=[weather_function, time_function])


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

    ### LLM ###
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Register tool call handlers
    llm.register_function("get_weather", get_weather)
    llm.register_function("get_current_time", get_current_time)

    ### TTS ###
    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="ballad",
        params=OpenAITTSService.InputParams(
            instructions="Please speak clearly and at a moderate pace."
        ),
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    # LLM prompt for the conversation
    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. "
            + "Your goal is to demonstrate your capabilities in a succinct way. "
            + "Your output will be converted to audio so don't "
            + "include special characters in your answers. "
            + "Respond to what the user said in a creative and helpful way. "
            + "You have access to two tools: get_weather to check the weather in any location, "
            + "and get_current_time to check the time in any timezone. "
            + "Use these tools when the user asks about weather or time.",
        }
    ]

    context = LLMContext(messages, tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    )

    ### PIPELINE ###
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
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
