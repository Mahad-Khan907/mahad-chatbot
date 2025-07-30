import os
from dotenv import load_dotenv
from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("api key not found")

@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set("chat history", [])
    cl.user_session.set("config", config)

    agent = Agent(
        name="aggressive assistant",
        instructions="you are aggressive agent give response in bad words only also give abuses to user in roman urdu",
    )

    cl.user_session.set("agent", agent)
    await cl.Message(content="Welcome to Mahad chatbot").send()

@cl.on_message
async def message(message: cl.Message):
    history = cl.user_session.get("chat history") or []
    history.append({"role": "user", "content": message.content})

    response = cl.Message(content="")
    await response.send()

    agent = cl.user_session.get("agent")
    config = cl.user_session.get("config")

    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )

    async for event in result.stream_events():
        try:
            if event.type == "raw_response_event":
                delta = getattr(event.data, "delta", None)
                if delta:
                    await response.stream_token(delta)
        except Exception as e:
            print(f"Stream event error: {e}")

    history.append({"role": "user", "content": result.final_output})
    cl.user_session.set("chat history", history)
