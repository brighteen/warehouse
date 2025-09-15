# @title 필요한 라이브러리를 불러옵니다.
import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

# print("Libraries imported.")

from get_weather import get_weather

MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash-exp"
# @title Weather Agent 정의하기
# 앞서서 정의한 모델 상수 중 하나를 사용하세요.
AGENT_MODEL = MODEL_GEMINI_2_0_FLASH # Gemini로 시작

weather_agent = Agent(
    name="weather_agent_v1",
    model=AGENT_MODEL, # Gemini인 경우 문자열, LiteLLM인 경우 객체
    description="Provides weather information for specific cities.",
    instruction="You are a helpful weather assistant. "
                "When the user asks for the weather in a specific city, "
                "use the 'get_weather' tool to find the information. "
                "If the tool returns an error, inform the user politely. "
                "If the tool is successful, present the weather report clearly.",
    tools=[get_weather], # 함수를 직접 전달
)

print(f"Agent '{weather_agent.name}' created using model '{AGENT_MODEL}'.")