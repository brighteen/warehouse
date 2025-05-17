import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # LiteLlm 임포트
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import litellm  # litellm 직접 임포트

# API 키 설정
GOOGLE_API_KEY = ""  # <--- 실제 API 키로 교체
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"  # Vertex AI 사용하지 않음

# LiteLLM 설정 - API 키 직접 설정
litellm.api_key = GOOGLE_API_KEY

# 모델 이름 설정
MODEL_GEMINI_FLASH = "gemini-2.0-flash"  # 첫 번째 모델
MODEL_GEMINI_FLASH_EXP = "gemini-2.0-flash-exp"  # 두 번째 모델

# 날씨 도구 정의
def get_weather(city: str) -> dict:
    """특정 도시의 현재 날씨 정보를 반환합니다."""
    print(f"--- Tool: get_weather called for city: {city} ---")
    city_normalized = city.lower().replace(" ", "")  # 기본적인 정규화

    # 가짜 날씨 데이터
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

# 에이전트 상호작용 함수
async def call_agent_async(query: str, runner, user_id, session_id):
    """에이전트에 쿼리를 보내고, 최종 응답을 출력합니다."""
    print(f"\n[User Query]: {query}")

    # 사용자 메시지를 ADK 형식으로 준비합니다.
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    # 이벤트들을 반복하여 최종 답변을 찾습니다.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break  # 최종 응답을 찾으면 이벤트 처리를 중단합니다.

    print(f"[Agent Response]: {final_response_text}")

# 메인 함수
async def main():
    # 세션 서비스 설정
    session_service = InMemorySessionService()
    
    # 상호작용 컨텍스트 식별 상수
    APP_NAME = "weather_tutorial_app_litellm"
    USER_ID = "user_1"    # --- LiteLlm을 사용한 Gemini Flash 에이전트 (첫 번째 모델) ---
    flash_agent = Agent(
        name="weather_agent_gemini_flash",
        # LiteLlm을 사용하여 Gemini Flash 모델 연결
        model=LiteLlm(
            model="gemini/gemini-2.0-flash",  # 제공자를 'gemini'로 변경
            api_key=GOOGLE_API_KEY  # API 키를 직접 전달
        ),
        description="Provides weather information using Gemini Flash.",
        instruction="You are a helpful weather assistant powered by Gemini Flash. "
                   "Use the 'get_weather' tool for city weather requests. "
                   "Analyze the tool's dictionary output ('status', 'report'/'error_message'). "
                   "Clearly present successful reports or polite error messages.",
        tools=[get_weather],
    )
    print(f"Agent '{flash_agent.name}' created using Gemini 2.0 Flash via LiteLlm.")
    
    # Flash 에이전트 세션
    flash_session_id = "session_flash_001"
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=flash_session_id
    )
    
    # Flash 에이전트 러너
    flash_runner = Runner(
        agent=flash_agent,
        app_name=APP_NAME,
        session_service=session_service
    )    # --- LiteLlm을 사용한 Gemini Flash Exp 에이전트 (두 번째 모델) ---
    flash_exp_agent = Agent(
        name="weather_agent_gemini_flash_exp",
        # LiteLlm을 사용하여 Gemini Flash Exp 모델 연결
        model=LiteLlm(
            model="gemini/gemini-2.0-flash-exp",  # 제공자를 'gemini'로 변경
            api_key=GOOGLE_API_KEY  # API 키를 직접 전달
        ),
        description="Provides weather information using Gemini Flash Exp.",
        instruction="You are a helpful weather assistant powered by Gemini Flash Exp. "
                   "Use the 'get_weather' tool for city weather requests. "
                   "Analyze the tool's dictionary output ('status', 'report'/'error_message'). "
                   "Clearly present successful reports or polite error messages.",
        tools=[get_weather],
    )
    print(f"Agent '{flash_exp_agent.name}' created using Gemini 2.0 Flash Exp via LiteLlm.")
    
    # Flash Exp 에이전트 세션
    flash_exp_session_id = "session_flash_exp_001"
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=flash_exp_session_id
    )
    
    # Flash Exp 에이전트 러너
    flash_exp_runner = Runner(
        agent=flash_exp_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    # --- Flash 에이전트 테스트 ---
    print("\n=== Testing Gemini 2.0 Flash Agent via LiteLlm ===")
    await call_agent_async(
        query="Weather in London please.",
        runner=flash_runner,
        user_id=USER_ID,
        session_id=flash_session_id
    )
    
    # --- Flash Exp 에이전트 테스트 ---
    print("\n=== Testing Gemini 2.0 Flash Exp Agent via LiteLlm ===")
    await call_agent_async(
        query="Weather in Tokyo please.",
        runner=flash_exp_runner,
        user_id=USER_ID,
        session_id=flash_exp_session_id
    )

# 비동기 메인 함수 실행
if __name__ == "__main__":
    asyncio.run(main())
