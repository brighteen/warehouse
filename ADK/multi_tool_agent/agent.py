# 날짜 및 시간 관련 기능을 사용하기 위한 datetime 모듈 가져오기
import datetime
# 타임존 정보를 처리하기 위한 ZoneInfo 클래스 가져오기
from zoneinfo import ZoneInfo
# Google ADK 에이전트 기능을 사용하기 위한 Agent 클래스 가져오기
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """
    Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    # 도시 이름이 'new york'인지 확인 (대소문자 구분 없이)
    if city.lower() == "new york":
        # 뉴욕의 날씨 정보를 담은 성공 응답 반환
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        # 지원되지 않는 도시인 경우 오류 메시지 반환
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """
    Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """
    # 도시 이름이 'new york'인지 확인 (대소문자 구분 없이)
    if city.lower() == "new york":
        # 뉴욕의 타임존 식별자 설정
        tz_identifier = "America/New_York"
    else:
        # 지원되지 않는 도시인 경우 오류 메시지 반환
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    # 지정된 타임존 객체 생성
    tz = ZoneInfo(tz_identifier)
    # 현재 시간을 해당 타임존으로 가져오기
    now = datetime.datetime.now(tz)
    # 시간을 포맷팅하여 응답 메시지 생성
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    # 성공 응답 반환
    return {"status": "success", "report": report}

# 에이전트 인스턴스 생성
root_agent = Agent(
    # 에이전트 이름 설정
    name="weather_time_agent",
    # 사용할 모델 지정 (Gemini 2.0 Flash)
    model="gemini-2.0-flash",
    # 에이전트 설명 정의 (영어 및 한국어)
    description=(
        "Agent to answer questions about the time and weather in a city. "
    ),
    # 에이전트의 행동 지침 정의 (영어 및 한국어)
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city. "
    ),
    # 에이전트가 사용할 도구(함수) 목록 설정
    tools=[get_weather, get_current_time],
)