# @title 에이전트 상호작용 함수 정의하기

from google.genai import types # 메시지 Content/Parts를 생성하기 위함

async def call_agent_async(query: str, runner, user_id, session_id):
  """에이전트에 쿼리를 보내고, 최종 응답을 출력합니다."""
  print(f"\n>>> User Query: {query}")

  # 사용자 메시지를 ADK 형식으로 준비합니다.
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # 핵심 컨셉: run_async는 에이전트의 로직을 실행하고 Event들을 생성합니다.
  # 최종 답변을 찾기 위해 이벤트들을 반복(iterate)합니다.
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      # 아래 줄의 주석을 해제하면 실행 중 발생하는 *모든* 이벤트를 확인할 수 있습니다.
      print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # 핵심 컨셉: is_final_response()는 해당 턴의 마지막 메시지임을 나타냅니다.
      if event.is_final_response():
          if event.content and event.content.parts:
             # 처음 부분에 텍스트 응답이 있다고 가정합니다.
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # 잠재적인 오류나 에스컬레이션 상황을 처리합니다.
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # 필요하다면 여기에서 추가적인 확인을 수행하세요 (예: 특정 오류 코드 등).
          break # 최종 응답을 찾으면 이벤트 처리를 중단합니다.

  print(f"<<< Agent Response: {final_response_text}")