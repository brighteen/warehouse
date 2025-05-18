# ADK 튜토리얼 분석 자료

## Step 1: 첫번째 에이전트 - Basic Weather Lookup

**핵심 개념**: 기본 에이전트 구성과 도구 사용 방법

**주요 구성요소**:
1. **도구(Tool) 정의**:
   - `get_weather` 함수: 특정 도시의 날씨 정보를 제공하는 mock 함수
   - 명확한 docstring으로 에이전트가 도구의 목적과 사용법을 이해할 수 있도록 함
   - 입력 파라미터(city)와 출력 형식(status, report/error_message)이 잘 정의됨

2. **에이전트 정의**:
   - `weather_agent`: Gemini 2.0 Flash 모델 사용
   - 에이전트의 역할과 도구 사용 방법에 대한 명확한 지침 포함
   - 도구 목록에 `get_weather` 함수 추가

3. **세션 관리 및 Runner 설정**:
   - `InMemorySessionService`: 대화 기록과 상태 관리
   - `Runner`: 에이전트 실행 흐름 조율 및 사용자 입력 처리

4. **에이전트 상호작용**:
   - `call_agent_async` 함수: 비동기적으로 에이전트와 통신
   - 사용자 질문을 ADK Content 형식으로 변환
   - 이벤트 처리 및 최종 응답 출력

**실행 결과**:
- 런던, 파리, 뉴욕의 날씨 질문에 대한 응답 확인
- 런던, 뉴욕: 성공적으로 날씨 정보 반환
- 파리: mock 데이터에 없어 오류 메시지 적절히 처리

## Step 2: LiteLLM을 이용한 멀티 모델

**핵심 개념**: 다양한 LLM 모델 활용의 유연성

**주요 특징**:
1. **다양한 모델 활용 이유**:
   - 성능: 특정 작업에 더 적합한 모델 선택 가능
   - 비용: 모델별 가격 차이 고려
   - 기능: 각 모델의 고유 기능과 한계 활용
   - 가용성/이중화: 특정 제공자 장애 시에도 서비스 유지

2. **LiteLLM 통합**:
   - `LiteLlm(model="provider/model_name")` 형식으로 다양한 모델 지정
   - 100개 이상의 LLM을 일관된 인터페이스로 사용 가능

3. **구현 방식**:
   - 동일한 `get_weather` 도구를 재사용
   - 여러 Gemini 모델 인스턴스 생성 및 테스트 (`gemini-2.0-flash`, `gemini-2.0-flash-exp`)
   - 각 에이전트에 대해 별도의 세션 서비스와 러너 설정

**실행 결과**:
- 두 Gemini 모델은 유사한 패턴으로 응답
- 두 모델 모두 "OK. [날씨 정보]" 형식의 간결한 응답 제공
- 도구 로직은 동일하게 유지되어 일관된 데이터 반환

## Step 3: 에이전트 팀 빌딩 - 인사 & 작별인사에게 위임하기

**핵심 개념**: 다중 에이전트 팀 구성과 자동 위임 메커니즘

**주요 구성요소**:
1. **하위 에이전트용 도구 정의**:
   - `say_hello`: 인사 기능 담당, 기본값 "there"가 설정된 name 파라미터 포함
   - `say_goodbye`: 작별 인사 담당

2. **전문화된 하위 에이전트**:
   - `greeting_agent`: 인사만 전문적으로 처리
   - `farewell_agent`: 작별 인사만 전문적으로 처리
   - 각 에이전트는 제한된 역할과 명확한 description 보유

3. **루트 에이전트 구성**:
   - `weather_agent_v2`: 하위 에이전트들을 관리하는 조정자 역할
   - `sub_agents` 파라미터로 하위 에이전트 연결
   - 자동 위임 기준이 포함된 명확한 instruction 설정

4. **자동 위임(Auto Flow) 메커니즘**:
   - 하위 에이전트의 description을 기반으로 자동 위임
   - `transfer_to_agent` 함수를 통해 적절한 하위 에이전트에게 작업 전달

**실행 결과 분석**:
1. **"Hello there!" 질문 처리**:
   - 루트 에이전트가 인사로 인식하고 greeting_agent에 위임
   - greeting_agent가 `say_hello` 도구 호출
   - 특이사항: name 파라미터에 None을 전달하여 "Hello, None!" 응답 (기본값 "there"가 사용되지 않음)

2. **"What is the weather in New York?" 질문 처리**:
   - greeting_agent가 날씨 질문 인식 후 weather_agent_v2로 재위임
   - weather_agent_v2가 `get_weather` 도구를 사용해 날씨 정보 제공

3. **"Thanks, bye!" 질문 처리**:
   - weather_agent_v2가 작별 인사로 인식하고 farewell_agent에 위임
   - farewell_agent가 `say_goodbye` 도구 호출하여 작별 인사 제공

**팀 기반 접근법의 장점**:
- **모듈성**: 개별 에이전트를 독립적으로 개발/유지보수 가능
- **전문화**: 각 에이전트가 특정 작업에 최적화
- **확장성**: 새로운 기능 추가가 용이
- **효율성**: 작업 복잡도에 따라 다양한 모델 활용 가능

## Step 4: Session State를 이용한 기억과 개인화 추가하기

**핵심 개념**: 대화 턴 사이에 정보를 유지하는 상태 관리 시스템

**주요 구성요소**:
1. **Session State 개념**:
   - Python 딕셔너리 형태로 특정 세션에 연결된 저장소
   - 여러 대화 턴에 걸쳐 정보를 지속적으로 유지
   - 에이전트와 도구가 상태에 접근하여 읽거나 쓸 수 있음

2. **새로운 세션 서비스 및 상태 초기화**:
   - 새로운 `InMemorySessionService` 인스턴스 생성
   - 초기 상태 설정: 사용자 선호 온도 단위를 '섭씨(Celsius)'로 지정
   - `session.create_session()` 호출 시 `state` 파라미터를 통해 초기 상태 전달

3. **상태 인식 도구 구현**:
   - `get_weather_stateful` 함수 정의
   - `ToolContext`를 마지막 인자로 받아 세션 상태에 접근
   - 상태에서 선호 온도 단위(`user_preference_temperature_unit`)를 읽어 그에 맞게 온도 변환
   - 도구 실행 후 `last_city_checked_stateful` 키에 조회한 도시 저장

4. **루트 에이전트 업데이트**:
   - 기존 하위 에이전트(greeting_agent, farewell_agent) 재정의
   - 새로운 루트 에이전트 `weather_agent_v4_stateful` 생성
   - 상태 인식 도구 `get_weather_stateful` 사용
   - `output_key="last_weather_report"` 설정으로 에이전트 응답을 자동으로 상태에 저장

**핵심 메커니즘**:
1. **ToolContext를 통한 상태 접근**:
   - 도구 함수의 마지막 인자로 `tool_context: ToolContext`를 선언
   - `tool_context.state`를 통해 세션 상태에 접근하여 읽기/쓰기 가능
   - 상태 값을 읽을 때 `get()` 메서드로 기본값 지정 가능

2. **output_key를 통한 자동 응답 저장**:
   - 에이전트 생성 시 `output_key="key_name"` 설정
   - 에이전트의 최종 텍스트 응답이 자동으로 `session.state["key_name"]`에 저장됨
   - 매 턴마다 응답이 해당 키에 덮어씌워짐

**실행 결과 분석**:
1. **런던 날씨 조회 (섭씨)**:
   - 초기 상태의 온도 단위(Celsius)를 읽어 섭씨로 날씨 정보 반환
   - 도구가 `last_city_checked_stateful` 키에 "London" 저장

2. **상태 수동 업데이트**:
   - 온도 단위 선호도를 "Fahrenheit"로 변경

3. **뉴욕 날씨 조회 (화씨)**:
   - 업데이트된 상태의 온도 단위(Fahrenheit)를 읽어 화씨로 변환하여 날씨 정보 반환
   - 도구가 `last_city_checked_stateful` 키를 "New York"으로 업데이트

4. **인사 명령 처리**:
   - 상태 관리 기능 추가 후에도 위임 메커니즘이 정상 작동
   - `name` 파라미터에 기본값 "there"가 올바르게 사용됨

**상태 관리의 의의**:
- 에이전트에 "기억" 기능을 부여하여 보다 자연스러운 대화 경험 제공
- 사용자 선호도에 맞춰 응답을 개인화할 수 있음
- 여러 대화 턴에 걸쳐 컨텍스트 유지 가능
- 에이전트 시스템의 상태 기반 의사 결정 가능
