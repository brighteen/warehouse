# 콜백 함수 (Callback Function)

## 콜백 함수란?
콜백 함수는 다른 함수의 매개변수(파라미터)로 전달되어, 특정 이벤트가 발생하거나 특정 시점에 도달했을 때 호출되는 함수입니다. 콜백 함수를 통해 프로그램의 실행 흐름을 제어하고 비동기 작업을 효과적으로 처리할 수 있습니다.

### 핵심 개념
- **함수를 매개변수로 전달**: 함수를 다른 함수의 인자로 전달합니다.
- **함수의 실행을 지연**: 특정 조건이 만족될 때까지 함수의 실행을 지연시킵니다.
- **비동기 처리**: 시간이 오래 걸리는 작업을 처리하면서도 프로그램이 계속 실행될 수 있게 합니다.

### 사용 목적
1. **순차적 실행 보장**: 작업이 순차적으로 실행되어야 할 때 사용합니다.
2. **비동기 작업 처리**: 파일 읽기/쓰기, 네트워크 요청, 타이머 등 비동기 작업에서 결과를 처리할 때 활용합니다.
3. **이벤트 처리**: 사용자 입력, 센서 데이터 수신 등의 이벤트가 발생했을 때 반응하기 위해 사용합니다.

## 예제 파일 구성

이 디렉토리에는 콜백 함수의 다양한 사용 패턴을 보여주는 간단한 예제들이 포함되어 있습니다:

1. **[01.example_callback.py](./01.example_callback.py)**: 기본적인 콜백 함수의 사용법을 보여주는 간단한 예제
2. **[02.example_callback_advanced.py](./02.example_callback_advanced.py)**: 더하기, 나누기, 타이머 등 일상적인 예제로 콜백 함수 이해하기
3. **[03.promise_pattern.py](./03.promise_pattern.py)**: 약속(Promise) 패턴을 통한 콜백 지옥 해결 방법
4. **[04.practical_callbacks.py](./04.practical_callbacks.py)**: 계산기, 파일 처리, 이벤트 시스템 등 실용적인 콜백 함수 예제
5. **[05.async_await_style.py](./05.async_await_style.py)**: 콜백 대신 async/await를 사용한 현대적인 비동기 프로그래밍 방식 예제

## 실행 방법

각 예제 파일은 독립적으로 실행할 수 있습니다:

```bash
python 01.example_callback.py
python 02.example_callback_advanced.py
python 03.promise_pattern.py
python 04.practical_callbacks.py
python 05.async_await_style.py
```

## 콜백 함수 활용 예시

간단한 콜백 함수 활용 예시:

```python
def 함수1(콜백함수):
    print("함수1 실행")
    콜백함수()  # 전달받은 콜백 함수 실행

def 함수2():
    print("함수2 실행 (콜백)")

# 함수2를 콜백으로 전달
함수1(함수2)
```

더 다양한 예제와 활용법은 예제 파일들을 통해 확인할 수 있습니다.
1. **순차적 실행 보장**: 작업이 순차적으로 실행되어야 할 때 사용합니다.
2. **비동기 작업 처리**: 파일 읽기/쓰기, 네트워크 요청, 타이머 등 비동기 작업에서 결과를 처리할 때 활용합니다.
3. **이벤트 처리**: 사용자 입력, 센서 데이터 수신 등의 이벤트가 발생했을 때 반응하기 위해 사용합니다.

## 예제 파일 구성

이 디렉토리에는 콜백 함수의 다양한 사용 패턴을 보여주는 예제들이 포함되어 있습니다:

1. **[example_callback.py](./example_callback.py)**: 기본적인 콜백 함수의 사용법을 보여주는 간단한 예제
2. **[example_callback_advanced.py](./example_callback_advanced.py)**: 성공/실패 콜백, 비동기 콜백, 이벤트 기반 콜백 등 다양한 콜백 패턴 예제
3. **[promise_pattern.py](./promise_pattern.py)**: JavaScript의 Promise와 유사한 패턴을 Python으로 구현한 예제
4. **[practical_callbacks.py](./practical_callbacks.py)**: 파일 처리, API 요청, UI 이벤트 처리 등 실용적인 시나리오에서의 콜백 사용 예제
5. **[async_await_style.py](./async_await_style.py)**: 콜백 대신 async/await를 사용한 현대적인 비동기 프로그래밍 방식 예제

## 콜백 함수의 장단점

### 장점
- **유연성**: 함수의 동작을 동적으로 변경할 수 있습니다.
- **코드 재사용**: 동일한 함수를 다양한 콜백과 함께 사용할 수 있습니다.
- **비동기 처리**: 시간이 오래 걸리는 작업을 비동기적으로 처리할 수 있습니다.
- **이벤트 기반 프로그래밍**: 이벤트에 반응하는 코드를 쉽게 작성할 수 있습니다.

### 단점
- **콜백 지옥(Callback Hell)**: 중첩된 콜백이 많아지면 코드가 복잡해집니다.
- **오류 처리**: 비동기 콜백에서의 오류 처리가 복잡할 수 있습니다.
- **코드 추적**: 실행 흐름이 복잡해져 디버깅이 어려울 수 있습니다.
- **컨텍스트 관리**: 콜백 내에서 `this`나 지역 변수의 스코프 관리가 어려울 수 있습니다.

## 콜백 대안 및 현대적 접근법

최근의 프로그래밍 언어들은 콜백 함수의 단점을 보완하기 위한 여러 기능을 제공합니다:

1. **Promise/Future**: 비동기 작업의 결과를 나타내는 객체로, 체이닝을 통해 콜백 지옥을 방지합니다.
2. **async/await**: Promise 기반 비동기 코드를 동기 코드처럼 작성할 수 있게 해주는 문법적 기능입니다.
3. **Observable/Stream**: 시간에 따라 발생하는 이벤트 스트림을 처리하는 패턴입니다.
4. **제너레이터(Generator)**: 함수 실행을 일시 중지하고 다시 시작할 수 있는 기능입니다.

## 실행 방법

각 예제 파일은 독립적으로 실행할 수 있습니다:

```bash
python example_callback.py
python example_callback_advanced.py
python promise_pattern.py
python practical_callbacks.py
python async_await_style.py
```

각 예제의 출력을 확인하며 콜백 함수의 다양한 사용법과 패턴을 이해할 수 있습니다.