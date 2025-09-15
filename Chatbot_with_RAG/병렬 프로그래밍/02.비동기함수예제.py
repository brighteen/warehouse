import asyncio  # asyncio 라이브러리를 가져옵니다. 비동기 프로그래밍을 위한 파이썬 표준 라이브러리입니다.
import datetime

# 비동기적으로 실행될 함수를 정의합니다. 'async def'를 사용하여 정의합니다.
async def task(seconds):
    print(f'[작업시작] {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # 시작 메시지를 출력합니다. 작업이 몇 초 후에 끝날지 알려줍니다.
    print(f"이 작업은 {seconds} 초 뒤 종료됩니다.")
    
    # asyncio.sleep 함수를 사용하여 비동기적으로 지정된 시간 동안 대기합니다.
    # 'await'는 이 함수가 완료될 때까지 현재 코루틴의 실행을 일시 중지합니다.
    await asyncio.sleep(seconds)
    
    # 대기 시간이 끝나면, 작업 완료 메시지를 출력합니다.
    print(f"작업이 끝났습니다.")
    print(f'[작업종료] {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# 메인 함수를 정의합니다. 이 함수도 비동기적으로 실행됩니다.
async def main():
    # asyncio.gather를 사용하여 여러 코루틴(task 함수 호출)을 동시에 실행합니다.
    # 이렇게 하면 task(1), task(2), task(3)이 거의 동시에 시작됩니다.
    await asyncio.gather(
        task(1),
        task(2),
        task(3)
    )

# asyncio.run을 사용하여 메인 함수를 실행합니다. 이는 프로그램의 시작점입니다.
# 이벤트 루프를 시작하고 main 코루틴을 실행합니다.
asyncio.run(main())