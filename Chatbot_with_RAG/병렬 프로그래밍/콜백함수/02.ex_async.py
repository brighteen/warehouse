import asyncio

# 비동기 함수 정의
async def hello():
    print("안녕하세요!")
    # asyncio.sleep은 비동기 대기를 구현합니다
    await asyncio.sleep(2)  # 1초 대기
    print("반갑습니다!")
    return "hello(courutine!)"  # 코루틴이 반환하는 값
    
# 비동기 함수 실행하기
# async def main():
#     await hello()  # await 키워드로 코루틴 실행

# hello() -> 코루틴 객체만 생성됨.(함수 내부 코드 실행 안됨)
asyncio.run(hello())  # asyncio.run()으로 이벤트 루프 실행
print(f'\n[print문으로 실행]\n{asyncio.run(hello())}')  # 코루틴 객체가 반환됨