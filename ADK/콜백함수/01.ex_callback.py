"""
이 파일은 기본적인 콜백 함수의 사용법을 보여주는 예제입니다.
여러 함수를 순차적으로 실행하기 위해 콜백 체인을 구성하는 방법과
비동기적인 실행 방식의 차이를 보여줍니다.
"""

import time

def first(callback=None):
    print('first 함수 실행')
    time.sleep(1)
    print('1초 경과')
    
    # 콜백 함수가 전달되었다면 실행
    if callback:
        callback()

def second(callback=None):
    print('second 함수 실행')
    time.sleep(3)
    print('3초 경과')
    
    # 콜백 함수가 전달되었다면 실행
    if callback:
        callback()

def third(callback=None):
    print('third 함수 실행')
    time.sleep(2)
    print('2초 경과')
    
    # 콜백 함수가 전달되었다면 실행
    if callback:
        callback()

def fourth():
    print('fourth 함수 실행 - 실행 완료!')

# 콜백 함수 체인 만들기 - 순차적 실행
first(lambda: second(lambda: third(fourth)))
print('콜백 체인 실행 완료!')

# 비동기적 실행 (실제로는 동기적 순차 실행)
first()
second()
third()
fourth()
print('비동기적 실행 완료!')