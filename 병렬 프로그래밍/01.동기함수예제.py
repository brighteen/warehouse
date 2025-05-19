import time
import datetime

def task(seconds):
    print(f'[작업시작] {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # 시작 메시지를 출력합니다. 작업이 몇 초 후에 끝날지 알려줍니다.
    print(f"이 작업은 {seconds} 초 뒤 종료됩니다.")
    
    # time.sleep 함수를 사용하여 지정된 시간 동안 대기합니다.
    time.sleep(seconds)
    
    # 대기 시간이 끝나면, 작업 완료 메시지를 출력합니다.
    print(f'[작업종료] {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"작업이 끝났습니다.\n")

def main():
    # task(1), task(2), task(3)을 순차적으로 호출합니다.
    task(1)
    task(2)
    task(3)
    
main()