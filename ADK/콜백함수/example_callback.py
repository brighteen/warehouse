import time
def first():
    '''
    1초 뒤에 second()콜백하기
    '''
    print('first')
    # 1초 뒤에 second()를 호출
    # time.sleep(1)
    time.sleep(1)

def second():
    '''
    3초 뒤에 third()콜백하기
    '''
    print('second')
    time.sleep(3)

def third():
    '''
    2초 뒤에 fourth()콜백하기
    '''
    print('third')
    time.sleep(2)

first(second(third))