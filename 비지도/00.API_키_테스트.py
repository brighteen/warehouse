import os
from dotenv import load_dotenv

def test_google_api_key():
    """
    .env 파일에서 Google API 키를 불러와 테스트하는 함수
    """
    # .env 파일 로드
    load_dotenv()
    
    # Google API 키 가져오기
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    # API 키 확인
    if google_api_key:
        print("Google API 키를 성공적으로 불러왔습니다!")
        print(f"API 키 앞 부분: {google_api_key[:5]}{'*' * 15}")  # 보안을 위해 API 키 전체를 출력하지 않음
        return True
    else:
        print("Google API 키를 불러오는데 실패했습니다. .env 파일을 확인해주세요.")
        return False

def test_simple_api_request():
    """
    간단한 API 요청을 테스트하는 함수 (선택적)
    """
    try:
        import requests
        
        # .env 파일 로드
        load_dotenv()
        
        # Google API 키 가져오기
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            print("API 키가 없습니다.")
            return False
            
        # 테스트용 API 요청 (예: Google Maps API 간단 테스트)
        # 실제 사용하는 API에 맞게 URL과 매개변수를 수정하세요
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address=Seoul&key={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            print("API 요청 성공!")
            print("응답 결과 (일부):", response.json().get("status"))
            return True
        else:
            print(f"API 요청 실패: 상태 코드 {response.status_code}")
            print(response.text)
            return False
            
    except ImportError:
        print("requests 라이브러리가 설치되어 있지 않습니다. 'pip install requests'를 실행하세요.")
        return False
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

if __name__ == "__main__":
    print("==== Google API 키 테스트 시작 ====")
    key_loaded = test_google_api_key()
    
    if key_loaded:
        print("\n==== API 요청 테스트 시작 ====")
        user_input = input("API 요청 테스트를 진행하시겠습니까? (y/n): ")
        if user_input.lower() == 'y':
            test_simple_api_request()
        else:
            print("API 요청 테스트를 건너뜁니다.")
    
    print("\n==== 테스트 완료 ====")