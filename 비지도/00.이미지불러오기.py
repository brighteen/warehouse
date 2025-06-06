import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import base64
import io
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
import random

# 이미지를 저장할 디렉토리 설정
# 현재 폴더의 상위 폴더에 있는 이미지 파일 경로
base_dir = r'C:\Users\brigh\Documents\GitHub\warehouse\비지도\artwork_data'  # 이 경로는 위에서 본 3_1/6/ 폴더에 이미지들이 있음

# 이미지 파일 목록 가져오기
image_files = glob.glob(os.path.join(base_dir, '*.jpg'))
print(f"발견된 이미지 파일 수: {len(image_files)}")

# 일부 이미지 파일 출력
for i, img_path in enumerate(image_files[:5]):
    print(f"이미지 {i+1}: {os.path.basename(img_path)}")

# 이미지 불러와서 시각화하기
plt.figure(figsize=(15, 10))

for i, img_path in enumerate(image_files[:5]):
    try:
        img = Image.open(img_path)
        plt.subplot(1, 5, i+1)
        plt.imshow(np.array(img))
        plt.title(os.path.basename(img_path))
        plt.axis('off')
    except Exception as e:
        print(f"이미지 {img_path} 로드 중 오류 발생: {e}")

plt.tight_layout()
plt.show()

# 이미지 크기 정보 출력
print("\n이미지 크기 정보:")
for i, img_path in enumerate(image_files[:5]):
    try:
        img = Image.open(img_path)
        print(f"이미지 {os.path.basename(img_path)}: {img.size}, 모드: {img.mode}")
    except Exception as e:
        print(f"이미지 {img_path} 정보 확인 중 오류 발생: {e}")

# AI Agent를 사용한 이미지 분석
async def main():
    """메인 실행 함수"""
    if image_files:
        # 무작위로 이미지 하나 선택
        selected_image = select_random_image()
        print(f"\n🎨 선택된 이미지: {os.path.basename(selected_image)}")
        
        # 선택된 이미지 표시
        plt.figure(figsize=(8, 6))
        img = Image.open(selected_image)
        plt.imshow(np.array(img))
        plt.title(f"선택된 작품: {os.path.basename(selected_image)}")
        plt.axis('off')
        plt.show()
        
        # Agent를 사용하여 이미지 분석
        await analyze_artwork_with_agent(selected_image)
    else:
        print("분석할 이미지가 없습니다.")

# 비동기 함수 실행
if __name__ == "__main__":
    # Jupyter 환경에서는 이미 이벤트 루프가 실행 중일 수 있으므로 체크
    try:
        asyncio.run(main())
    except RuntimeError:
        # Jupyter 환경인 경우 asyncio 이벤트 루프 처리
        try:
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        except:
            print("비동기 실행 오류가 발생했습니다. 동기 버전으로 실행합니다.")
            # 동기 버전으로 실행
            if image_files:
                selected_image = select_random_image()
                print(f"\n🎨 선택된 이미지: {os.path.basename(selected_image)}")
                
                # 선택된 이미지 표시
                plt.figure(figsize=(8, 6))
                img = Image.open(selected_image)
                plt.imshow(np.array(img))
                plt.title(f"선택된 작품: {os.path.basename(selected_image)}")
                plt.axis('off')
                plt.show()
                
                # 기본 정보 출력
                print(describe_artwork(selected_image))
                print("\n" + "="*50)
                print(analyze_image_content(selected_image))

# API 키 설정
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

def image_to_base64(image_path):
    """이미지를 base64로 인코딩하는 함수"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_artwork(image_path: str) -> str:
    """선택된 이미지에 대한 상세한 설명을 제공하는 도구"""
    try:
        # 이미지가 존재하는지 확인
        if not os.path.exists(image_path):
            return f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}"
        
        # 이미지 정보 수집
        img = Image.open(image_path)
        filename = os.path.basename(image_path)
        
        # 이미지를 base64로 인코딩
        base64_image = image_to_base64(image_path)
        
        description = f"""
이미지 기본 정보:
- 파일명: {filename}
- 이미지 크기: {img.size[0]} x {img.size[1]} 픽셀
- 이미지 모드: {img.mode}
- 파일 크기: {round(os.path.getsize(image_path)/1024, 2)} KB

이 이미지는 artwork_data 폴더에 저장된 예술 작품입니다.
AI가 이미지 내용을 분석하여 상세한 설명을 제공해드립니다.
        """
        
        return description
        
    except Exception as e:
        return f"이미지 분석 중 오류가 발생했습니다: {str(e)}"

def analyze_image_content(image_path: str) -> str:
    """Gemini Vision API를 사용하여 이미지 내용을 분석하는 도구"""
    try:
        # API 키 설정
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Vision 모델 초기화
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # 이미지 읽기
        img = Image.open(image_path)
        
        # 프롬프트 작성
        prompt = """
        이 예술 작품 이미지를 자세히 분석해주세요. 다음 내용을 포함해주세요:
        
        1. 작품의 스타일과 장르 (예: 회화, 조각, 사진 등)
        2. 주요 색상과 색조
        3. 구성 요소와 주제
        4. 예술적 기법이나 특징
        5. 전체적인 분위기나 감정
        6. 추정되는 시대나 스타일
        
        한국어로 상세하고 흥미롭게 설명해주세요.
        """
        
        # 이미지 분석 요청
        response = model.generate_content([prompt, img])
        
        return response.text
        
    except Exception as e:
        return f"AI 이미지 분석 중 오류가 발생했습니다: {str(e)}"

# 이미지 설명 Agent 정의
image_description_agent = Agent(
    name="artwork_description_agent",
    model="gemini-2.0-flash",
    description="예술 작품 이미지를 분석하고 상세한 설명을 제공하는 전문 에이전트",
    instruction="""당신은 예술 작품 이미지를 분석하는 전문가입니다. 
    사용자가 이미지에 대한 설명을 요청하면 다음 단계를 따르세요:
    1. 먼저 describe_artwork 도구를 사용하여 이미지의 기본 정보를 확인하세요.
    2. 그다음 analyze_image_content 도구를 사용하여 AI 기반 상세 분석을 수행하세요.
    3. 두 분석 결과를 종합하여 친근하고 교육적인 설명을 제공해주세요.""",
    tools=[describe_artwork, analyze_image_content]
)

# 세션 서비스 및 러너 설정
session_service = InMemorySessionService()
APP_NAME = "artwork_analysis_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)

runner = Runner(
    agent=image_description_agent,
    app_name=APP_NAME,
    session_service=session_service
)

async def analyze_artwork_with_agent(image_path):
    """Agent를 사용하여 이미지를 분석하는 함수"""
    query = f"이 이미지에 대해 분석해주세요: {image_path}"
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    print(f"\n=== AI Agent 이미지 분석 ===")
    print(f"분석할 이미지: {os.path.basename(image_path)}")
    print("분석 중...")
    
    final_response = ""
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text
            break
    
    print(f"\n📝 Agent 분석 결과:")
    print(final_response)
    return final_response

def select_random_image():
    """무작위로 이미지 하나를 선택하는 함수"""
    if image_files:
        selected_image = random.choice(image_files)
        return selected_image
    return None