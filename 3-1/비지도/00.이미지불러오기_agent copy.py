import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import base64
import io
import random
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

# ADK 관련 import
try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    print("ADK가 설치되지 않았습니다. 기본 기능만 사용합니다.")
    ADK_AVAILABLE = False

# Gemini Vision API import
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    print("google-generativeai가 설치되지 않았습니다. Vision 분석 기능이 제한됩니다.")
    GENAI_AVAILABLE = False

# API 키 설정
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# 이미지를 저장할 디렉토리 설정
base_dir = r'C:\Users\brigh\Documents\GitHub\warehouse\비지도\artwork_data'

# 이미지 파일 목록 가져오기
image_files = glob.glob(os.path.join(base_dir, '*.jpg'))
print(f"발견된 이미지 파일 수: {len(image_files)}")

# 일부 이미지 파일 출력
for i, img_path in enumerate(image_files[:5]):
    print(f"이미지 {i+1}: {os.path.basename(img_path)}")

def image_to_base64(image_path):
    """이미지를 base64로 인코딩하는 함수"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def select_random_images(count=3):
    """무작위로 지정된 개수의 이미지들을 선택하는 함수"""
    if len(image_files) < count:
        print(f"경고: 요청된 이미지 수({count})가 사용 가능한 이미지 수({len(image_files)})보다 많습니다.")
        return image_files[:count] if image_files else []
    
    selected_images = random.sample(image_files, count)
    return selected_images

def select_random_image():
    """무작위로 이미지 하나를 선택하는 함수 (기존 호환성 유지)"""
    if image_files:
        selected_image = random.choice(image_files)
        return selected_image
    return None

def describe_artwork(image_path: str) -> str:
    """선택된 이미지에 대한 상세한 설명을 제공하는 도구"""
    try:
        # 이미지가 존재하는지 확인
        if not os.path.exists(image_path):
            return f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}"
        
        # 이미지 정보 수집
        img = Image.open(image_path)
        filename = os.path.basename(image_path)
        
        description = f"""
        📋 이미지 기본 정보:
        - 파일명: {filename}
        - 이미지 크기: {img.size[0]} x {img.size[1]} 픽셀
        - 이미지 모드: {img.mode}
        - 파일 크기: {round(os.path.getsize(image_path)/1024, 2)} KB

        이 이미지는 artwork_data 폴더에 저장된 예술 작품입니다.
        """
        
        return description
        
    except Exception as e:
        return f"이미지 분석 중 오류가 발생했습니다: {str(e)}"

def analyze_image_content(image_path: str) -> str:
    """Gemini Vision API를 사용하여 이미지 내용을 분석하는 도구"""
    if not GENAI_AVAILABLE:
        return "google-generativeai 패키지가 설치되지 않았습니다. pip install google-generativeai를 실행해주세요."
    
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
        
        return f"🎨 AI 예술 분석 결과:\n{response.text}"
        
    except Exception as e:
        return f"AI 이미지 분석 중 오류가 발생했습니다: {str(e)}"

def display_basic_images():
    """기본 이미지 표시 기능"""
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

# ADK Agent 설정 (ADK가 설치된 경우에만)
if ADK_AVAILABLE:
    # 이미지 설명 Agent 정의
    image_description_agent = Agent(
        name="artwork_description_agent",
        model="gemini-2.0-flash",
        description="예술 작품 이미지를 분석하고 간략한 설명을 제공하는 전문 에이전트",        instruction="""당신은 예술 작품 이미지를 분석하는 전문가입니다. 
        사용자가 이미지에 대한 설명을 요청하면 다음 단계를 따르세요:
        1. 먼저 describe_artwork 도구를 사용하여 이미지의 기본 정보를 확인하세요.
        2. 그다음 analyze_image_content 도구를 사용하여 AI 기반 상세 분석을 수행하세요.
        3. 두 분석 결과를 종합하여 친근하고 교육적인 설명을 제공해주세요.
        4. 한국어로 분석 결과를 간략하게게 작성해주세요.""",
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
        
        print(f"\n=== 🤖 AI Agent 이미지 분석 ===")
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

    async def main_with_agent():
        """ADK Agent를 사용한 메인 실행 함수"""
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

def main_without_agent():
    """ADK 없이 기본 기능만 사용하는 메인 함수"""
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
        
        # 기본 정보 출력
        print(describe_artwork(selected_image))
        print("\n" + "="*50)
        print(analyze_image_content(selected_image))
    else:
        print("분석할 이미지가 없습니다.")

# 특정 이미지 분석 함수
def analyze_specific_image(image_filename):
    """지정된 이미지 파일명을 찾아서 분석하는 함수"""
    # 지정된 이미지 파일 찾기
    target_image = None
    for img_path in image_files:
        if os.path.basename(img_path) == image_filename:
            target_image = img_path
            break
    
    if not target_image:
        print(f"❌ 오류: '{image_filename}' 이미지를 찾을 수 없습니다.")
        print("사용 가능한 이미지 파일들:")
        for i, img_path in enumerate(image_files[:10]):  # 처음 10개만 표시
            print(f"  - {os.path.basename(img_path)}")
        if len(image_files) > 10:
            print(f"  ... 외 {len(image_files)-10}개 더")
        return None
    
    return target_image

async def analyze_specific_with_agent(image_filename):
    """특정 이미지를 Agent로 분석"""
    target_image = analyze_specific_image(image_filename)
    if not target_image:
        return
    
    print(f"\n🎨 분석 대상 이미지: {image_filename}")
    
    # 선택된 이미지 표시
    plt.figure(figsize=(10, 8))
    img = Image.open(target_image)
    plt.imshow(np.array(img))
    plt.title(f"분석 대상 작품: {image_filename}")
    plt.axis('off')
    plt.show()
    
    # Agent를 사용하여 이미지 분석
    await analyze_artwork_with_agent(target_image)

def analyze_specific_without_agent(image_filename):
    """특정 이미지를 기본 기능으로 분석"""
    target_image = analyze_specific_image(image_filename)
    if not target_image:
        return
    
    print(f"\n🎨 분석 대상 이미지: {image_filename}")
    
    # 선택된 이미지 표시
    plt.figure(figsize=(10, 8))
    img = Image.open(target_image)
    plt.imshow(np.array(img))
    plt.title(f"분석 대상 작품: {image_filename}")
    plt.axis('off')
    plt.show()
    
    # 기본 정보 출력
    print(describe_artwork(target_image))
    print("\n" + "="*50)
    print(analyze_image_content(target_image))

async def analyze_multiple_images_with_agent(image_count=3):
    """여러 이미지를 Agent로 분석하고 JSON으로 저장하는 함수"""
    if len(image_files) < image_count:
        print(f"❌ 오류: 분석할 이미지 수({image_count})가 사용 가능한 이미지 수({len(image_files)})보다 많습니다.")
        return
    
    # 무작위로 이미지들 선택
    selected_images = select_random_images(image_count)
    analysis_results = {}
    
    print(f"\n🎨 {image_count}개 이미지 분석을 시작합니다...")
    print("="*60)
    
    for i, image_path in enumerate(selected_images, 1):
        image_name = os.path.basename(image_path)
        print(f"\n📸 [{i}/{image_count}] 분석 중: {image_name}")
        
        # 이미지 표시
        plt.figure(figsize=(8, 6))
        img = Image.open(image_path)
        plt.imshow(np.array(img))
        plt.title(f"분석 중인 작품 {i}: {image_name}")
        plt.axis('off')
        plt.show()
        
        # Agent를 사용하여 이미지 분석
        try:
            query = f"이 이미지에 대해 간결하고 명확하게 분석해주세요: {image_path}"
            content = types.Content(role='user', parts=[types.Part(text=query)])
            
            print(f"🤖 AI Agent 분석 중...")
            
            final_response = ""
            async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text
                    break            # 결과 저장 - 파일명을 키로, 분석결과를 값으로 하는 간단한 구조
            analysis_results[image_name] = final_response
            
            print(f"✅ 분석 완료!")
            print(f"📝 결과 미리보기: {final_response[:100]}...")
            
        except Exception as e:
            error_msg = f"분석 중 오류 발생: {str(e)}"
            analysis_results[f"사진{i}"] = {
                "파일명": image_name,
                "분석결과": error_msg
            }
            print(f"❌ 오류: {error_msg}")
    
    # JSON 파일로 저장
    save_analysis_to_json(analysis_results)
    return analysis_results

def analyze_multiple_images_without_agent(image_count=3):
    """여러 이미지를 기본 기능으로 분석하고 JSON으로 저장하는 함수"""
    if len(image_files) < image_count:
        print(f"❌ 오류: 분석할 이미지 수({image_count})가 사용 가능한 이미지 수({len(image_files)})보다 많습니다.")
        return
    
    # 무작위로 이미지들 선택
    selected_images = select_random_images(image_count)
    analysis_results = {}
    
    print(f"\n🎨 {image_count}개 이미지 분석을 시작합니다...")
    print("="*60)
    
    for i, image_path in enumerate(selected_images, 1):
        image_name = os.path.basename(image_path)
        print(f"\n📸 [{i}/{image_count}] 분석 중: {image_name}")
        
        # 이미지 표시
        plt.figure(figsize=(8, 6))
        img = Image.open(image_path)
        plt.imshow(np.array(img))
        plt.title(f"분석 중인 작품 {i}: {image_name}")
        plt.axis('off')
        plt.show()
        
        # 기본 분석 수행
        try:
            basic_info = describe_artwork(image_path)
            ai_analysis = analyze_image_content(image_path)
            
            combined_analysis = f"{basic_info}\n\n{ai_analysis}"
            
            # 결과 저장
            analysis_results[f"사진{i}"] = {
                "파일명": image_name,
                "분석결과": combined_analysis
            }
            
            print(f"✅ 분석 완료!")
            print(f"📝 결과 미리보기: {combined_analysis[:100]}...")
            
        except Exception as e:
            error_msg = f"분석 중 오류 발생: {str(e)}"
            analysis_results[f"사진{i}"] = {
                "파일명": image_name,
                "분석결과": error_msg
            }
            print(f"❌ 오류: {error_msg}")
    
    # JSON 파일로 저장
    save_analysis_to_json(analysis_results)
    return analysis_results

def save_analysis_to_json(analysis_results):
    """분석 결과를 JSON 파일로 저장하는 함수"""
    try:
        # 현재 실행 파일과 동일한 위치에 저장
        current_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"artwork_analysis_{timestamp}.json"
        json_filepath = os.path.join(current_dir, json_filename)
        
        # JSON 파일로 저장 (한국어 지원)
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 분석 결과가 저장되었습니다:")
        print(f"📁 파일 경로: {json_filepath}")
        print(f"📊 분석된 이미지 수: {len(analysis_results)}")
        
        return json_filepath
        
    except Exception as e:
        print(f"❌ JSON 저장 중 오류 발생: {str(e)}")
        return None

# 메인 실행 부분
if __name__ == "__main__":
    print("="*60)
    print("🎨 예술 작품 이미지 분석 시스템")
    print("="*60)
    
    # 3개 이미지 무작위 선택 및 분석
    image_count = 3
    
    print(f"\n🔍 {image_count}개 이미지 무작위 선택 및 상세 분석")
    print("="*60)
    
    if ADK_AVAILABLE:
        # ADK Agent를 사용한 분석
        try:
            asyncio.run(analyze_multiple_images_with_agent(image_count))
        except RuntimeError:
            # Jupyter 환경인 경우
            try:
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                loop.run_until_complete(analyze_multiple_images_with_agent(image_count))
            except:
                print("비동기 실행 오류가 발생했습니다. 기본 기능으로 실행합니다.")
                analyze_multiple_images_without_agent(image_count)
    else:
        # ADK 없이 기본 기능만 사용
        analyze_multiple_images_without_agent(image_count)
