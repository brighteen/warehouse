import winreg
from ultralytics import YOLO
import cv2
import os
import ctypes

def get_wallpaper_path():
    key_path = r"Control Panel\Desktop"
    try:
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ)
        wallpaper_path, _ = winreg.QueryValueEx(registry_key, "WallPaper")
        winreg.CloseKey(registry_key)
        return wallpaper_path
    except Exception as e:
        print("Error:", e)
        return None

def set_wallpaper(image_path):
    SPI_SETDESKWALLPAPER = 20
    # 배경화면을 설정 (3: 업데이트된 변경사항 저장)
    result = ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, image_path, 3)
    if not result:
        print("배경화면 설정에 실패했습니다.")
    else:
        print("배경화면이 성공적으로 변경되었습니다.")

# 현재 배경화면 경로 가져오기
wallpaper_path = get_wallpaper_path()
if not wallpaper_path or not os.path.exists(wallpaper_path):
    print("배경화면 경로를 가져오지 못했거나, 파일이 존재하지 않습니다.")
    exit()

print("현재 배경화면 경로:", wallpaper_path)

# YOLO 모델 불러오기 및 배경화면 이미지에 대해 객체 검출 수행
model = YOLO("yolov8s.pt")  # 원하는 모델을 사용하세요
result = model.predict(wallpaper_path)

# 예측 결과 이미지에 검출 박스 그리기
detected_img = result[0].plot()

# 결과 이미지를 저장할 경로 설정 (원본과 같은 폴더에 저장)
new_wallpaper_path = os.path.join(os.path.dirname(wallpaper_path), "wallpaper_detected.jpg")
cv2.imwrite(new_wallpaper_path, detected_img)
print("검출 결과 이미지가 저장되었습니다:", new_wallpaper_path)

# 검출 결과 이미지를 새로운 배경화면으로 설정
set_wallpaper(new_wallpaper_path)
