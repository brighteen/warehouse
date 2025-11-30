import winreg

def get_wallpaper_path():
    # 레지스트리 키 경로
    key_path = r"Control Panel\Desktop"
    try:
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ)
        # "WallPaper" 값 읽기
        wallpaper_path, _ = winreg.QueryValueEx(registry_key, "WallPaper")
        winreg.CloseKey(registry_key)
        return wallpaper_path
    except Exception as e:
        return f"Error: {e}"

print("현재 윈도우 배경화면 경로:", get_wallpaper_path())
