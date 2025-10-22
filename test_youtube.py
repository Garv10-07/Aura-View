# file: test_youtube.py

import yt_dlp

# Wahi YouTube link yahan daalo jo aap use kar rahe ho
YOUTUBE_URL = "https://www.youtube.com/live/FyFAqPHBKiQ?si=VmZeUAUDMU_wNz52"

print(f"📡 YouTube stream ({YOUTUBE_URL}) se direct link nikaalne ki koshish...")

try:
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(YOUTUBE_URL, download=False)
        stream_url = info['url']
    
    print("\n✅✅✅ SUCCESS! Direct stream link mil gaya hai:")
    print(stream_url)

except Exception as e:
    print("\n❌❌❌ ERROR! Link nikaalne mein problem aa rahi hai.")
    print("Error Details:", e)