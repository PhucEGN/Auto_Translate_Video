import os
from pathlib import Path
from dotenv import load_dotenv

# Tự động load file .env ở thư mục gốc
load_dotenv()

# Đường dẫn thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent

# Đường dẫn thư mục output
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"

# Tự động tạo thư mục nếu chưa tồn tại
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Lấy API Key và Models
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TTS_VOICE = "vi-VN-HoaiMyNeural"  # Giọng đọc tiếng Việt mượt mà
GROQ_LLM_MODEL = "llama-3.1-8b-instant"
GROQ_STT_MODEL = "whisper-large-v3"