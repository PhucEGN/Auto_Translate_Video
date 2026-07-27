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
DEFAULT_VOICE = "Minh Đức"
DEFAULT_STYLE = "tu_nhien"
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_STT_MODEL = "whisper-large-v3"

# --- CẤU HÌNH CLONE GIỌNG (VOICE CLONING) ---
ENABLE_VOICE_CLONE = False       # Trạng thái mặc định
REF_AUDIO_FILENAME = "ref.wav"    # Tên file audio mẫu dùng để nhái giọng
REF_AUDIO_DURATION = 10.0         # Độ dài tối đa của file audio mẫu (giây)