<div align="center">

# 🎬 Quy Trình Tự Động Thuyết Minh & Chèn Phụ Đề Video

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![yt-dlp](https://img.shields.io/badge/yt--dlp-mới%20nhất-red.svg?style=for-the-badge)](https://github.com/yt-dlp/yt-dlp)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Đã%20hỗ%20trợ-0078D7.svg?style=for-the-badge&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Groq](https://img.shields.io/badge/Groq-Whisper%20%26%20LLaMA3-orange.svg?style=for-the-badge)](https://groq.com/)
[![Giấy phép: MIT](https://img.shields.io/badge/Giấy%20phép-MIT-yellow.svg?style=for-the-badge)](LICENSE)

Quy trình tự động hóa khép kín (End-to-End) bằng Python hỗ trợ: Tải video ngắn từ nhiều nền tảng, nhận diện giọng nói, dịch thuật chuẩn ngữ cảnh, tạo giọng đọc thuyết minh tiếng Việt và chèn phụ đề cứng đẹp mắt.

[Tính năng](#-tính-năng-nổi-bật) • [Kiến trúc Pipeline](#-kiến-trúc-pipeline) • [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống) • [Cài đặt](#-hướng-dẫn-cài-đặt) • [Cách sử dụng](#-hướng-dẫn-sử-dụng)

</div>

---

## 🌟 Tính Năng Nổi Bật

- 📥 **Tải Video Đa Nền Tảng**: Hỗ trợ tải video ngắn chất lượng cao nhất từ YouTube Shorts, TikTok, Facebook Reels,... bằng `yt-dlp`.
- 🔊 **Xử Lý Âm Thanh Chuẩn**: Tự động tách, chuyển đổi codec âm thanh về AAC 192kbps chuẩn hóa tần số lấy mẫu ($44.1\text{ kHz}$), đảm bảo không bị rè/ù hay câm tiếng.
- 🎙️ **Nhận Dạng Giọng Nói (STT)**: Chuyển đổi giọng nói nước ngoài thành văn bản kèm mốc thời gian chính xác từng từ bằng **Groq Whisper**.
- 🤖 **Dịch Thuật Thông Minh**: Dịch thuật tự nhiên sang tiếng Việt bằng **Groq LLaMA 3.3**, đảm bảo sát ngữ cảnh và khớp độ dài câu.
- 🗣️ **Tạo Giọng Đọc AI (TTS)**: Tạo giọng thuyết minh tiếng Việt truyền cảm, mượt mà bằng **Edge-TTS**.
- 🎬 **Biên Tập & Chèn Phụ Đề**: Tự động ghép nối âm thanh mới và ép phụ đề cứng (Burn-in Subtitles) tùy biến kiểu dáng vào video bằng **FFmpeg**.

---

## 🔄 Kiến Trúc Pipeline

```text
[ Link Video Đầu Vào ]
        │
        ▼
 (1) Tải Video & Tách Âm Thanh (yt-dlp + FFmpeg)
        │
        ├────────────────────────┐
        ▼                        ▼
 [ File Audio (.m4a/.mp3) ]   [ File Video (.mp4) ]
        │                        │
        ▼                        │
 (2) Nhận Dạng Giọng Nói         │
     (Groq Whisper)              │
        │                        │
        ▼                        │
 (3) Dịch Thuật & Khớp Thời Gian │
     (Groq LLaMA 3.3)            │
        │                        │
        ├────────────────┐       │
        ▼                ▼       │
 [ Văn Bản Tiếng Việt ] [ .SRT ] │
        │                │       │
        ▼                │       │
 (4) Tạo Giọng Đọc AI    │       │
     (Edge-TTS)          │       │
        │                │       │
        ▼                ▼       ▼
 (5) Ghép Âm Thanh & Đè Phụ Đề (FFmpeg)
        │
        ▼
[ Video Hoàn Chỉnh (.mp4) ]
```
# 📋 Yêu Cầu Hệ Thống
1. **Python 3.10+**
2. **Node.js (LTS) (Bắt buộc cho yt-dlp để giải mã JavaScript của YouTube)**
3. **FFmpeg (Bắt buộc để ghép video/audio và vẽ phụ đề)**

## Hướng dẫn cài nhanh FFmpeg trên Windows (qua PowerShell / CMD):
```bash
winget install Gyan.FFmpeg
```

# ⚙️ Hướng Dẫn Cài Đặt
1. **Clone (Tải) thư mục dự án về máy:**
```bash
git clone https://github.com/PhucEGN/Auto_Translate_Video
cd Auto_Translate_Video
```
2. **Tạo môi trường ảo Python (Khuyên dùng):**
```bash
python -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

3. **Cài đặt các thư viện phụ thuộc:**
```bash
pip install -r requirements.txt
```

4. **Cấu hình Biến môi trường:** 
- Tạo 1 file tên .env tại thư mục gốc và thêm API Key của bạn:
```bash
GROQ_API_KEY=key_groq_api_cua_ban_tai_day
```

# 🚀 Hướng Dẫn Sử Dụng
1. **Nhập API Key vào ".env"**
# 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ chính: Python**

- **Xử lý Media: yt-dlp, FFmpeg, moviepy**

- **Mô hình AI: Groq API (whisper-large-v3, llama-3.3-70b), edge-tts**

# 📝 Giấy Phép (License)
- **MIT**