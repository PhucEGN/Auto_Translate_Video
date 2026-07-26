import json
import os
import re
import sys
import time
import threading
import subprocess
import urllib.parse
import requests
from pathlib import Path
from datetime import datetime
import yt_dlp
from tqdm import tqdm

# Đảm bảo import được config dù chạy ở thư mục nào
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class QuietLogger:
    """Class dùng để chặn toàn bộ log/warning rác từ yt-dlp."""
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


class VideoDownloader:
    """Class quản lý việc tải video và tách âm thanh chuẩn định dạng từ URL (Hỗ trợ Douyin, YouTube, v.v.)."""

    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir if temp_dir else config.TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_douyin_url(self, raw_input: str) -> str:
        """Trích xuất URL từ chuỗi chia sẻ Douyin và theo dõi redirect đến link gốc."""
        # 1. Trích xuất URL từ đoạn văn bản chia sẻ
        url_match = re.search(r'https?://[^\s]+', raw_input)
        if not url_match:
            return raw_input
        
        extracted_url = url_match.group(0)

        # 2. Xử lý link rút gọn v.douyin.com
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        try:
            res = requests.get(extracted_url, headers=headers, allow_redirects=True, timeout=10)
            final_url = res.url
            # Xóa các tham số tracking không cần thiết trên URL Douyin
            parsed = urllib.parse.urlparse(final_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            return clean_url
        except Exception:
            return extracted_url

    def _monitor_progress(self, folder_path: Path, filesize_estimate: int, stop_event: threading.Event):
        """Thread chạy ngầm quét dung lượng tất cả file tạm trong folder để cập nhật tqdm khi dùng aria2c."""
        if filesize_estimate > 0:
            pbar = tqdm(
                total=filesize_estimate,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="📥 [Downloading Video]",
                ncols=80,
                leave=True
            )
        else:
            pbar = tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="📥 [Downloading Video]",
                ncols=80,
                leave=True
            )

        last_bytes = 0
        while not stop_event.is_set():
            current_bytes = 0
            for file in folder_path.iterdir():
                if file.is_file() and not file.name.endswith(('.json', '.mp3', '.aria2')):
                    try:
                        current_bytes += file.stat().st_size
                    except FileNotFoundError:
                        pass

            if current_bytes > last_bytes:
                diff = current_bytes - last_bytes
                if filesize_estimate > 0:
                    increment = min(diff, pbar.total - pbar.n)
                    if increment > 0:
                        pbar.update(increment)
                else:
                    pbar.update(diff)
                last_bytes = current_bytes

            time.sleep(0.2)

        if filesize_estimate > 0 and pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
        pbar.close()

    def _get_cookie_file(self) -> str | None:
        """Kiểm tra và trả về đường dẫn file cookies.txt nếu có."""
        # Kiểm tra file cookies.txt trong thư mục gốc của dự án hoặc thư mục temp
        project_root = Path(__file__).resolve().parent.parent
        possible_paths = [
            project_root / "cookies.txt",
            self.temp_dir / "cookies.txt",
            self.temp_dir / "douyin_cookies.txt"
        ]

        for p in possible_paths:
            if p.exists() and p.stat().st_size > 0:
                return str(p)

        # Nếu chưa có, tự động tạo 1 file cookie giả định cơ bản cho Douyin
        default_cookie = self.temp_dir / "douyin_cookies.txt"
        with open(default_cookie, "w", encoding="utf-8") as f:
            f.write("# Netscape HTTP Cookie File\n")
            f.write(".douyin.com\tTRUE\t/\tFALSE\t0\tpassport_csrf_token\tdummy_token\n")
        return str(default_cookie)

    def download_video(self, raw_url: str, verbose: bool = True):
        """Tải video Douyin/YouTube và xuất ra thư mục DD-MM-YYYY_HH-MM-SS."""
        
        # Xử lý URL nếu là link Douyin
        video_url = self._resolve_douyin_url(raw_url)

        # Cấu hình Headers chuẩn trình duyệt
        common_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Referer': 'https://www.douyin.com/',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
        }

        cookie_file = self._get_cookie_file()

        quiet_opts = {
            'quiet': True,
            'no_warnings': True,
            'logger': QuietLogger(),
            'http_headers': common_headers,
        }
        if cookie_file:
            quiet_opts['cookiefile'] = cookie_file

        # 1. Trích xuất thông tin video
        filesize_estimate = 0
        try:
            if verbose:
                print("📡 Đang kết nối và lấy thông tin video...")
                
            with yt_dlp.YoutubeDL(quiet_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                video_title = info.get('title', 'Unknown Title')
                duration = info.get('duration', 0)
                uploader = info.get('uploader', 'Unknown')
                
                filesize_estimate = info.get('filesize') or info.get('filesize_estimate') or 0
                if not filesize_estimate and 'requested_formats' in info:
                    for fmt in info['requested_formats']:
                        filesize_estimate += fmt.get('filesize') or fmt.get('filesize_estimate') or 0
        except Exception:
            video_title = "Douyin Video"
            duration = 0
            uploader = "Unknown"

        # 2. Tạo thư mục chứa dữ liệu
        created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        folder_path = self.temp_dir / created_at
        folder_path.mkdir(parents=True, exist_ok=True)

        video_path = folder_path / "file.mp4"
        audio_path = folder_path / "file.mp3"
        json_path = folder_path / "info.json"

        # 3. Chuẩn bị metadata JSON
        metadata = {
            "title": video_title,
            "url": video_url,
            "created_at": created_at,
            "duration": duration,
            "uploader": uploader,
            "transcript": None,
            "translated_text": None,
            "status": "downloaded"
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        # Cấu hình tải chính
        ydl_opts = {
            'outtmpl': str(folder_path / 'file.%(ext)s'),
            'format': 'best[vcodec~=h264][acodec!=none]/best[vcodec!=none][acodec!=none]/best',
            'merge_output_format': 'mp4',
            'overwrite': True,
            'quiet': True,
            'no_warnings': True,
            'logger': QuietLogger(),
            'http_headers': common_headers,
            
            # --- CẤU HÌNH TỐI ƯU ARIA2C ---
            'external_downloader': 'aria2c',
            'external_downloader_args': [
                '-x16', '-s16', '-k1M',
                '--min-split-size=1M',
                '--max-connection-per-server=16',
                '--summary-interval=0',
                '--console-log-level=warn',
                '--quiet=true'
            ],
            'nocheckcertificate': True,
        }

        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file

        try:
            if verbose:
                print(f"🎬 Tiêu đề: {video_title}")

            # Thread giám sát tiến trình
            stop_event = threading.Event()
            monitor_thread = None
            if verbose:
                monitor_thread = threading.Thread(
                    target=self._monitor_progress,
                    args=(folder_path, filesize_estimate, stop_event),
                    daemon=True
                )
                monitor_thread.start()

            # Thực hiện tải file
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            if monitor_thread:
                stop_event.set()
                monitor_thread.join()

            # Chuẩn hóa tên file xuất ra thành file.mp4
            downloaded_mp4 = folder_path / "file.mp4"
            if not downloaded_mp4.exists():
                for pf in folder_path.glob("file.*"):
                    if pf.suffix in ['.mp4', '.mkv', '.webm', '.flv']:
                        pf.rename(video_path)
                        break

            # 4. Tách file.mp3 từ file.mp4 bằng FFmpeg
            if verbose:
                print("🎙️ Đang tách âm thanh ra file.mp3...")
            
            self._extract_mp3(video_path, audio_path)

            if verbose:
                print("✅ Tải và xử lý hoàn tất!")
                print(f" 📂 Thư mục: {folder_path.name}")
                print(f" ├─ Video: {video_path.name}")
                print(f" ├─ Audio: {audio_path.name}")
                print(f" └─ Metadata: {json_path.name}\n")

            return True, folder_path, video_path, audio_path, json_path

        except Exception as e:
            if verbose:
                print(f"❌ Có lỗi xảy ra khi tải video: {e}")
            raise e

    def _extract_mp3(self, input_video: Path, output_mp3: Path):
        """Tách luồng audio ra MP3 chuẩn Mono 64kbps / 16kHz (tối ưu cho Whisper)."""
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_video),
            '-vn',
            '-c:a', 'libmp3lame',
            '-b:a', '64k',
            '-ac', '1',
            '-ar', '16000',
            str(output_mp3)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


if __name__ == "__main__":
    link_douyin = "🔗 Link Video: https://v.douyin.com/vwz_mEohrm0/ 复制此链接，打开Dou音搜索，直接观看视频！"
    downloader = VideoDownloader()
    downloader.download_video(link_douyin, verbose=True)