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
        url_match = re.search(r'https?://[^\s]+', raw_input)
        if not url_match:
            return raw_input
        
        extracted_url = url_match.group(0)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        try:
            res = requests.get(extracted_url, headers=headers, allow_redirects=True, timeout=10)
            final_url = res.url
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
                if file.is_file() and not file.name.endswith(('.json', '.mp3', '.aria2', '.wav')):
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
        project_root = Path(__file__).resolve().parent.parent
        possible_paths = [
            project_root / "cookies.txt",
            self.temp_dir / "cookies.txt",
            self.temp_dir / "douyin_cookies.txt"
        ]

        for p in possible_paths:
            if p.exists() and p.stat().st_size > 0:
                return str(p)

        default_cookie = self.temp_dir / "douyin_cookies.txt"
        with open(default_cookie, "w", encoding="utf-8") as f:
            f.write("# Netscape HTTP Cookie File\n")
            f.write(".douyin.com\tTRUE\t/\tFALSE\t0\tpassport_csrf_token\tdummy_token\n")
        return str(default_cookie)

    def extract_reference_audio(self, input_audio: Path, output_ref_path: Path, duration: float = 10.0) -> Path:
        """Cắt một đoạn audio ngắn (mặc định 10s) ở đầu file làm mẫu clone giọng."""
        cmd = [
            'ffmpeg', '-y',
            '-ss', '00:00:01',  # Bỏ 1s đầu tránh giật
            '-i', str(input_audio),
            '-t', str(duration),
            '-c:a', 'pcm_s16le',
            '-ar', '24000',      # Sample rate phù hợp cho TTS clone
            '-ac', '1',
            str(output_ref_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_ref_path

    def separate_background_audio(
        self, 
        audio_or_video_path: Path, 
        output_dir: Path, 
        bg_volume: float = 0.3, 
        verbose: bool = True
    ) -> Path:
        """
        Sử dụng Demucs để tách 4 stem (drums, bass, other, vocals), 
        sau đó gộp (drums + bass + other) thành background.wav với bộ lọc cân bằng dải động
        và điều chỉnh âm lượng nền theo tỷ lệ phần trăm.
        
        :param bg_volume: Tỷ lệ âm lượng nhạc nền (ví dụ: 0.5 tương đương 50%, 1.0 là 100%).
        """
        audio_or_video_path = Path(audio_or_video_path)
        output_dir = Path(output_dir)

        # Ưu tiên sử dụng file.mp3 trong cùng thư mục nếu có
        mp3_candidate = output_dir / "file.mp3"
        target_audio_path = mp3_candidate if mp3_candidate.exists() else audio_or_video_path

        if verbose:
            print(f"🎵 Đang phân tách dải âm thanh & hiệu ứng từ '{target_audio_path.name}' bằng Demucs...")

        # 1. Gọi trực tiếp demucs.separate.main bằng Python Import
        try:
            from demucs.separate import main as demucs_main

            old_argv = sys.argv
            sys.argv = [
                'demucs',
                '-n', 'htdemucs',
                '-o', str(output_dir),
                str(target_audio_path)
            ]

            try:
                demucs_main()
                if verbose:
                    print("✅ Demucs đã hoàn thành tiến trình phân tách!")
            finally:
                sys.argv = old_argv

        except ImportError:
            raise ImportError("❌ Không thể import 'demucs.separate'. Vui lòng kiểm tra lại thư viện Demucs.")
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi trong quá trình chạy Demucs: {e}")

        # 2. Định vị thư mục stem chứa các file .wav kết quả
        htdemucs_dir = output_dir / "htdemucs"
        stem_dir = None

        if htdemucs_dir.exists():
            subdirs_stem = [d for d in htdemucs_dir.iterdir() if d.is_dir()]
            if subdirs_stem:
                stem_dir = subdirs_stem[0]
            elif list(htdemucs_dir.glob("*.wav")):
                stem_dir = htdemucs_dir

        if not stem_dir:
            stem_dir_alt = output_dir / "stem"
            if stem_dir_alt.exists():
                stem_dir = stem_dir_alt

        if not stem_dir or not stem_dir.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy thư mục chứa các file stem tại: {output_dir}")

        # 3. Lấy danh sách các dải âm thanh hiệu ứng (bỏ qua Vocals)
        drums_file = stem_dir / "drums.wav"
        bass_file = stem_dir / "bass.wav"
        other_file = stem_dir / "other.wav"

        available_inputs = []
        for f in [drums_file, bass_file, other_file]:
            if f.exists():
                available_inputs.extend(["-i", str(f)])

        if not available_inputs:
            raise FileNotFoundError(f"❌ Không tìm thấy dải hiệu ứng nào (drums, bass, other) trong {stem_dir}")

        # 4. Trộn bằng FFmpeg, áp dụng dynaudnorm để cân bằng tiếng động nhỏ/nhạc to và áp dụng bg_volume
        final_bg_path = output_dir / "background.wav"
        num_inputs = len(available_inputs) // 2

        # Filter explanation:
        # amix: Trộn các stem lại với nhau
        # dynaudnorm=f=150:g=15: Cân bằng động (Dynamic Audio Normalization) giúp kéo tiếng nhỏ lên và ghìm nhạc to xuống
        # volume={bg_volume}: Nhân tỷ lệ âm lượng tổng của nền (ví dụ 0.5)
        filter_str = f"amix=inputs={num_inputs}:duration=longest:dropout_transition=0,dynaudnorm=f=150:g=15,volume={bg_volume}"

        mix_cmd = [
            'ffmpeg', '-y',
            *available_inputs,
            '-filter_complex', filter_str,
            str(final_bg_path)
        ]

        if verbose:
            print(f"🎛️ Đang cân bằng âm thanh & điều chỉnh mức nền ({int(bg_volume * 100)}%)...")

        try:
            subprocess.run(mix_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"❌ Lỗi FFmpeg khi trộn âm thanh: {e.stderr}")

        if final_bg_path.exists():
            if verbose:
                size_mb = final_bg_path.stat().st_size / (1024 * 1024)
                print(f"🎉 Hoàn tất trích xuất nhạc nền/hiệu ứng: {final_bg_path.name} ({size_mb:.2f} MB)\n")
            return final_bg_path
        else:
            raise FileNotFoundError(f"❌ File '{final_bg_path.name}' không được tạo thành công.")
        
    def download_video(self, raw_url: str, verbose: bool = True):
        """Tải video Douyin/YouTube và xuất ra thư mục DD-MM-YYYY_HH-MM-SS."""
        
        video_url = self._resolve_douyin_url(raw_url)

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

        created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        folder_path = self.temp_dir / created_at
        folder_path.mkdir(parents=True, exist_ok=True)

        video_path = folder_path / "file.mp4"
        audio_path = folder_path / "file.mp3"
        json_path = folder_path / "info.json"

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

        ydl_opts = {
            'outtmpl': str(folder_path / 'file.%(ext)s'),
            'format': 'best[vcodec~=h264][acodec!=none]/best[vcodec!=none][acodec!=none]/best',
            'merge_output_format': 'mp4',
            'overwrite': True,
            'quiet': True,
            'no_warnings': True,
            'logger': QuietLogger(),
            'http_headers': common_headers,
            
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

            stop_event = threading.Event()
            monitor_thread = None
            if verbose:
                monitor_thread = threading.Thread(
                    target=self._monitor_progress,
                    args=(folder_path, filesize_estimate, stop_event),
                    daemon=True
                )
                monitor_thread.start()

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            if monitor_thread:
                stop_event.set()
                monitor_thread.join()

            downloaded_mp4 = folder_path / "file.mp4"
            if not downloaded_mp4.exists():
                for pf in folder_path.glob("file.*"):
                    if pf.suffix in ['.mp4', '.mkv', '.webm', '.flv']:
                        pf.rename(video_path)
                        break

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
    link_douyin = "https://youtube.com/shorts/Q-sZRMZ0qyg?si=OToNnOVULfAewwLx"
    downloader = VideoDownloader()
    downloader.download_video(link_douyin, verbose=True)