import json
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional
from groq import Groq

# Đảm bảo import được config dù chạy ở thư mục nào
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class AudioTranscriber:
    """Class quản lý việc chuyển đổi file MP3 sang văn bản kèm mốc thời gian (timestamps) và lưu vào info.json."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key else getattr(config, "GROQ_API_KEY", None)
        if not self.api_key:
            raise ValueError("❌ Không tìm thấy GROQ_API_KEY trong config.py!")
        
        self.client = Groq(api_key=self.api_key)

    def _compress_audio_if_needed(self, audio_path: Path, max_mb: float = 20.0) -> Path:
        """Nếu dung lượng file vượt quá max_mb, tự động nén về Mono 64kbps để đáp ứng giới hạn của Groq API."""
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb <= max_mb:
            return audio_path

        compressed_path = audio_path.parent / f"{audio_path.stem}_compressed.mp3"
        print(f"📦 File audio ({file_size_mb:.1f}MB) vượt quá ngưỡng an toàn {max_mb}MB.")
        print(f"⚡ Đang tự động nén file âm thanh về định dạng tối ưu (Mono 64kbps)...")

        cmd = [
            'ffmpeg', '-y',
            '-i', str(audio_path),
            '-vn',                 # Bỏ luồng video
            '-c:a', 'libmp3lame',  # Codec MP3
            '-b:a', '64k',         # Giảm bitrate xuống 64kbps
            '-ac', '1',            # Chuyển về Mono (1 kênh)
            '-ar', '16000',        # Sample rate 16kHz chuẩn cho Whisper
            str(compressed_path)
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            compressed_size = compressed_path.stat().st_size / (1024 * 1024)
            print(f"✅ Nén hoàn tất! Dung lượng mới: {compressed_size:.1f}MB")
            return compressed_path
        except Exception as e:
            print(f"⚠️ Nén thất bại, sẽ dùng file gốc: {e}")
            return audio_path

    def transcribe(
        self, 
        audio_path: Path, 
        json_path: Path, 
        language: str = "zh", 
        verbose: bool = False
    ) -> dict:
        """Chuyển đổi file audio (mp3) sang text kèm timestamp và cập nhật vào file info.json.

        Args:
            audio_path (Path): Đường dẫn tới file .mp3
            json_path (Path): Đường dẫn file info.json để ghi kết quả.
            language (str): Mã ngôn ngữ phát hiện ('zh', 'en', 'vi',...).
            verbose (bool): In thông báo tiến trình ra terminal nếu True.
        """
        audio_path = Path(audio_path)
        json_path = Path(json_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy file audio tại: {audio_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy file JSON tại: {json_path}")

        # Tự động nén audio nếu file quá lớn (> 20MB)
        target_audio = self._compress_audio_if_needed(audio_path, max_mb=20.0)

        if verbose:
            print(f"🎙️ [Transcriber] Đang gửi file '{target_audio.name}' lên Groq Whisper (lấy timestamps)...")

        try:
            # Gửi audio lên Groq API yêu cầu trả về verbose_json
            with open(target_audio, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(target_audio.name, file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",  # Bắt buộc để lấy chi tiết từng câu & timestamp
                    language=language
                )

            # Lấy toàn bộ văn bản gốc
            full_text = transcription.text.strip()
            
            # Trích xuất danh sách các phân đoạn kèm mốc thời gian start & end
            segments = []
            if hasattr(transcription, "segments") and transcription.segments:
                for seg in transcription.segments:
                    # Chuyển object/dict thành format gọn nhẹ
                    seg_dict = seg if isinstance(seg, dict) else seg.__dict__
                    segments.append({
                        "id": seg_dict.get("id"),
                        "start": round(seg_dict.get("start", 0.0), 2),
                        "end": round(seg_dict.get("end", 0.0), 2),
                        "text": seg_dict.get("text", "").strip()
                    })

            # Ghi trực tiếp cả transcript lẫn segments vào info.json
            self._update_json(json_path, full_text, segments, verbose)

            # Dọn dẹp file tạm nén nếu có
            if target_audio != audio_path and target_audio.exists():
                try:
                    target_audio.unlink()
                except Exception:
                    pass

            return {"transcript": full_text, "segments": segments}

        except Exception as e:
            # Dọn dẹp file tạm nếu bị lỗi
            if target_audio != audio_path and target_audio.exists():
                try:
                    target_audio.unlink()
                except Exception:
                    pass

            if verbose:
                print(f"❌ Lỗi khi chuyển đổi Audio sang Text: {e}")
            raise e

    def _update_json(self, json_path: Path, transcript_text: str, segments: list, verbose: bool):
        """Cập nhật văn bản nhận diện và mốc thời gian segments vào info.json."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["transcript"] = transcript_text
            data["segments"] = segments  # Chứa danh sách mốc thời gian từng câu
            data["status"] = "transcribed"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            if verbose:
                print(f"📝 Đã cập nhật transcript & segments vào: {json_path.name}")

        except Exception as e:
            if verbose:
                print(f"⚠️ Không thể cập nhật file JSON: {e}")


if __name__ == "__main__":
    sample_dir = config.TEMP_DIR / "26-07-2026_12-56-50"
    sample_audio = sample_dir / "file.mp3"
    sample_json = sample_dir / "info.json"

    if sample_audio.exists() and sample_json.exists():
        transcriber = AudioTranscriber()
        transcriber.transcribe(
            audio_path=sample_audio,
            json_path=sample_json,
            language="zh",
            verbose=True
        )