import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from pydub import AudioSegment
from vieneu import Vieneu

# Đảm bảo import và sử dụng thông tin từ config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def clean_text_for_display(text: str) -> str:
    """Xóa các nhãn biểu cảm [cười], [thở dài]... để dùng làm phụ đề hiển thị sạch."""
    return re.sub(r"\[.*?\]", "", text).strip()


def condense_vietnamese_text(text: str) -> str:
    """
    Tự động lọc bớt các từ đệm/từ nối không bắt buộc trong tiếng Việt 
    khi phát hiện câu bị quá dài so với thời lượng video gốc.
    """
    # Lưu lại các nhãn biểu cảm để không làm mất hiệu ứng Vieneu
    tags = re.findall(r"\[.*?\]", text)
    clean_txt = re.sub(r"\[.*?\]", " __TAG__ ", text)

    # Danh sách các từ đệm có thể bỏ bớt mà không làm mất nghĩa cốt lõi
    filler_words = [
        r"\bthì\b", r"\blà\b", r"\bmà\b", r"\brằng\b", r"\bđó\b", 
        r"\bhãy\b", r"\bnhé\b", r"\bnha\b", r"\bấy\b", r"\bđấy\b"
    ]
    
    for filler in filler_words:
        clean_txt = re.sub(filler, "", clean_txt, flags=re.IGNORECASE)

    # Trả lại các nhãn biểu cảm vào đúng vị trí
    for tag in tags:
        clean_txt = clean_txt.substitute("__TAG__", tag) if hasattr(clean_txt, 'substitute') else clean_txt.replace("__TAG__", tag, 1)

    # Làm sạch khoảng trắng thừa
    return re.sub(r"\s+", " ", clean_txt).strip()


def process_audio_smooth(
    wave_data: np.ndarray, 
    sample_rate: int = 48000, 
    pad_tail_ms: int = 150,
    fade_ms: int = 20
) -> AudioSegment:
    """
    Xử lý mượt biên độ âm thanh:
    - Chuyển float32 sang PCM int16.
    - Áp dụng Fade In/Out cực nhẹ để triệt tiêu tiếng ngắt giật (click/pop).
    - Thêm một chút padding cuối câu để nhịp ngắt câu tự nhiên.
    """
    wave_int16 = (np.clip(wave_data, -1.0, 1.0) * 32767).astype(np.int16)

    audio = AudioSegment(
        data=wave_int16.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1
    )

    if len(audio) > (fade_ms * 2):
        audio = audio.fade_in(fade_ms).fade_out(fade_ms)

    if pad_tail_ms > 0:
        silence = AudioSegment.silent(duration=pad_tail_ms, frame_rate=sample_rate)
        audio = audio + silence

    return audio


class SegmentTTSGenerator:
    """
    Class tạo giọng đọc AI tiếng Việt theo từng segment (Sử dụng Vieneu trên GPU).
    Tích hợp tự động tối ưu nhịp điệu, chống méo tiếng và xuất định dạng MP3.
    """

    def __init__(
        self, 
        voice: Optional[str] = None, 
        style: str = "tu_nhien", 
        sample_rate: int = 48000,
        max_words_per_sec: float = 3.5,  # Tốc độ an toàn tối đa (từ/giây) chống méo tiếng
        pad_tail_ms: int = 150,
        vieneu_instance: Optional[Vieneu] = None
    ):
        self.voice = voice if voice else getattr(config, "DEFAULT_VOICE", "Trúc Ly")
        self.style = style
        self.sample_rate = sample_rate
        self.max_words_per_sec = max_words_per_sec
        self.pad_tail_ms = pad_tail_ms
        self.vieneu = vieneu_instance if vieneu_instance is not None else Vieneu()

    def generate_speech_for_segments(self, json_path: Path, verbose: bool = True) -> List[Dict[str, Any]]:
        """Đọc info.json, tối ưu văn bản, sinh giọng đọc Vieneu MP3 và cập nhật lại file JSON."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy file JSON tại: {json_path}")

        folder_path = json_path.parent

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        if not segments:
            if verbose:
                print("⚠️ Không tìm thấy 'segments' trong info.json!")
            return []

        tts_dir = folder_path / "translated_audio_segments"
        tts_dir.mkdir(parents=True, exist_ok=True)

        valid_segments = []
        text_list = []

        # 1. Tiền xử lý văn bản & Kiểm tra nguy cơ méo tiếng
        for seg in segments:
            vi_text = seg.get("translated_text", "").strip()
            if not vi_text:
                continue

            # Tính thời lượng mong muốn của segment (nếu JSON gốc có timestamp start/end)
            start_time = seg.get("start", 0.0)
            end_time = seg.get("end", 0.0)
            duration = max(end_time - start_time, 0.5)

            # Đếm số từ thực tế (loại trừ các nhãn biểu cảm)
            clean_words = clean_text_for_display(vi_text).split()
            word_count = len(clean_words)

            # Cảnh báo & Tự động cô đọng nếu tốc độ nói vượt ngưỡng an toàn
            if duration > 0.5 and (word_count / duration) > self.max_words_per_sec:
                if verbose:
                    print(f"⚠️ [Seg {seg.get('id')}] Phát hiện nguy cơ méo tiếng ({word_count} từ / {duration:.1f}s). Đang tự động rút gọn...")
                vi_text = condense_vietnamese_text(vi_text)

            # Thêm dấu ngắt câu tự nhiên nếu thiếu
            if not vi_text.endswith((".", "!", "?", "...", "]")):
                vi_text += "."

            seg["processed_text"] = vi_text
            valid_segments.append(seg)
            text_list.append(vi_text)

        if not text_list:
            if verbose:
                print("⚠️ Không có câu thoại nào hợp lệ để tạo giọng đọc!")
            return segments

        if verbose:
            print(f"🔊 [Vieneu Smooth TTS] Bắt đầu tạo {len(text_list)} segments ({self.voice}) trên GPU...")

        # 2. Chạy Batch Inference trên GPU bằng Vieneu
        raw_audios = self.vieneu.infer_batch(
            text_list, 
            voice=self.voice, 
            style=self.style
        )

        # 3. Xử lý âm thanh mượt mà và lưu file MP3
        for seg, wave in zip(valid_segments, raw_audios):
            seg_id = seg.get("id", 0)
            output_filename = f"vi_seg_{seg_id:03d}.mp3"
            output_filepath = tts_dir / output_filename

            # Áp dụng thuật toán làm mượt chống méo/nổ tiếng
            smooth_audio = process_audio_smooth(
                wave_data=wave,
                sample_rate=self.sample_rate,
                pad_tail_ms=self.pad_tail_ms
            )

            # Xuất file MP3 chất lượng cao
            smooth_audio.export(str(output_filepath), format="mp3", bitrate="192k")

            # Cập nhật thông tin vào object segment
            seg["translated_audio_file"] = f"translated_audio_segments/{output_filename}"

            if verbose:
                print(f"  ✨ [Seg {seg_id}] Đã xuất MP3 mượt: {output_filename}")

        # 4. Cập nhật lại file info.json
        data["segments"] = segments
        data["status"] = "tts_generated"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"\n✅ Đã hoàn thành toàn bộ {len(valid_segments)} file MP3 lồng tiếng!")
            print(f"📂 Thư mục chứa: {tts_dir.name}")
            print(f"📝 Cập nhật thành công: {json_path.name}\n")

        return segments


if __name__ == "__main__":
    temp_dir = getattr(config, "TEMP_DIR", Path("output/temp"))

    if temp_dir.exists():
        subdirs = sorted([d for d in temp_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)

        if subdirs:
            target_folder = subdirs[0]
            target_json = target_folder / "info.json"

            if target_json.exists():
                print(f"🔍 Đang chạy test Vieneu TTS Generator trên folder: {target_folder.name}\n")

                tts = SegmentTTSGenerator(voice="Trúc Ly")
                tts.generate_speech_for_segments(json_path=target_json, verbose=True)
            else:
                print(f"❌ Không tìm thấy file 'info.json' trong folder: {target_folder}")
        else:
            print(f"⚠️ Thư mục '{temp_dir}' đang trống.")
    else:
        print(f"❌ Thư mục TEMP_DIR không tồn tại: {temp_dir}")