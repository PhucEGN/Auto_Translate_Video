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
    tags = re.findall(r"\[.*?\]", text)
    clean_txt = re.sub(r"\[.*?\]", " __TAG__ ", text)

    filler_words = [
        r"\bthì\b", r"\blà\b", r"\bmà\b", r"\brằng\b", r"\bđó\b", 
        r"\bhãy\b", r"\bnhé\b", r"\bnha\b", r"\bấy\b", r"\bđấy\b"
    ]
    
    for filler in filler_words:
        clean_txt = re.sub(filler, "", clean_txt, flags=re.IGNORECASE)

    for tag in tags:
        clean_txt = clean_txt.substitute("__TAG__", tag) if hasattr(clean_txt, 'substitute') else clean_txt.replace("__TAG__", tag, 1)

    return re.sub(r"\s+", " ", clean_txt).strip()


def process_audio_smooth(
    wave_data: np.ndarray, 
    sample_rate: int = 48000, 
    pad_tail_ms: int = 150,
    fade_ms: int = 20
) -> AudioSegment:
    """Xử lý mượt biên độ âm thanh."""
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
    """Class tạo giọng đọc AI tiếng Việt theo từng segment (Sử dụng Vieneu trên GPU)."""

    def __init__(
        self, 
        voice: Optional[str] = None, 
        style: str = "tu_nhien", 
        sample_rate: int = 48000,
        max_words_per_sec: float = 4.5,
        pad_tail_ms: int = 150,
        enable_clone: bool = False,
        ref_audio_path: Optional[Path] = None,
        vieneu_instance: Optional[Vieneu] = None
    ):
        self.voice = voice if voice else getattr(config, "DEFAULT_VOICE", "Minh Đức")
        self.style = style
        self.sample_rate = sample_rate
        self.max_words_per_sec = max_words_per_sec
        self.pad_tail_ms = pad_tail_ms
        self.enable_clone = enable_clone
        self.ref_audio_path = Path(ref_audio_path) if ref_audio_path else None
        self.vieneu = vieneu_instance if vieneu_instance is not None else Vieneu()

        # Sử dụng ONNX backend làm mặc định nếu không truyền vieneu_instance
        self.vieneu = vieneu_instance if vieneu_instance is not None else Vieneu(backend="onnx")
        
    def generate_speech_for_segments(self, json_path: Path, verbose: bool = True) -> List[Dict[str, Any]]:
        """Đọc info.json, tối ưu văn bản, sinh giọng đọc Vieneu MP3 (Mặc định hoặc Clone) và cập nhật lại file JSON."""
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

        for seg in segments:
            vi_text = seg.get("translated_text", "").strip()
            if not vi_text:
                continue

            start_time = seg.get("start", 0.0)
            end_time = seg.get("end", 0.0)
            duration = max(end_time - start_time, 0.5)

            clean_words = clean_text_for_display(vi_text).split()
            word_count = len(clean_words)

            if duration > 0.5 and (word_count / duration) > self.max_words_per_sec:
                if verbose:
                    print(f"⚠️ [Seg {seg.get('id')}] Phát hiện nguy cơ méo tiếng ({word_count} từ / {duration:.1f}s). Đang tự động rút gọn...")
                vi_text = condense_vietnamese_text(vi_text)

            if not vi_text.endswith((".", "!", "?", "...", "]")):
                vi_text += "."

            seg["processed_text"] = vi_text
            valid_segments.append(seg)
            text_list.append(vi_text)

        if not text_list:
            if verbose:
                print("⚠️ Không có câu thoại nào hợp lệ để tạo giọng đọc!")
            return segments

        # Phân nhánh xử lý: Giọng Clone hoặc Giọng Mặc Định
        if self.enable_clone and self.ref_audio_path and self.ref_audio_path.exists():
            if verbose:
                print(f"🎭 [Vieneu Clone TTS] Bắt đầu nhái giọng từ mẫu: {self.ref_audio_path.name}...")
            
            raw_audios = self.vieneu.infer_batch(
                text_list, 
                ref_audio=str(self.ref_audio_path),
                style=self.style
            )
        else:
            if self.enable_clone and verbose:
                print("⚠️ Không tìm thấy file audio mẫu clone, tự động chuyển về giọng mặc định!")
                
            if verbose:
                print(f"🔊 [Vieneu Standard TTS] Tạo {len(text_list)} segments ({self.voice}) trên GPU...")

            raw_audios = self.vieneu.infer_batch(
                text_list, 
                voice=self.voice, 
                style=self.style
            )

        for seg, wave in zip(valid_segments, raw_audios):
            seg_id = seg.get("id", 0)
            output_filename = f"vi_seg_{seg_id:03d}.mp3"
            output_filepath = tts_dir / output_filename

            smooth_audio = process_audio_smooth(
                wave_data=wave,
                sample_rate=self.sample_rate,
                pad_tail_ms=self.pad_tail_ms
            )

            smooth_audio.export(str(output_filepath), format="mp3", bitrate="192k")
            seg["translated_audio_file"] = f"translated_audio_segments/{output_filename}"

            if verbose:
                print(f"  ✨ [Seg {seg_id}] Đã xuất MP3 mượt: {output_filename}")

        data["segments"] = segments
        data["status"] = "tts_generated"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"\n✅ Đã hoàn thành toàn bộ {len(valid_segments)} file MP3 lồng tiếng!")
            print(f"📂 Thư mục chứa: {tts_dir.name}")
            print(f"📝 Cập nhật thành công: {json_path.name}\n")

        return segments