import asyncio
import json
import sys
from pathlib import Path
from typing import Optional
import edge_tts

# Đảm bảo import và sử dụng thông tin từ config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class SegmentTTSGenerator:
    """Class tạo giọng đọc AI tiếng Việt theo từng segment dựa vào mốc thời gian."""

    def __init__(self, voice: Optional[str] = None):
        # Lấy cấu hình giọng đọc từ config, mặc định dùng giọng Nữ Hoài My (vi-VN-HoaiMyNeural)
        # Giọng Nam gợi ý: 'vi-VN-NamMinhNeural'
        self.voice = voice if voice else getattr(config, "DEFAULT_VOICE", "vi-VN-HoaiMyNeural")

    def generate_speech_for_segments(self, json_path: Path, verbose: bool = True) -> list:
        """Đọc info.json, sinh giọng đọc tiếng Việt cho từng câu và cập nhật lại file JSON."""
        return asyncio.run(self._process_segments(json_path, verbose))

    async def _process_segments(self, json_path: Path, verbose: bool) -> list:
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

        # Tạo thư mục con chứa các file âm thanh tiếng Việt đã chuyển đổi
        tts_dir = folder_path / "translated_audio_segments"
        tts_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"🔊 [TTS] Bắt đầu tạo giọng đọc ({self.voice}) cho {len(segments)} segments...")

        for seg in segments:
            seg_id = seg.get("id", 0)
            
            # Lấy câu tiếng Việt đã dịch ở bước Translator
            vi_text = seg.get("translated_text", "").strip()
            if not vi_text:
                continue

            output_filename = f"vi_seg_{seg_id:03d}.mp3"
            output_filepath = tts_dir / output_filename

            # Sinh audio tiếng Việt cho phân đoạn này
            await self._create_single_tts(vi_text, output_filepath)

            # Cập nhật đường dẫn tương đối vào segment
            seg["translated_audio_file"] = f"translated_audio_segments/{output_filename}"
            
            if verbose:
                print(f"  ✨ [Seg {seg_id}] Đã tạo âm thanh: {output_filename}")

        # Cập nhật thông tin vào info.json
        data["segments"] = segments
        data["status"] = "tts_generated"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"\n✅ Đã tạo xong toàn bộ {len(segments)} file âm thanh lồng tiếng!")
            print(f"📂 Thư mục chứa: {tts_dir.name}")
            print(f"📝 Đã cập nhật file: {json_path.name}\n")

        return segments

    async def _create_single_tts(self, text: str, output_path: Path):
        """Hàm bất đồng bộ gọi Edge-TTS sinh file mp3 cho 1 câu."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))


if __name__ == "__main__":
    # Tự động tìm folder gần nhất trong TEMP_DIR để test
    temp_dir = getattr(config, "TEMP_DIR", Path("output/temp"))
    
    if temp_dir.exists():
        subdirs = sorted([d for d in temp_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        
        if subdirs:
            target_folder = subdirs[0]
            target_json = target_folder / "info.json"

            if target_json.exists():
                print(f"🔍 Đang chạy test TTS Generator trên folder: {target_folder.name}\n")
                
                tts = SegmentTTSGenerator()
                tts.generate_speech_for_segments(json_path=target_json, verbose=True)
            else:
                print(f"❌ Không tìm thấy file 'info.json' trong folder: {target_folder}")
        else:
            print(f"⚠️ Thư mục '{temp_dir}' đang trống.")
    else:
        print(f"❌ Thư mục TEMP_DIR không tồn tại: {temp_dir}")