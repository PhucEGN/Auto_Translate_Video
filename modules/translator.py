import json
import re
import sys
from pathlib import Path
from typing import Optional
from groq import Groq

# Đảm bảo import và sử dụng thông tin từ config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class SegmentTranslator:
    """Class quản lý việc dịch văn bản từng segment từ ngôn ngữ gốc sang tiếng Việt."""

    def __init__(self, api_key: Optional[str] = None):
        # Lấy API Key trực tiếp từ config
        self.api_key = api_key if api_key else getattr(config, "GROQ_API_KEY", None)
        if not self.api_key:
            raise ValueError("❌ Không tìm thấy GROQ_API_KEY trong config.py!")

        self.client = Groq(api_key=self.api_key)

    def _split_long_or_list_segment(self, segments: list) -> list:
        """Tự động kiểm tra và tách các segment bị Whisper gom cục (ví dụ: chuỗi từ vựng cách nhau bằng dấu phẩy 、, hoặc ,)."""
        if len(segments) != 1:
            return segments

        single_seg = segments[0]
        raw_text = single_seg.get("text", "").strip()

        # Tách chuỗi theo dấu phẩy tiếng Trung (、), dấu phẩy thường (,), hoặc chấm phẩy (;)
        items = [
            item.strip()
            for item in re.split(r"[、,;，]+", raw_text)
            if item.strip()
        ]

        # Nếu chỉ có 1 từ/câu đơn lẻ thực sự, không làm gì cả
        if len(items) <= 1:
            return segments

        # Nếu phát hiện danh sách gồm nhiều từ/câu, bổ nhỏ segment ra
        total_duration = single_seg.get("end", 0) - single_seg.get("start", 0)
        total_chars = max(len(raw_text), 1)

        new_segments = []
        current_start = single_seg.get("start", 0)

        for idx, item in enumerate(items):
            item_duration = (len(item) / total_chars) * total_duration
            current_end = round(current_start + item_duration, 2)

            new_segments.append(
                {
                    "id": idx,
                    "start": round(current_start, 2),
                    "end": current_end,
                    "text": item,
                }
            )
            current_start = current_end

        return new_segments

    def _translate_batch(self, batch_segments: list, target_lang: str) -> dict:
        """Hàm trợ lý dịch một nhóm (batch) câu để tránh quá tải Token limit của LLM."""
        if not batch_segments:
            return {}

        # 1. Tạo bản đồ tạm: Đánh số index từ 0 -> len-1 để gửi cho LLM
        temp_items = []
        index_to_original_id = {}

        for idx, seg in enumerate(batch_segments):
            temp_items.append({"id": idx, "text": seg.get("text", "")})
            index_to_original_id[idx] = seg.get("id")

        total_items = len(temp_items)

        # 2. Prompt chuẩn hóa và bắt buộc cấu trúc rõ ràng
        prompt = f"""Bạn là một biên dịch viên chuyên nghiệp. Hãy dịch danh sách các từ vựng hoặc câu thoại sau đây sang {target_lang}.

YÊU CẦU BẮT BUỘC:
1. Dịch chính xác, tự nhiên, thuần Việt. Nếu là từ vựng đơn lẻ, hãy dịch ngắn gọn đúng nghĩa của từ đó.
2. Giữ nguyên ý nghĩa, không tự ý thêm bớt hoặc rút gọn nội dung.
3. BẮT BUỘC dịch đầy đủ TẤT CẢ {total_items} mục trong danh sách (từ id 0 đến id {total_items - 1}).
4. Kết quả BẮT BUỘC trả về đúng định dạng JSON object chứa danh sách 'translations'.

Danh sách các mục cần dịch:
{json.dumps(temp_items, ensure_ascii=False)}

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (Chỉ trả về JSON, không kèm lời giải thích):
{{
  "translations": [
    {{"id": 0, "translated_text": "nội dung dịch 0"}},
    {{"id": 1, "translated_text": "nội dung dịch 1"}}
  ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content
            translated_data = json.loads(raw_content)

            # Extract danh sách từ response JSON
            translated_list = []
            if isinstance(translated_data, dict):
                for key in ["translations", "data", "items"]:
                    if key in translated_data and isinstance(
                        translated_data[key], list
                    ):
                        translated_list = translated_data[key]
                        break
                if not translated_list:
                    for val in translated_data.values():
                        if isinstance(val, list):
                            translated_list = val
                            break
            elif isinstance(translated_data, list):
                translated_list = translated_data

            # Map ngược từ temp_id (0..N-1) về original_id
            trans_map = {}
            for item in translated_list:
                if isinstance(item, dict) and "id" in item:
                    temp_id = item["id"]
                    if temp_id in index_to_original_id:
                        original_id = index_to_original_id[temp_id]
                        trans_text = item.get("translated_text", "").strip()
                        trans_map[original_id] = trans_text

            return trans_map

        except Exception as e:
            print(f"⚠️ Lỗi khi dịch batch: {e}")
            return {}

    def translate_segments(
        self,
        json_path: Path,
        target_lang: str = "Vietnamese",
        batch_size: int = 30,
        verbose: bool = True,
    ) -> list:
        """Đọc info.json, tự động làm sạch/tách segment nếu bị dính chùm, dịch từng segment và cập nhật lại file JSON."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(
                f"❌ Không tìm thấy file JSON tại: {json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        if not segments:
            if verbose:
                print("⚠️ Không tìm thấy 'segments' trong info.json!")
            return []

        # TỰ ĐỘNG XỬ LÝ: Kiểm tra nếu Whisper gom toàn bộ chuỗi từ vựng vào 1 segment
        segments = self._split_long_or_list_segment(segments)

        if verbose:
            print(
                f"🌐 [Translator] Bắt đầu dịch {len(segments)} segments sang {target_lang} (Chia làm các batch {batch_size} câu)..."
            )

        # Chia segments thành từng batch nhỏ hơn để xử lý an toàn
        full_trans_map = {}
        total_batches = (len(segments) + batch_size - 1) // batch_size

        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            current_batch_num = (i // batch_size) + 1

            if verbose:
                print(
                    f" 🔄 Đang dịch batch {current_batch_num}/{total_batches} ({len(batch)} câu)..."
                )

            batch_map = self._translate_batch(batch, target_lang)
            full_trans_map.update(batch_map)

        # Áp dụng bản dịch vào từng segment
        full_translated_text = []
        for seg in segments:
            seg_id = seg.get("id")
            translated_str = full_trans_map.get(seg_id, "").strip()

            # Fallback nếu Llama không dịch được
            if not translated_str:
                translated_str = seg.get("text", "").strip()

            seg["translated_text"] = translated_str
            full_translated_text.append(translated_str)

        # Cập nhật tổng thể vào info.json
        data["segments"] = segments
        data["translated_text"] = " ".join(full_translated_text)
        data["status"] = "translated"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"✅ Đã dịch thành công {len(segments)} câu/từ vựng!")
            print(f"📝 Đã cập nhật bản dịch vào: {json_path.name}\n")

        return segments


if __name__ == "__main__":
    temp_dir = getattr(config, "TEMP_DIR", Path("output/temp"))

    if temp_dir.exists():
        subdirs = sorted(
            [d for d in temp_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if subdirs:
            target_folder = subdirs[0]
            target_json = target_folder / "info.json"

            if target_json.exists():
                print(
                    f"🔍 Đang chạy test Translator trên folder: {target_folder.name}\n"
                )

                translator = SegmentTranslator()
                translated_segments = translator.translate_segments(
                    json_path=target_json, verbose=True
                )

                print("--- KẾT QUẢ DỊCH THỬ NGHỆM (5 câu đầu) ---")
                for seg in translated_segments[:5]:
                    print(f"[{seg['start']}s -> {seg['end']}s]")
                    print(f" 🔴 Gốc : {seg['text']}")
                    print(f" 🟢 Dịch: {seg['translated_text']}")
                    print("-" * 40)
            else:
                print(
                    f"❌ Không tìm thấy file 'info.json' trong folder: {target_folder}"
                )
        else:
            print(
                f"⚠️ Thư mục '{temp_dir}' đang trống, hãy chạy downloader & transcriber trước."
            )
    else:
        print(f"❌ Thư mục TEMP_DIR không tồn tại: {temp_dir}")