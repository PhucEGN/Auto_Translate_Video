import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests

# Import cấu hình
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class SegmentTranslator:
    """Module dịch ngữ cảnh tự động gộp các câu thoại quá ngắn/quá nhanh trước khi dịch."""

    def __init__(self, api_key: str = None, min_segment_duration: float = 1.2):
        """
        - min_segment_duration: Thời lượng tối thiểu (giây). 
          Nếu 1 segment ngắn hơn mức này, hệ thống sẽ đề xuất gộp với segment kế tiếp.
        """
        self.api_key = api_key if api_key else getattr(config, "GROQ_API_KEY", "")
        self.min_duration = min_segment_duration

    def _merge_fast_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Thuật toán gom cụm: Phát hiện các câu nói quá nhanh/quắn câu và gộp 
        thời lượng (start-end) cùng văn bản gốc lại thành 1 segment thống nhất.
        """
        if not segments:
            return []

        merged = []
        curr = segments[0].copy()

        for next_seg in segments[1:]:
            curr_duration = curr.get("end", 0.0) - curr.get("start", 0.0)
            
            # Tính khoảng nghỉ giữa 2 câu
            gap = next_seg.get("start", 0.0) - curr.get("end", 0.0)

            # ĐIỀU KIỆN GỘP: 
            # 1. Câu hiện tại quá ngắn (< min_duration) HOẶC câu nối tiếp quá ngắn
            # 2. VÀ Khoảng nghỉ giữa 2 câu rất nhỏ (< 0.5s) để đảm bảo tính liền mạch
            if (curr_duration < self.min_duration or (next_seg.get("end", 0.0) - next_seg.get("start", 0.0)) < self.min_duration) and gap < 0.5:
                # Gộp text gốc
                curr["text"] = f"{curr.get('text', '').strip()} {next_seg.get('text', '').strip()}"
                # Cập nhật mốc thời gian kết thúc
                curr["end"] = next_seg.get("end", curr.get("end"))
            else:
                merged.append(curr)
                curr = next_seg.copy()

        merged.append(curr)

        # Đánh lại ID cho các segment sau khi gộp
        for idx, seg in enumerate(merged):
            seg["id"] = idx

        return merged

    def translate_segments(self, json_path: Path, verbose: bool = True) -> List[Dict[str, Any]]:
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy file JSON tại: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_segments = data.get("segments", [])
        if not raw_segments:
            return []

        # 1. TIỀN XỬ LÝ: Gộp các câu quá nhanh lại với nhau
        original_count = len(raw_segments)
        segments = self._merge_fast_segments(raw_segments)
        
        if verbose:
            print(f"🧩 [Translator] Đã gộp các câu nói nhanh: từ {original_count} câu -> {len(segments)} cụm câu liền mạch.")

        # 2. Chuẩn bị danh sách gửi LLM dịch
        items_to_translate = [
            {
                "id": seg["id"], 
                "text": seg.get("text", ""),
                "duration": round(seg.get("end", 0.0) - seg.get("start", 0.0), 2)
            } 
            for seg in segments
        ]

        total_items = len(items_to_translate)

        # 3. PROMPT ĐIỀU CHỈNH: Cho phép LLM chủ động tổng hợp ý thành 1 câu mượt
        prompt = f"""You are a professional film dubbing translator and scriptwriter.
Translate the following list of dialogue entries from the source language into Vietnamese.

CORE OBJECTIVE:
Translate into concise, highly natural, and idiomatic Vietnamese suitable for voiceover. Some input items contain merged fast-spoken phrases—synthesize their core meaning into SINGLE, COMPLETE, AND SMOOTH VIETNAMESE SENTENCES without awkward pauses or literal line-by-line fragments.

MANDATORY RULES:
1. STRICT CONCISENESS & TIMING: Vietnamese phrases tend to be longer. Eliminate unnecessary filler words (e.g., "thì", "là", "mà", "rằng", "đó") so the generated AI voice reads at a natural, non-rushed pace.
2. EXACT COUNT REQUIREMENT: You MUST translate ALL {total_items} items (from id 0 to id {total_items - 1}).
3. NO FOREIGN CHARACTERS: Absolutely NO original Chinese or English characters allowed in the translation.
4. NO EXPLANATIONS: Do NOT add any notes, side comments, or extra text outside the translated sentences.
5. PROPER NAMES & SPECIES: 
   - For human names: Replace with an equivalent Vietnamese name or a Vietnamese-transliterated name that stays close to the original sound/meaning. MAINTAIN 100% NAME CONSISTENCY across all entries (never change the same character's name halfway).
   - For species, insects, or specialized terms: Translate accurately into common/standard Vietnamese terms.
6. NATURAL SPOKEN REGISTER: Use everyday, accessible spoken language. Avoid overly academic, stiff, or formal vocabulary.

AUTOMATIC EMOTION TAGS (Vieneu Tags):
Based on context, naturally embed emotional cues `[...]` directly into `translated_text` where appropriate:
- Supported tags: [cười], [thở dài], [hắng giọng], [ngập ngừng]
*(Only insert tags when the emotion is clearly evident from the context).*

INPUT DIALOGUE LIST:
{json.dumps(items_to_translate, ensure_ascii=False)}

OUTPUT FORMAT (Return STRICTLY a valid JSON object):
{{
  "translations": [
    {{"id": 0, "translated_text": "Smooth merged translation with optional [hắng giọng] tag..."}},
    {{"id": 1, "translated_text": "Next translated sentence..."}}
  ]
}}"""

        # 4. Gọi API Llama 3.3 / Groq để dịch
        translated_map = self._call_llm_api(prompt)

        # 5. Cập nhật câu dịch vào danh sách segment đã gộp
        for seg in segments:
            seg_id = seg["id"]
            seg["translated_text"] = translated_map.get(seg_id, seg.get("text", ""))

        # 6. Ghi đè lại cấu trúc segments mới vào info.json
        data["segments"] = segments
        data["status"] = "translated"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"✅ [Translator] Đã dịch xong toàn bộ {len(segments)} cụm câu thoại!")

        return segments

    def _call_llm_api(self, prompt: str) -> Dict[int, str]:
        """Hàm phụ trách gửi request tới API LLM và parse kết quả JSON."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            res.raise_for_status()
            res_data = res.json()
            content = res_data["choices"][0]["message"]["content"]
            
            parsed = json.loads(content)
            translations = parsed.get("translations", [])
            
            # Chuyển về dạng dict {id: translated_text}
            return {item["id"]: item["translated_text"] for item in translations}

        except Exception as e:
            print(f"❌ Lỗi khi gọi LLM API Translator: {e}")
            return {}