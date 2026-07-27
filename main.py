import sys
import shutil
import argparse
from pathlib import Path
from vieneu import Vieneu

# Import các thông số cấu hình từ config.py
import config

# Import 5 module chính trong hệ thống
from modules.downloader import VideoDownloader
from modules.transcriber import AudioTranscriber
from modules.translator import SegmentTranslator
from modules.tts_generator import SegmentTTSGenerator
from modules.audio_video_merger import AudioVideoMerger


def process_video_pipeline(video_url: str, vieneu_instance: Vieneu = None):
    """Luồng xử lý tự động chuyển đổi video từ Link sang Video Tiếng Việt lồng tiếng hoàn chỉnh."""
    print("=" * 70)
    print("🚀 BẮT ĐẦU TIẾN TRÌNH TỰ ĐỘNG TẠO VIDEO LỒNG TIẾNG TIẾNG VIỆT")
    print("=" * 70)
    print(f"🔗 Link Video: {video_url}\n")

    temp_folder = None

    try:
        # -----------------------------------------------------------------
        # BƯỚC 1: Tải Video & Khởi tạo thư mục tạm trong TEMP_DIR
        # -----------------------------------------------------------------
        print("📌 [1/5] Tải video và trích xuất dữ liệu...")
        downloader = VideoDownloader()
        success, folder_path, video_path, audio_path, json_path = downloader.download_video(video_url, verbose=True)
        if not success:
            print("❌ Quá trình tải video thất bại, vui lòng kiểm tra lại link hoặc kết nối mạng.")
            sys.exit(1)
        temp_folder = json_path.parent
        audio_file_path = temp_folder / "file.mp3"
        print(f"   📂 Thư mục làm việc tạm: {temp_folder.name}\n")

        # -----------------------------------------------------------------
        # BƯỚC 2: Nhận diện giọng nói & Timestamp bằng Groq Whisper
        # -----------------------------------------------------------------
        print("📌 [2/5] Nhận diện giọng nói & trích xuất mốc thời gian (Whisper)...")
        transcriber = AudioTranscriber()
        transcriber.transcribe(
            audio_path=audio_file_path,
            json_path=json_path,
            language="zh",
            verbose=True
        )

        # -----------------------------------------------------------------
        # BƯỚC 3: Dịch ngữ cảnh từng câu sang Tiếng Việt bằng Llama 3.3 70B
        # -----------------------------------------------------------------
        print("📌 [3/5] Dịch nội dung sang tiếng Việt (Llama 3.3 70B)...")
        translator = SegmentTranslator()
        translator.translate_segments(json_path, verbose=True)

        # -----------------------------------------------------------------
        # BƯỚC 4: Tạo giọng đọc AI tiếng Việt mượt mà bằng Vieneu (GPU)
        # -----------------------------------------------------------------
        print("📌 [4/5] Tạo file âm thanh lồng tiếng AI (Vieneu TTS)...")
        tts_voice = getattr(config, "DEFAULT_VOICE", "Minh Đức")
        tts_style = getattr(config, "DEFAULT_STYLE", "tu_nhien")

        tts = SegmentTTSGenerator(
            voice=tts_voice, 
            style=tts_style,
            vieneu_instance=vieneu_instance
        )
        tts.generate_speech_for_segments(json_path, verbose=True)

        # -----------------------------------------------------------------
        # BƯỚC 5: Xếp Timeline và Ghép Video + Audio Tiếng Việt
        # -----------------------------------------------------------------
        print("📌 [5/5] Căn chỉnh timeline và gộp video thành phẩm...")
        merger = AudioVideoMerger()
        temp_final_video = merger.merge_audio_video(json_path, verbose=True)

        # -----------------------------------------------------------------
        # DỌN DẸP & DỜI THÀNH PHẨM SANG THƯ MỤC OUTPUT
        # -----------------------------------------------------------------
        print("🧹 [Hoàn tất] Đang di chuyển sản phẩm về thư mục Output và xóa thư mục tạm...")
        
        output_base_dir = getattr(config, "OUTPUT_DIR", Path("output"))
        output_base_dir.mkdir(parents=True, exist_ok=True)

        # Đặt tên thư mục kết quả trong output theo tên folder tạm
        final_output_folder = output_base_dir / temp_folder.name

        # Nếu thư mục đích đã tồn tại trong output thì xóa trước để đè mới
        if final_output_folder.exists():
            shutil.rmtree(final_output_folder)

        # Chuyển toàn bộ thư mục kết quả từ temp/ sang output/
        shutil.move(str(temp_folder), str(final_output_folder))

        print("\n" + "=" * 70)
        print("🎉 TẤT CẢ CÁC BƯỚC ĐÃ HOÀN THÀNH XUẤT SẮC!")
        print(f"📂 Thư mục thành phẩm: {final_output_folder.resolve()}")
        print(f"🎬 File Video cuối cùng: {final_output_folder / 'final_output.mp4'}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ ĐÃ XẢY RA LỖI TRONG QUÁ TRÌNH XỬ LÝ: {e}")
        # Dọn dẹp thư mục tạm nếu có lỗi phát sinh
        if temp_folder and temp_folder.exists():
            print(f"🗑️ Đang dọn dẹp thư mục tạm bị lỗi: {temp_folder}")
            shutil.rmtree(temp_folder, ignore_errors=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Auto Translate & Dub Video Pipeline")
    parser.add_argument(
        "--url", "-u", 
        type=str, 
        help="Đường dẫn URL của video (YouTube, Facebook, TikTok, ...)"
    )
    args = parser.parse_args()

    video_url = args.url
    
    # Nếu không truyền qua tham số dòng lệnh --url, yêu cầu nhập từ phím
    if not video_url:
        print("=" * 70)
        print("🎥 HỆ THỐNG TỰ ĐỘNG DỊCH VÀ LỒNG TIẾNG VIDEO (VIENEU TTS)")
        print("=" * 70)
        video_url = input("👉 Nhập link video cần xử lý: ").strip()

    if not video_url:
        print("❌ Link video không được để trống!")
        sys.exit(1)

    # Khởi tạo Vieneu 1 lần duy nhất trên GPU
    print("\n⚙️ Đang khởi tạo mô hình Vieneu TTS...")
    vieneu_instance = Vieneu()

    process_video_pipeline(video_url, vieneu_instance=vieneu_instance)


if __name__ == "__main__":
    main()