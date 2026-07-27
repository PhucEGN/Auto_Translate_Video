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


def process_video_pipeline(video_url: str, enable_clone: bool = False, vieneu_instance: Vieneu = None):
    """Luồng xử lý tự động chuyển đổi video từ Link sang Video Tiếng Việt lồng tiếng hoàn chỉnh."""
    print("=" * 70)
    print("🚀 BẮT ĐẦU TIẾN TRÌNH TỰ ĐỘNG TẠO VIDEO LỒNG TIẾNG TIẾNG VIỆT")
    print(f"🎭 Chế độ Nhái Giọng (Voice Clone): {'BẬT' if enable_clone else 'TẮT'}")
    print("=" * 70)
    print(f"🔗 Link Video: {video_url}\n")

    temp_folder = None

    try:
        # -----------------------------------------------------------------
        # BƯỚC 1: Tải Video & Khởi tạo thư mục tạm trong TEMP_DIR
        # -----------------------------------------------------------------
        print("📌 [1/6] Tải video và trích xuất dữ liệu...")
        downloader = VideoDownloader()
        success, folder_path, video_path, audio_path, json_path = downloader.download_video(video_url, verbose=True)
        if not success:
            print("❌ Quá trình tải video thất bại, vui lòng kiểm tra lại link hoặc kết nối mạng.")
            sys.exit(1)
        temp_folder = json_path.parent
        audio_file_path = temp_folder / "file.mp3"
        print(f"   📂 Thư mục làm việc tạm: {temp_folder.name}\n")

        # -----------------------------------------------------------------
        # BƯỚC 1.5: Tách Nhạc Nền / Hiệu Ứng Gốc Bằng Demucs (Meta AI)
        # -----------------------------------------------------------------
        print("📌 [1.5/6] Tách nhạc nền & hiệu ứng âm thanh bằng Demucs...")
        bg_volume = getattr(config, "BG_VOLUME", 0.5)  # Giảm xuống 50% mặc định
        try:
            downloader.separate_background_audio(
                audio_or_video_path=video_path, 
                output_dir=temp_folder, 
                bg_volume=bg_volume,
                verbose=True
            )
        except Exception as bg_err:
            print(f"⚠️ Không thể tách nhạc nền ({bg_err}). Hệ thống sẽ tiếp tục mà không có nhạc nền gốc.\n")

        # -----------------------------------------------------------------
        # BƯỚC 2: Nhận diện giọng nói & Timestamp bằng Groq Whisper
        # -----------------------------------------------------------------
        print("📌 [2/6] Nhận diện giọng nói & trích xuất mốc thời gian (Whisper)...")
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
        print("📌 [3/6] Dịch nội dung sang tiếng Việt (Llama 3.3 70B)...")
        translator = SegmentTranslator()
        translator.translate_segments(json_path, verbose=True)

        # -----------------------------------------------------------------
        # BƯỚC 4: Tạo giọng đọc AI tiếng Việt mượt mà bằng Vieneu (GPU)
        # -----------------------------------------------------------------
        print("📌 [4/6] Tạo file âm thanh lồng tiếng AI (Vieneu TTS)...")
        tts_voice = getattr(config, "DEFAULT_VOICE", "Minh Đức")
        tts_style = getattr(config, "DEFAULT_STYLE", "tu_nhien")

        ref_audio_path = None
        if enable_clone:
            ref_filename = getattr(config, "REF_AUDIO_FILENAME", "ref.wav")
            ref_duration = getattr(config, "REF_AUDIO_DURATION", 10.0)
            ref_audio_path = temp_folder / ref_filename
            print(f"🎙️ Đang trích xuất {ref_duration}s âm thanh gốc làm mẫu nhái giọng...")
            downloader.extract_reference_audio(audio_file_path, ref_audio_path, duration=ref_duration)

        tts = SegmentTTSGenerator(
            voice=tts_voice, 
            style=tts_style,
            enable_clone=enable_clone,
            ref_audio_path=ref_audio_path,
            vieneu_instance=vieneu_instance
        )
        tts.generate_speech_for_segments(json_path, verbose=True)

        # -----------------------------------------------------------------
        # BƯỚC 5: Xếp Timeline và Ghép Video + Audio Tiếng Việt (+ Nhạc Nền)
        # -----------------------------------------------------------------
        print("📌 [5/6] Căn chỉnh timeline và gộp video thành phẩm...")
        merger = AudioVideoMerger()
        temp_final_video = merger.merge_audio_video(json_path, verbose=True)

        # -----------------------------------------------------------------
        # DỌN DẸP & DỜI THÀNH PHẨM SANG THƯ MỤC OUTPUT
        # -----------------------------------------------------------------
        print("🧹 [Hoàn tất] Đang di chuyển sản phẩm về thư mục Output và xóa thư mục tạm...")
        
        output_base_dir = getattr(config, "OUTPUT_DIR", Path("output"))
        output_base_dir.mkdir(parents=True, exist_ok=True)

        final_output_folder = output_base_dir / temp_folder.name

        if final_output_folder.exists():
            shutil.rmtree(final_output_folder)

        shutil.move(str(temp_folder), str(final_output_folder))

        print("\n" + "=" * 70)
        print("🎉 TẤT CẢ CÁC BƯỚC ĐÃ HOÀN THÀNH XUẤT SẮC!")
        print(f"📂 Thư mục thành phẩm: {final_output_folder.resolve()}")
        print(f"🎬 File Video cuối cùng: {final_output_folder / 'final_output.mp4'}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ ĐÃ XẢY RA LỖI TRONG QUÁ TRÌNH XỬ LÝ: {e}")
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
    parser.add_argument(
        "--clone", "-c",
        type=int,
        choices=[0, 1],
        help="Bật nhái giọng nhân vật (1: Bật, 0: Tắt)"
    )
    args = parser.parse_args()

    video_url = args.url
    clone_choice = args.clone
    
    if not video_url:
        print("=" * 70)
        print("🎥 HỆ THỐNG TỰ ĐỘNG DỊCH VÀ LỒNG TIẾNG VIDEO (VIENEU TTS)")
        print("=" * 70)
        video_url = input("👉 Nhập link video cần xử lý: ").strip()

    if not video_url:
        print("❌ Link video không được để trống!")
        sys.exit(1)

    state = "pytorch"
    if clone_choice is None:
        user_input = input("👉 Nhập lựa chọn nhái giọng (1: Bật, 0: Tắt) [Mặc định 0]: ").strip()
        enable_clone = True if user_input == "1" else False
    else:
        state = "onnx"
        enable_clone = (clone_choice == 1)

    print(f"\n⚙️ Đang khởi tạo mô hình Vieneu TTS ({state.upper()})...")
    vieneu_instance = Vieneu(backend=state)

    process_video_pipeline(video_url, enable_clone=enable_clone, vieneu_instance=vieneu_instance)


if __name__ == "__main__":
    main()