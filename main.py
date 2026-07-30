import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from vieneu import Vieneu

# Import các thông số cấu hình từ config.py
import config

# Import 5 module chính trong hệ thống
from modules.downloader import VideoDownloader
from modules.transcriber import AudioTranscriber
from modules.translator import SegmentTranslator
from modules.tts_generator import SegmentTTSGenerator
from modules.audio_video_merger import AudioVideoMerger


def scan_available_voices(voices_dir: Path = Path("voices")) -> List[Path]:
    """Quét thư mục voices/ để lấy danh sách các file giọng đọc (.pt, .wav, .mp3)."""
    if not voices_dir.exists():
        voices_dir.mkdir(parents=True, exist_ok=True)
        return []

    valid_extensions = {".pt", ".wav", ".mp3"}
    voice_files = [
        f for f in voices_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    return sorted(voice_files, key=lambda x: x.name)


def select_voice_interactive(voice_files: List[Path]) -> Tuple[Optional[Path], str]:
    """Hiển thị danh sách giọng đọc sẵn có và cho phép người dùng lựa chọn."""
    print("\n" + "=" * 50)
    print("🎙️ DANH SÁCH GIỌNG ĐỌC SẴN CÓ (Thư mục voices/)")
    print("=" * 50)

    # Lựa chọn 0: Sử dụng giọng đọc mặc định của hệ thống
    default_voice_name = getattr(config, "DEFAULT_VOICE", "Minh Đức")
    print(f"  [0] Giọng mặc định hệ thống ({default_voice_name})")

    for idx, voice_path in enumerate(voice_files, start=1):
        file_type = "Profile (.pt)" if voice_path.suffix.lower() == "Audio (.wav/.mp3)" else "Audio (.wav/.mp3)"
        print(f"  [{idx}] {voice_path.stem} [{file_type}]")

    print("=" * 50)

    while True:
        choice = input(f"👉 Chọn giọng đọc (0-{len(voice_files)}) [Mặc định 0]: ").strip()
        
        if not choice or choice == "0":
            return None, default_voice_name

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(voice_files):
                selected_file = voice_files[idx - 1]
                return selected_file, selected_file.stem

        print("⚠️ Lựa chọn không hợp lệ, vui lòng nhập lại!")


def process_video_pipeline(
    video_url: str, 
    selected_voice_path: Optional[Path] = None,
    selected_voice_name: str = "Minh Đức",
    vieneu_instance: Vieneu = None
):
    """Luồng xử lý tự động chuyển đổi video từ Link sang Video Tiếng Việt lồng tiếng hoàn chỉnh."""
    print("=" * 70)
    print("🚀 BẮT ĐẦU TIẾN TRÌNH TỰ ĐỘNG TẠO VIDEO LỒNG TIẾNG TIẾNG VIỆT")
    print(f"🎙️ Giọng đọc sử dụng: {selected_voice_name}")
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
            return False

        temp_folder = json_path.parent
        audio_file_path = temp_folder / "file.mp3"
        print(f"   📂 Thư mục làm việc tạm: {temp_folder.name}\n")

        # -----------------------------------------------------------------
        # BƯỚC 1.5: Tách Nhạc Nền / Hiệu Ứng Gốc Bằng Demucs (Meta AI)
        # -----------------------------------------------------------------
        print("📌 [1.5/6] Tách nhạc nền & hiệu ứng âm thanh bằng Demucs...")
        bg_volume = getattr(config, "BG_VOLUME", 0.5)
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
        tts_style = getattr(config, "DEFAULT_STYLE", "tu_nhien")

        # Cấu hình chế độ clone từ file giọng đã chọn (nếu có)
        enable_clone = selected_voice_path is not None and selected_voice_path.exists()

        tts = SegmentTTSGenerator(
            voice=selected_voice_name, 
            style=tts_style,
            enable_clone=enable_clone,
            ref_audio_path=selected_voice_path,
            vieneu_instance=vieneu_instance,
            volume_gain_db=4.0
        )
        tts.generate_speech_for_segments(json_path, verbose=True)

        # -----------------------------------------------------------------
        # BƯỚC 5: Xếp Timeline và Ghép Video + Audio Tiếng Việt (+ Nhạc Nền)
        # -----------------------------------------------------------------
        print("📌 [5/6] Căn chỉnh timeline và gộp video thành phẩm...")
        merger = AudioVideoMerger()
        merger.merge_audio_video(json_path, verbose=True)

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
        return True

    except Exception as e:
        print(f"\n❌ ĐÃ XẢY RA LỖI TRONG QUÁ TRÌNH XỬ LÝ: {e}")
        if temp_folder and temp_folder.exists():
            print(f"🗑️ Đang dọn dẹp thư mục tạm bị lỗi: {temp_folder}")
            shutil.rmtree(temp_folder, ignore_errors=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Auto Translate & Dub Video Pipeline")
    parser.add_argument(
        "--url", "-u", 
        type=str, 
        help="Đường dẫn URL của video (YouTube, Facebook, TikTok, ...)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🎥 HỆ THỐNG TỰ ĐỘNG DỊCH VÀ LỒNG TIẾNG VIDEO (VIENEU TTS)")
    print("=" * 70)
    print("⚙️ Đang khởi tạo mô hình Vieneu TTS trên GPU...")
    
    # Khởi tạo mô hình 1 lần duy nhất trên GPU CUDA
    vieneu_instance = Vieneu(backbone_device="cuda")
    print("✅ Load Model Vieneu thành công vào VRAM/RAM!\n")

    first_run_url = args.url

    # VÒNG LẶP XỬ LÝ NHIỀU VIDEO CONTINUOUSLY
    while True:
        if first_run_url:
            video_url = first_run_url
            first_run_url = None  # Reset lại sau khi dùng lần đầu
        else:
            print("-" * 70)
            video_url = input("👉 Nhập link video mới (hoặc gõ 'q' / 'exit' để thoát): ").strip()
            
            if video_url.lower() in ['q', 'exit', 'quit']:
                print("👋 Cảm ơn bạn đã sử dụng hệ thống! Đã thoát.")
                break

            if not video_url:
                print("⚠️ Link video không được để trống!")
                continue

        # 1. Quét danh sách file giọng trong thư mục voices/
        voice_files = scan_available_voices(Path("voices"))

        # 2. Cho người dùng tương tác chọn giọng đọc
        selected_voice_path, selected_voice_name = select_voice_interactive(voice_files)

        # 3. Thực thi pipeline xử lý video
        process_video_pipeline(
            video_url=video_url, 
            selected_voice_path=selected_voice_path,
            selected_voice_name=selected_voice_name,
            vieneu_instance=vieneu_instance
        )


if __name__ == "__main__":
    main()