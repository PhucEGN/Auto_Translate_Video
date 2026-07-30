import json
import sys
import subprocess
from pathlib import Path
from typing import Optional
from pydub import AudioSegment

# Đảm bảo import và sử dụng thông tin từ config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class AudioVideoMerger:
    """Class xử lý việc căn chỉnh thời gian âm thanh dịch và gộp vào video gốc (hỗ trợ nhạc nền gốc & mirror video)."""

    def __init__(self):
        pass

    def merge_audio_video(
        self, 
        json_path: Path, 
        bg_volume: float = 0.7,      # Âm lượng nhạc nền gốc (0.7 = 70%)
        speech_volume: float = 1.2,  # Âm lượng giọng lồng tiếng (1.2 = 120%)
        mirror_video: bool = True,  # Bổ sung option Lật gương (Flip ngang)
        verbose: bool = True
    ) -> Path:
        """Đọc info.json, xếp toàn bộ âm thanh vào đúng timeline và gộp vào video (kèm nhạc nền/mirror nếu có)."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy file JSON tại: {json_path}")

        folder_path = json_path.parent
        input_video = folder_path / "file.mp4"

        if not input_video.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy video gốc tại: {input_video}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        if not segments:
            raise ValueError("⚠️ Không tìm thấy 'segments' trong info.json!")

        # 1. Lấy tổng độ dài của video gốc (tính bằng giây)
        video_duration = self._get_video_duration(input_video)
        if verbose:
            print(f"🎬 [Merger] Độ dài video gốc: {video_duration:.2f}s")
            if mirror_video:
                print("🪞 [Merger] Đã bật chế độ Lật gương (Mirror/Horizontal Flip) cho Video!")
            print(f"🔊 [Merger] Đang xử lý và xếp timeline cho {len(segments)} segments...")

        # Tạo thư mục tạm lưu các file audio đã được căn chỉnh tốc độ
        adjusted_dir = folder_path / "adjusted_audio_segments"
        adjusted_dir.mkdir(parents=True, exist_ok=True)

        # 2. Khai báo 1 đường timeline âm thanh im lặng đúng bằng độ dài video (tính bằng ms)
        master_audio = AudioSegment.silent(duration=int(video_duration * 1000))

        for idx, seg in enumerate(segments):
            seg_id = seg.get("id", idx)
            start_time_sec = float(seg.get("start", 0.0))
            end_time_sec = float(seg.get("end", 0.0))
            target_duration = max(end_time_sec - start_time_sec, 0.5)

            audio_rel_path = seg.get("translated_audio_file")
            if not audio_rel_path:
                continue

            audio_abs_path = folder_path / audio_rel_path
            if not audio_abs_path.exists():
                if verbose:
                    print(f"⚠️ [Seg {seg_id}] Không tìm thấy file: {audio_abs_path}")
                continue

            # Đo độ dài file MP3 hiện tại
            seg_audio_raw = AudioSegment.from_file(audio_abs_path)
            actual_duration = seg_audio_raw.duration_seconds

            final_audio_path = audio_abs_path

            # CHỈ XỬ LÝ NẾU ÂM THANH DÀI HƠN THỜI GIAN MỤC TIÊU
            if actual_duration > target_duration:
                speed_factor = actual_duration / target_duration
                # Khống chế mức nén tối đa 1.4x để tránh bị quá nhanh/méo tiếng
                speed_factor = min(speed_factor, 1.4)

                adjusted_filepath = adjusted_dir / f"adj_seg_{seg_id:03d}.mp3"
                self._adjust_audio_speed(audio_abs_path, adjusted_filepath, speed_factor)
                final_audio_path = adjusted_filepath
                
                if verbose:
                    print(f"  ⚡ [Seg {seg_id}] Dài hơn gốc ({actual_duration:.2f}s > {target_duration:.2f}s) -> Ép tốc độ {speed_factor:.2f}x")
            else:
                if verbose:
                    print(f"  ✅ [Seg {seg_id}] Ngắn hơn/bằng gốc ({actual_duration:.2f}s <= {target_duration:.2f}s) -> Giữ nguyên tốc độ")

            # Load file audio cuối cùng và đè vào vị trí start_time trên Master Timeline
            processed_seg_audio = AudioSegment.from_file(final_audio_path)
            start_ms = int(start_time_sec * 1000)
            master_audio = master_audio.overlay(processed_seg_audio, position=start_ms)

        # 3. Xuất file âm thanh giọng lồng tiếng tổng hợp
        master_audio_path = folder_path / "full_translated_audio.wav"
        master_audio.export(master_audio_path, format="wav")

        # 4. Kiểm tra file nhạc nền/hiệu ứng gốc (background.wav)
        bg_audio_path = folder_path / "background.wav"
        output_video = folder_path / "final_output.mp4"

        if bg_audio_path.exists():
            if verbose:
                print("🎼 [Merger] Phát hiện nhạc nền/hiệu ứng gốc! Đang hòa trộn nhạc nền + giọng lồng tiếng...")
            self._render_final_video_with_bg(
                input_video, 
                master_audio_path, 
                bg_audio_path, 
                output_video,
                bg_volume=bg_volume,
                speech_volume=speech_volume,
                mirror_video=mirror_video
            )
        else:
            if verbose:
                print("⚙️ [Merger] Không tìm thấy nhạc nền, đang gộp đường âm thanh lồng tiếng vào video...")
            self._render_final_video(
                input_video, 
                master_audio_path, 
                output_video, 
                mirror_video=mirror_video
            )

        # Cập nhật JSON
        data["status"] = "completed"
        data["output_video"] = "final_output.mp4"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"\n✅ Đã xuất thành công video hoàn chỉnh!")
            print(f" 🎥 File kết quả: {output_video}\n")

        return output_video

    def _get_video_duration(self, video_path: Path) -> float:
        """Lấy chính xác tổng số giây của video bằng ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())

    def _adjust_audio_speed(self, input_mp3: Path, output_mp3: Path, speed_factor: float):
        """Dùng FFmpeg atempo để tăng tốc độ nếu câu nói quá dài."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_mp3),
            "-filter:a", f"atempo={speed_factor}",
            "-vn",
            str(output_mp3)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def _render_final_video(
        self, 
        video_path: Path, 
        audio_path: Path, 
        output_path: Path, 
        mirror_video: bool = False
    ):
        """Ghép video gốc với file WAV tổng (Hỗ trợ lật gương nếu mirror_video=True)."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path)
        ]

        if mirror_video:
            # Re-encode video với filter hflip (lật ngang)
            cmd.extend([
                "-vf", "hflip",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22"
            ])
        else:
            # Giữ nguyên video stream -> Siêu nhanh
            cmd.extend(["-c:v", "copy"])

        cmd.extend([
            "-map", "0:v:0",      # Lấy luồng video từ file 0
            "-map", "1:a:0",      # Lấy luồng audio tổng từ file 1
            "-c:a", "aac",        # Encode âm thanh tiếng Việt mới
            "-b:a", "192k",
            str(output_path)
        ])
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def _render_final_video_with_bg(
        self, 
        video_path: Path, 
        speech_audio_path: Path, 
        bg_audio_path: Path, 
        output_path: Path,
        bg_volume: float = 0.7,
        speech_volume: float = 1.2,
        mirror_video: bool = False
    ):
        """Trộn giọng lồng tiếng Việt với Nhạc nền/Hiệu ứng gốc rồi xuất video (Hỗ trợ lật gương)."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),          # Input 0: Video gốc
            "-i", str(speech_audio_path),   # Input 1: Giọng lồng tiếng Việt
            "-i", str(bg_audio_path)        # Input 2: Nhạc nền & hiệu ứng gốc
        ]

        # Xử lý luồng Audio
        filter_complex = (
            f"[1:a]volume={speech_volume}[a1];"
            f"[2:a]volume={bg_volume}[a2];"
            f"[a1][a2]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )

        # Xử lý Video Filter (Mirror)
        if mirror_video:
            filter_complex = f"[0:v]hflip[vout];" + filter_complex
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22"
            ])
        else:
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "0:v:0",
                "-c:v", "copy"
            ])

        cmd.extend([
            "-map", "[aout]",               # Lấy luồng audio đã hòa trộn
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path)
        ])
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


if __name__ == "__main__":
    # Test trực tiếp module
    temp_dir = getattr(config, "TEMP_DIR", Path("output/temp"))

    if temp_dir.exists():
        subdirs = sorted([d for d in temp_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)

        if subdirs:
            target_folder = subdirs[0]
            target_json = target_folder / "info.json"

            if target_json.exists():
                print(f"🔍 Đang chạy test AudioVideoMerger trên folder: {target_folder.name}\n")

                merger = AudioVideoMerger()
                # Thử truyền mirror_video=True để test lật gương
                final_file = merger.merge_audio_video(
                    json_path=target_json, 
                    mirror_video=True, 
                    verbose=True
                )
            else:
                print(f"❌ Không tìm thấy 'info.json' trong: {target_folder}")
        else:
            print(f"⚠️ Thư mục '{temp_dir}' đang trống.")
    else:
        print(f"❌ Thư mục TEMP_DIR không tồn tại: {temp_dir}")