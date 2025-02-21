from typing import List, NamedTuple
import logging
import os
from moviepy.editor import VideoFileClip
from classes.util.progress_tracker import ProgressTracker, ProcessingStage

class Segment(NamedTuple):
    start: float  # Changed from start_time for consistency
    end: float    # Changed from end_time for consistency
 
class AudioAnalyzer:
    def __init__(self, video_path: str, output_dir: str, progress_tracker: ProgressTracker = None):
        self.video_path = video_path
        self.output_dir = output_dir
        self.temp_dir = os.path.join('uploads', 'temp_files')
        os.makedirs(self.temp_dir, exist_ok=True)
        self.logger = logging.getLogger('youtube_shorts')
        self.progress_tracker = progress_tracker

    def _extract_audio(self) -> str:
        """Extract audio from video to temporary file."""
        try:
            video_filename = os.path.basename(self.video_path)
            temp_audio_filename = f"{os.path.splitext(video_filename)[0]}_temp.mp3"
            temp_audio_path = os.path.join(self.temp_dir, temp_audio_filename)
            
            if os.path.exists(temp_audio_path):
                self.logger.info("Using existing temp audio file")
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        ProcessingStage.AUDIO_EXTRACTION,
                        1.0,
                        "Using existing audio file"
                    )
                return temp_audio_path
            
            # First try to check if the video file is valid using ffmpeg directly
            import subprocess
            try:
                # Try to get video information using ffprobe
                probe_cmd = [
                    'ffprobe', 
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    self.video_path
                ]
                video_codec = subprocess.check_output(probe_cmd, stderr=subprocess.PIPE).decode().strip()
                self.logger.debug(f"Video codec: {video_codec}")
                
                # Extract audio directly using ffmpeg first
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', self.video_path,
                    '-vn',  # No video
                    '-acodec', 'libmp3lame',
                    '-ab', '192k',
                    '-ar', '44100',
                    '-y',  # Overwrite output file
                    temp_audio_path
                ]
                
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        ProcessingStage.AUDIO_EXTRACTION,
                        0.3,
                        "Extracting audio using ffmpeg"
                    )
                
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                if os.path.exists(temp_audio_path):
                    if self.progress_tracker:
                        self.progress_tracker.update_progress(
                            ProcessingStage.AUDIO_EXTRACTION,
                            1.0,
                            "Audio extraction complete"
                        )
                    return temp_audio_path
                    
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"FFmpeg extraction failed: {e.stderr.decode()}, falling back to MoviePy")
            
            # If ffmpeg direct extraction fails, try MoviePy as fallback
            self.logger.debug(f"Extracting audio using MoviePy from: {self.video_path}")
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.AUDIO_EXTRACTION,
                    0.3,
                    "Extracting audio using MoviePy"
                )

            with VideoFileClip(self.video_path) as video:
                if video.audio is None:
                    self.logger.error("No audio track found in video")
                    return None
                    
                self.logger.info(f"Extracting audio to {temp_audio_path}")
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        ProcessingStage.AUDIO_EXTRACTION,
                        0.3,
                        "Extracting audio from video"
                    )
                
                video.audio.write_audiofile(
                    temp_audio_path,
                    fps=44100,  # Standard audio quality
                    nbytes=2,   # 16-bit audio
                    codec='libmp3lame',  # MP3 codec
                    logger=None  # Disable moviepy's internal logging
                )
                
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        ProcessingStage.AUDIO_EXTRACTION,
                        1.0,
                        "Audio extraction complete"
                    )
            
            return temp_audio_path
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {str(e)}")
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.AUDIO_EXTRACTION,
                    1.0,
                    f"Audio extraction failed: {str(e)}"
                )
            return None
