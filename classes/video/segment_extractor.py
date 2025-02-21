import os
from moviepy.editor import VideoFileClip
import logging
from typing import List, NamedTuple
from classes.util.progress_tracker import ProgressTracker, ProcessingStage

class Segment(NamedTuple):
    start: float
    end: float

class SegmentExtractor:
    """
    Handles the extraction of video segments based on timestamp ranges.
    Uses MoviePy for video manipulation but abstracts the implementation details.
    """
    def __init__(self, progress_tracker: ProgressTracker = None):
        self.logger = logging.getLogger('youtube_shorts')
        self.progress_tracker = progress_tracker

    def extract_video_segments(self, video_path: str, segments: List[Segment], output_dir: str) -> List[Segment]:
        """Extract segments from source video based on timestamp ranges."""
        extracted_segments = []
        try:
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.VIDEO_EXTRACTION,
                    0.1,
                    f"Loading video file: {os.path.basename(video_path)}"
                )
            
            self.logger.debug(f"Opening video file: {video_path}")
            with VideoFileClip(video_path) as video:
                total_segments = len(segments)
                self.logger.info(f"Extracting {total_segments} segments from video")
                
                for i, segment in enumerate(segments):
                    start = segment.start
                    end = segment.end
                    segment_duration = end - start
                    
                    if 15 <= segment_duration <= 65:  # Updated to match segment selector constraints
                        if self.progress_tracker:
                            self.progress_tracker.update_progress(
                                ProcessingStage.VIDEO_EXTRACTION,
                                0.1 + (0.8 * (i / total_segments)),
                                f"Extracting segment {i+1}/{total_segments} ({segment_duration:.1f}s)"
                            )
                        
                        output_path = os.path.join(
                            output_dir, 
                            f"segment_{i}_{os.path.basename(video_path)}"
                        )
                        
                        self.logger.debug(
                            f"Extracting segment {i+1}: "
                            f"{start:.1f}s to {end:.1f}s "
                            f"(duration: {segment_duration:.1f}s)"
                        )
                        
                        # Extract the subclip
                        subclip = video.subclip(start, end)
                        
                        # Write the new video file
                        subclip.write_videofile(
                            output_path,
                            codec='libx264',
                            audio_codec='aac',
                            temp_audiofile=os.path.join(output_dir, 'temp-audio.m4a'),
                            remove_temp=True,
                            logger=None  # Disable moviepy's internal logging
                        )
                        
                        extracted_segments.append(segment)
                        self.logger.info(
                            f"Extracted segment {i+1}: "
                            f"{start:.2f}s to {end:.2f}s -> {output_path}"
                        )
                
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.VIDEO_EXTRACTION,
                    1.0,
                    f"Successfully extracted {len(extracted_segments)} segments"
                )
            
            self.logger.info(f"Extracted {len(extracted_segments)} video segments")
            return extracted_segments
            
        except Exception as e:
            self.logger.error(f"Failed to extract video segments: {str(e)}", exc_info=True)
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.VIDEO_EXTRACTION,
                    1.0,
                    f"Video extraction failed: {str(e)}"
                )
            return []