import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import logging
from typing import List, NamedTuple
from classes.video.segment_extractor import Segment

class VideoEditor:
    def __init__(self, output_dir: str = 'uploads/segments'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger('youtube_shorts')

    def extract_video_segments(self, video_path: str, segments: List[NamedTuple], output_dir: str) -> List[NamedTuple]:
        """Extracts video segments using MoviePy."""
        extracted_segments = []
        try:
            with VideoFileClip(video_path) as video:
                for i, segment in enumerate(segments):
                    start_time = segment.start
                    end_time = segment.end
                    segment_duration = end_time - start_time
                    
                    if 15 <= segment_duration <= 65:  # Updated to match segment selector constraints
                        output_path = os.path.join(output_dir, f"segment_{i}_{os.path.basename(video_path)}")
                        
                        # Extract the subclip
                        subclip = video.subclip(start_time, end_time)
                        
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
            
            self.logger.info(f"Extracted {len(extracted_segments)} video segments")
            return extracted_segments
            
        except Exception as e:
            self.logger.error(f"Failed to extract video segments: {str(e)}")
            return []

    def combine_segments(self, video_path: str, segments: List[Segment], output_filename: str) -> str:
        """Combine multiple segments into a single video file."""
        clips = []
        with VideoFileClip(video_path) as video:
            for segment in segments:
                start = segment.start
                end = segment.end
                segment_duration = end - start
                
                if 10 <= segment_duration <= 60:
                    # Extract the subclip
                    subclip = video.subclip(start, end)
                    clips.append(subclip)

        if clips:
            # Combine all clips
            final_clip = concatenate_videoclips(clips)
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Write the combined video
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac'
            )
            
            return output_path
            
        return None
