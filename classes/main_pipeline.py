from typing import List, Optional, Dict, Any, Generator
import logging
import os
from dataclasses import dataclass
from classes.audio.audio_analyzer import AudioAnalyzer
from classes.video.segment_extractor import SegmentExtractor, Segment
from classes.audio.transcript import Transcript
from classes.audio.speaker_diarizer import SpeakerDiarizer
from classes.audio.transcript_merger import merge_transcript_and_diarization
from classes.nlp.sentiment_analysis import SentimentAnalysis
from classes.util.segment_selector import SegmentSelector
from classes.audio.audio_feature_analyzer import AudioFeatureAnalyzer
from classes.util.progress_tracker import ProgressTracker, ProcessingStage, ProgressUpdate
import json

@dataclass
class ProcessingResult:
    success: bool
    message: str
    segments: List[Segment]
    artifacts: Dict[str, str]
    progress: Dict[str, Any]

class MainPipeline: 
    """
    Orchestrates the processing pipeline and manages artifacts for video analysis.
    Coordinates between different components (audio, transcription, sentiment, etc.)
    and maintains processing state.
    """
    def __init__(self, base_dir: str = 'uploads'):
        self.logger = logging.getLogger('youtube_shorts')
        self.base_dir = base_dir
        self.temp_dir = os.path.join(base_dir, 'temp_files')
        self.segments_dir = os.path.join(base_dir, 'segments')
        
        # Ensure directories exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.segments_dir, exist_ok=True)
        
        # Initialize components
        self.progress_tracker = ProgressTracker()
        self.transcript_engine = Transcript(self.progress_tracker)
        self.speaker_diarizer = SpeakerDiarizer(self.progress_tracker)
        self.sentiment_analyzer = SentimentAnalysis(self.progress_tracker)
        self.segment_extractor = SegmentExtractor(self.progress_tracker)
        self.segment_selector = SegmentSelector(self.progress_tracker)
        self.audio_feature_analyzer = AudioFeatureAnalyzer(self.progress_tracker)
        self.progress_callback = None
        self._last_result = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def get_last_result(self) -> Optional[ProcessingResult]:
        return self._last_result

    def process_video(self, video_path: str) -> Generator[str, None, None]:
        """
        Process a video file to extract interesting segments.
        Yields progress updates as strings and stores final result.
        """
        try:
            self.progress_tracker.reset()
            self._last_result = None
            artifacts = {}
            
            # 1. Extract audio
            self.logger.info(f"Starting processing pipeline for {video_path}")
            self.progress_tracker.update_progress(
                ProcessingStage.AUDIO_EXTRACTION, 
                0.0, 
                "Starting audio extraction"
            )
            
            audio_analyzer = AudioAnalyzer(video_path, self.temp_dir, self.progress_tracker)
            audio_path = audio_analyzer._extract_audio()
            
            if self.progress_callback:
                yield self.progress_callback(self.progress_tracker.get_last_update())

            if not audio_path:
                self._last_result = ProcessingResult(
                    success=False,
                    message="Failed to extract audio",
                    segments=[],
                    artifacts={},
                    progress=self.progress_tracker.to_dict()
                )
                return
            
            artifacts['audio'] = audio_path
            
            self.progress_tracker.update_progress(
                ProcessingStage.AUDIO_EXTRACTION, 
                1.0, 
                "Audio extraction complete"
            )
            
            # 2. Generate transcript
            transcript_path = self.transcript_engine.transcribe(audio_path, self.temp_dir)
            if self.progress_callback:
                yield self.progress_callback(self.progress_tracker.get_last_update())

            if not transcript_path:
                self._last_result = ProcessingResult(
                    success=False,
                    message="Failed to generate transcript",
                    segments=[],
                    artifacts=artifacts,
                    progress=self.progress_tracker.to_dict()
                )
                return

            artifacts['transcript'] = transcript_path
            
            # Check if speaker information already exists in the transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            has_speaker_info = any('speaker' in segment for segment in transcript_data.get('segments', []))
            
            # Only run speaker diarization if needed
            if not has_speaker_info:
                self.logger.info("No speaker information found. Running speaker diarization...")
                # Process speaker diarization
                diarization_data = self.speaker_diarizer.process_audio(audio_path)
                
                # Merge transcript with diarization results
                merged_transcript = merge_transcript_and_diarization(transcript_data, diarization_data)
                
                # Save merged transcript
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_transcript, f, indent=2, ensure_ascii=False)
            else:
                self.logger.info("Using existing speaker information from transcript")
                merged_transcript = transcript_data

            # 3. Analyze sentiment
            sentiment_path = self.sentiment_analyzer.analyze_transcript(transcript_path, self.temp_dir)
            if self.progress_callback:
                yield self.progress_callback(self.progress_tracker.get_last_update())

            if not sentiment_path:
                self._last_result = ProcessingResult(
                    success=False,
                    message="Failed to analyze sentiment",
                    segments=[],
                    artifacts=artifacts,
                    progress=self.progress_tracker.to_dict()
                )
                return

            artifacts['sentiment'] = sentiment_path

            # 4. Analyze audio features
            self.progress_tracker.update_progress(
                ProcessingStage.AUDIO_FEATURE_ANALYSIS,
                0.0,
                "Starting audio feature analysis"
            )
            
            self.logger.info("Analyzing audio features")
            audio_features = self.audio_feature_analyzer.analyze_audio(audio_path)
            if self.progress_callback:
                yield self.progress_callback(self.progress_tracker.get_last_update())

            if not audio_features:
                self._last_result = ProcessingResult(
                    success=False,
                    message="Failed to analyze audio features",
                    segments=[],
                    artifacts=artifacts,
                    progress=self.progress_tracker.to_dict()
                )
                return

            self.progress_tracker.update_progress(
                ProcessingStage.SEGMENT_SELECTION,
                0.0,
                "Starting segment selection"
            )
            
            # 5. Select segments using combined features
            selected_segments = self.segment_selector.select_interesting_segments(
                transcript_path, 
                sentiment_path,
                audio_features
            )
            if self.progress_callback:
                yield self.progress_callback(self.progress_tracker.get_last_update())

            # Create Segment objects from tuples
            segments = [
                Segment(start=start, end=end)
                for start, end in selected_segments
            ]
            
            self.progress_tracker.update_progress(
                ProcessingStage.SEGMENT_SELECTION,
                1.0,
                f"Selected {len(segments)} segments"
            )
            
            # 6. Extract video segments
            if segments:
                extracted_segments = self.segment_extractor.extract_video_segments(
                    video_path, segments, self.segments_dir
                )
                if self.progress_callback:
                    yield self.progress_callback(self.progress_tracker.get_last_update())
                
                self.progress_tracker.update_progress(
                    ProcessingStage.VIDEO_EXTRACTION,
                    1.0,
                    f"Extracted {len(extracted_segments)} video segments"
                )
                if self.progress_callback:
                    yield self.progress_callback(self.progress_tracker.get_last_update())
                
                self._last_result = ProcessingResult(
                    success=True,
                    message=f"Video processed successfully. Found {len(extracted_segments)} segments.",
                    segments=extracted_segments,
                    artifacts=artifacts,
                    progress=self.progress_tracker.to_dict()
                )
                
                # Clean up temporary files after successful processing
                self.cleanup_artifacts(video_path)
                return

            self._last_result = ProcessingResult(
                success=True,
                message="Processing complete but no suitable segments found",
                segments=[],
                artifacts=artifacts,
                progress=self.progress_tracker.to_dict()
            )
            
            # Clean up even if no segments were found
            self.cleanup_artifacts(video_path)
            return
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            self._last_result = ProcessingResult(
                success=False,
                message=f"Processing failed: {str(e)}",
                segments=[],
                artifacts=artifacts,
                progress=self.progress_tracker.to_dict()
            )
            # Don't clean up on error so we can debug issues
            return
    
    def get_processing_status(self, video_path: str) -> Dict[str, Any]:
        """
        Get the current processing status and available artifacts for a video
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Check for various artifacts
        artifacts = {
            'audio': os.path.join(self.temp_dir, f"{base_name}_temp.mp3"),
            'transcript': os.path.join(self.temp_dir, f"{base_name}_transcript.json"),
            'sentiment': os.path.join(self.temp_dir, f"{base_name}_sentiment.json")
        }
        
        # Check which artifacts exist
        status = {
            'audio_extracted': os.path.exists(artifacts['audio']),
            'transcript_generated': os.path.exists(artifacts['transcript']),
            'sentiment_analyzed': os.path.exists(artifacts['sentiment']),
            'artifacts': {k: v for k, v in artifacts.items() if os.path.exists(v)}
        }
        
        return status
    
    def cleanup_artifacts(self, video_path: str, keep_transcript: bool = False, remove_source: bool = True) -> None:
        """
        Clean up temporary processing artifacts and optionally the source video.
        Args:
            video_path: Path to the original video file
            keep_transcript: Whether to keep the transcript file (default: False)
            remove_source: Whether to remove the original video file (default: True)
        """
        try:
            # Get all temp files related to this video
            status = self.get_processing_status(video_path)
            self.logger.debug(f"Cleaning up artifacts for {video_path}")
            self.logger.debug(f"Found artifacts: {status['artifacts']}")
            
            for artifact_type, path in status['artifacts'].items():
                if not (keep_transcript and artifact_type == 'transcript'):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                            self.logger.debug(f"Removed artifact: {path}")
                        else:
                            self.logger.warning(f"Artifact not found: {path}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove artifact {path}: {str(e)}")
            
            # Remove the source video if requested
            if remove_source:
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        self.logger.info(f"Removed source video: {video_path}")
                    else:
                        self.logger.warning(f"Source video not found: {video_path}")
                except Exception as e:
                    self.logger.error(f"Failed to remove source video: {str(e)}")
            
            # Clean up any other temp files with this video's name
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            self.logger.debug(f"Looking for additional temp files with base name: {base_name}")
            
            try:
                for temp_file in os.listdir(self.temp_dir):
                    if temp_file.startswith(base_name):
                        temp_path = os.path.join(self.temp_dir, temp_file)
                        try:
                            os.remove(temp_path)
                            self.logger.debug(f"Removed temp file: {temp_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove temp file {temp_path}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error while scanning temp directory: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup artifacts: {str(e)}", exc_info=True)