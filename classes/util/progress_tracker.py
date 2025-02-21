from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json

class ProcessingStage(Enum):
    AUDIO_EXTRACTION = "audio_extraction"
    TRANSCRIPTION = "transcription"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    AUDIO_FEATURE_ANALYSIS = "audio_feature_analysis"
    SEGMENT_SELECTION = "segment_selection"
    VIDEO_EXTRACTION = "video_extraction"

@dataclass
class ProgressUpdate:
    stage: ProcessingStage
    progress: float  # 0 to 1
    message: str
    details: Optional[Dict[str, Any]] = None

class ProgressTracker:
    """
    Tracks progress of video processing and provides updates that can be sent to the frontend
    """
    def __init__(self):
        self.logger = logging.getLogger('youtube_shorts')
        self._current_stage: Optional[ProcessingStage] = None
        self._progress: Dict[ProcessingStage, float] = {
            stage: 0.0 for stage in ProcessingStage
        }
        self._stage_weights = {
            ProcessingStage.AUDIO_EXTRACTION: 0.1,
            ProcessingStage.TRANSCRIPTION: 0.3,
            ProcessingStage.SENTIMENT_ANALYSIS: 0.2,
            ProcessingStage.AUDIO_FEATURE_ANALYSIS: 0.1,
            ProcessingStage.SEGMENT_SELECTION: 0.1,
            ProcessingStage.VIDEO_EXTRACTION: 0.2
        }
        self._last_update: Optional[ProgressUpdate] = None

    def update_progress(
        self, 
        stage: ProcessingStage, 
        progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> ProgressUpdate:
        """
        Update progress for a specific stage and return a progress update object
        progress should be between 0 and 1
        """
        self._current_stage = stage
        self._progress[stage] = min(max(progress, 0.0), 1.0)
        
        update = ProgressUpdate(
            stage=stage,
            progress=self.total_progress,
            message=message,
            details=details
        )
        
        # Store the update
        self._last_update = update
        
        # Log the update
        self.logger.debug(
            f"Progress update: {stage.value} - {progress:.1%} - {message}"
        )
        
        return update

    def get_last_update(self) -> Optional[ProgressUpdate]:
        """
        Get the most recent progress update
        """
        return self._last_update

    @property
    def total_progress(self) -> float:
        """
        Calculate total progress across all stages
        """
        weighted_progress = sum(
            self._progress[stage] * weight
            for stage, weight in self._stage_weights.items()
        )
        return weighted_progress

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert current progress state to a dictionary
        """
        return {
            "current_stage": self._current_stage.value if self._current_stage else None,
            "total_progress": self.total_progress,
            "stage_progress": {
                stage.value: progress
                for stage, progress in self._progress.items()
            }
        }

    def reset(self) -> None:
        """
        Reset progress tracking
        """
        self._current_stage = None
        self._progress = {stage: 0.0 for stage in ProcessingStage}
        self._last_update = None
        self.logger.debug("Progress tracking reset")