import os
import logging
import librosa
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from classes.util.progress_tracker import ProgressTracker, ProcessingStage

# Configure logging for librosa and soundfile
logging.getLogger('librosa').setLevel(logging.ERROR)
logging.getLogger('soundfile').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.ERROR)

@dataclass
class AudioFeatures:
    start: float  # Changed from start_time for consistency
    end: float    # Changed from end_time for consistency
    volume_rms: float
    volume_peak: float
    speech_rate: float
    silence_ratio: float
    intensity_score: float

class AudioFeatureAnalyzer:
    """
    Analyzes audio features like volume, intensity, speech rate, etc.
    to help identify engaging segments.
    """
    def __init__(self, progress_tracker: ProgressTracker = None):
        self.logger = logging.getLogger('youtube_shorts')
        self.progress_tracker = progress_tracker
        # Parameters for analysis
        self.segment_duration = 1.0  # Duration in seconds for feature extraction
        self.min_silence_duration = 0.3  # Minimum silence duration in seconds
        self.silence_threshold = -40  # dB threshold for silence detection

    def analyze_audio(self, audio_path: str) -> List[AudioFeatures]:
        """
        Analyze audio file and return features for each segment
        """
        try:
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.AUDIO_FEATURE_ANALYSIS,
                    0.1,
                    "Loading audio file"
                )

            self.logger.debug(f"Loading audio file: {audio_path}")
            # Load audio file
            y, sr = librosa.load(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            
            self.logger.debug(f"Audio duration: {duration:.2f}s, Sample rate: {sr}Hz")
            
            # Calculate features for each segment
            features = []
            total_segments = int(np.ceil(duration / self.segment_duration))
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.AUDIO_FEATURE_ANALYSIS,
                    0.2,
                    f"Analyzing {total_segments} audio segments"
                )
            
            # Calculate update interval to show roughly 20 updates during processing
            update_interval = max(1, total_segments // 20)
            
            for i, start in enumerate(np.arange(0, duration - self.segment_duration, self.segment_duration)):
                end = min(start + self.segment_duration, duration)
                
                # Get segment samples
                start_idx = int(start * sr)
                end_idx = int(end * sr)
                segment = y[start_idx:end_idx]
                
                # Calculate features
                rms = librosa.feature.rms(y=segment)[0].mean()
                peak = np.abs(segment).max()
                
                # Calculate silence ratio
                silence_mask = librosa.amplitude_to_db(np.abs(segment)) < self.silence_threshold
                silence_ratio = np.mean(silence_mask)
                
                # Estimate speech rate using zero-crossing rate as a proxy
                zcr = librosa.feature.zero_crossing_rate(segment)[0].mean()
                speech_rate = zcr * 100  # Scale for readability
                
                # Calculate overall intensity score
                intensity_score = self._calculate_intensity_score(
                    rms, peak, silence_ratio, speech_rate
                )
                
                features.append(AudioFeatures(
                    start=start,  # Updated field name
                    end=end,      # Updated field name
                    volume_rms=float(rms),
                    volume_peak=float(peak),
                    speech_rate=float(speech_rate),
                    silence_ratio=float(silence_ratio),
                    intensity_score=float(intensity_score)
                ))
                
                # Update progress less frequently
                if self.progress_tracker and i % update_interval == 0:
                    progress = 0.2 + (0.7 * (i / total_segments))
                    self.progress_tracker.update_progress(
                        ProcessingStage.AUDIO_FEATURE_ANALYSIS,
                        progress,
                        f"Analyzed {i}/{total_segments} segments"
                    )
                
                # Log interesting features for debugging
                if intensity_score > 0.8:  # Log high-intensity segments
                    self.logger.debug(
                        f"High intensity segment at {start:.1f}s: "
                        f"score={intensity_score:.2f}, "
                        f"volume={peak:.2f}, "
                        f"speech_rate={speech_rate:.2f}"
                    )
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.AUDIO_FEATURE_ANALYSIS,
                    1.0,
                    f"Completed audio analysis of {len(features)} segments"
                )
            
            self.logger.info(
                f"Audio analysis complete. "
                f"Found {sum(1 for f in features if f.intensity_score > 0.8)} "
                f"high-intensity segments"
            )
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to analyze audio features: {str(e)}", exc_info=True)
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.AUDIO_FEATURE_ANALYSIS,
                    1.0,
                    f"Audio analysis failed: {str(e)}"
                )
            return []

    def _calculate_intensity_score(
        self,
        rms: float,
        peak: float,
        silence_ratio: float,
        speech_rate: float
    ) -> float:
        """
        Calculate a combined intensity score from various features
        """
        # Normalize each component to 0-1 range (these weights can be tuned)
        volume_score = (rms + peak) / 2  # Combine RMS and peak volume
        speech_activity_score = 1 - silence_ratio  # More speech = higher score
        rate_score = min(speech_rate / 200, 1)  # Cap speech rate score
        
        # Weight and combine scores (adjust weights based on importance)
        weights = {
            'volume': 0.4,
            'speech_activity': 0.4,
            'speech_rate': 0.2
        }
        
        intensity_score = (
            volume_score * weights['volume'] +
            speech_activity_score * weights['speech_activity'] +
            rate_score * weights['speech_rate']
        )
        
        return intensity_score