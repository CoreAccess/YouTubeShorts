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
    start: float
    end: float
    volume_rms: float
    volume_peak: float
    speech_rate: float
    silence_ratio: float
    intensity_score: float
    speech_pattern_score: float = 0.0  # New: Score for natural speech patterns
    conversation_likelihood: float = 0.0  # New: Likelihood this is part of a conversation

class AudioFeatureAnalyzer:
    """
    Analyzes audio features like volume, intensity, speech rate, etc.
    to help identify engaging segments.
    """
    def __init__(self, progress_tracker: ProgressTracker = None):
        self.logger = logging.getLogger('youtube_shorts')
        self.progress_tracker = progress_tracker
        
        # Refined parameters for better feature extraction
        self.segment_duration = 0.5  # Reduced for finer granularity
        self.min_silence_duration = 0.25  # Better detection of natural pauses
        self.silence_threshold = -35  # Slightly less sensitive to silence
        
        # Energy and intensity thresholds
        self.low_energy_threshold = 0.2
        self.high_energy_threshold = 0.8
        self.speech_rate_threshold = 150  # Words per minute approx

    def analyze_audio(self, audio_path: str) -> List[AudioFeatures]:
        try:
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.AUDIO_FEATURE_ANALYSIS,
                    0.1,
                    "Loading audio file"
                )

            # Load audio with resampling for memory efficiency
            y, sr = librosa.load(audio_path, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Pre-calculate features with smaller hop length for better precision
            hop_length = int(self.segment_duration * sr)  # 0.5s segments
            
            # Enhanced feature extraction - ensure numpy arrays
            rms = np.array(librosa.feature.rms(y=y, hop_length=hop_length)[0])
            zcr = np.array(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0])
            spec_cent = np.array(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0])
            
            # Dynamic threshold calculation based on audio statistics
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            silence_threshold_db = float(librosa.amplitude_to_db(np.array([rms_mean - rms_std]))) # Fix array type
            
            features = []
            total_segments = len(rms)
            update_interval = max(1, total_segments // 20)
            
            # Rolling window for smoother feature calculation
            window_size = 3  # Number of segments to consider for smoothing
            
            for i in range(total_segments):
                start = float(i * self.segment_duration)
                end = float(min(start + self.segment_duration, duration))
                
                # Get windowed features for smoother analysis
                window_start = max(0, i - window_size // 2)
                window_end = min(total_segments, i + window_size // 2 + 1)
                
                # Calculate smoothed features - ensure float conversions
                rms_smooth = float(np.mean(rms[window_start:window_end]))
                zcr_smooth = float(np.mean(zcr[window_start:window_end]))
                spec_cent_smooth = float(np.mean(spec_cent[window_start:window_end]))
                
                # Normalize features
                rms_norm = float(rms_smooth / np.max(rms))
                spec_cent_norm = float(spec_cent_smooth / np.mean(spec_cent))
                
                # Enhanced speech rate estimation using both ZCR and spectral centroid
                speech_rate = float((zcr_smooth * 150 + spec_cent_norm * 50) / 2)
                
                # More sophisticated silence detection
                segment_db = float(librosa.amplitude_to_db(np.array([rms_smooth])))  # Fix array type
                silence_ratio = float(segment_db < silence_threshold_db)
                
                # Calculate refined intensity score
                intensity_score = self._calculate_intensity_score(
                    rms_norm,
                    min(1.0, float(rms_smooth * 2 / np.max(rms))),
                    silence_ratio,
                    speech_rate
                )
                
                features.append(AudioFeatures(
                    start=start,
                    end=end,
                    volume_rms=float(rms_norm),
                    volume_peak=float(min(1.0, rms_smooth * 2 / np.max(rms))),
                    speech_rate=float(speech_rate),
                    silence_ratio=float(silence_ratio),
                    intensity_score=float(intensity_score),
                    speech_pattern_score=0.0,  # Will be set in post-processing
                    conversation_likelihood=0.0  # Will be set in post-processing
                ))
                
                if self.progress_tracker and i % update_interval == 0:
                    progress = 0.2 + (0.7 * (i / total_segments))
                    self.progress_tracker.update_progress(
                        ProcessingStage.AUDIO_FEATURE_ANALYSIS,
                        progress,
                        f"Analyzed {i}/{total_segments} segments"
                    )
            
            # Post-process features to identify conversation patterns
            self._post_process_features(features)
            
            self.logger.info(
                f"Audio analysis complete. "
                f"Found {sum(1 for f in features if f.intensity_score > 0.7)} "
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

    def _post_process_features(self, features: List[AudioFeatures]) -> None:
        if not features:
            return
            
        window_size = 5  # Look at 2.5 seconds before and after
        
        for i in range(len(features)):
            # Get surrounding features
            start_idx = max(0, i - window_size)
            end_idx = min(len(features), i + window_size + 1)
            window = features[start_idx:end_idx]
            
            # Calculate speech pattern metrics with more lenient thresholds
            intensities = np.array([f.intensity_score for f in window], dtype=np.float64)
            silence_ratios = np.array([f.silence_ratio for f in window], dtype=np.float64)
            speech_rates = np.array([f.speech_rate for f in window], dtype=np.float64)
            
            intensity_variance = float(np.std(intensities))
            silence_pattern = float(np.mean(silence_ratios))
            speech_rate_variance = float(np.std(speech_rates))
            
            # Even more lenient speech pattern scoring
            speech_pattern_score = 0.0
            
            # More tolerant intensity variation range
            if 0.12 <= intensity_variance <= 0.48:  # Widened from 0.15-0.45
                speech_pattern_score += 0.4
            elif 0.08 <= intensity_variance <= 0.55:  # Widened from 0.1-0.5
                speech_pattern_score += 0.25  # Increased from 0.2
            
            # More tolerant pause patterns
            if 0.12 <= silence_pattern <= 0.48:  # Widened from 0.15-0.45
                speech_pattern_score += 0.3
            elif 0.08 <= silence_pattern <= 0.55:  # Widened from 0.1-0.5
                speech_pattern_score += 0.2  # Increased from 0.15
            
            # More tolerant speech rate variation
            if 0.06 <= speech_rate_variance <= 0.38:  # Widened from 0.08-0.35
                speech_pattern_score += 0.3
            elif 0.04 <= speech_rate_variance <= 0.45:  # Widened from 0.05-0.4
                speech_pattern_score += 0.2  # Increased from 0.15
            
            # Calculate conversation likelihood with adjusted thresholds
            conversation_indicators = [
                1.0 if 0.12 <= intensity_variance <= 0.48 else  # Widened from 0.15-0.45
                0.6 if 0.08 <= intensity_variance <= 0.55 else 0.0,  # Increased partial credit from 0.5
                
                1.0 if 0.12 <= silence_pattern <= 0.48 else  # Widened from 0.15-0.45
                0.6 if 0.08 <= silence_pattern <= 0.55 else 0.0,  # Increased partial credit from 0.5
                
                1.0 if 0.06 <= speech_rate_variance <= 0.38 else  # Widened from 0.08-0.35
                0.6 if 0.04 <= speech_rate_variance <= 0.45 else 0.0,  # Increased partial credit from 0.5
                
                1.0 if features[i].intensity_score >= 0.45 else  # Reduced from 0.5
                0.6 if features[i].intensity_score >= 0.35 else 0.0  # Reduced from 0.4, increased credit from 0.5
            ]
            
            conversation_likelihood = sum(conversation_indicators) / len(conversation_indicators)
            
            # Update feature scores
            features[i].speech_pattern_score = float(speech_pattern_score)
            features[i].conversation_likelihood = float(conversation_likelihood)
            
            # More lenient score adjustments with stronger boosts
            if speech_pattern_score > 0.55:  # Reduced from 0.6
                features[i].intensity_score = float(min(1.0, features[i].intensity_score * 1.3))  # Increased from 1.25
            elif speech_pattern_score > 0.35:  # Reduced from 0.4
                features[i].intensity_score = float(min(1.0, features[i].intensity_score * 1.15))  # Increased from 1.1
            elif speech_pattern_score < 0.25:  # Reduced from 0.3
                features[i].intensity_score = float(features[i].intensity_score * 0.97)  # Less penalty from 0.95
                
            # More lenient conversation flow adjustments
            if conversation_likelihood > 0.55:  # Reduced from 0.6
                features[i].intensity_score = float(min(1.0, features[i].intensity_score * 1.25))  # Increased from 1.2
            elif conversation_likelihood > 0.35:  # Reduced from 0.4
                features[i].intensity_score = float(min(1.0, features[i].intensity_score * 1.15))  # Increased from 1.1
            elif conversation_likelihood < 0.25:  # Reduced from 0.3
                features[i].intensity_score = float(features[i].intensity_score * 0.92)  # Less penalty from 0.9

    def _calculate_intensity_score(
        self,
        rms: float,
        peak: float,
        silence_ratio: float,
        speech_rate: float
    ) -> float:
        """Calculate a refined intensity score from various features"""
        # Normalize volume features - be more lenient with quieter speech
        volume_score = (rms * 1.2 + peak) / 2  # Boost RMS weight
        
        # Speech activity score with more tolerance for pauses
        speech_activity_score = 1.0 - (silence_ratio * 1.1)  # Reduced silence penalty from 1.2
        
        # More tolerant speech rate scoring
        rate_score = 1.0 - abs(speech_rate - self.speech_rate_threshold) / (self.speech_rate_threshold * 1.2)  # Increased tolerance
        rate_score = max(0.0, min(1.0, rate_score))
        
        # Adjusted weights to be more lenient
        weights = {
            'volume': 0.25,        # Reduced from 0.3
            'speech_activity': 0.4, # Kept the same
            'speech_rate': 0.35    # Increased from 0.3
        }
        
        intensity_score = (
            volume_score * weights['volume'] +
            speech_activity_score * weights['speech_activity'] +
            rate_score * weights['speech_rate']
        )
        
        return max(0.0, min(1.0, intensity_score))