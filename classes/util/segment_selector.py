import logging
from typing import List, Tuple, Dict, Any, Optional
import json
import numpy as np
from dataclasses import dataclass
from classes.util.progress_tracker import ProgressTracker, ProcessingStage

@dataclass
class SegmentScore:
    start: float    # Changed from start_time
    end: float      # Changed from end_time
    sentiment_score: float
    audio_score: float
    combined_score: float
    sentiment_type: str
    text: str

class SegmentSelector:
    """
    Helper class for selecting and processing video segments based on 
    sentiment analysis and transcript data.
    """
    def __init__(self, progress_tracker: ProgressTracker = None):
        self.logger = logging.getLogger('youtube_shorts')
        self.progress_tracker = progress_tracker
        self.audio_features = []  # Store audio features for use across methods
        # Define emotions that indicate interesting content - increased weights
        self.interesting_emotions = {
            'fear': 1.3,
            'joy': 1.2,
            'surprise': 1.25,
            'sadness': 1.2,
            'anger': 1.3,
            'disgust': 1.2  # Added disgust as it might indicate interesting reactions
        }
        # Adjusted weights to favor sentiment and audio more
        self.weights = {
            'sentiment': 0.5,  # Increased from 0.4
            'audio': 0.35,     # Increased from 0.3
            'context': 0.15    # Reduced context importance
        }

    def select_interesting_segments(
        self, 
        transcript_path: str, 
        sentiment_path: str,
        audio_features: List[Any]  # AudioFeatures from AudioFeatureAnalyzer
    ) -> List[Tuple[float, float]]:
        """
        Select segments based on sentiment analysis, audio features, and transcript timing.
        Returns a list of (start, end) tuples for interesting segments.
        """
        try:
            self.audio_features = audio_features  # Store for use in other methods
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.SEGMENT_SELECTION,
                    0.1,
                    "Loading sentiment and transcript data"
                )

            # Load data
            with open(sentiment_path, 'r', encoding='utf-8') as f:
                sentiment_data = json.load(f)
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)

            self.logger.debug(
                f"Processing {len(sentiment_data)} sentiment segments "
                f"and {len(audio_features)} audio features"
            )

            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.SEGMENT_SELECTION,
                    0.3,
                    "Calculating segment scores"
                )

            # Calculate scores for each potential segment
            segment_scores = self._calculate_segment_scores(
                sentiment_data, 
                audio_features,
                transcript_data['segments']
            )

            self.logger.debug(f"Found {len(segment_scores)} potential segments before filtering")
            # Log distribution of scores
            if segment_scores:
                scores = [s.combined_score for s in segment_scores]
                self.logger.debug(
                    f"Score stats - Min: {min(scores):.2f}, "
                    f"Max: {max(scores):.2f}, "
                    f"Mean: {sum(scores)/len(scores):.2f}"
                )

            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.SEGMENT_SELECTION,
                    0.7,
                    "Selecting best segments"
                )

            # Select the best non-overlapping segments
            selected_segments = self._select_best_segments(segment_scores)
            
            if selected_segments:
                # Log detailed information about selected segments
                for seg in selected_segments:
                    self.logger.debug(
                        f"Selected segment {seg.start:.2f}-{seg.end:.2f}\n"
                        f"Sentiment: {seg.sentiment_type} (score: {seg.sentiment_score:.2f})\n"
                        f"Audio score: {seg.audio_score:.2f}\n"
                        f"Combined score: {seg.combined_score:.2f}\n"
                        f"Text preview: {seg.text[:100]}..."
                    )
                
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        ProcessingStage.SEGMENT_SELECTION,
                        1.0,
                        f"Selected {len(selected_segments)} interesting segments"
                    )
                
                return [(seg.start, seg.end) for seg in selected_segments]
            else:
                self.logger.warning("No segments met the selection criteria")
                # Log some statistics about why segments might have been filtered
                if segment_scores:
                    above_threshold = len([s for s in segment_scores if s.combined_score >= 0.4])
                    self.logger.debug(
                        f"Out of {len(segment_scores)} segments:\n"
                        f"- {above_threshold} segments above min_score threshold (0.4)\n"
                    )
                
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        ProcessingStage.SEGMENT_SELECTION,
                        1.0,
                        "No suitable segments found"
                    )
                return []
            
        except Exception as e:
            self.logger.error(f"Failed to select segments: {str(e)}", exc_info=True)
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.SEGMENT_SELECTION,
                    1.0,
                    f"Segment selection failed: {str(e)}"
                )
            raise

    def _calculate_segment_scores(
        self,
        sentiment_data: List[Dict],
        audio_features: List[Any],
        transcript_segments: List[Dict]
    ) -> List[SegmentScore]:
        scores = []
        total_segments = len(sentiment_data)
        
        # Create a mapping of time ranges to audio features for faster lookup
        audio_map = {(f.start, f.end): f for f in audio_features}
        
        for i, sentiment in enumerate(sentiment_data):
            # Map the old field names to new ones
            start = sentiment.get('start_time') or sentiment.get('start', 0)
            end = sentiment.get('end_time') or sentiment.get('end', 0)
            
            # Get audio features for this time range
            audio_score = self._get_audio_score(start, end, audio_features)
            
            # Calculate sentiment score with emotion weighting
            base_sentiment_score = sentiment['score']
            emotion_multiplier = self.interesting_emotions.get(
                sentiment['sentiment'].lower(), 1.0
            )
            sentiment_score = base_sentiment_score * emotion_multiplier
            
            # Include context score if available
            context_multiplier = sentiment.get('context_score', 1.0)
            
            # Combine scores using weights
            combined_score = (
                sentiment_score * self.weights['sentiment'] +
                audio_score * self.weights['audio'] +
                context_multiplier * self.weights['context']
            )
            
            self.logger.debug(
                f"Segment {i+1}/{total_segments}: "
                f"Time: {start:.1f}-{end:.1f}, "
                f"Emotion: {sentiment['sentiment']}, "
                f"Score: {combined_score:.2f}"
            )
            
            scores.append(SegmentScore(
                start=start,
                end=end,
                sentiment_score=sentiment_score,
                audio_score=audio_score,
                combined_score=combined_score,
                sentiment_type=sentiment['sentiment'],
                text=sentiment['text']
            ))
        
        return scores

    def _get_audio_score(
        self,
        start: float,
        end: float,
        audio_features: List[Any]
    ) -> float:
        """
        Get the average audio intensity score for a time range
        """
        # Changed to include features that overlap with the segment
        relevant_features = [
            f for f in audio_features
            if (f.start <= end and f.end >= start)  # Features that overlap with segment
        ]
        
        if not relevant_features:
            return 0.5  # Default score if no features found
        
        # Weight features by how much they overlap with the segment
        total_weight = 0
        weighted_sum = 0
        segment_duration = end - start
        
        for feature in relevant_features:
            overlap_start = max(start, feature.start)
            overlap_end = min(end, feature.end)
            overlap_duration = overlap_end - overlap_start
            weight = overlap_duration / segment_duration
            
            weighted_sum += feature.intensity_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _select_best_segments(
        self,
        scored_segments: List[SegmentScore],
        min_score: float = 0.4,
        max_segments: int = 8,
        target_duration: float = 60.0,  # Target duration in seconds
        max_silence_ratio: float = 0.4  # Maximum allowed silence ratio
    ) -> List[SegmentScore]:
        """
        Select the best non-overlapping segments, prioritizing:
        1. Segments close to target duration (60s)
        2. Clean sentence boundaries
        3. High engagement (minimal silences)
        4. High combined scores
        """
        # First, filter out segments with too much silence and too short duration
        qualified_segments = [
            s for s in scored_segments
            if any(
                f.silence_ratio <= max_silence_ratio
                for f in self.audio_features
                if f.start >= s.start and f.end <= s.end
            ) and (s.end - s.start) >= 30  # Minimum 30 seconds before considering
        ]
        
        # Group nearby segments that could potentially be merged
        potential_groups = []
        current_group = []
        for i, segment in enumerate(qualified_segments):
            if not current_group:
                current_group.append(segment)
            else:
                prev_segment = current_group[-1]
                gap = segment.start - prev_segment.end
                
                # More aggressive merging for short segments
                max_gap = 5.0 if (segment.end - segment.start) < 45 else 2.0
                if gap <= max_gap and (segment.end - current_group[0].start) <= 90.0:
                    current_group.append(segment)
                else:
                    if len(current_group) > 1:
                        potential_groups.append(current_group[:])
                    current_group = [segment]
        
        if current_group and len(current_group) > 1:
            potential_groups.append(current_group)
        
        # Score each potential group and individual segment
        scored_candidates = []
        
        # Add individual segments
        for segment in qualified_segments:
            duration = segment.end - segment.start
            if 30 <= duration <= 90:  # Expanded duration range
                # Modified duration scoring to more heavily penalize short durations
                if duration < target_duration:
                    duration_score = (duration / target_duration) ** 1.5  # Exponential penalty for short duration
                else:
                    duration_score = 1.0 - (duration - target_duration) / (90 - target_duration)
                
                boundary_score = self._calculate_boundary_score(segment)
                silence_score = 1.0 - self._calculate_average_silence(segment)
                
                total_score = (
                    segment.combined_score * 0.3 +  # Reduced original score weight
                    duration_score * 0.5 +          # Increased duration importance
                    boundary_score * 0.1 +          # Reduced boundary importance
                    silence_score * 0.1             # Engagement/silence weight
                )
                
                # Additional multiplier for segments close to target duration
                if 50 <= duration <= 70:
                    total_score *= 1.2
                
                scored_candidates.append((total_score, [(segment.start, segment.end)], [segment]))
        
        # Add potential merged groups with higher priority
        for group in potential_groups:
            merged_start = group[0].start
            merged_end = group[-1].end
            duration = merged_end - merged_start
            
            if 45 <= duration <= 90:  # Prefer longer merged segments
                if duration < target_duration:
                    duration_score = (duration / target_duration) ** 1.5
                else:
                    duration_score = 1.0 - (duration - target_duration) / (90 - target_duration)
                
                boundary_score = (
                    self._calculate_boundary_score(group[0]) * 0.6 +
                    self._calculate_boundary_score(group[-1]) * 0.4
                )
                silence_score = 1.0 - self._calculate_average_silence_for_range(merged_start, merged_end)
                
                avg_combined_score = sum(s.combined_score for s in group) / len(group)
                
                total_score = (
                    avg_combined_score * 0.25 +    # Reduced original scores weight
                    duration_score * 0.55 +        # Increased duration importance
                    boundary_score * 0.1 +         # Clean boundaries
                    silence_score * 0.1            # Engagement/silence weight
                )
                
                # Bonus for merged segments close to target duration
                if 50 <= duration <= 70:
                    total_score *= 1.25
                
                scored_candidates.append((
                    total_score,
                    [(s.start, s.end) for s in group],
                    group
                ))
        
        # Sort by total score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select non-overlapping segments/groups
        selected = []
        used_ranges = []
        
        for score, ranges, segments in scored_candidates:
            # Check if any of the ranges overlap with already selected ones
            overlaps = False
            for start, end in ranges:
                for used_start, used_end in used_ranges:
                    if (start <= used_end and end >= used_start):
                        overlaps = True
                        break
                if overlaps:
                    break
            
            if not overlaps:
                selected.extend(segments)
                used_ranges.extend(ranges)
                
                if len(selected) >= max_segments:
                    break
        
        # Sort final selections by start time
        selected.sort(key=lambda x: x.start)
        return selected

    def _calculate_boundary_score(self, segment: SegmentScore) -> float:
        """Calculate how well the segment aligns with sentence boundaries."""
        score = 0.0
        text = segment.text.strip()
        
        # Start of segment
        if text and text[0].isupper() and not text.startswith('...'):
            score += 0.6  # Higher weight for clean start
            
        # End of segment
        if text and text.endswith(('.', '!', '?')) and not text.endswith('...'):
            score += 0.4  # Lower weight for clean end
            
        return score

    def _calculate_average_silence(self, segment: SegmentScore) -> float:
        """Calculate the average silence ratio for a segment."""
        relevant_features = [
            f for f in self.audio_features
            if f.start >= segment.start and f.end <= segment.end
        ]
        
        if not relevant_features:
            return 0.5
        
        return sum(f.silence_ratio for f in relevant_features) / len(relevant_features)

    def _calculate_average_silence_for_range(self, start: float, end: float) -> float:
        """Calculate the average silence ratio for a time range."""
        relevant_features = [
            f for f in self.audio_features
            if f.start >= start and f.end <= end
        ]
        
        if not relevant_features:
            return 0.5
        
        return sum(f.silence_ratio for f in relevant_features) / len(relevant_features)

    def _find_sentence_boundaries(
        self, 
        transcript_segments: List[dict], 
        sentiment_start: float
    ) -> Tuple[float, float]:
        """
        Find the natural sentence boundaries around a given sentiment start time.
        Returns a tuple of (start, end) or None if invalid boundaries.
        Tries to get as close to 60 seconds as possible while ending at a sentence boundary.
        Will expand beyond the sentiment bounds to ensure we capture complete sentences.
        """
        # Find starting point (beginning of sentence)
        start_segment_idx = 0
        for idx, seg in enumerate(transcript_segments):
            if seg['start'] <= sentiment_start <= seg['end']:
                start_segment_idx = idx
                # Travel backwards until we find a clear sentence start
                # Look for up to 5 segments back to find a good starting point
                searches_back = 0
                while start_segment_idx > 0 and searches_back < 5:
                    prev_text = transcript_segments[start_segment_idx - 1]['text'].strip()
                    curr_text = transcript_segments[start_segment_idx]['text'].strip()
                    
                    # Check if previous segment ends with sentence end
                    if prev_text.endswith(('.', '!', '?')):
                        # Add a small buffer by including part of the previous segment
                        start_segment_idx -= 1
                        break
                    # Check if current segment starts with capital letter
                    if not curr_text or curr_text[0].isupper():
                        # Add a small buffer by including part of the previous segment if available
                        if start_segment_idx > 0:
                            # Look at previous segment duration
                            prev_duration = transcript_segments[start_segment_idx]['start'] - transcript_segments[start_segment_idx - 1]['start']
                            # If it's short enough (< 2s), include it for a smoother start
                            if prev_duration <= 2.0:
                                start_segment_idx -= 1
                        break
                    start_segment_idx -= 1
                    searches_back += 1
                break
        
        # Get the actual start time, looking back slightly to avoid cutting off start
        segment_start = transcript_segments[start_segment_idx]['start']
        if start_segment_idx > 0:
            prev_end = transcript_segments[start_segment_idx - 1]['end']
            # If there's a small gap between segments, start a bit earlier
            if segment_start - prev_end <= 0.5:  # If gap is less than 0.5 seconds
                segment_start = prev_end
        
        # Find end point (try to get as close to 60 seconds as possible)
        target_end_time = segment_start + 60.0
        end_segment_idx = start_segment_idx
        
        # First, find a segment that goes beyond our target time
        while end_segment_idx < len(transcript_segments) - 1:
            if transcript_segments[end_segment_idx + 1]['start'] >= target_end_time:
                break
            end_segment_idx += 1

        # Now work backwards until we find a complete sentence end
        original_end_idx = end_segment_idx
        while end_segment_idx > start_segment_idx:
            curr_text = transcript_segments[end_segment_idx]['text'].strip()
            next_text = transcript_segments[end_segment_idx + 1]['text'].strip() if end_segment_idx < len(transcript_segments) - 1 else ""
            
            # Check if this is a good ending point
            if curr_text.endswith(('.', '!', '?')):
                # Also check if next segment starts a new sentence
                if not next_text or next_text[0].isupper():
                    break
            end_segment_idx -= 1
        
        # If we had to backtrack too far, try going forward instead
        if transcript_segments[end_segment_idx]['end'] < target_end_time - 20:
            end_segment_idx = original_end_idx
            # Try going forward to find a good ending point
            while end_segment_idx < len(transcript_segments) - 1:
                curr_text = transcript_segments[end_segment_idx]['text'].strip()
                next_text = transcript_segments[end_segment_idx + 1]['text'].strip()
                
                # Calculate duration if we were to include the next segment
                next_end = transcript_segments[end_segment_idx + 1]['end']
                potential_duration = next_end - segment_start
                
                # Check if current segment is a good ending point
                good_end_point = (curr_text.endswith(('.', '!', '?')) and 
                               (not next_text or next_text[0].isupper()))
                
                if good_end_point:
                    # If we have a good end point and decent length, stop here
                    if transcript_segments[end_segment_idx]['end'] - segment_start >= 45:
                        break
                
                # If adding next segment exceeds 65 seconds, stop at current if it's a decent endpoint
                if potential_duration > 65 and good_end_point:
                    break
                
                # Otherwise, continue if we haven't exceeded max duration
                if potential_duration <= 67:  # Allow slightly longer for proper sentence completion
                    end_segment_idx += 1
                else:
                    break
        
        segment_end = transcript_segments[end_segment_idx]['end']
        
        # Always include a small buffer at the end to avoid cutting off words
        if end_segment_idx < len(transcript_segments) - 1:
            next_start = transcript_segments[end_segment_idx + 1]['start']
            next_end = transcript_segments[end_segment_idx + 1]['end']
            # Look ahead up to 2 seconds to include sentence completion
            if next_start - segment_end <= 0.5:  # If there's a small gap
                # Check if including the next segment would make it too long
                if next_end - segment_start <= 67:  # Still within our max duration
                    segment_end = next_end
        
        # Only return if segment is of acceptable length
        duration = segment_end - segment_start
        if 30 <= duration <= 90:  # More flexible duration range 
            return (segment_start, segment_end)
        return None

    def _merge_overlapping_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]: 
        """
        Merge any overlapping segments into single continuous segments.
        """
        if not segments:
            self.logger.info("No segments to merge")
            return []
            
        # Sort segments by start time
        segments.sort()
        merged_segments = []
        
        # Initialize with the first segment
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if start <= current_end:
                # Segments overlap, merge them
                current_end = max(current_end, end)
            else:
                # No overlap, add current segment and start new one
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add the last segment
        merged_segments.append((current_start, current_end))
        
        self.logger.debug(f"Merged {len(segments)} segments into {len(merged_segments)} segments")
        return merged_segments