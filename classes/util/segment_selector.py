import logging
from typing import List, Tuple, Dict, Any, Optional
import json
import numpy as np
from dataclasses import dataclass
from classes.util.progress_tracker import ProgressTracker, ProcessingStage

@dataclass
class SegmentScore:
    start: float
    end: float
    sentiment_score: float
    audio_score: float
    combined_score: float
    sentiment_type: str
    text: str
    speaker: Optional[str] = None  # Add speaker information
    conversation_id: Optional[str] = None  # Track related segments
    speaker_count: Optional[int] = None  # Number of speakers in conversation

class SegmentSelector:
    """
    Helper class for selecting and processing video segments based on 
    sentiment analysis and transcript data.
    """
    def __init__(self, progress_tracker: ProgressTracker = None):
        self.logger = logging.getLogger('youtube_shorts')
        self.progress_tracker = progress_tracker
        self.audio_features = []
        
        # Refined emotion weights with more lenient neutral handling
        self.interesting_emotions = {
            'fear': 1.2,
            'joy': 1.15,
            'surprise': 1.2,
            'sadness': 1.1,
            'anger': 1.2,
            'disgust': 1.1,
            'neutral': 0.85  # Increased to allow more neutral segments if they're engaging
        }
        
        # Adjusted weights to favor longer conversations
        self.weights = {
            'sentiment': 0.25,    
            'audio': 0.3,        
            'context': 0.2,      # Slightly reduced to give more weight to duration
            'conversation': 0.25  # Increased to favor good conversations
        }
        
        # Duration parameters optimized for longer segments
        self.min_duration = 30.0  # Maintain minimum
        self.max_duration = 90.0  # Keep maximum
        self.ideal_duration = 60.0  # Target duration
        self.duration_tolerance = 15.0  # How far from ideal before heavy penalties
        self.duration_flexibility = 0.2  # Stricter adherence to duration preferences

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

    def _calculate_duration_score(self, duration: float) -> float:
        """Calculate how well a segment matches our target duration."""
        # Stronger preference for segments closer to ideal duration
        if duration < self.min_duration or duration > self.max_duration:
            return 0.0
        
        diff = abs(duration - self.ideal_duration)
        if diff <= 15.0:  # Strong preference for 45-75 second range
            return 1.0 - (diff / 15.0) * 0.5
        return max(0.5, 1.0 - (diff / (self.max_duration - self.ideal_duration)))

    def _calculate_conversation_score(self, segments: List[SegmentScore]) -> float:
        """Calculate how engaging a conversation is based on speaker patterns."""
        if not segments:
            return 0.0

        # Get unique speakers
        speakers = set(s.speaker for s in segments if s.speaker != 'UNKNOWN')
        speaker_count = len(speakers)
        
        # Calculate speaker turns with enhanced pattern detection
        turns = 0
        prev_speaker = None
        consecutive_same_speaker = 0
        
        for seg in segments:
            if seg.speaker != prev_speaker and seg.speaker != 'UNKNOWN':
                turns += 1
                consecutive_same_speaker = 0
            else:
                consecutive_same_speaker += 1
            prev_speaker = seg.speaker

        # Calculate turn density with penalties for long monologues
        duration = segments[-1].end - segments[0].start
        turn_density = (turns * 60) / duration if duration > 0 else 0
        
        # Penalize segments with long monologues
        monologue_penalty = 1.0 - (min(consecutive_same_speaker, 5) * 0.1)
        
        # Ideal turn density is about 4-8 turns per minute for engaging conversation
        turn_score = min(1.0, turn_density / 6.0) * monologue_penalty
        
        # Prefer 2-3 speakers with stronger weight on speaker interaction
        speaker_score = min(1.0, speaker_count / 3.0)
        
        # Weight turn patterns more heavily than speaker count
        return (turn_score * 0.7 + speaker_score * 0.3)

    def _calculate_segment_scores(
        self,
        sentiment_data: List[Dict],
        audio_features: List[Any],
        transcript_segments: List[Dict]
    ) -> List[SegmentScore]:
        # First, group by pre-detected conversations
        conversation_groups = {}
        for segment in sentiment_data:
            conv_id = segment.get('conversation_id')
            if conv_id:
                if conv_id not in conversation_groups:
                    conversation_groups[conv_id] = []
                conversation_groups[conv_id].append(segment)
        
        scores = []
        for conv_id, segments in conversation_groups.items():
            # Skip empty groups
            if not segments:
                continue
                
            # Sort segments by start time
            segments.sort(key=lambda x: x['start'])
            start = segments[0]['start']
            end = segments[-1]['end']
            duration = end - start
            
            # Skip conversations outside duration bounds
            if duration < self.min_duration or duration > self.max_duration:
                continue
                
            # Calculate conversation quality metrics
            speakers = set(s.get('speaker', 'UNKNOWN') for s in segments)
            speaker_turns = sum(1 for i in range(1, len(segments))
                              if segments[i].get('speaker') != segments[i-1].get('speaker'))
            
            # Conversation engagement score
            conversation_score = (
                min(len(speakers) / 3.0, 1.0) * 0.5 +  # Speaker variety
                min(speaker_turns / 5.0, 1.0) * 0.5     # Interaction density
            )
            
            # Audio engagement score
            audio_score = self._get_audio_score(start, end, audio_features)
            
            # Calculate average sentiment and emotion intensity
            sentiment_scores = []
            for segment in segments:
                if segment['sentiment'].lower() == 'neutral' and len(segments) > 1:
                    continue
                    
                emotion_weight = self.interesting_emotions.get(
                    segment['sentiment'].lower(), 1.0
                )
                weighted_score = segment['score'] * emotion_weight
                sentiment_scores.append(weighted_score)
            
            if not sentiment_scores:  # Skip if all segments were neutral
                continue
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Calculate final score with emphasis on conversation quality
            combined_score = (
                conversation_score * self.weights['conversation'] +
                audio_score * self.weights['audio'] +
                avg_sentiment * self.weights['sentiment']
            )
            
            # Add duration preference
            duration_preference = 1.0 - abs(60.0 - duration) / 30.0  # Prefer ~60s segments
            combined_score *= max(0.8, duration_preference)  # Apply duration preference with minimum impact
            
            scores.append(SegmentScore(
                start=start,
                end=end,
                sentiment_score=avg_sentiment,
                audio_score=audio_score,
                combined_score=combined_score,
                sentiment_type=max(segments, key=lambda x: x['score'])['sentiment'],
                text=" ".join(s['text'] for s in segments),
                speaker="|".join(speakers),
                conversation_id=conv_id,
                speaker_count=len(speakers)
            ))
        
        return scores

    def _group_by_time_proximity(self, segments: List[Dict]) -> List[List[Dict]]:
        """Group segments that are close in time."""
        if not segments:
            return []
            
        segments = sorted(segments, key=lambda x: x['start'])
        groups = []
        current_group = []
        
        for segment in segments:
            # If this is the first segment or it's close to the previous segment
            if not current_group:
                current_group = [segment]
            else:
                prev_end = current_group[-1]['end']
                current_start = segment['start']
                
                # If there's a significant gap, start a new group
                if current_start - prev_end > 2.0:  # 2 second gap threshold
                    # Only save groups that could make valid segments
                    group_duration = current_group[-1]['end'] - current_group[0]['start']
                    if self.min_duration <= group_duration <= self.max_duration:
                        groups.append(current_group)
                    current_group = [segment]
                else:
                    current_group.append(segment)
        
        # Add the last group if it meets duration requirements
        if current_group:
            group_duration = current_group[-1]['end'] - current_group[0]['start']
            if self.min_duration <= group_duration <= self.max_duration:
                groups.append(current_group)
        
        # Log grouping results
        self.logger.debug(
            f"Grouped {len(segments)} segments into {len(groups)} groups "
            f"(min duration: {self.min_duration}s, max duration: {self.max_duration}s)"
        )
        
        return groups

    def _calculate_content_variety(self, segments: List[Dict]) -> float:
        """Calculate how varied and interesting the content is."""
        # Get unique words to measure topic variety
        words = set()
        emotions = set()
        
        for segment in segments:
            words.update(word.lower() for word in segment['text'].split())
            emotions.add(segment['sentiment'].lower())
        
        # Score based on vocabulary richness and emotion variety
        vocab_score = min(1.0, len(words) / 100.0)  # Cap at 100 unique words
        emotion_score = len(emotions) / 6.0  # Normalize by total possible emotions
        
        return (vocab_score + emotion_score) / 2

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

    def has_significant_overlap(self, start: float, end: float, used_ranges: List[Tuple[float, float]]) -> bool:
        """
        Enhanced overlap detection focusing on natural conversation patterns and speaker transitions.
        """
        for used_start, used_end in used_ranges:
            overlap_start = max(start, used_start)
            overlap_end = min(end, used_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                total_duration = end - start
                
                # Analyze speech patterns in the overlap
                overlap_features = [
                    f for f in self.audio_features
                    if f.start <= overlap_end and f.end >= overlap_start
                ]
                
                if not overlap_features:
                    continue
                
                # Track speaker transitions and speech continuity
                speakers = set()
                continuous_speech = 0
                prev_end = None
                
                for feat in sorted(overlap_features, key=lambda x: x.start):
                    if hasattr(feat, 'speaker_id'):
                        speakers.add(feat.speaker_id)
                    
                    # Check speech continuity
                    if prev_end is not None:
                        gap = feat.start - prev_end
                        if gap < 0.5:  # Small gaps indicate continuous conversation
                            continuous_speech += 1
                    prev_end = feat.end
                
                # Decision logic for overlap significance
                is_significant = (
                    # Long overlap with continuous speech
                    (overlap_duration > 4.0 and continuous_speech >= 2) or
                    # Multiple speakers actively conversing
                    (len(speakers) >= 2 and overlap_duration > 3.0) or
                    # Substantial portion of the segment overlaps
                    (overlap_duration / total_duration > 0.25)
                )
                
                if is_significant:
                    return True
                    
        return False

    def _select_best_segments(
        self,
        scored_segments: List[SegmentScore],
        min_score: float = 0.25,
        max_segments: int = None,
        max_silence_ratio: float = 0.65
    ) -> List[SegmentScore]:
        if not scored_segments:
            return []
            
        video_duration = max(seg.end for seg in scored_segments)
        base_segments = 10
        max_segments = max(5, min(20, int((video_duration / 800) * base_segments)))
        
        # Enhanced filtering with conversation pattern analysis
        filtered_segments = []
        conversation_ranges = []
        
        for segment in scored_segments:
            # Calculate conversation metrics with enhanced gap detection
            relevant_features = [
                f for f in self.audio_features 
                if f.start >= segment.start and f.end <= segment.end
            ]
            
            if not relevant_features:
                continue
                
            # Analyze conversation continuity
            duration = segment.end - segment.start
            avg_conversation_likelihood = sum(
                f.conversation_likelihood for f in relevant_features
            ) / len(relevant_features)
            
            # Boost scores for longer, coherent conversations
            duration_factor = min(1.0, (duration - 30) / 30)  # Scales up to 60s
            if duration >= 45.0 and avg_conversation_likelihood > 0.4:
                segment.combined_score *= (1.0 + duration_factor * 0.3)
            
            # Enhanced score threshold considering duration
            min_score_adjusted = min_score * (0.9 if duration >= 50.0 else 1.0)
            if (segment.combined_score < min_score_adjusted and 
                avg_conversation_likelihood < 0.35):
                continue
                
            # Check for natural conversation flow with previous segments
            has_overlap = False
            for conv_start, conv_end in conversation_ranges:
                if self.has_significant_overlap(segment.start, segment.end, [(conv_start, conv_end)]):
                    has_overlap = True
                    break
            
            if has_overlap:
                continue
                
            # Analyze speech patterns
            silence_info = self._analyze_silence_patterns(segment.start, segment.end)
            if silence_info['long_silence_ratio'] > max_silence_ratio and silence_info['natural_pause_ratio'] < 0.2:
                continue
                
            filtered_segments.append(segment)
            # Add to conversation ranges with adaptive spacing
            buffer = 1.0 if duration < 45.0 else 1.5  # Larger buffer for longer segments
            conversation_ranges.append((segment.start - buffer, segment.end + buffer))
        
        # Sort by score and duration preference
        filtered_segments.sort(
            key=lambda x: (x.combined_score * (1.0 + min(0.3, (x.end - x.start - 30) / 100))),
            reverse=True
        )
        
        # Final selection with improved spacing
        selected = []
        used_ranges = []
        
        for segment in filtered_segments:
            if any(
                abs(segment.start - end) < 2.0 or abs(segment.end - start) < 2.0
                for start, end in used_ranges
            ):
                continue
            
            # Variety check with duration consideration
            if len(selected) >= 3:
                similar_segments = sum(
                    1 for s in selected[-3:]
                    if abs(s.sentiment_score - segment.sentiment_score) < 0.3 and
                    s.sentiment_type == segment.sentiment_type
                )
                if similar_segments >= 3:
                    continue
            
            selected.append(segment)
            used_ranges.append((segment.start, segment.end))
            
            if len(selected) >= max_segments:
                break
        
        selected.sort(key=lambda x: x.start)
        return selected

    def _analyze_silence_patterns(self, start: float, end: float) -> Dict[str, float]:
        """Analyze different types of silence in a segment"""
        relevant_features = [
            f for f in self.audio_features
            if f.start >= start and f.end <= end
        ]
        
        if not relevant_features:
            return {
                'total_silence_ratio': 0.5,
                'long_silence_ratio': 0.5,
                'natural_pause_ratio': 0.0
            }
        
        # Count different types of silence
        long_silences = 0
        natural_pauses = 0
        total_segments = len(relevant_features)
        
        consecutive_silence = 0
        for feat in relevant_features:
            if feat.silence_ratio > 0.8:
                consecutive_silence += 1
                if consecutive_silence >= 3:  # 1.5 seconds of silence
                    long_silences += 1
            else:
                if 1 <= consecutive_silence <= 2:  # 0.5-1.0 second pause
                    natural_pauses += 1
                consecutive_silence = 0
        
        return {
            'total_silence_ratio': sum(f.silence_ratio for f in relevant_features) / total_segments,
            'long_silence_ratio': long_silences / total_segments,
            'natural_pause_ratio': natural_pauses / total_segments
        }

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
        Find conversation boundaries with more natural speech pattern detection.
        Allows conversations to start mid-sentence for more natural segments.
        """
        # Find starting segment
        start_segment_idx = 0
        for idx, seg in enumerate(transcript_segments):
            if seg['start'] <= sentiment_start <= seg['end']:
                start_segment_idx = idx
                break
        
        # Instead of looking back for sentence boundaries, analyze conversation flow
        searches_back = 0
        while start_segment_idx > 0 and searches_back < 5:  # Keep 5 segment look-back limit
            prev_segment = transcript_segments[start_segment_idx - 1]
            curr_segment = transcript_segments[start_segment_idx]
            
            # Check for natural conversation breaks (pauses, speaker changes, topic shifts)
            time_gap = curr_segment['start'] - prev_segment['end']
            speaker_change = curr_segment.get('speaker') != prev_segment.get('speaker')
            
            # Allow starting at any word if there's a natural break in conversation
            if time_gap > 0.75 or speaker_change:  # Reduced from 1.0s gap
                break
            
            # If there's strong conversation continuity, include previous segment
            if time_gap < 0.3:  # Quick speech indicates conversation flow
                start_segment_idx -= 1
            
            searches_back += 1
        
        segment_start = transcript_segments[start_segment_idx]['start']
        
        # Find end segment - target ~60s with natural conversation end
        target_end_time = segment_start + 60.0
        end_segment_idx = start_segment_idx
        prev_speaker = transcript_segments[end_segment_idx].get('speaker')
        
        while end_segment_idx < len(transcript_segments) - 1:
            curr_duration = transcript_segments[end_segment_idx]['end'] - segment_start
            next_duration = transcript_segments[end_segment_idx + 1]['end'] - segment_start
            curr_segment = transcript_segments[end_segment_idx]
            next_segment = transcript_segments[end_segment_idx + 1]
            
            # Stop if next segment would make it too long
            if next_duration > 90.0:
                break
            
            # Look for natural conversation endpoints
            time_gap = next_segment['start'] - curr_segment['end']
            speaker_change = next_segment.get('speaker') != curr_segment.get('speaker')
            
            # Consider ending if we're in target duration range and hit a natural break
            if curr_duration >= 45.0:
                if time_gap > 0.75 or speaker_change:  # Natural conversation break
                    # Strong preference for ending near target duration
                    if abs(curr_duration - 60.0) <= 20.0:
                        break
                    # Or if adding next segment would make it too far from target
                    if abs(next_duration - 60.0) > 25.0:
                        break
            
            end_segment_idx += 1
            prev_speaker = curr_segment.get('speaker')
        
        segment_end = transcript_segments[end_segment_idx]['end']
        
        # Small adjustment for word completion
        if end_segment_idx < len(transcript_segments) - 1:
            next_segment = transcript_segments[end_segment_idx + 1]
            
            # Include next segment if it's very close and maintains conversation flow
            if (next_segment['start'] - segment_end <= 0.3 and 
                next_segment['end'] - segment_start <= 90.0 and
                next_segment.get('speaker') == prev_speaker):
                segment_end = next_segment['end']
        
        # Verify duration
        duration = segment_end - segment_start
        if 30.0 <= duration <= 90.0:
            return (segment_start, segment_end)
        
        return None

    def _merge_overlapping_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]: 
        """
        Merge any overlapping segments into single continuous segments.
        Enforces strict duration limits and spacing requirements.
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
            current_duration = current_end - current_start
            new_duration = end - current_start
            gap = start - current_end
            
            # If there's a small gap (< 3s), check if we should merge
            if gap < 3.0 and new_duration <= self.max_duration:
                # Only merge if the gap contains speech (check audio features)
                gap_intensity = self._get_audio_score(current_end, start, self.audio_features)
                if gap_intensity > 0.4:  # There's significant speech in the gap
                    current_end = end
                    continue
            
            # Save current segment and start new one
            if self.min_duration <= current_duration <= self.max_duration:
                merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
        
        # Handle the final segment
        final_duration = current_end - current_start
        if self.min_duration <= final_duration <= self.max_duration:
            merged_segments.append((current_start, current_end))
        
        # Ensure minimum spacing between segments
        final_segments = []
        for i, (start, end) in enumerate(merged_segments):
            if i == 0 or start - final_segments[-1][1] >= 3.0:  # Minimum 3s gap
                final_segments.append((start, end))
            else:
                self.logger.debug(
                    f"Skipping segment {start:.1f}-{end:.1f} due to insufficient "
                    f"spacing from previous segment (gap: {start - final_segments[-1][1]:.1f}s)"
                )
        
        self.logger.debug(
            f"Merged {len(segments)} segments into {len(final_segments)} non-overlapping segments"
        )
        return final_segments