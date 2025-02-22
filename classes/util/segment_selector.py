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
        
        # More relaxed scoring weights
        self.interesting_emotions = {
            'fear': 1.2,
            'joy': 1.1,
            'surprise': 1.2,
            'sadness': 1.1,
            'anger': 1.2,
            'disgust': 1.1,
            'neutral': 0.9  # Increased from 0.8
        }
        
        # More balanced weights to allow for variety
        self.weights = {
            'sentiment': 0.25,
            'audio': 0.35,    # Increased audio weight
            'context': 0.2,
            'conversation': 0.2
        }
        
        # More flexible duration parameters
        self.min_duration = 40.0  # Reduced from 50.0
        self.max_duration = 90.0  # Increased from 67.0
        self.target_duration = 60.0
        self.duration_tolerance = 20.0  # More flexible duration tolerance
        
        # Overlap prevention
        self.max_overlap_duration = 0.0  # No overlap allowed
        self.min_segment_gap = 1.0  # Minimum gap between segments

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
        diff = abs(duration - self.target_duration)
        if diff <= self.duration_tolerance:
            return 1.0 - (diff / self.duration_tolerance)
        return 0.0

    def _calculate_conversation_score(self, segments: List[SegmentScore]) -> float:
        """Calculate how engaging a conversation is based on speaker patterns."""
        if not segments:
            return 0.0

        # Get unique speakers
        speakers = set(s.speaker for s in segments if s.speaker != 'UNKNOWN')
        speaker_count = len(speakers)
        
        # Calculate speaker turns
        turns = 0
        prev_speaker = None
        for seg in segments:
            if seg.speaker != prev_speaker and seg.speaker != 'UNKNOWN':
                turns += 1
            prev_speaker = seg.speaker

        # Calculate turn density (turns per minute)
        duration = segments[-1].end - segments[0].start
        turn_density = (turns * 60) / duration if duration > 0 else 0

        # Ideal turn density is about 4-8 turns per minute for engaging conversation
        turn_score = min(1.0, turn_density / 6.0)
        
        # Prefer 2-3 speakers
        speaker_score = min(1.0, speaker_count / 3.0)

        return (turn_score * 0.6 + speaker_score * 0.4)

    def _calculate_segment_scores(
        self,
        sentiment_data: List[Dict],
        audio_features: List[Any],
        transcript_segments: List[Dict]
    ) -> List[SegmentScore]:
        scores = []
        
        # First pass: identify major conversation clusters in the timeline
        conversation_clusters = []
        current_cluster = []
        last_end = 0
        
        # Sort by start time to find temporal clusters
        sorted_segments = sorted(sentiment_data, key=lambda x: x['start'])
        
        for segment in sorted_segments:
            if not current_cluster:
                current_cluster = [segment]
            else:
                gap = segment['start'] - last_end
                # If gap is more than 30 seconds, consider it a new cluster
                if gap > 30.0:
                    conversation_clusters.append(current_cluster)
                    current_cluster = [segment]
                else:
                    current_cluster.append(segment)
            last_end = segment['end']
        
        if current_cluster:
            conversation_clusters.append(current_cluster)
            
        # Calculate cluster importance scores
        cluster_scores = []
        for cluster in conversation_clusters:
            cluster_duration = cluster[-1]['end'] - cluster[0]['start']
            unique_speakers = len(set(s.get('speaker', 'UNKNOWN') for s in cluster))
            speaker_turns = sum(1 for i in range(1, len(cluster))
                              if cluster[i].get('speaker') != cluster[i-1].get('speaker'))
            
            # Score the cluster based on its properties
            cluster_score = (
                (unique_speakers / 3) * 0.4 +  # Weight for speaker variety
                (min(speaker_turns / 10, 1.0) * 0.4) +  # Weight for interaction
                (min(cluster_duration / 180, 1.0) * 0.2)  # Weight for sustained conversation
            )
            
            cluster_scores.append((cluster_score, cluster))
        
        # Sort clusters by score
        cluster_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Process segments with preference for better clusters
        for cluster_score, cluster in cluster_scores:
            # Use existing conversation_id from sentiment data
            conversation_groups = {}
            for segment in cluster:
                conv_id = segment.get('conversation_id')
                if not conv_id:
                    conv_id = f"single_{segment['start']}"
                
                if conv_id not in conversation_groups:
                    conversation_groups[conv_id] = []
                conversation_groups[conv_id].append(segment)
            
            # Process each conversation group within the cluster
            for conv_id, group_segments in conversation_groups.items():
                # Sort segments by start time within the group
                group_segments.sort(key=lambda x: x['start'])
                
                # Calculate group duration
                duration = group_segments[-1]['end'] - group_segments[0]['start']
                
                # Skip if duration is outside our bounds
                if not (self.min_duration <= duration <= self.max_duration):
                    continue

                # Get unique speakers in group
                speakers = set(s.get('speaker', 'UNKNOWN') for s in group_segments)
                speaker_count = len(speakers)
                
                # Calculate content variety score
                variety_score = self._calculate_content_variety(group_segments)
                
                # Calculate group-level conversation quality
                conversation_quality = self._calculate_conversation_score([
                    SegmentScore(
                        start=s['start'],
                        end=s['end'],
                        sentiment_score=0,
                        audio_score=0,
                        combined_score=0,
                        sentiment_type=s['sentiment'],
                        text=s['text'],
                        speaker=s.get('speaker', 'UNKNOWN'),
                        conversation_id=conv_id,
                        speaker_count=speaker_count
                    ) for s in group_segments
                ])
                
                # Apply cluster quality boost
                cluster_boost = cluster_score * 0.2  # Up to 20% boost for being in a good cluster
                
                # Process each segment in the conversation
                for segment in group_segments:
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    
                    # Basic feature scores
                    audio_score = self._get_audio_score(start, end, audio_features)
                    base_sentiment_score = segment['score']
                    
                    # Apply emotion weights
                    emotion_multiplier = self.interesting_emotions.get(
                        segment['sentiment'].lower(), 1.0
                    )
                    sentiment_score = base_sentiment_score * emotion_multiplier
                    
                    # Duration score for the entire conversation
                    duration_score = self._calculate_duration_score(duration)
                    
                    # Combined score with conversation quality and cluster boost
                    combined_score = (
                        sentiment_score * self.weights['sentiment'] +
                        audio_score * self.weights['audio'] +
                        variety_score * self.weights['context'] +
                        conversation_quality * self.weights['conversation']
                    ) * duration_score * (1 + cluster_boost)

                    scores.append(SegmentScore(
                        start=start,
                        end=end,
                        sentiment_score=sentiment_score,
                        audio_score=audio_score,
                        combined_score=combined_score,
                        sentiment_type=segment['sentiment'],
                        text=segment['text'],
                        speaker=segment.get('speaker', 'UNKNOWN'),
                        conversation_id=conv_id,
                        speaker_count=speaker_count
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
        """Check if a segment has significant overlap with existing segments."""
        segment_duration = end - start
        
        for used_start, used_end in used_ranges:
            # Calculate overlap
            overlap_start = max(start, used_start)
            overlap_end = min(end, used_end)
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                used_duration = used_end - used_start
                overlap_ratio = overlap_duration / min(segment_duration, used_duration)
                
                if overlap_duration > self.max_overlap_duration:
                    self.logger.debug(
                        f"Segment {start:.1f}-{end:.1f} rejected: "
                        f"overlap duration {overlap_duration:.1f}s exceeds max {self.max_overlap_duration}s"
                    )
                    return True
                
                if overlap_ratio > 0.25:
                    self.logger.debug(
                        f"Segment {start:.1f}-{end:.1f} rejected: "
                        f"overlap ratio {overlap_ratio:.1%} exceeds 25%"
                    )
                    return True
        return False

    def _select_best_segments(
        self,
        scored_segments: List[SegmentScore],
        min_score: float = 0.3,
        max_segments: int = 8,
        target_duration: float = 60.0,
        max_silence_ratio: float = 0.4
    ) -> List[SegmentScore]:
        if not scored_segments:
            return []

        video_duration = max(seg.end for seg in scored_segments)
        
        # Calculate group scores first
        group_scores = []
        groups_by_id = {}
        
        # Group segments by conversation_id
        for segment in scored_segments:
            if segment.conversation_id not in groups_by_id:
                groups_by_id[segment.conversation_id] = []
            groups_by_id[segment.conversation_id].append(segment)
        
        # Calculate scores for each group
        for group_id, segments in groups_by_id.items():
            segments.sort(key=lambda x: x.start)
            start = segments[0].start
            end = segments[-1].end
            duration = end - start
            
            # Strictly enforce duration limits
            if duration < self.min_duration or duration > self.max_duration:
                self.logger.debug(
                    f"Skipping segment {start:.1f}-{end:.1f} due to duration {duration:.1f}s "
                    f"(min: {self.min_duration}s, max: {self.max_duration}s)"
                )
                continue
                
            # Calculate group level metrics
            avg_score = sum(s.combined_score for s in segments) / len(segments)
            speaker_variety = len(set(s.speaker for s in segments if s.speaker))
            duration_score = self._calculate_duration_score(duration)
            
            # Combined group score
            group_score = (
                avg_score * 0.4 +
                (speaker_variety / 3.0) * 0.3 +  # Normalize by expecting up to 3 speakers
                duration_score * 0.3
            )
            
            if group_score >= min_score:
                group_scores.append((group_score, group_id, segments, start, end))
        
        # Sort groups by score
        group_scores.sort(key=lambda x: x[0], reverse=True)

        # Find natural breaks in conversation for zone boundaries
        def find_natural_zone_boundaries() -> List[float]:
            """Find natural breaking points in the video based on conversation flow."""
            all_boundaries = set()
            
            # Add start and end of video
            all_boundaries.add(0.0)
            all_boundaries.add(video_duration)
            
            # Add points where conversations naturally break
            current_conv = None
            for segment in sorted(scored_segments, key=lambda x: x.start):
                if segment.conversation_id != current_conv:
                    all_boundaries.add(segment.start)
                    if current_conv is not None:
                        # Add end of previous conversation
                        all_boundaries.add(prev_end)
                    current_conv = segment.conversation_id
                prev_end = segment.end
            
            # Ensure we have enough boundaries
            boundaries = sorted(list(all_boundaries))
            if len(boundaries) < max_segments * 2:
                # Add evenly spaced points if needed
                target_points = max_segments * 2
                current_points = len(boundaries)
                points_to_add = target_points - current_points
                
                if points_to_add > 0:
                    for i in range(1, points_to_add + 1):
                        point = (i * video_duration) / (points_to_add + 1)
                        boundaries.append(point)
                    boundaries.sort()
            
            return boundaries

        # Create zones based on natural boundaries
        zone_boundaries = find_natural_zone_boundaries()
        num_zones = len(zone_boundaries) - 1
        zones_used = [False] * num_zones
        
        def get_segment_zones(start: float, end: float) -> List[int]:
            """Get which zones a segment spans"""
            zones = []
            for i in range(num_zones):
                zone_start = zone_boundaries[i]
                zone_end = zone_boundaries[i + 1]
                # Check if segment overlaps with this zone
                if start < zone_end and end > zone_start:
                    zones.append(i)
            return zones

        def check_zone_availability(start: float, end: float, min_zones_between: int = 2) -> bool:
            """Check if the zones for this segment are available and maintain spacing"""
            segment_zones = get_segment_zones(start, end)
            
            # Check if any of these zones are already used
            if any(zones_used[z] for z in segment_zones):
                return False
            
            # Find closest used zones and enforce stricter spacing
            used_zone_indices = [i for i, used in enumerate(zones_used) if used]
            if used_zone_indices:
                for zone in segment_zones:
                    # Find distance to closest used zone
                    distances = [abs(zone - used_zone) for used_zone in used_zone_indices]
                    if min(distances) <= min_zones_between:
                        return False
            
            return True

        def check_for_overlap(start: float, end: float, used_ranges: List[Tuple[float, float]], buffer: float = 1.0) -> bool:
            """Check if a segment overlaps with any existing segments, including a buffer zone"""
            for used_start, used_end in used_ranges:
                # Add buffer to both ends of segments when checking
                if (start - buffer) < used_end and (end + buffer) > used_start:
                    return True
            return False
        
        def mark_zones_used(start: float, end: float):
            """Mark the zones this segment uses as taken"""
            for zone in get_segment_zones(start, end):
                zones_used[zone] = True

        # Select segments ensuring no overlap and proper spacing
        selected = []
        used_ranges = []
        
        for _, group_id, segments, start, end in group_scores:
            duration = end - start
            
            # Double-check duration constraints
            if duration < self.min_duration or duration > self.max_duration:
                continue
            
            # Check for any overlap with existing selections, including buffer zones
            if check_for_overlap(start, end, used_ranges):
                self.logger.debug(
                    f"Skipping segment {start:.1f}-{end:.1f} due to overlap with existing segments"
                )
                continue
            
            # Check zone availability and spacing
            if not check_zone_availability(start, end):
                self.logger.debug(
                    f"Skipping segment {start:.1f}-{end:.1f} due to zone conflict"
                )
                continue
            
            selected.extend(segments)
            used_ranges.append((start, end))
            mark_zones_used(start, end)
            
            zone_indices = get_segment_zones(start, end)
            self.logger.debug(
                f"Selected segment {start:.1f}-{end:.1f} "
                f"(duration: {duration:.1f}s, "
                f"zones: {zone_indices}, "
                f"speakers: {len(set(s.speaker for s in segments))})"
            )
            
            if len(selected) >= max_segments:
                break

        if selected:
            # Sort segments by start time
            selected.sort(key=lambda x: x.start)
            
            # Validate final selection for overlaps and durations
            final_selected = []
            final_ranges = []
            
            for segment in selected:
                duration = segment.end - segment.start
                if self.min_duration <= duration <= self.max_duration:
                    if not check_for_overlap(segment.start, segment.end, final_ranges):
                        final_selected.append(segment)
                        final_ranges.append((segment.start, segment.end))
            
            # Log final distribution info
            used_zone_count = sum(1 for z in zones_used if z)
            total_duration = sum(end - start for start, end in final_ranges)
            
            self.logger.info(
                f"Selected {len(final_selected)} segments across {used_zone_count}/{num_zones} zones:\n"
                f"Total content duration: {total_duration:.1f}s\n" +
                "\n".join(
                    f"Segment at {seg.start:.1f}-{seg.end:.1f} "
                    f"(duration: {seg.end-seg.start:.1f}s, "
                    f"zones: {get_segment_zones(seg.start, seg.end)})"
                    for seg in final_selected
                )
            )
            
            return final_selected
        
        return []

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
            
            # If there's a significant gap, or merging would exceed duration limits,
            # save current segment and start new one
            if gap >= self.min_segment_gap or new_duration > self.max_duration:
                if self.min_duration <= current_duration <= self.max_duration:
                    merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
            else:
                # Extend current segment if it won't exceed max duration
                if new_duration <= self.max_duration:
                    current_end = max(current_end, end)
                else:
                    # If current segment meets duration requirements, save it
                    if self.min_duration <= current_duration <= self.max_duration:
                        merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
        
        # Handle the final segment
        final_duration = current_end - current_start
        if self.min_duration <= final_duration <= self.max_duration:
            merged_segments.append((current_start, current_end))
        
        # Verify no overlaps in final result
        for i in range(len(merged_segments)-1):
            current_end = merged_segments[i][1]
            next_start = merged_segments[i+1][0]
            if next_start - current_end < self.min_segment_gap:
                self.logger.warning(f"Found segments too close together after merging: {current_end:.1f} to {next_start:.1f}")
                # Skip segments that are too close together
                return [s for i, s in enumerate(merged_segments) if i == 0 or 
                       merged_segments[i][0] - merged_segments[i-1][1] >= self.min_segment_gap]
        
        self.logger.debug(
            f"Merged {len(segments)} segments into {len(merged_segments)} segments\n" +
            "\n".join(f"Duration: {end-start:.1f}s ({start:.1f}-{end:.1f})" 
                     for start, end in merged_segments)
        )
        return merged_segments