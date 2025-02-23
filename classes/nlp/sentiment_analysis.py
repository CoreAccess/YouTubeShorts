import os
import json
import logging
import torch
from transformers import pipeline
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
from classes.util.progress_tracker import ProgressTracker, ProcessingStage
import sys

@dataclass
class SentimentResult:
    text: str
    start: float
    end: float
    sentiment: str
    score: float
    speaker: str  # Added speaker field
    context_score: Optional[float] = None
    conversation_id: Optional[str] = None  # Added to track conversation groups

class SentimentAnalysis:
    def __init__(self, progress_tracker: ProgressTracker = None):
        self.logger = logging.getLogger('youtube_shorts')
        self.progress_tracker = progress_tracker
        # Use CUDA if available
        self.device = 0 if torch.cuda.is_available() else -1
        self.logger.info(f"Initializing sentiment analysis model on {'GPU' if self.device == 0 else 'CPU'}")
        
        try:
            # Initialize the sentiment analysis pipeline
            self.classifier = pipeline(
                "sentiment-analysis",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=self.device
            )
            self.logger.debug("Sentiment analysis model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load sentiment model: {str(e)}", exc_info=True)
            raise
        
        # Configuration for analysis
        self.max_sequence_length = 512
        self.context_window = 3  # Number of segments to consider for context
        self.segments_per_group = 20
        self.emotion_weights = {
            'fear': 1.2,
            'joy': 1.1,
            'surprise': 1.15,
            'sadness': 1.1,
            'anger': 1.2,
            'disgust': 1.1,
            'neutral': 0.8
        }
        self.max_silence_between_speakers = 2.0  # Increased from 1.0
        self.min_segment_duration = 40.0  # Reduced from 50.0
        self.max_segment_duration = 90.0  # Increased from 65.0
        self.target_duration = 60.0  # Target duration for segments
        self.min_speaker_turns = 1  # Reduced from 2 to allow single-speaker segments

    def analyze_transcript(self, transcript_path: str, temp_dir: str) -> str:
        """
        Analyze the sentiment of a transcript file and save results
        Returns the path to the saved sentiment analysis file
        """
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(transcript_path))[0]
            sentiment_path = os.path.join(temp_dir, f"{base_name}_sentiment.json")
            
            # If sentiment analysis already exists, return it
            if os.path.exists(sentiment_path):
                self.logger.info(f"Using existing sentiment analysis: {sentiment_path}")
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        ProcessingStage.SENTIMENT_ANALYSIS,
                        1.0,
                        "Using existing sentiment analysis"
                    )
                return sentiment_path
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.SENTIMENT_ANALYSIS,
                    0.1,
                    "Loading transcript data"
                )
            
            # Load transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Check if speaker information already exists
            has_speaker_info = any('speaker' in segment for segment in transcript_data.get('segments', []))
            if not has_speaker_info:
                self.logger.warning("No speaker information found in transcript. Speaker diarization may be required.")
            else:
                self.logger.info("Using existing speaker information from transcript")
            
            self.logger.debug(f"Processing {len(transcript_data['segments'])} transcript segments")
            
            # Group segments
            grouped_segments = self._group_segments(transcript_data['segments'])
            self.logger.debug(f"Grouped into {len(grouped_segments)} segment groups")
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.SENTIMENT_ANALYSIS,
                    0.2,
                    f"Analyzing {len(grouped_segments)} segment groups"
                )
            
            # Process grouped segments
            results = []
            for i, group in enumerate(tqdm(grouped_segments, desc="Analyzing sentiment")):
                # Group segments by speaker
                speaker_segments = {}
                for seg in group:
                    speaker = seg.get('speaker', 'UNKNOWN')
                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []
                    speaker_segments[speaker].append(seg)
                
                conversation_id = f"conv_{i}"
                
                # Analyze each speaker's contribution separately
                for speaker, segs in speaker_segments.items():
                    combined_text = " ".join([seg['text'] for seg in segs])
                    if not combined_text.strip():
                        continue
                        
                    sentiment = self._analyze_text(combined_text)
                    context_score = self._calculate_context_score(
                        i, grouped_segments, sentiment['label']
                    )
                    
                    weighted_score = sentiment['score'] * self.emotion_weights.get(
                        sentiment['label'].lower(), 1.0
                    )
                    
                    results.append(SentimentResult(
                        text=combined_text,
                        start=segs[0]['start'],
                        end=segs[-1]['end'],
                        sentiment=sentiment['label'],
                        score=weighted_score,
                        speaker=speaker,
                        context_score=context_score,
                        conversation_id=conversation_id
                    ).__dict__)
                
                if self.progress_tracker:
                    progress = (i + 1) / len(grouped_segments)
                    self.progress_tracker.update_progress(
                        ProcessingStage.SENTIMENT_ANALYSIS,
                        0.2 + (progress * 0.7),
                        f"Analyzed {i+1}/{len(grouped_segments)} segments"
                    )
                
                self.logger.debug(
                    f"Segment {i+1}: {sentiment['label']} "
                    f"(score: {weighted_score:.2f}, context: {context_score:.2f})"
                )
            
            # Save results
            with open(sentiment_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.SENTIMENT_ANALYSIS,
                    1.0,
                    f"Completed sentiment analysis of {len(results)} segments"
                )
            
            self.logger.info(f"Sentiment analysis completed and saved to {sentiment_path}")
            return sentiment_path
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}", exc_info=True)
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.SENTIMENT_ANALYSIS,
                    1.0,
                    f"Sentiment analysis failed: {str(e)}"
                )
            return None

    def _group_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Enhanced grouping with more lenient requirements for conversations
        """
        # Group by existing conversation IDs first
        conversation_groups = {}
        used_time_ranges = []
        
        for segment in segments:
            conv_id = segment.get('conversation_id')
            if conv_id:
                if conv_id not in conversation_groups:
                    # More lenient overlap check
                    has_overlap = False
                    for start, end in used_time_ranges:
                        overlap_start = max(segment['start'], start)
                        overlap_end = min(segment['end'], end)
                        if overlap_end > overlap_start:
                            overlap_duration = overlap_end - overlap_start
                            if overlap_duration > 4.0:  # Increased from 3.5
                                has_overlap = True
                                break
                    
                    if not has_overlap:
                        conversation_groups[conv_id] = []
                        used_time_ranges.append((segment['start'], segment['end']))
                        
                if conv_id in conversation_groups:
                    conversation_groups[conv_id].append(segment)
        
        # Sort groups by start time
        grouped = [
            group for group in conversation_groups.values()
            if group
        ]
        grouped.sort(key=lambda g: g[0]['start'])
        
        # Validate and adjust groups with more lenient criteria
        final_groups = []
        last_end_time = 0
        
        for group in grouped:
            duration = group[-1]['end'] - group[0]['start']
            start_time = group[0]['start']
            
            # More lenient gap requirement
            if start_time - last_end_time < 2.0:  # Reduced from 2.5
                continue
            
            # Duration bounds with strict minimum
            if not (30.0 <= duration <= 95.0):  # Maintain 30s minimum, allow up to 95s max
                continue
            
            # Split long groups at natural break points
            if duration > 80.0:  # Increased from 75.0
                subgroups = []
                current_subgroup = []
                current_duration = 0
                
                for segment in group:
                    seg_duration = segment['end'] - segment['start']
                    new_duration = current_duration + seg_duration
                    
                    # More lenient splitting conditions but maintain 30s minimum
                    if (new_duration > 95.0 or  # Increased from 90.0
                        (current_duration >= 30.0 and  # Maintain 30s minimum
                         segment['text'].strip().endswith(('.', '!', '?'))) or
                        (subgroups and segment['start'] - subgroups[-1][-1]['end'] < 2.0)):  # Reduced from 2.5
                        if current_subgroup:
                            subgroups.append(current_subgroup)
                        current_subgroup = [segment]
                        current_duration = seg_duration
                    else:
                        current_subgroup.append(segment)
                        current_duration = new_duration
                
                if current_subgroup:
                    subgroups.append(current_subgroup)
                
                # Add subgroups with more lenient spacing but maintain duration requirements
                for subgroup in subgroups:
                    subgroup_duration = subgroup[-1]['end'] - subgroup[0]['start']
                    if (30.0 <= subgroup_duration <= 95.0 and  # Maintain 30s minimum
                        (not final_groups or subgroup[0]['start'] - final_groups[-1][-1]['end'] >= 2.0)):  # Reduced from 2.5
                        final_groups.append(subgroup)
            else:
                final_groups.append(group)
            
            if final_groups:
                last_end_time = final_groups[-1][-1]['end']
        
        return final_groups

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a piece of text, handling long sequences appropriately
        """
        # If text is too long, split into sentences and average the results
        if len(text.split()) > self.max_sequence_length:
            sentences = self._split_into_sentences(text)
            sentiments = []
            
            for sentence in sentences:
                if sentence.strip():  # Only process non-empty sentences
                    result = self.classifier(sentence)[0]
                    sentiments.append(result)
            
            # Aggregate results
            if sentiments:
                # Find the dominant sentiment
                positive_score = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
                negative_score = len(sentiments) - positive_score
                
                # Calculate average confidence
                avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
                
                return {
                    'label': 'POSITIVE' if positive_score > negative_score else 'NEGATIVE',
                    'score': avg_score
                }
            else:
                return {'label': 'NEUTRAL', 'score': 0.5}
        else:
            # Process normally if text is within length limits
            return self.classifier(text)[0]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, trying to keep context while staying within length limits
        """
        # Simple sentence splitting on common punctuation
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_context_score(
        self, 
        current_idx: int, 
        grouped_segments: List[List[Dict]], 
        current_sentiment: str
    ) -> float:
        """
        Enhanced context scoring with more lenient thresholds
        """
        context_score = 1.0
        window = self.context_window
        
        # Look at surrounding segments
        for i in range(max(0, current_idx - window), min(current_idx + window + 1, len(grouped_segments))):
            if i == current_idx:
                continue
                
            group = grouped_segments[i]
            speakers = set(seg.get('speaker', 'UNKNOWN') for seg in group)
            
            # More lenient scoring for conversations
            if len(speakers) > 1:
                context_score += 0.3  # Increased from 0.25
            elif len(speakers) == 1 and any(seg.get('text', '').strip().endswith(('?', '!', '...')) for seg in group):
                context_score += 0.2  # Increased from 0.15, added '...' as engaging marker
            
            # Analyze sentiment progression with more lenient scoring
            group_text = " ".join([seg['text'] for seg in group])
            group_sentiment = self._analyze_text(group_text)
            
            # More generous bonuses for emotional variety
            if group_sentiment['label'] != current_sentiment:
                context_score += 0.25  # Increased from 0.2
            
            # More lenient bonus for emotional content
            if group_sentiment['score'] > 0.6:  # Reduced from 0.7
                context_score += 0.2  # Increased from 0.15
        
        # Allow higher boost for engaging contexts
        return min(2.5, context_score)  # Increased from 2.2