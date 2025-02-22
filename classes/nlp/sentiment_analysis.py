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
        Group segments into conversations based on speaker changes and timing.
        Ensures proper conversation boundaries and maintains speaker coherence.
        """
        grouped = []
        current_group = []
        current_speakers = set()
        current_conversation_start = 0
        conversation_id = 0
        
        for i, segment in enumerate(segments):
            # Get the speaker first before any other operations
            speaker = segment.get('speaker', 'UNKNOWN')
            
            if current_group:
                current_duration = segment['end'] - segments[current_conversation_start]['start']
                gap_to_previous = segment['start'] - current_group[-1]['end']
                
                # Conditions for starting a new group:
                # 1. Current group would exceed max duration
                # 2. Long pause between speakers
                # 3. Too many different speakers
                # 4. Natural conversation boundary (long pause + speaker change)
                should_start_new = (
                    current_duration > self.max_segment_duration or
                    gap_to_previous > self.max_silence_between_speakers or
                    (len(current_speakers) >= 3 and speaker not in current_speakers) or
                    (gap_to_previous > 1.0 and speaker not in current_speakers)
                )
                
                if should_start_new:
                    # Only save group if it meets criteria
                    group_duration = current_group[-1]['end'] - current_group[0]['start']
                    speaker_turns = sum(1 for j in range(1, len(current_group)) 
                                     if current_group[j].get('speaker') != current_group[j-1].get('speaker'))
                    
                    if (group_duration >= self.min_segment_duration and 
                        speaker_turns >= self.min_speaker_turns):
                        # Assign conversation ID to all segments in group
                        for seg in current_group:
                            seg['conversation_id'] = f"conv_{conversation_id}"
                        grouped.append(current_group)
                        conversation_id += 1
                    
                    current_group = []
                    current_speakers = set()
                    current_conversation_start = i
            
            current_group.append(segment)
            current_speakers.add(speaker)
        
        # Handle final group
        if current_group:
            group_duration = current_group[-1]['end'] - current_group[0]['start']
            speaker_turns = sum(1 for j in range(1, len(current_group)) 
                             if current_group[j].get('speaker') != current_group[j-1].get('speaker'))
            
            if (group_duration >= self.min_segment_duration and 
                speaker_turns >= self.min_speaker_turns):
                # Assign conversation ID to all segments in final group
                for seg in current_group:
                    seg['conversation_id'] = f"conv_{conversation_id}"
                grouped.append(current_group)
        
        self.logger.debug(
            f"Grouped {len(segments)} segments into {len(grouped)} conversation groups. "
            f"Average group size: {sum(len(g) for g in grouped)/len(grouped) if grouped else 0:.1f} segments"
        )
        
        return grouped

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
        Calculate a context score based on surrounding segments and speaker interactions
        Higher score if surrounding segments show engaging conversation
        """
        context_score = 1.0
        window = self.context_window
        
        # Look at surrounding segments
        for i in range(max(0, current_idx - window), min(current_idx + window + 1, len(grouped_segments))):
            if i == current_idx:
                continue
                
            group = grouped_segments[i]
            speakers = set(seg.get('speaker', 'UNKNOWN') for seg in group)
            
            # Higher score for multi-speaker segments (active conversation)
            if len(speakers) > 1:
                context_score += 0.15
            
            # Check sentiment continuity
            group_text = " ".join([seg['text'] for seg in group])
            group_sentiment = self._analyze_text(group_text)
            if group_sentiment['label'] == current_sentiment:
                context_score += 0.1
        
        return context_score