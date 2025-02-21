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
    start: float       # Changed from start_time for consistency
    end: float         # Changed from end_time for consistency
    sentiment: str
    score: float
    context_score: Optional[float] = None  # Added for contextual scoring

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
                # Combine text while keeping timing info
                combined_text = " ".join([seg['text'] for seg in group])
                start = group[0]['start']    # Updated variable name
                end = group[-1]['end']       # Updated variable name
                
                # Process text through sentiment analysis
                sentiment = self._analyze_text(combined_text)
                
                # Get contextual score by looking at surrounding segments
                context_score = self._calculate_context_score(
                    i, grouped_segments, sentiment['label']
                )
                
                # Apply emotion weight to the score
                weighted_score = sentiment['score'] * self.emotion_weights.get(
                    sentiment['label'].lower(), 1.0
                )
                
                results.append(SentimentResult(
                    text=combined_text,
                    start=start,             # Updated field name
                    end=end,                 # Updated field name
                    sentiment=sentiment['label'],
                    score=weighted_score,
                    context_score=context_score
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
        Group segments together for better contextual analysis.
        Groups are formed based on timing and natural breaks.
        """
        grouped = []
        current_group = []
        
        for segment in segments:
            current_group.append(segment)
            
            # Check if we should start a new group
            if len(current_group) >= self.segments_per_group:
                # Check if there's a significant pause that might indicate a scene break
                if len(current_group) > 1:
                    pause_duration = current_group[-1]['start'] - current_group[-2]['end']
                    if pause_duration > 2.0:  # More than 2 seconds of silence
                        grouped.append(current_group[:-1])  # Add group without the last segment
                        current_group = [current_group[-1]]  # Start new group with last segment
                        continue
                
                grouped.append(current_group)
                current_group = []
        
        # Add any remaining segments
        if current_group:
            grouped.append(current_group)
            
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
        Calculate a context score based on surrounding segments.
        Higher score if surrounding segments have similar sentiment.
        """
        context_score = 1.0
        window = self.context_window
        
        # Look at previous segments
        for i in range(max(0, current_idx - window), current_idx):
            prev_text = " ".join([seg['text'] for seg in grouped_segments[i]])
            prev_sentiment = self._analyze_text(prev_text)
            if prev_sentiment['label'] == current_sentiment:
                context_score += 0.1  # Boost score for matching sentiment
        
        # Look at following segments
        for i in range(current_idx + 1, min(current_idx + window + 1, len(grouped_segments))):
            next_text = " ".join([seg['text'] for seg in grouped_segments[i]])
            next_sentiment = self._analyze_text(next_text)
            if next_sentiment['label'] == current_sentiment:
                context_score += 0.1  # Boost score for matching sentiment
        
        return context_score