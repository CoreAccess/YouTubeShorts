import os
import logging
import torch
from faster_whisper import WhisperModel
from classes.util.progress_tracker import ProgressTracker, ProcessingStage
import json

class Transcript:
    def __init__(self, progress_tracker: ProgressTracker = None):
        self.logger = logging.getLogger('youtube_shorts')
        self.progress_tracker = progress_tracker
        
        # Check if CUDA is available and initialize Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        self.logger.info(f"Initializing Whisper model on {device} with compute type {compute_type}")
        self.model = WhisperModel("large-v2", device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, temp_dir: str) -> str:
        """
        Transcribe audio file using Faster-Whisper and save transcript with word-level timestamps
        Returns the path to the saved transcript file
        """
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            transcript_path = os.path.join(temp_dir, f"{base_name}_transcript.json")
            
            # If transcript already exists, return it
            if os.path.exists(transcript_path):
                self.logger.info(f"Using existing transcript: {transcript_path}")
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        ProcessingStage.TRANSCRIPTION,
                        1.0,
                        "Using existing transcript"
                    )
                return transcript_path
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.TRANSCRIPTION,
                    0.1,
                    "Starting transcription"
                )
            
            self.logger.info(f"Starting transcription of {audio_path}")
            
            # Transcribe with word-level timestamps
            segments, info = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                temperature=0.0,
                initial_prompt="This is a TV show or movie transcript."
            )
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.TRANSCRIPTION,
                    0.7,
                    "Processing transcription results"
                )
            
            # Format and save transcript
            transcript_data = {
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "words": [
                            {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "probability": word.probability
                            }
                            for word in (segment.words or [])
                        ]
                    }
                    for segment in segments
                ],
                "language": info.language,
                "language_probability": info.language_probability
            }
            
            # Save transcript with pretty formatting for readability
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.TRANSCRIPTION,
                    1.0,
                    f"Transcription completed ({len(transcript_data['segments'])} segments)"
                )
            
            self.logger.info(f"Transcription completed and saved to {transcript_path}")
            return transcript_path
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.TRANSCRIPTION,
                    1.0,
                    f"Transcription failed: {str(e)}"
                )
            return None