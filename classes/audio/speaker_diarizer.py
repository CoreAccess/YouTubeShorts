import os
# Set SpeechBrain to use copy mode before any imports
os.environ["SB_SAVE_MODE"] = "copy"

import logging
import torch
from pyannote.audio import Pipeline
from typing import Dict, List, Any
from classes.util.progress_tracker import ProgressTracker, ProcessingStage
from dotenv import load_dotenv
import warnings
from speechbrain.utils.fetching import fetch

# Suppress specific warnings about version mismatches
warnings.filterwarnings('ignore', message='Model was trained with')

# Configure logging levels for verbose libraries
logging.getLogger('speechbrain').setLevel(logging.ERROR)
logging.getLogger('pyannote').setLevel(logging.ERROR)
logging.getLogger('torchaudio').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

class SpeakerDiarizer:
    def __init__(self, progress_tracker: ProgressTracker = None):
        load_dotenv()
        
        self.logger = logging.getLogger('youtube_shorts')
        self.progress_tracker = progress_tracker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diarization_pipeline = None

        # Initialize speaker diarization model
        self.logger.info("Initializing speaker diarization model")
        self.logger.info("Speaker Diarization is Using Device: " + self.device)
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                self.logger.warning("HF_TOKEN environment variable not set. Speaker diarization will not be available.")
                return

            os.environ["TORCH_AUDIO_BACKEND"] = "soundfile"
            self.logger.info("Setting up diarization pipeline...")
            
            # Use a local cache directory in the project folder
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=hf_token,
                cache_dir=cache_dir
            )

            if self.device == "cuda":
                self.diarization_pipeline.to(torch.device(self.device))
            self.logger.info("Diarization pipeline initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize diarization: {str(e)}")
            self.diarization_pipeline = None

    def process_audio(self, audio_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process audio file for speaker diarization
        Returns dictionary with speaker segments
        """
        if not self.diarization_pipeline:
            return {"segments": []}

        try:
            if self.progress_tracker:
                self.progress_tracker.update_progress(
                    ProcessingStage.TRANSCRIPTION,
                    0.5,
                    "Processing speaker diarization"
                )

            diarization = self.diarization_pipeline(audio_path)
            
            # Convert pyannote output to usable format
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": f"SPEAKER_{speaker}"
                })

            return {"segments": segments}

        except Exception as e:
            self.logger.error(f"Speaker diarization failed: {str(e)}")
            return {"segments": []}