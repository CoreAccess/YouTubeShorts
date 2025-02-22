from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from bisect import bisect_left

@dataclass
class TimeSegment:
    start: float
    end: float

class SpeakerTurn:
    def __init__(self, start: float, end: float, speaker: str):
        self.start = start
        self.end = end
        self.speaker = speaker
        self.text = ""
        self.words = []

def find_overlapping_speaker(time: float, speaker_segments: List[Dict[str, Any]]) -> str:
    """Find which speaker was talking at a given timestamp"""
    for segment in speaker_segments:
        if segment["start"] <= time <= segment["end"]:
            return segment["speaker"]
    return "UNKNOWN"

def find_speaker_turn(time: float, turns: List[SpeakerTurn]) -> Optional[SpeakerTurn]:
    """Find the speaker turn that contains the given time"""
    for turn in turns:
        if turn.start <= time <= turn.end:
            return turn
    return None

def merge_transcript_and_diarization(transcript_data: Dict[str, Any], diarization_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Merge transcript segments with speaker diarization data.
    Maintains conversation flow and ensures proper speaker attribution.
    """
    logger = logging.getLogger('youtube_shorts')
    
    try:
        # If no diarization data, return original transcript
        if not diarization_data["segments"]:
            return transcript_data
            
        # First, create speaker turns from diarization data
        speaker_turns = []
        for segment in diarization_data["segments"]:
            turn = SpeakerTurn(
                start=segment["start"],
                end=segment["end"],
                speaker=segment["speaker"]
            )
            speaker_turns.append(turn)
        
        # Sort turns by start time
        speaker_turns.sort(key=lambda x: x.start)
        
        # Process transcript segments and assign to speaker turns
        processed_segments = []
        current_conversation_id = 0
        last_speaker = None
        last_end_time = 0
        
        for segment in transcript_data["segments"]:
            segment_mid = (segment["start"] + segment["end"]) / 2
            current_turn = find_speaker_turn(segment_mid, speaker_turns)
            
            # Create new segment with speaker information
            new_segment = segment.copy()
            
            if current_turn:
                new_segment["speaker"] = current_turn.speaker
                
                # Check if this is part of the same conversation
                time_gap = segment["start"] - last_end_time
                speaker_changed = last_speaker != current_turn.speaker
                
                # Start new conversation if:
                # 1. Large time gap (> 2s)
                # 2. Speaker changed and gap > 0.5s
                if (time_gap > 2.0) or (speaker_changed and time_gap > 0.5):
                    current_conversation_id += 1
                
                new_segment["conversation_id"] = f"conv_{current_conversation_id}"
                last_speaker = current_turn.speaker
            else:
                new_segment["speaker"] = "UNKNOWN"
                # Unknown speaker segments start new conversations
                current_conversation_id += 1
                new_segment["conversation_id"] = f"conv_{current_conversation_id}"
                last_speaker = None
            
            last_end_time = segment["end"]
            
            # Process words if available
            if "words" in segment:
                for word in segment["words"]:
                    word_mid = (word["start"] + word["end"]) / 2
                    word_turn = find_speaker_turn(word_mid, speaker_turns)
                    word["speaker"] = word_turn.speaker if word_turn else "UNKNOWN"
            
            processed_segments.append(new_segment)
        
        # Update transcript with processed segments
        transcript_data["segments"] = processed_segments
        
        # Add conversation metadata
        conversations = {}
        for segment in processed_segments:
            conv_id = segment["conversation_id"]
            if conv_id not in conversations:
                conversations[conv_id] = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speakers": set([segment["speaker"]]),
                    "turns": 1
                }
            else:
                conv = conversations[conv_id]
                conv["end"] = segment["end"]
                if segment["speaker"] not in conv["speakers"]:
                    conv["speakers"].add(segment["speaker"])
                    conv["turns"] += 1
        
        # Add conversation metadata to transcript
        transcript_data["conversations"] = [
            {
                "id": conv_id,
                "start": data["start"],
                "end": data["end"],
                "speaker_count": len(data["speakers"]),
                "turn_count": data["turns"]
            }
            for conv_id, data in conversations.items()
        ]
        
        # Add list of unique speakers
        transcript_data["speakers"] = list(set(
            segment["speaker"] 
            for segment in processed_segments 
            if segment["speaker"] != "UNKNOWN"
        ))
        
        return transcript_data
        
    except Exception as e:
        logger.error(f"Failed to merge transcript and diarization: {str(e)}")
        return transcript_data