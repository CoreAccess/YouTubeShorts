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

def merge_transcript_and_diarization(
    transcript_data: Dict[str, Any], 
    diarization_data: Dict[str, List[Dict[str, Any]]],
    audio_features: Optional[List[Any]] = None
) -> Dict[str, Any]:
    logger = logging.getLogger('youtube_shorts')
    
    def is_overlapping_conversation(new_start: float, new_end: float, conversations: Dict) -> bool:
        """Check if a new conversation would overlap with existing ones"""
        for conv_data in conversations.values():
            overlap_start = max(new_start, conv_data["start"])
            overlap_end = min(new_end, conv_data["end"])
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > 3.5 and (overlap_duration / (new_end - new_start)) > 0.15:  # More lenient overlap check
                    return True
        return False

    try:
        # If no diarization data, return original transcript
        if not diarization_data["segments"]:
            return transcript_data

        def get_audio_intensity(start: float, end: float) -> float:
            """Get average audio intensity for a time range"""
            if not audio_features:
                return 0.5
            
            relevant_features = [
                f for f in audio_features
                if f.start <= end and f.end >= start
            ]
            if not relevant_features:
                return 0.5
            
            weighted_sum = 0
            total_weight = 0
            for feature in relevant_features:
                overlap_start = max(start, feature.start)
                overlap_end = min(end, feature.end)
                overlap_duration = overlap_end - overlap_start
                weight = overlap_duration / (end - start)
                weighted_sum += feature.intensity_score * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.5
            
        # Create speaker turns from diarization data
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
        conversation_speakers = set()
        conversation_start = 0
        conversation_segments = []
        conversations = {}  # Track all conversations for overlap checking
        
        for segment in transcript_data["segments"]:
            segment_mid = (segment["start"] + segment["end"]) / 2
            current_turn = find_speaker_turn(segment_mid, speaker_turns)
            
            new_segment = segment.copy()
            
            if current_turn:
                new_segment["speaker"] = current_turn.speaker
                time_gap = segment["start"] - last_end_time
                speaker_changed = last_speaker != current_turn.speaker
                current_duration = segment["end"] - conversation_start
                
                # Enhanced conversation break detection
                should_start_new = False
                
                # Check audio intensity during gap
                gap_intensity = get_audio_intensity(last_end_time, segment["start"]) if time_gap > 0.5 else 1.0
                
                # More lenient conversation break conditions but maintain duration limits
                if (
                    # Long silence gap with very low audio intensity
                    (time_gap > 2.5 and gap_intensity < 0.25) or
                    # Speaker change with significant gap and low intensity
                    (speaker_changed and time_gap > 0.75 and gap_intensity < 0.35) or
                    # Strict duration limit
                    current_duration > 90.0 or
                    # Speaker limit slightly relaxed
                    (current_turn.speaker not in conversation_speakers and len(conversation_speakers) >= 4) or
                    # Natural conversation end with more lenient conditions
                    (speaker_changed and time_gap > 0.5 and 
                     conversation_segments and conversation_segments[-1]["text"].strip().endswith(('.', '!', '?'))) or
                    # Check overlap with existing conversations
                    (conversation_segments and 
                     is_overlapping_conversation(conversation_start, segment["end"], conversations))
                ):
                    # Only break if current conversation meets minimum requirements
                    if conversation_segments:
                        conv_duration = conversation_segments[-1]["end"] - conversation_segments[0]["start"]
                        conv_intensity = get_audio_intensity(conversation_segments[0]["start"], conversation_segments[-1]["end"])
                        
                        # Maintain strict 30s minimum but be more lenient with other criteria
                        if (30.0 <= conv_duration <= 90.0 and 
                            (len(conversation_speakers) > 1 or conv_intensity > 0.55)):  # More lenient intensity requirement
                            conversations[f"conv_{current_conversation_id}"] = {
                                "start": conversation_segments[0]["start"],
                                "end": conversation_segments[-1]["end"]
                            }
                            should_start_new = True
                
                if should_start_new:
                    current_conversation_id += 1
                    conversation_speakers = {current_turn.speaker}
                    conversation_start = segment["start"]
                    conversation_segments = []
                else:
                    conversation_speakers.add(current_turn.speaker)
                
                new_segment["conversation_id"] = f"conv_{current_conversation_id}"
                last_speaker = current_turn.speaker
                conversation_segments.append(new_segment)
            else:
                new_segment["speaker"] = "UNKNOWN"
                current_conversation_id += 1
                new_segment["conversation_id"] = f"conv_{current_conversation_id}"
                last_speaker = None
                conversation_speakers = set()
                conversation_start = segment["start"]
                conversation_segments = [new_segment]
            
            last_end_time = segment["end"]
            
            # Process words with speaker attribution
            if "words" in segment:
                for word in segment["words"]:
                    word_mid = (word["start"] + word["end"]) / 2
                    word_turn = find_speaker_turn(word_mid, speaker_turns)
                    word["speaker"] = word_turn.speaker if word_turn else "UNKNOWN"
            
            processed_segments.append(new_segment)
        
        # Update transcript with processed segments
        transcript_data["segments"] = processed_segments
        
        # Add conversation metadata with audio features and ensure no significant overlaps
        valid_conversations = {}
        for conv_id, data in conversations.items():
            # Skip conversations that overlap with already validated ones
            if not is_overlapping_conversation(data["start"], data["end"], valid_conversations):
                if (30.0 <= data["duration"] <= 90.0 and
                    (len(data["speakers"]) > 1 or data["intensity"] > 0.6)):
                    valid_conversations[conv_id] = data
            else:
                logger.debug(f"Skipping overlapping conversation {conv_id}")
        
        # Add filtered conversation metadata to transcript
        transcript_data["conversations"] = [
            {
                "id": conv_id,
                "start": data["start"],
                "end": data["end"],
                "speaker_count": len(data["speakers"]),
                "turn_count": data["turns"],
                "duration": data["duration"],
                "intensity": data["intensity"]
            }
            for conv_id, data in valid_conversations.items()
        ]
        
        # Update unique speakers list
        transcript_data["speakers"] = list(set(
            segment["speaker"] 
            for segment in processed_segments 
            if segment["speaker"] != "UNKNOWN"
        ))
        
        return transcript_data
        
    except Exception as e:
        logger.error(f"Failed to merge transcript and diarization: {str(e)}")
        return transcript_data