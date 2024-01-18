from pyannote.audio.pipelines import SpeakerDiarization
import json

def diarize_audio(input_audio):
    # Load pretrained pipeline
    pipeline = SpeakerDiarization.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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"
    )

    # Apply pretrained pipeline
    diarization = pipeline({'uri': 'input', 'audio': input_audio})

    # Create a mapping to replace speaker names
    speaker_mapping = {'SPEAKER_00': 'Gernot', 'SPEAKER_01': 'Varun'}

    # Create a list to store tuples of time intervals, original speaker, new speaker, and transcript
    results = []

    # Store time intervals, original speaker, new speaker, and transcript
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = round(turn.start, 1)
        end_time = round(turn.end, 1)
        original_speaker_id = speaker  # Directly use the speaker string

        # Replace speaker name if in the mapping
        new_speaker_name = speaker_mapping.get(original_speaker_id, original_speaker_id)

        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'original_speaker_id': original_speaker_id,
            'speaker_id': new_speaker_name
        })

    # Sort the results based on start time
    results.sort(key=lambda x: x['start_time'])

    return results

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    input_audio_path = "Python_AI.mp3"  # Replace with your actual audio file path
    output_file_path = "diarization_results.json"

    diarization_results = diarize_audio(input_audio_path)
    save_results(diarization_results, output_file_path)

