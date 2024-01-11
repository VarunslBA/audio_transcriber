import diarizationspeaker
import audiocrop
import whisperaudio

if __name__ == "__main__":
    input_file_path = "Python_AI.mp3"
    output_folder_path = "cropped_audio"

    # Diarize audio
    diarization_results = diarizationspeaker.diarize_audio(input_file_path)
    diarizationspeaker.save_results(diarization_results, "diarization_results.json")

    # Replace speaker names
    for entry in diarization_results:
        entry['speaker_id'] = entry['speaker_id'].replace("SPEAKER_00", "Gernot").replace("SPEAKER_01", "Varun")

    # Crop audio
    time_intervals = [(entry['start_time'], entry['end_time']) for entry in diarization_results]
    audiocrop.crop_mp3(input_file_path, output_folder_path, time_intervals)

    # Transcribe audio
    transcription_results = whisperaudio.transcribe_audio(output_folder_path)
    whisperaudio.save_results(transcription_results, "transcription_results.json")

    # Print combined results
    for diarization_entry, transcription_entry in zip(diarization_results, transcription_results):
        print(f"File: {transcription_entry['filename']}")
        print(f"Timestamp: {diarization_entry['start_time']}s - {diarization_entry['end_time']}s")
        print(f"Speaker: {diarization_entry['speaker_id']}")
        print(f"Detected Language: {transcription_entry['detected_language']}")
        print(f"Transcript: {transcription_entry['transcript']}")
        print("\n")

