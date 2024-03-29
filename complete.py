import diarizationspeaker
import audiocrop
import whisperaudio

def combine_results(diarization_results, transcription_results):
    combined_results = []
    current_speaker = None
    current_transcript = ""

    for diarization_entry, transcription_entry in zip(diarization_results, transcription_results):
        speaker_id = diarization_entry['speaker_id']
        transcript = transcription_entry['transcript']

        if current_speaker is None:
            current_speaker = speaker_id
            current_transcript = transcript
        elif current_speaker == speaker_id:
            current_transcript += " " + transcript
        else:
            combined_results.append({
                'speaker_id': current_speaker,
                'start_time': diarization_entry['start_time'],
                'end_time': diarization_entry['end_time'],
                'detected_language': transcription_entry['detected_language'],
                'transcript': current_transcript.strip()
            })

            current_speaker = speaker_id
            current_transcript = transcript

    # Add the last entry
    combined_results.append({
        'speaker_id': current_speaker,
        'start_time': diarization_results[-1]['start_time'],
        'end_time': diarization_results[-1]['end_time'],
        'detected_language': transcription_results[-1]['detected_language'],
        'transcript': current_transcript.strip()
    })

    return combined_results

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


    # # Print combined results
    # for diarization_entry, transcription_entry in zip(diarization_results, transcription_results):
    #     print(f"File: {transcription_entry['filename']}")
    #     print(f"Timestamp: {diarization_entry['start_time']}s - {diarization_entry['end_time']}s")
    #     print(f"Speaker: {diarization_entry['speaker_id']}")
    #     print(f"Detected Language: {transcription_entry['detected_language']}")
    #     print(f"Transcript: {transcription_entry['transcript']}")
    #     print("\n")


# Combine and print results
combined_results = combine_results(diarization_results, transcription_results)

current_speaker = combined_results[0]['speaker_id']

for entry in combined_results:
    print("\n")
    print("\n" if current_speaker != entry['speaker_id'] else "", end="")
    print(f"{entry['speaker_id']}: {entry['transcript']}", end="")
    current_speaker = entry['speaker_id']

print("\n")
#




#Take the diarization result and Transcription result directly from JSON file after running them separately
# import json
#
# def read_results_from_json(json_file):
#     with open(json_file, 'r') as f:
#         return json.load(f)
#
# def combine_results(diarization_results, transcription_results):
#     combined_results = []
#     current_speaker = None
#     current_transcript = ""
#
#     for diarization_entry, transcription_entry in zip(diarization_results, transcription_results):
#         speaker_id = diarization_entry['speaker_id']
#         transcript = transcription_entry['transcript']
#
#         if current_speaker is None:
#             current_speaker = speaker_id
#             current_transcript = transcript
#         elif current_speaker == speaker_id:
#             current_transcript += " " + transcript
#         else:
#             combined_results.append({
#                 'speaker_id': current_speaker,
#                 'start_time': diarization_entry['start_time'],
#                 'end_time': diarization_entry['end_time'],
#                 'detected_language': transcription_entry['detected_language'],
#                 'transcript': current_transcript.strip()
#             })
#
#             current_speaker = speaker_id
#             current_transcript = transcript
#
#     # Add the last entry
#     combined_results.append({
#         'speaker_id': current_speaker,
#         'start_time': diarization_results[-1]['start_time'],
#         'end_time': diarization_results[-1]['end_time'],
#         'detected_language': transcription_results[-1]['detected_language'],
#         'transcript': current_transcript.strip()
#     })
#
#     return combined_results
#
# if __name__ == "__main__":
#     diarization_file_path = "diarization_results.json"
#     transcription_file_path = "transcription_results.json"
#
#     # Read diarization and transcription results from JSON files
#     diarization_results = read_results_from_json(diarization_file_path)
#     transcription_results = read_results_from_json(transcription_file_path)
#
#     # Combine and print results
#     combined_results = combine_results(diarization_results, transcription_results)
#
#     current_speaker = combined_results[0]['speaker_id']
#
#     for entry in combined_results:
#         print("\n")
#         print("\n" if current_speaker != entry['speaker_id'] else "", end="")
#         print(f"{entry['speaker_id']}: {entry['transcript']}", end="")
#         current_speaker = entry['speaker_id']
#
#     print("\n")





