import whisper
import audiocrop
import os
from pydub import AudioSegment
import re



# Load audio and process in chunks
audio_path = "audio/German.mp3"


# Load the audio
# Get audio duration using pydub
audio = AudioSegment.from_file(audio_path)
audio_length = len(audio) / 1000  # Convert milliseconds to seconds
print(f"audio_length: {audio_length}")

interval_duration = 30

time_intervals = [(start_time, start_time + interval_duration) for start_time in range(0, int(audio_length), interval_duration)]
print(time_intervals)
output_folder_path = "cropped_test"
audiocrop.crop_mp3(audio_path, output_folder_path, time_intervals)

def human_sorted(iterable):
    """Sorts the input iterable in a human-friendly order."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(iterable, key=alphanum_key)

def transcribe_audio(audio_directory):
    model = whisper.load_model("base")
    results = []

    # Get a human-friendly sorted list of audio files in the directory
    audio_files = human_sorted([
        filename
        for filename in os.listdir(audio_directory)
        if filename.endswith(".mp3")
    ])

    for filename in audio_files:
        audio_path = os.path.join(audio_directory, filename)
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        # print the recognized text
        results.append(result.text)

    print(results)


transcription_results = transcribe_audio(output_folder_path)



















# # Calculate the total number of chunks needed
# num_chunks = len(audio) // (chunk_size_seconds * sample_rate)
#
# # Initialize an empty string to store the final recognized text
# final_text = ""
#
# # Define decoding options
# options = whisper.DecodingOptions(fp16=False)
#
# # Iterate through each chunk
# for i in range(num_chunks):
#     # Calculate the start and end indices for the current chunk
#     start_idx = i * chunk_size_seconds * sample_rate
#     end_idx = (i + 1) * chunk_size_seconds * sample_rate
#
#     # Extract the current chunk
#     chunk = audio[start_idx:end_idx]
#
#     # Make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(chunk).to(model.device)
#
#     # Detect the spoken language
#     _, probs = model.detect_language(mel)
#     detected_language = max(probs, key=probs.get)
#     print(f"Detected language: {detected_language}")
#
#     # Decode the audio
#     result = whisper.decode(model, mel, options)
#
#     # Append the recognized text to the final result
#     final_text += result.text
#
# # Handle the last chunk (even if it's shorter than chunk_size_seconds)
# last_chunk = audio[num_chunks * chunk_size_seconds * sample_rate:]
# if len(last_chunk) > 0:
#     mel_last = whisper.log_mel_spectrogram(last_chunk).to(model.device)
#     _, probs_last = model.detect_language(mel_last)
#     detected_language_last = max(probs_last, key=probs_last.get)
#     print(f"Detected language (last chunk): {detected_language_last}")
#
#     result_last = whisper.decode(model, mel_last, options)
#     final_text += result_last.text
#
# # Print the final recognized text
# print("Final Recognized Text:")
# print(final_text)