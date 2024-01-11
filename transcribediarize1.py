import torch
import whisper
from pyannote.core import Segment
from pyannote.audio import Pipeline

# Load the model
model = whisper.load_model("base")

# Load audio and process in chunks
audio_path = "Python_AI.mp3"
chunk_size_seconds = 8  # Adjust as needed
sample_rate = 16000  # Adjust based on your audio data

# Load the audio
audio = whisper.load_audio(audio_path, sample_rate)

# Calculate the total number of chunks needed
num_chunks = len(audio) // (chunk_size_seconds * sample_rate)

# Initialize an empty list to store timestamp, speaker, and text information
transcription_results = []

# Define the target shape for mel spectrogram
target_mel_shape = (80, 3000)

# Iterate through each chunk
for i in range(num_chunks):
    # Calculate the start and end indices for the current chunk
    start_idx = i * chunk_size_seconds * sample_rate
    end_idx = (i + 1) * chunk_size_seconds * sample_rate

    # Extract the current chunk
    chunk = audio[start_idx:end_idx]

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(chunk).to(model.device)

    # Pad the mel spectrogram to match the target shape
    pad_width = (target_mel_shape[1] - mel.shape[1]) // 2
    mel = torch.nn.functional.pad(mel, (pad_width, pad_width))

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    # Decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # Append the recognized text, timestamp, and speaker information to the list
    start_time = i * chunk_size_seconds
    end_time = (i + 1) * chunk_size_seconds
    transcription_results.append({
        "segment": Segment(start_time, end_time),
        "speaker": None,  # Placeholder for speaker information
        "text": result.text
    })

# ... (rest of the code remains unchanged)
# Create diarization using the pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")
diarization = pipeline('Python_AI.mp3')

# # Load diarization results and update the speaker information in the transcription results
# for turn, _, speaker in diarization.itertracks(yield_label=True):
#     for result in transcription_results:
#         if result["segment"].start < turn.end and result["segment"].end > turn.start:
#             result["speaker"] = f"speaker_{speaker}"
#
# # # Print the final synchronized results
# # print("Final Synchronized Results:")
# # for result in transcription_results:
# #     print(f"start={result['segment'].start:.1f}s stop={result['segment'].end:.1f}s {result['speaker']} text='{result['text']}'")
# #
# # for turn, _, speaker in diarization.itertracks(yield_label=True):
# #     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
#
# # Print the final synchronized results
# print("Final Synchronized Results:")
# for result in transcription_results:
#     print(f"start={result['segment'].start:.1f}s stop={result['segment'].end:.1f}s {result['speaker']} text='{result['text']}'")

# Load diarization results into a list
diarization_results = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    diarization_results.append({
        "segment": turn,
        "speaker": f"speaker_{speaker}"
    })

# Update the speaker information in the transcription results
for result in transcription_results:
    closest_diarization = min(diarization_results, key=lambda x: abs(result["segment"].start + result["segment"].end - x["segment"].start - x["segment"].end) / 2)
    result["speaker"] = closest_diarization["speaker"]

# # Print the final synchronized results
# print("Final Synchronized Results:")
# for result in transcription_results:
#     print(f"start={result['segment'].start:.1f}s stop={result['segment'].end:.1f}s {result['speaker']} text='{result['text']}'")

# Print the final synchronized results
for result in transcription_results:
    print(f"start={result['segment'].start:.1f}s stop={result['segment'].end:.1f}s {result['speaker']} text='{result['text']}'")

    # # If the speaker information is available, print the corresponding diarization segments
    # if result['speaker'] is not None:
    #     speaker_segments = [s for s in diarization_results if s['speaker'] == result['speaker']]
    #     for segment in speaker_segments:
    #         print(f"start={segment['segment'].start:.1f}s stop={segment['segment'].end:.1f}s {result['speaker']}")