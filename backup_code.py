# from pyannote.core import Segment
# import whisper
#
# # instantiate the speaker diarization pipeline
# from pyannote.audio import Pipeline
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.0",
#     use_auth_token="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"
# )
#
# # run the diarization pipeline on an audio file
# diarization = pipeline("Python_AI.mp3")
#
# # dump the diarization output to disk using RTTM format
# with open("audio.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)
#
# # Load the Whisper ASR model
# model = whisper.load_model("base")
#
# # Load audio and process in chunks
# audio_path = "Python_AI.mp3"
# chunk_size_seconds = 30  # Adjust as needed
# sample_rate = 16000  # Adjust based on your audio data
#
# # Load the audio
# audio = whisper.load_audio(audio_path, sample_rate)
#
# # Calculate the total number of chunks needed
# num_chunks = len(audio) // (chunk_size_seconds * sample_rate)
#
# # Initialize a counter for speakers
# speaker_counter = 1
#
# # Initialize an empty string to store the final recognized text
# final_text = ""
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
#     options = whisper.DecodingOptions(fp16=False)
#     result = whisper.decode(model, mel, options)
#
#     # Get the corresponding speaker label for the current segment
#     current_segment = Segment(start=start_idx / sample_rate, end=end_idx / sample_rate)
#     labels = list(diarization.get_labels(current_segment))
#     if labels:
#         speaker = f"S{labels[0]}"
#     else:
#         # Assign a new speaker label if no label is available
#         speaker = f"S{speaker_counter}"
#         speaker_counter += 1
#
#     # Append the recognized text to the final result with speaker label
#     final_text += f"Speaker {speaker}: {result.text}\n"
#
# # Print the final recognized text
# print("Final Recognized Text:")
# print(final_text)

