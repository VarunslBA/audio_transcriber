from transformers import pipeline

# Load the audio transcription pipeline directly from the Hugging Face Model Hub
transcriber = pipeline(task="automatic-speech-recognition", model="mfreihaut/finetuned-audio-transcriber")

# Provide the path to your audio file
audio_path = "Python_AI.mp3"  # Change this to your actual audio file path

# Perform transcription
transcription_result = transcriber(audio_path)

# Print the transcription result
print("Transcription:")
print(transcription_result)
