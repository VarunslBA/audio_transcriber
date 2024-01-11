import whisper
import os

model = whisper.load_model("base")

# Set the directory containing the audio files
audio_directory = "cropped_audio"


# Iterate through each audio file in the directory
for filename in os.listdir(audio_directory):
    if filename.endswith(".mp3"):
        audio_path = os.path.join(audio_directory, filename)

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(fp16 = False)
        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(f"File name: {filename}, Result: {result.text}")




