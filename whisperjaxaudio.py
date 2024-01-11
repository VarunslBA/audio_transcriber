import whisper
import os
import json

model = whisper.load_model("base")
audio = whisper.load_audio("cropped_audio/segment_1.mp3")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
_, probs = model.detect_language(mel)
detected_language = max(probs, key=probs.get)
options = whisper.DecodingOptions(fp16=False)
result = whisper.decode(model, mel, options)
transcript = result.text
print(transcript)


