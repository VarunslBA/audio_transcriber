from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
import torchaudio
import torch

# Load the saved model
model_path = "./whisper-small-hi"  # Replace with the actual path to your saved model
model = WhisperForConditionalGeneration.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path)

# Load a new audio file for transcription
audio_path = "path/to/your/new_audio_file.wav"  # Replace with the actual path to your audio file
waveform, sample_rate = torchaudio.load(audio_path)

# Preprocess the audio for the model
input_features = processor.feature_extractor(waveform, sampling_rate=sample_rate).input_features[0]

# Perform inference
input_ids = model.encode(input_features, return_tensors="pt").input_ids
with torch.no_grad():
    output_ids = model.generate(input_ids)

# Decode the output to text
transcription = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

# Print or use the transcription as needed
print("Transcription:", transcription)
