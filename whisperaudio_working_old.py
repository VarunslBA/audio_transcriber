import os
import torch
import whisper
#import diarizationspeaker

# Load the model
model = whisper.load_model("base")

# Path to the directory containing audio files
audio_directory = "cropped_audio"

j = 0

# Process each audio file in the directory
for filename in os.listdir(audio_directory):
    if filename.endswith(".mp3") or filename.endswith(".wav"):  # Adjust file extensions as needed
        audio_path = os.path.join(audio_directory, filename)
        j += 1
        print(f"j: {j}")
        print(f"file length: {len(filename)}")

        # Load audio and process in chunks
        chunk_size_seconds = 30  # Process one second at a time
        sample_rate = 16000  # Adjust based on your audio data
        target_mel_shape = (80, 3000)

        # Load the audio
        audio = whisper.load_audio(audio_path, sample_rate)

        # Calculate the total number of chunks needed
        num_chunks = max(1, len(audio) // (chunk_size_seconds * sample_rate))

        # # Get the speaker information for the current file
        # speaker_info = diarizationspeaker.speaker_list
        # #print(f"Speaker info: {speaker_info}")

        # Iterate through each chunk (each second)
        for i in range(num_chunks):
            # Calculate the start and end indices for the current chunk
            start_idx = i * chunk_size_seconds * sample_rate
            end_idx = (i + 1) * chunk_size_seconds * sample_rate

            # Extract the current chunk
            chunk = audio[start_idx:end_idx]

            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(chunk).to(model.device)

            # Pad the mel spectrogram to match the target shape
            pad_width = target_mel_shape[1] - mel.shape[1]
            if pad_width > 0:
                mel = torch.nn.functional.pad(mel, (0, pad_width))

            # Decode the audio
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(model, mel, options)

            # Print the recognized text for the current second and associated speaker
            print(f"File: {filename}, start={start_idx:.1f}s stop={end_idx:.1f}, Second {j + 1}: {result.text}")






