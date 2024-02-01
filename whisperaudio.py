import whisper
import os
import json
import re
from pydub import AudioSegment
import audiocrop

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f)


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
        audio = AudioSegment.from_file(audio_path)
        audio_length = len(audio) / 1000  # Convert milliseconds to seconds
        if audio_length > 30:
            transcript = transcribe_audio_more30(filename)
        else:
            transcript = transcribe_audio_less30(audio_directory,filename)
        results.append({
            'filename': filename,
            # 'detected_language': detected_language,
            'transcript': transcript
        })

    return results


def transcribe_audio_less30(audio_directory, filename):
    model = whisper.load_model("base")
    audio_path = os.path.join(audio_directory, filename)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    transcript = result.text
    return transcript



def transcribe_audio_more30(filename):
    audio = AudioSegment.from_file(filename)
    audio_length = len(audio) / 1000  # Convert milliseconds to seconds
    interval_duration = 30
    time_intervals = [(start_time, start_time + interval_duration) for start_time in
                      range(0, int(audio_length), interval_duration)]
    print(time_intervals)
    output_folder_path = "cropped_test"
    audiocrop.crop_mp3(filename, output_folder_path, time_intervals)
    model = whisper.load_model("base")

    # Get a human-friendly sorted list of audio files in the directory
    audio_files = human_sorted([
        filename
        for filename in os.listdir(output_folder_path)
        if filename.endswith(".mp3")
    ])

    for filename in audio_files:
        results30 = []
        audio_path = os.path.join(output_folder_path, filename)
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
        transcript = result.text

        return transcript




if __name__ == "__main__":
    audio_directory_path = "cropped_audio"

    transcription_results = transcribe_audio(audio_directory_path)
    save_results(transcription_results, "transcription_results.json")
#



