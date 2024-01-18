from pydub import AudioSegment
import diarizationspeaker
import os
import shutil
import whisperaudio  # assuming this module is correctly implemented

def crop_mp3(input_file, output_folder, intervals, min_duration=500):
    audio = AudioSegment.from_mp3(input_file)

    # Convert intervals to milliseconds
    interval_milliseconds = [(start * 1000, end * 1000) for start, end in intervals]

    # Clear the output directory
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)

    for i, (start, end) in enumerate(interval_milliseconds):
        cropped_audio = audio[start:end]

        # Check if the duration is more than 1 second
        if len(cropped_audio) >= min_duration:
            output_file = f"{output_folder}/{i + 1}.mp3"
            cropped_audio.export(output_file, format="mp3")

if __name__ == "__main__":
    input_file_path = "Python_AI.mp3"
    output_folder_path = "cropped_audio"

    time_intervals = diarizationspeaker.diarize_audio(input_file_path)
    diarizationspeaker.save_results(time_intervals, "diarization_results.json")  # Save results
    time_intervals = [(entry['start_time'], entry['end_time']) for entry in time_intervals]

    crop_mp3(input_file_path, output_folder_path, time_intervals)


