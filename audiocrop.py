from pydub import AudioSegment
import diarizationspeaker

def crop_mp3(input_file, output_folder, intervals):
    audio = AudioSegment.from_mp3(input_file)

    # Convert intervals to milliseconds
    interval_milliseconds = [(start * 1000, end * 1000) for start, end in intervals]

    for i, (start, end) in enumerate(interval_milliseconds):
        cropped_audio = audio[start:end]
        output_file = f"{output_folder}/{i + 1}.mp3"
        cropped_audio.export(output_file, format="mp3")

if __name__ == "__main__":
    input_file_path = "Python_AI.mp3"
    output_folder_path = "cropped_audio"

    time_intervals = diarizationspeaker.diarize_audio(input_file_path)
    diarizationspeaker.save_results(time_intervals, "diarization_results.json")  # Save results
    time_intervals = [(entry['start_time'], entry['end_time']) for entry in time_intervals]

    import os
    os.makedirs(output_folder_path, exist_ok=True)

    crop_mp3(input_file_path, output_folder_path, time_intervals)
