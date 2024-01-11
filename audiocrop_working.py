from pydub import AudioSegment
import diarizationspeaker

def crop_mp3(input_file, output_folder, intervals):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(input_file)

    # Convert intervals to milliseconds
    interval_milliseconds = [(start * 1000, end * 1000) for start, end in intervals]

    for i, (start, end) in enumerate(interval_milliseconds):
        # Crop the audio segment
        cropped_audio = audio[start:end]

        # Define the output file name
        output_file = f"{output_folder}/segment_{i + 1}.mp3"

        # Export the cropped audio segment
        cropped_audio.export(output_file, format="mp3")

if __name__ == "__main__":
    # Input MP3 file path
    input_file_path = "Python_AI.mp3"

    # Output folder for cropped segments
    output_folder_path = "cropped_audio"

    # Define time intervals in seconds
    time_intervals = diarizationspeaker.time_intervals
    # time_intervals = [(0, 30), (30, 60), (60, 90)]  # Example intervals

    # Create the output folder if it doesn't exist
    import os
    os.makedirs(output_folder_path, exist_ok=True)

    # Crop the MP3 file into defined intervals
    crop_mp3(input_file_path, output_folder_path, time_intervals)
