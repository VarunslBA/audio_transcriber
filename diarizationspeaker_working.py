from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# apply pretrained pipeline
diarization = pipeline("Python_AI.mp3")

# Create a list to store tuples of time intervals
time_intervals = []
speaker_list = []

# Print the result and store time intervals
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_time = turn.start
    end_time = turn.end
    speaker_id = f"speaker_{speaker}"
    print(f"start={start_time:.1f}s stop={end_time:.1f}s {speaker_id}")

# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...
    # Append the time interval as a tuple to the list with only 1 decimal point
    time_intervals.append((round(start_time, 1), round(end_time, 1)))
    speaker_list.append((round(start_time, 1), round(end_time, 1), speaker_id))

# Now 'time_intervals' contains a list of tuples representing each time interval with only 1 decimal point
print("List of Time Intervals:")
print(time_intervals)
print(speaker_list)
