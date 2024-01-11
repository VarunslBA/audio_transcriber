from transformers import pipeline
import json

# Load the summarization pipeline with the pre-trained model
summarizer = pipeline("summarization", model="kabita-choudhary/finetuned-bart-for-conversation-summary")

# Load diarization and transcription results from files
with open("diarization_results.json", "r") as f:
    diarization_results = json.load(f)

with open("transcription_results.json", "r") as f:
    transcription_results = json.load(f)

# Combine diarization and transcription results
combined_results = []

for diarization_entry, transcription_entry in zip(diarization_results, transcription_results):
    combined_entry = {
        'speaker_id': diarization_entry['speaker_id'],
        'detected_language': transcription_entry['detected_language'],
        'transcript': transcription_entry['transcript']
    }
    combined_results.append(combined_entry)

# Convert the combined results into a string
conversation = ""
for entry in combined_results:
    conversation += f"Speaker {entry['speaker_id']}: {entry['transcript']}\n"

# Generate a summary
summary = summarizer(conversation, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)

# Print the summarized output
print("Summary:", summary[0]['summary_text'])
