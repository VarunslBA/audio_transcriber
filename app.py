from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
import os
import diarizationspeaker
import audiocrop
import whisperaudio
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the summarization pipeline with the pre-trained model
summarizer = pipeline("summarization", model="kabita-choudhary/finetuned-bart-for-conversation-summary")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        diarization_results = diarizationspeaker.diarize_audio(file_path)
        diarizationspeaker.save_results(diarization_results, "diarization_results.json")
        time_intervals = [(entry['start_time'], entry['end_time']) for entry in diarization_results]

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        audiocrop.crop_mp3(file_path, app.config['UPLOAD_FOLDER'], time_intervals)

        transcription_results = whisperaudio.transcribe_audio(app.config['UPLOAD_FOLDER'])
        whisperaudio.save_results(transcription_results, "transcription_results.json")

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
        summary_text = summary[0]['summary_text']

        return render_template('result.html', diarization_results=diarization_results,
                               transcription_results=transcription_results, summary_text=summary_text)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
