from gtts import gTTS
import os

def text_to_speech(text, language='en', filename='audio/sample_output.mp3'):
    # Create a gTTS object
    tts = gTTS(text=text, lang=language, slow=False)

    # Save the speech as an MP3 file
    tts.save(filename)

    # Play the generated audio (optional)
    os.system(f"start {filename}")  # This command opens the default audio player

# Example usage
doctor_text = "Doctor: I'm sorry to hear that you're not feeling well..."
patient_text = "Patient: Thank you, doctor. I appreciate your thorough explanation..."


doctor_text = "Please send in the next patient."

# [speaker:Patient]  Hello doctor, good morning.
#
# [speaker:Doctor]  Good morning, have a seat. Please tell me what happened.
#
# [speaker:Patient]  (showing the knee) For the past few months, I have been experiencing a severe pain in my left knee whenever I stand up or walk long distances.
#
# [speaker:Doctor]  (checking the knee) Yes, it is slightly swollen, but probably nothing is broken. Can you please stand up for me?
#
# [speaker:Patient]  (stands up) It really hurts when I try to stand after being seated for a while.
#
# [speaker:Doctor]  Did you fall down or hit your knee somewhere?
#
# [speaker:Patient] No doctor, as far as I remember, I didn’t hurt my knees.
#
# [speaker:Doctor]  Okay, so I’m giving you Ibuprofen; it will help bring down the swelling and pain. Once the swelling goes down, you can take some tests which will help me judge why you have this constant pain. If you don’t find Ibuprofen in the medical store, you can ask them to give you Paracetamol 600. It will also help ease the pain. Have the medicines for two days and come back for another check-up once the swelling is gone.
#
# [speaker:Patient] Sure doctor. Thank you.
#
# [speaker:Doctor]  You are welcome.

# Convert text to speech for doctor and patient
text_to_speech(doctor_text, filename='doctor.mp3')
text_to_speech(patient_text, filename='patient.mp3')
