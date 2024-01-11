from transformers import BartTokenizer, BartForConditionalGeneration

# Load the tokenizer and model
model = BartForConditionalGeneration.from_pretrained("pytorch_model.bin")
tokenizer = BartTokenizer.from_pretrained("tokenizer.json")


# Your conversation or text to be summarized
conversation = '''
Patient – Good Morning, doctor. May I come in?
Doctor – Good Morning. How are you? You do look quite pale this morning.
# ... (rest of the conversation)
Doctor – Welcome.
'''

# Tokenize the input text
input_ids = tokenizer.encode(conversation, return_tensors="pt")

# Generate a summary
outputs = model.generate(input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)

# Decode and print the summarized output
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Summary:", summary)
