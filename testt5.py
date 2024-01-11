from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline
from transformers import AutoTokenizer

text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."


summarizer = pipeline("summarization", model="D:/pythonProject/ai/model")
summarizer(text)



tokenizer = AutoTokenizer.from_pretrained("D:/pythonProject/ai/model")
inputs = tokenizer(text, return_tensors="pt").input_ids


model = AutoModelForSeq2SeqLM.from_pretrained("D:/pythonProject/ai/model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

tokenizer.decode(outputs[0], skip_special_tokens=True)