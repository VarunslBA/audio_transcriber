from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer
from datasets import Audio
import gradio as gr
from multiprocessing import Pool
import os

# notebook_login()

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train[:10]+validation[:10]", token="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")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test[:10]", token="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")
# common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", token="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")
# common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", token="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")

#print(common_voice)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

#print(common_voice)

### Load WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

### Load WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

### Combine To Create A WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

### Prepare Data
#print(common_voice["train"][0])

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Prepare dataset without using multiprocessing
# common_voice = common_voice.map(
#     prepare_dataset,
#     remove_columns=common_voice.column_names["train"],
#     num_proc=1  # Set num_proc to 1 to avoid multiprocessing
# )
# #print(common_voice["train"][0])

if __name__ == '__main__':
    # with Pool(processes=2) as pool:
    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],
        num_proc=1  # Set num_proc to 1 to avoid multiprocessing
    )

## Training and Evaluation

### Define a Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

### Load a Pre-Trained Checkpoint
model_checkpoint = "./whisper-small-hi-checkpoint"  # Specify the path to the saved checkpoint
if os.path.exists(model_checkpoint):
    model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
else:
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

### Define the Training Configuration
#import accelerate`

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a directory path of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=1,
    gradient_checkpointing=True,
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Create a trainer without training (for evaluation)
eval_trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Evaluate and print WER before fine-tuning
eval_results_before_fine_tuning = trainer.evaluate()
print("WER before fine-tuning:", eval_results_before_fine_tuning["eval_wer"])

# Fine-tune the model
trainer.train()

# Evaluate and print WER after fine-tuning
eval_results_after_fine_tuning = trainer.evaluate()
print("WER after fine-tuning:", eval_results_after_fine_tuning["eval_wer"])

# Save the trained model locally
trainer.save_model("./whisper-small-hi-checkpoint")