import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("presencesw/summarize_t5")
# dataset = load_dataset("Amod/mental_health_counseling_conversations")

# Define the T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# # Define a function to preprocess the data
# def preprocess_data(example):
#     inputs = tokenizer(example['Context'], return_tensors="pt", max_length=512, truncation=True)
#     labels = tokenizer(example['Response'], return_tensors="pt", max_length=150, truncation=True)
#     return {'input_ids': inputs['input_ids'], 'labels': labels['input_ids']}
def preprocess_data(example):
    input_text = example['texts']
    output_text = example['targets']

    # Check if either input or output is empty
    if not input_text or not output_text:
        return None

    # Tokenize input and output
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    # print(inputs)
    labels = tokenizer(output_text, return_tensors="pt", max_length=150, truncation=True)

    # Check if the tokenized input and output have the same length
    if inputs['input_ids'].size(1) != labels['input_ids'].size(1):
        return None

    return {'input_ids': inputs['input_ids'], 'labels': labels['input_ids']}


# Apply preprocessing to the dataset
train_dataset = dataset['train'].map(preprocess_data)

# Define the DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=550, shuffle=True)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Fine-tune the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
model.train()

num_epochs = 1
for epoch in range(num_epochs):
    for batch in train_dataloader:
        #print(batch[1])
        optimizer.zero_grad()

        # Access 'Context' and 'Response' from each example in the batch
        context = " ".join([str(sentence) for sentence in batch['texts']])
        response = " ".join([str(sentence) for sentence in batch['targets']])

        # Tokenize context and response
        input_ids = tokenizer(context, return_tensors="pt", max_length=512, truncation=True)['input_ids'].squeeze()
        labels = tokenizer(response, return_tensors="pt", max_length=150, truncation=True)['input_ids'].squeeze()

        # Ensure both input_ids and labels have the batch dimension
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)

        # Forward pass: compute predicted y by passing inputs to the model
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits

        # Compute the loss using the specified loss function
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update the model parameters
        optimizer.step()

    # Save the fine-tuned model
model.save_pretrained("fine_tuned_mental_health_model1")

#         outputs = model(input_ids, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#
# # Save the fine-tuned model
# model.save_pretrained("fine_tuned_mental_health_model")
#
# # Save the fine-tuned model and tokenizer
# model.save_pretrained("fine_tuned_mental_health_model")
# tokenizer.save_pretrained("fine_tuned_mental_health_model")

# # Load the fine-tuned model and tokenizer
# loaded_model = T5ForConditionalGeneration.from_pretrained("fine_tuned_mental_health_model")
# loaded_tokenizer = T5Tokenizer.from_pretrained("fine_tuned_mental_health_model")
#
# # Example new context
# new_context = "I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it. How can I change my feeling of being worthless to everyone?"
#
# # Tokenize the new context
# input_ids = loaded_tokenizer(new_context, return_tensors="pt", max_length=512, truncation=True)[
#     'input_ids'].squeeze()
# input_ids = input_ids.unsqueeze(0)
#
# # Generate summary using the loaded model
# generated_ids = loaded_model.generate(input_ids, max_length=150, num_beams=2, length_penalty=2.0, early_stopping=True)
#
# # Decode and print the generated summary
# generated_summary = loaded_tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
# print("Generated Summary:", generated_summary)


