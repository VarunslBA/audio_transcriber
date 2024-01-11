import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load the fine-tuned model and tokenizer
loaded_model = T5ForConditionalGeneration.from_pretrained("fine_tuned_mental_health_model1")
loaded_tokenizer = T5Tokenizer.from_pretrained("fine_tuned_mental_health_model")

# Example new context
#new_context = "I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it. How can I change my feeling of being worthless to everyone?"
#new_context = "Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in \"Harry Potter and the Order of the Phoenix\" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. \"I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar,\" he told an Australian interviewer earlier this month. \"I don't think I'll be particularly extravagant. \"The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs.\" At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film \"Hostel: Part II,\" currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. \"I'll definitely have some sort of party,\" he said in an interview. \"Hopefully none of you will be reading about it.\" Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. \"People are always looking to say 'kid star goes off the rails,'\" he told reporters last month. \"But I try very hard not to go that way because it would be too easy for them.\" His latest outing as the boy wizard in \"Harry Potter and the Order of the Phoenix\" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called \"My Boy Jack,\" about author Rudyard Kipling and his son, due for release later this year. He will also appear in \"December Boys,\" an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's \"Equus.\" Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: \"I just think I'm going to be more sort of fair game,\" he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed."
new_context = '''Patient – Good Morning, doctor. May I come in?

Doctor – Good Morning. How are you? You do look quite pale this morning.

Patient – Yes, doctor. I’ve not been feeling well for the past few days. I’ve been having a stomach ache for a few days and feeling a bit dizzy since yesterday.

Doctor – Okay, let me check. (applies pressure on the stomach and checks for pain) Does it hurt here?

Patient – Yes, doctor, the pain there is the sharpest.

Doctor – Well, you are suffering from a stomach infection, that’s the reason you are having a stomach ache and also getting dizzy. Did you change your diet recently or have something unhealthy?

Patient – Actually, I went to a fair last week and ate food from the stalls there.

Doctor – Okay, so you are probably suffering from food poisoning. Since the food stalls in fairs are quite unhygienic, there’s a high chance those uncovered food might have caused food poisoning.

Patient – I think I will never eat from any unhygienic place in the future.

Doctor – That’s good. I’m prescribing some medicines, have them for one week and come back for a checkup next week. And please try to avoid spicy and fried foods for now.

Patient – Okay, doctor, thank you.

Doctor – Welcome.'''

# Tokenize the new context
input_ids = loaded_tokenizer(new_context, return_tensors="pt", max_length=512, truncation=True)[
    'input_ids'].squeeze()
input_ids = input_ids.unsqueeze(0)

# Generate summary using the loaded model
generated_ids = loaded_model.generate(input_ids, max_length=150, num_beams=2, length_penalty=2.0, early_stopping=True)

# Decode and print the generated summary
generated_summary = loaded_tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
print("Generated Summary:", generated_summary)