
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def process_data(text, tokenizer, model):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        scores = model(**encoded_input).logits
        predicted_label = scores.argmax(-1).item()
        response = tokenizer.decode([predicted_label])
    return response

text = "What are the key differences between renewable and non-renewable energy sources?"
model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

# Load the model and tokenizer
tokenizer, model = load_model(model_name)

# Process the data
response = process_data(text, tokenizer, model)

print(response)