# import chromadb
# import torch
import transformers
import langchain
import faiss
import numpy as np 
import pandas as pd
import re
import torch
import spacy
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,  Trainer, TrainingArguments, AdamW, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import GPT2TokenizerFast , OpenAIGPTTokenizerFast
import datetime

with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
from nltk.tokenize import sent_tokenize

texts = sent_tokenize(text)

def preprocess_text(text):
    # Remove newlines and excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove unwanted characters except for dots, which we'll handle separately
    text = re.sub(r'[^A-Za-z0-9.,;!?()\-\'\" ]', '', text)
    # Remove sequences of dots longer than 3
    text = re.sub(r'\.{2,}', '.', text)
    return text.strip()

# nlp = spacy.load('en_core_web_sm')
def clean_text(text):
    # Remove unwanted characters and symbols
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\.\s*', '. ', text)  # Ensure periods are followed by a single space
    text = re.sub(r'\s*\.\s*', '. ', text)  # Remove spaces before periods
    return text.strip()

# Function to tokenize sentences using SpaCy
def spacyST(text):
    doc = nlp(clean_text(text))
    return [sent.text for sent in doc.sents]

# Preprocess the texts
texts = [preprocess_text(text) for text in texts]
# print(texts)
    
def save_model(model, tokenizer, path='models/sales_agent_gpt2'):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def load_model(path='sales_agent_gpt2'):
    model = GPT2LMHeadModel.from_pretrained(path)
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    return model, tokenizer

# Save the model
# save_model(gpt2_lm_model, tokenizer)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2Model.from_pretrained('gpt2')
gpt2_lm_model = GPT2LMHeadModel.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

# Load the model (example usage)
# gpt2_lm_model, tokenizer = load_model()

def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = gpt2_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Generate and store embeddings in FAISS
embeddings = generate_embeddings(texts)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Query FAISS to get relevant information based on a user query
def query_index(query_text, top_k=2):
    query_embedding = generate_embeddings([query_text])
    distances, indices = index.search(query_embedding, top_k)
    return [texts[i] for i in indices[0]]

# Function to generate sales response using GPT-2
def generate_sales_response(context, query, user_persona):
    prompt = f"User Persona: {user_persona}\n\nContext: {context}\n\nQuery: {query}\n\nResponse: You will love this product because {context}. It perfectly matches your needs as {user_persona}. Don't miss out on this amazing opportunity! What do you think?"
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = gpt2_lm_model.generate(inputs.input_ids, max_length=200,max_new_tokens=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Sales agent chatbot function
def sales_agent_chatbot(query, user_persona):
    relevant_info = query_index(query)
    context = "\n".join(relevant_info)
    response = generate_sales_response(context, query, user_persona)
    return response

# # Example usage
# user_query = "Tell me apple vision pro"
# user_persona = "technology enthusiast"
# response = sales_agent_chatbot(user_query, user_persona)
# print(response)
# Gradio Interface
# Gradio Interface gradio
import gradio as gr
def main():
    def chatbot_interface(user_query, user_persona):
        if user_query and user_persona:
            response = sales_agent_chatbot(user_query, user_persona)
            return response
        else:
            return "Please enter both a query and a persona."

    # Define Gradio inputs and outputs
    inputs = [
        gr.Textbox(lines=2, placeholder="Enter your query", label="User Query"),
        gr.Textbox(lines=1, placeholder="Enter your persona (e.g., tech enthusiast)", label="User Persona")
    ]
#     outputs = gr.Textbox(label="Chatbot Response")

    # Launch the Gradio interface within Jupyter notebook
    interface = gr.Interface(fn=chatbot_interface, inputs=inputs, outputs=["text"], title="Sales Agent Chatbot")
    try:
        interface.launch(share=True)
    except Exception as e:
        print(f"Error launching Gradio interface: {str(e)}")

if __name__ == "__main__":
    main()