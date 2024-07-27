# LLM RAG ChatBot

[Uploading Screencast from 2024-07-25 21-47-35.webm…]()

This repository contains the implementation of a Sales Agent tuned ChatBot based on the GPT-2 model and Faiss (Facebook AI Similarity Search). Retrieval Augmented Generation used inorder to create prompts and context for the base LLM. 
## Table of Contents
- Overview
- Features
- Results

## Overview 
Text data was extracted from pdfs (Apple vision pro brochure) and from the apple vision pro website.The Data was embedded using the GPT2 Tokenizer. It utilizes GPT-2 knowledge base to answer questions. FAISS is used as 
a vector database for making a query similarity search which finds the top-k documents whose embeddings are most similar to the query which are fed as context to the LLM along with the original query. The prompt is passed
to GPT-2 modified inorder to pass the context and create the Sales agent persona. Gradio is then used to deploy the project.

REPOSITORY CONTENTS : 
- index.py : This file contains the main code which includes a Retriver and a Generator
- dataExt.py : This file contains the script for scraping all text from the pages and other links the page contains. This also extracts text from all pdfs present.

## Features
- GPT-2 Based: Utilizes GPT-2 models for natural language understanding and question answering.
- FAISS: For performing query embedding similarity search . Initially chromadb was tried but faiss was preferred
- Gradio : Gradio is used for deploying the project and building frontend
- Web Scraping Bot: Processes and cleans text data for training.

## Results
 Example => User Query: Tell me about Apple Vision Pro!
            User Persona: I am a tech enthusiast 
            Output: You will love this product because Apple Vision Pro Privacy Overview Learn how Apple Vision Pro and visionOS protect your data February 2024 Introducing Apple Vision Pro.3Privacy by design.3Surroundings.4Input.6Optic ID.8Guest User.8Persona.9EyeSight.10In-store demo and purchase.11Apples commitment to privacy.11 Apple Vision Pro Privacy Overview  February 20242ContentsIntroducing Apple Vision Pro At Apple, we believe privacy is a fundamental human right.
Apple, the Apple logo, Apple Pay, Face ID, FaceTime, iPad, iPhone, Mac, macOS, Safari, and Touch ID, are trademarks of Apple Inc., registered in the U.S. and other countries.. It perfectly matches your needs as technology enthusiast. Don't miss out on this amazing opportunity! What do you think?

 
