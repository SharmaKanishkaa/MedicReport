import streamlit as st
import os
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import PyPDF2
import pytesseract
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
import torch
# Function to extract text from a text-based PDF
def extract_text_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text
# Function to extract text from an image-based PDF using OCR
def extract_text_image_pdf(file_path):
    images = convert_from_path(file_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text
# Function to extract text from JPG/PNG images
def extract_text_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text
# Function to handle PDF, JPG, and PNG formats
def extract_text(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()

    # For PDF files
    if file_ext == '.pdf':
        try:
            return extract_text_pdf(file_path)  # Try extracting text-based PDF
        except:
            return extract_text_image_pdf(file_path)  # Fallback to image-based PDF

    # For JPG or PNG images
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        return extract_text_image(file_path)

    else:
        raise ValueError("Unsupported file format!")
# Load BioBERT model and tokenizer
@st.cache_resource
def load_biobert_model():
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")
    return tokenizer, model

# Function to perform NER using BioBERT
def run_biobert_ner(text):
    tokenizer, model = load_biobert_model()

    # Tokenize input text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**tokens)

    # Get predictions (entity tags)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Map tokens to their corresponding labels
    token_predictions = predictions[0].numpy()  # First batch
    tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    labels = [model.config.id2label[pred] for pred in token_predictions]

    entities = []
    current_entity = None
    for token, label in zip(tokens, labels):
        if label != "O":  # If token is not outside an entity
            if current_entity and label == current_entity['entity']:
                current_entity['text'] += " " + token.replace("##", "")  # Continue current entity
            else:
                if current_entity:
                    entities.append(current_entity)  # Append completed entity
                current_entity = {'entity': label, 'text': token.replace("##", "")}
        else:
            if current_entity:
                entities.append(current_entity)  # Append completed entity
                current_entity = None

    return entities

# Function to extract text from an image file using Tesseract
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Function to extract text from a PDF using pdf2image and Tesseract
def extract_text_from_pdf(pdf_file):
    images = pdf2image.convert_from_bytes(pdf_file.read())
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text
# Load BioBERT model and tokenizer for NER
@st.cache_resource
def load_biobert_model():
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")
    return tokenizer, model

# Load ClinicalBERT model for text simplification
@st.cache_resource
def load_clinicalbert_model():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModelForSeq2SeqLM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return tokenizer, model

# Load Summarization model (can use a general model like BART or T5)
@st.cache_resource
def load_summarization_model():
    summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarization_model

# Function to perform NER using BioBERT
def run_biobert_ner(text):
    tokenizer, model = load_biobert_model()

    # Tokenize input text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**tokens)

    # Get predictions (entity tags)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Map tokens to their corresponding labels
    token_predictions = predictions[0].numpy()  # First batch
    tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    labels = [model.config.id2label[pred] for pred in token_predictions]

    entities = []
    current_entity = None
    for token, label in zip(tokens, labels):
        if label != "O":  # If token is not outside an entity
            if current_entity and label == current_entity['entity']:
                current_entity['text'] += " " + token.replace("##", "")  # Continue current entity
            else:
                if current_entity:
                    entities.append(current_entity)  # Append completed entity
                current_entity = {'entity': label, 'text': token.replace("##", "")}
        else:
            if current_entity:
                entities.append(current_entity)  # Append completed entity
                current_entity = None

    return entities

# Function to simplify medical text using ClinicalBERT
def simplify_text(text):
    tokenizer, model = load_clinicalbert_model()

    # Tokenize and generate simplified text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    simplified_outputs = model.generate(**inputs)
    simplified_text = tokenizer.decode(simplified_outputs[0], skip_special_tokens=True)
    
    return simplified_text

# Function to summarize text
def summarize_text(text):
    summarization_model = load_summarization_model()
    
    # Summarize text
    summary = summarization_model(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to extract text from an image file using Tesseract
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Function to extract text from a PDF using pdf2image and Tesseract
def extract_text_from_pdf(pdf_file):
    images = pdf2image.convert_from_bytes(pdf_file.read())
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text
def load_scibert_model():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=3)  # Adjust num_labels based on classification task
    return tokenizer, model

# Function to classify text using SciBERT
def classify_text(text):
    tokenizer, model = load_scibert_model()

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).item()

    # Define class labels (for example purposes, update based on actual task)
    labels = ['Disease', 'Procedure', 'Medication']
    predicted_class = labels[predictions]

    return predicted_class
# Load BioBERT QA model for medical question answering
@st.cache_resource
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")
    return tokenizer, model

# Function to answer questions using BioBERT
def answer_question(question, context):
    tokenizer, model = load_qa_model()
    
    # Tokenize the input question and context (extracted text)
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    
    # Get the model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the most probable start and end token positions
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    # Decode the answer from the tokenized input
    answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
    
    return answer
# Streamlit app layout
st.title("MedicReport: AI-Driven Medical Report Analysis and Question Answering Tool")

# File uploader (one-time file upload)
uploaded_file = st.file_uploader("Upload a medical report (PDF/Image)", type=["pdf", "jpg", "jpeg", "png"])

# Extracted text
if uploaded_file is not None:
    file_type = uploaded_file.type

    # Extract text based on file type
    if "pdf" in file_type:
        st.info("Extracting text from PDF...")
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif "image" in file_type or file_type in ["image/jpeg", "image/png"]:
        st.info("Extracting text from image...")
        extracted_text = extract_text_from_image(uploaded_file)

    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Named Entity Recognition
    st.subheader("Named Entity Recognition (NER)")
    if st.button("Run NER"):
        entities = perform_ner(extracted_text)
        st.write(entities)

    # Text Simplification
    st.subheader("Text Simplification")
    if st.button("Simplify Medical Terms"):
        simplified_text = simplify_text(extracted_text)
        st.write(simplified_text)

    # Text Classification
    st.subheader("Text Classification")
    if st.button("Classify Text"):
        classification_result = classify_text(extracted_text)
        st.write(classification_result)

    # Medical Question Answering
    st.subheader("Medical Question Answering")
    question = st.text_input("Enter your medical question:")
    if st.button("Get Answer"):
        if question:
            answer = answer_question(question, extracted_text)
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")
