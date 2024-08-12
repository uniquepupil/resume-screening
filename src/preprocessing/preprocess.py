import PyPDF2
import docx
import pandas as pd
from typing import Tuple
from feature_extraction.feature_extraction import extract_features
from matching.matching import match_resume_to_job

def read_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def read_doc(file_path: str) -> str:
    doc = docx.Document(file_path)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return text

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

def preprocess_data(resumes_file: str, job_descriptions_file: str) -> float:
    resumes_text = read_pdf(resumes_file)
    job_descriptions_text = read_doc(job_descriptions_file)

    # Text preprocessing
    processed_resumes = preprocess_text(resumes_text)
    processed_job_descriptions = preprocess_text(job_descriptions_text)

    # Matching
    similarity_score = match_resume_to_job(processed_resumes, processed_job_descriptions)
    return similarity_score
