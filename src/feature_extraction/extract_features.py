import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(resumes, job_descriptions):
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Combine resumes and job descriptions for TF-IDF vectorization
    combined_texts = resumes + job_descriptions
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    
    # Split the TF-IDF matrix back into resumes and job descriptions
    num_resumes = len(resumes)
    resumes_tfidf = tfidf_matrix[:num_resumes]
    job_descriptions_tfidf = tfidf_matrix[num_resumes:]
    
    return resumes_tfidf, job_descriptions_tfidf

# Example usage
if __name__ == "__main__":
    # Sample data
    resumes = ["Resume text here"]
    job_descriptions = ["Job description text here"]
    
    resumes_tfidf, job_descriptions_tfidf = extract_features(resumes, job_descriptions)
    
    print("Extracted Features from Resumes:")
    print(resumes_tfidf)
    print("Extracted Features from Job Descriptions:")
    print(job_descriptions_tfidf)
