from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(resumes_text, job_descriptions_text):
    vectorizer = TfidfVectorizer()
    resume_features = vectorizer.fit_transform(resumes_text)
    job_description_features = vectorizer.transform(job_descriptions_text)
    
    return resume_features, job_description_features
