from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity_tfidf(resume_texts, job_description_texts):
    vectorizer = TfidfVectorizer()
    resume_vectors = vectorizer.fit_transform(resume_texts)
    job_description_vectors = vectorizer.transform(job_description_texts)
    similarity_matrix = cosine_similarity(resume_vectors, job_description_vectors)
    return similarity_matrix
