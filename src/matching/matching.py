from sklearn.metrics.pairwise import cosine_similarity
from feature_extraction.feature_extraction import extract_features

def match_resume_to_job(resume_text: str, job_description_text: str) -> float:
    """
    Calculate the cosine similarity between resume and job description texts.
    
    :param resume_text: Text of the resume.
    :param job_description_text: Text of the job description.
    :return: Similarity score.
    """
    features = extract_features([resume_text, job_description_text])
    similarity = cosine_similarity(features[0:1], features[1:2])
    return similarity[0][0]
