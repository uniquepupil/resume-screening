from sklearn.metrics.pairwise import cosine_similarity

def match_resumes_to_job_descriptions(resume_features, job_description_features):
    similarity_scores = cosine_similarity(resume_features, job_description_features)
    return similarity_scores
