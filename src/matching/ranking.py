def rank_resumes(resumes, job_description_text):
    """
    Rank resumes based on similarity score with the job description.
    
    :param resumes: List of resume texts.
    :param job_description_text: Text of the job description.
    :return: List of tuples (resume_text, similarity_score).
    """
    from matching.matching import match_resume_to_job

    ranked_resumes = [(resume, match_resume_to_job(resume, job_description_text)) for resume in resumes]
    return sorted(ranked_resumes, key=lambda x: x[1], reverse=True)
