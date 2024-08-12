from matching.ranking import rank_resumes

def test_ranking():
    resumes = ["Resume text 1", "Resume text 2", "Resume text 3"]
    job_description = "Job description text"
    ranked_resumes = rank_resumes(resumes, job_description)
    for resume, score in ranked_resumes:
        print(f"Resume: {resume}, Score: {score}")

if __name__ == "__main__":
    test_ranking()
