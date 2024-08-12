from preprocessing.preprocess import preprocess_data

def test_matching():
    resume_file = 'data/resumes/test_resume.pdf'
    job_description_file = 'data/job_descriptions/test_job_description.docx'
    score = preprocess_data(resume_file, job_description_file)
    print("Similarity Score:", score)

if __name__ == "__main__":
    test_matching()
