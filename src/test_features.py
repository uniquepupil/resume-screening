from feature_extraction.features import extract_features

sample_resumes = ["Software engineer with 5 years of experience", "Data scientist specializing in machine learning"]
sample_jobs = ["Looking for a software engineer", "Seeking an expert in machine learning"]

resume_features, job_features = extract_features(sample_resumes, sample_jobs)

print("Resume Features Shape:", resume_features.shape)
print("Job Features Shape:", job_features.shape)
