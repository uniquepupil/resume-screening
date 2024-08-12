from src.preprocessing.preprocess import preprocess_data
from src.feature_extraction.features import extract_features
from src.model.matching import match_resumes_to_job_descriptions
from src.evaluation.evaluate import evaluate_model

# Load test data
resumes_file = 'data/resumes/test_resumes.csv'
job_descriptions_file = 'data/job_descriptions/test_job_descriptions.csv'

resumes, job_descriptions = preprocess_data(resumes_file, job_descriptions_file)
resume_features, job_description_features = extract_features(resumes['processed_text'], job_descriptions['processed_text'])
similarity_scores = match_resumes_to_job_descriptions(resume_features, job_description_features)

# Assuming y_true is the ground truth labels
y_true = [...]  # Replace with actual labels
metrics = evaluate_model(y_true, similarity_scores)

print('Evaluation Metrics:')
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1 Score: {metrics['f1_score']}")
