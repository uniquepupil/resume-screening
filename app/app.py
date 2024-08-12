from flask import Flask, request, jsonify
from src.preprocessing.preprocess import preprocess_data
from src.feature_extraction.features import extract_features
from src.model.matching import match_resumes_to_job_descriptions

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_files():
    resumes_file = request.files['resumes']
    job_descriptions_file = request.files['job_descriptions']
    
    resumes, job_descriptions = preprocess_data(resumes_file, job_descriptions_file)
    
    resume_features, job_description_features = extract_features(resumes['processed_text'], job_descriptions['processed_text'])
    
    similarity_scores = match_resumes_to_job_descriptions(resume_features, job_description_features)
    
    # Assuming y_true is the ground truth labels, which may need to be provided separately
    y_true = [...]  # Replace with actual labels or provide a mechanism to upload or infer labels
    metrics = evaluate_model(y_true, similarity_scores)
    
    return jsonify({
        'similarity_scores': similarity_scores.tolist(),
        'metrics': metrics
    })

if __name__ == '__main__':
    app.run(debug=True)
