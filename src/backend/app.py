from flask import Flask, request, jsonify
from preprocessing.preprocess import preprocess_data

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to the Resume Screening App!'

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'resume' not in request.files or 'job_description' not in request.files:
        return jsonify({'error': 'Please upload both resume and job description files.'}), 400

    resume_file = request.files['resume']
    job_description_file = request.files['job_description']

    # Save files temporarily or process them directly
    resume_text = preprocess_data(resume_file, job_description_file)

    return jsonify({'message': 'Files processed successfully!', 'resume_text': resume_text})

if __name__ == '__main__':
    app.run(debug=True)
