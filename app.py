from flask import Flask, request, jsonify
import pdfplumber
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Root Route (For Testing API Status)
@app.route("/")
def home():
    return jsonify({"message": "API is working!"})

def download_pdf_from_url(pdf_url, save_path="temp_resume.pdf"):
    """Download the PDF from a given URL and save it locally."""
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()  # Raise error for bad response (4xx, 5xx)

        with open(save_path, "wb") as pdf_file:
            for chunk in response.iter_content(chunk_size=1024):
                pdf_file.write(chunk)
        
        return save_path  # Return the saved file path

    except requests.exceptions.RequestException as e:
        return None

def extract_text_from_pdf(pdf_path):
    """Extract text from a downloaded PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"

        return text.strip()
    except Exception:
        return None

def calculate_ats_score(resume_text, job_desc):
    """Calculate ATS score based on similarity between resume and job description."""
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return round(score * 100, 2)  # Convert to percentage
    except Exception:
        return 0

@app.route('/ats-score', methods=['POST'])
def ats_score():
    """API endpoint to calculate ATS score."""
    data = request.get_json()

    # Check if required fields exist
    if not data or 'resume_url' not in data or 'job_desc' not in data:
        return jsonify({"error": "Missing required fields: 'resume_url' and 'job_desc'"}), 400

    resume_url = data['resume_url']
    job_desc = data['job_desc']

    # Download PDF
    pdf_path = download_pdf_from_url(resume_url)
    if not pdf_path:
        return jsonify({"error": "Failed to download PDF from provided URL"}), 400

    # Extract text from PDF
    resume_text = extract_text_from_pdf(pdf_path)
    if not resume_text:
        return jsonify({"error": "Could not extract text from resume"}), 400

    # Compute ATS Score
    score = calculate_ats_score(resume_text, job_desc)

    # Clean up temporary file
    os.remove(pdf_path)

    return jsonify({"ats_score": score})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Vercel's assigned port
    app.run(host='0.0.0.0', port=port)

