import streamlit as st
import zipfile
import io
from PyPDF2 import PdfReader
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM  # Updated import for langchain-ibm
from langchain_community.llms.utils import enforce_stop_tokens  # Updated import

# Initialize the LLaMA model via IBM WatsonX
def initialize_model():
    apikey = os.getenv("apikey")
    project_id = os.getenv("projectid")
    verify = False
    model_id = 'meta-llama/llama-3-70b-instruct'

    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.1,
    }

    print("Initializing WatsonxLLM model...")  # Debug print
    model = WatsonxLLM(
        model_id=model_id,
        params=parameters,
        project_id=project_id,
        apikey=apikey,
        url="https://jp-tok.ml.cloud.ibm.com"
    )
    print("Model initialized successfully.")  # Debug print
    return model

llm = initialize_model()

# Function to generate a response using the LLaMA model
# Function to generate a response using the LLaMA model
def generate_response(llm, prompt):
    print(f"Generating response with prompt: {prompt}")  # Debug print
    try:
        response = llm.invoke(prompt)  # Ensure you're using the correct method
        print(f"Response received: {response}")  # Debug print
        if not response:
            print("Warning: Received empty response from the model.")  # Debug print
        return response
    except Exception as e:
        print(f"Error generating response: {type(e).__name__} - {str(e)}")  # Detailed debug print
        return None

# Function to calculate vector match percentage using CountVectorizer
def calculate_match_percentage(resume_text, jd_text):
    vectorizer = CountVectorizer().fit_transform([resume_text, jd_text])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0][1]
    return cosine_sim * 100

# Function to analyze resume and JD with LLaMA
def analyze_resume_with_llama(resume_text, jd_text):
    prompt = (
        "Evaluate the following resume against the job description. "
        "Provide a match score and feedback.\n\n"
        f"Job Description: {jd_text}\n\nResume: {resume_text}\n\n"
        "Analysis and Recommendations:"
    )
    print("Prompt being sent to the model:")
    print(prompt)
    
    response = generate_response(llm, prompt)
    
    print("Response received:")
    print(response)
    
    feedback = response if response else "No response received from the model."
    print("Feedback generated:")
    print(feedback)
    
    return feedback



# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(f"Extracted text: {text[:500]}...")  # Debug print to show a snippet of the text
    return text

# Function to process ZIP file containing PDFs
def process_zip_file(zip_file):
    resume_texts = []
    with zipfile.ZipFile(zip_file, 'r') as z:
        for file_info in z.infolist():
            if file_info.filename.endswith('.pdf'):
                with z.open(file_info) as f:
                    pdf_text = extract_text_from_pdf(f)
                    resume_texts.append((file_info.filename, pdf_text))
                    print(f"Processed file: {file_info.filename}")  # Debug print
    return resume_texts

# Streamlit Dashboard
st.header('LATENT 500', divider='rainbow')

st.title("Optimized ATS with IBM WatsonX LLaMA")
tabs = st.tabs(["Applicant", "Recruiter"])

with tabs[0]:
    st.header("Applicant Tab")
    jd_text = st.text_area("Paste the Job Description here", key="applicant_jd_text")
    
    uploaded_resume = st.file_uploader("Upload Your Resume (PDF)", type="pdf", help="Please upload your resume in PDF format.")
    
    if uploaded_resume and jd_text:
        resume_text = extract_text_from_pdf(uploaded_resume)
        vector_match_score = calculate_match_percentage(resume_text, jd_text)
        
        st.write(f"Vector Match Score: {vector_match_score:.2f}%")
        
        if st.button("Generate AI Match Score and Feedback"):
            feedback = analyze_resume_with_llama(resume_text, jd_text)
            st.write("AI Match Score and Feedback:")
            st.write(feedback)

with tabs[1]:
    st.header("Recruiter Tab")
    uploaded_zip = st.file_uploader("Upload a ZIP file containing resumes", type="zip", help="Please upload a ZIP file with PDF resumes.")
    
    if uploaded_zip:
        resume_texts = process_zip_file(uploaded_zip)
        jd_text = st.text_area("Paste the Job Description here", key="recruiter_jd_text")
        
        if jd_text:
            results = []
            for filename, resume_text in resume_texts:
                feedback = analyze_resume_with_llama(resume_text, jd_text)
                vector_match_score = calculate_match_percentage(resume_text, jd_text)
                results.append({
                    "Filename": filename,
                    "Vector Match Score": vector_match_score,
                    "AI Match Score": feedback
                })
            
            # Sort results by Vector Match Score and get top 3
            sorted_results = sorted(results, key=lambda x: x["Vector Match Score"], reverse=True)[:3]
            
            st.write("Top 3 Shortlisted Resumes based on ATS Match Score:")
            for result in sorted_results:
                st.write(f"Filename: {result['Filename']}")
                st.write(f"Vector Match Score: {result['Vector Match Score']:.2f}%")
                st.write(f"AI Match Score: {result['AI Match Score']}")
                st.checkbox(f"{result['Filename']} - Interviewed", key=f"{result['Filename']}_checkbox")

