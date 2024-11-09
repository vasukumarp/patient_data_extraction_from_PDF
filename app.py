import streamlit as st
import PyPDF2
from groq import Groq
from pymongo import MongoClient
import json
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq and MongoDB clients
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["Patient_Data"]
collection = db["insurance"]

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() for page in reader.pages)

def extract_data_with_groq(text):
    prompt = """
    Extract key information from this PDF text about patient payments and insurance details.
    Format the extracted information as a JSON object. Include fields like patient name, 
    patient ID, payment amount, payment date, insurance provider, claim number, etc. 
    Return only the JSON object.
    
    Text: {text}
    """
    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": prompt.format(text=text)}],
        model="llama-3.1-70b-versatile"
    )
    return response.choices[0].message.content

def extract_json_from_text(text):
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return None

st.title("Patient Data Extractor")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and st.button("Extract Data"):
    with st.spinner("Processing..."):
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        # Extract structured data using Groq
        extracted_text = extract_data_with_groq(pdf_text)
        
        # Parse JSON from extracted text
        data = extract_json_from_text(extracted_text)
        
        if data:
            # Store in MongoDB
            result = collection.insert_one(data)
            st.success(f"Data stored in MongoDB with ID: {result.inserted_id}")
            
            # Display extracted data
            st.subheader("Extracted Data")
            st.json(data)
        else:
            st.error("Failed to extract structured data. Please try again.")

st.markdown("---")
st.write("Powered by Streamlit, Groq, and MongoDB")