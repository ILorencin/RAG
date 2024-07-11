import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Set the OpenAI API key
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o"

# Function to summarize email content
def summarize_email(email_content):
    response = client.chat.completions.create(
            model=MODEL,
                messages=[
                    {"role": "system", "content": """
                    You are generating a transcript summary. Create a summary of the provided transcription. Please recognize from original language and provide summary in English.
                    """},
                    {"role": "user", "content": f"The audio transcription is: {email_content}"},
                ],
                temperature=0.1,
            )

    summary = response.choices[0].message.content
    return(summary)

# Function to list all PDF files in a folder
def list_pdfs_in_folder(folder_path):
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
    return pdf_files

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_paths):
    pdf_texts = []
    for path in pdf_paths:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        pdf_texts.append(text)
    return pdf_texts

# Function to retrieve relevant sections from PDFs based on the summary
def retrieve_relevant_sections(summary, pdf_texts):
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode the documents and the summary
    doc_vectors = model.encode(pdf_texts)
    summary_vector = model.encode([summary])[0]
    
    # Build the FAISS index
    dimension = doc_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance metric
    index.add(doc_vectors)
    
    # Search for the most relevant sections
    k = 5  # Number of top relevant sections to retrieve
    distances, indices = index.search(summary_vector.reshape(1, -1), k)
    
    # Get the relevant sections
    relevant_sections = [pdf_texts[i] for i in indices[0]]
    return relevant_sections

# Function to generate recommendations based on the summary and relevant sections
def generate_recommendations(summary, relevant_sections):
    prompt = f"Based on the following email summary, recommend a ski trip:\n\nSummary: {summary}\n\nRelevant information:\n"
    for section in relevant_sections:
        prompt += f"{section}\n\n"
    prompt += "Recommendation:"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
                    {"role": "system", "content": """
                    You are agent in a travel company. Plaese provide contact info at the end of every message. Create detailed reccomendation on croatian language.
                    """},
                    {"role": "user", "content": f"The audio transcription is: {prompt}"},
                ],
                temperature=0.1,
        
        max_tokens=3000
    )
    recommendation = response.choices[0].message.content
    return recommendation

# Streamlit app
def main():

    # Apply custom CSS for branding
    st.markdown(
        """
        <style>
            .main {
                background-color: #022b65;
                color: #fffefe;
            }
            .stButton>button {
                background-color: #008CBA;
                color: #fffefe;
            }
            .stFileUploader {
                color: #008CBA;
            }
            .stTextInput {
                background-color: #022b65;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set the title and logo
    st.image("image.png", width=300)

    st.title("AI asistent za preporuku skijanja")
    
    email_content = st.text_area("Bok, ja sam asistent za preporuku skijanja. Molim Vas unesite svoj upit.", height=200)
    
    if st.button("Generaj preporuku"):
        if not email_content:
            st.error("Please enter email content.")
        else:
            with st.spinner("Generiramo preporuku..."):
                # Step 1: Summarize the email content
                email_summary = summarize_email(email_content)
                
                # Step 2: List all PDF files in the folder
                pdf_folder_path = "path"
                pdf_paths = list_pdfs_in_folder(pdf_folder_path)
                
                # Step 3: Extract text from PDFs
                pdf_texts = extract_text_from_pdfs(pdf_paths)
                
                # Step 4: Retrieve relevant sections from the PDFs based on the summary
                relevant_sections = retrieve_relevant_sections(email_summary, pdf_texts)
                
                # Step 5: Generate recommendations based on the summary and relevant sections
                recommendation = generate_recommendations(email_summary, relevant_sections)
                
                #st.subheader("Email Summary")
                #st.write(email_summary)
                
                st.subheader("Moja preporuka:")
                st.write(recommendation)


               
if __name__ == "__main__":
    main()
    st.markdown(
                    """
                    <footer>
                    <div style="padding: 10px; text-align: center; color: #008CBA;">
                        <p>Powered by FIPU</p>
                        <a href="https://web" style="color: #008CBA; text-decoration: none;">Visit our website</a>
                    </div>
                    </footer>
                    """,
                    unsafe_allow_html=True
)
