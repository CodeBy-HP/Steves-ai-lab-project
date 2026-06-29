import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from utils import search_chroma


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

st.title("Medical Chatbot")

question = st.text_input("Ask a question")

if question:
    docs = search_chroma(question)
    
    context = "\n\n".join(docs)

    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    st.write(response.text)

    with st.expander("Retrieved documents"):
        for doc in docs:
            st.write(doc)
