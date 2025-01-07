import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

# Get GEMINI API key
google_api_key = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=google_api_key)

# Initialize Sentence Transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load datasets
df1 = pd.read_csv("Machine_failure.csv")
df2 = pd.read_csv("Reason.csv")

# Preprocessing: Label Encoding for Type column
le = LabelEncoder()
df1['Type'] = le.fit_transform(df1['Type'])
df2['Type'] = le.transform(df2['Type'])

type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Streamlit app configuration
st.set_page_config(page_title="Skavch Predictive Maintenance Engine", page_icon="📊", layout="wide")

# Add image to the header
st.image("bg1.jpg", use_column_width=True)

st.title("Skavch Predictive Maintenance Engine")

# Input Form
st.header("Enter Machine Parameters")
type_input_raw = st.selectbox('Type', ['L', 'M', 'H'])
type_input = type_mapping[type_input_raw]
air_temp_input = st.number_input('Air temperature [K]')
process_temp_input = st.number_input('Process temperature [K]')
rot_speed_input = st.number_input('Rotational speed [rpm]')
torque_input = st.number_input('Torque [Nm]')
tool_wear_input = st.number_input('Tool wear [min]')

# Function to extract and chunk text from PDF files
def extract_and_chunk_pdfs(pdf_files):
    chunks = []
    for pdf_file in pdf_files:
        with open(pdf_file, "r", encoding="utf-8") as f:
            text = f.read()
        chunk_size = 500
        overlap = 50
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
    return chunks

# Load and preprocess PDF knowledge base
pdf_files = ["PWguvM6DWT.pdf"]
text_chunks = extract_and_chunk_pdfs(pdf_files)
text_embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

# Retrieval function
def retrieve_relevant_chunks(query, text_chunks, text_embeddings, top_k=3):
    query_embedding = embedding_model.encode(query)
    similarities = cosine_similarity([query_embedding], text_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [text_chunks[i] for i in top_indices]

# Prompt template
prompt_template = """
You are a machine failure consultant assisting with root cause analysis and recommendations for maintenance. Based on the provided context, analyze the failure reason and suggest actionable solutions.

Context:
{context}

Failure Reason: {failure_reason}

Provide a detailed analysis and solutions in a concise and professional tone:
"""

# Function to query generative AI
def query_generative_ai(context, failure_reason):
    model = "models/chat-bison-001"
    final_prompt = prompt_template.format(context=context, failure_reason=failure_reason)
    response = genai.chat(
        model=model,
        messages=[{"content": final_prompt}],
        temperature=0.3
    )
    return response["candidates"][0]["content"]

# Predict Machine Failure
if st.button('Predict Machine Failure'):
    input_data = [[type_input, air_temp_input, process_temp_input, rot_speed_input, torque_input, tool_wear_input]]
    X = df1.drop('Machine failure', axis=1)
    y = df1['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model1 = RandomForestClassifier()
    model1.fit(X_train, y_train)

    failure_prob = model1.predict_proba(input_data)[0][1]
    machine_failure_pred = 1 if failure_prob >= 0.5 else 0

    if machine_failure_pred == 0:
        st.success(f"No Failure detected with probability of {1 - failure_prob:.2f}")
    else:
        st.error(f"Failure detected with probability of {failure_prob:.2f}")

        X2 = df2.drop('reason', axis=1)
        y2 = df2['reason']

        model2 = RandomForestClassifier()
        model2.fit(X2, y2)

        reason_pred = model2.predict(input_data)[0]

        # Retrieve relevant knowledge
        query = f"Failure reason: {reason_pred}. Provide root cause analysis and maintenance solutions."
        relevant_chunks = retrieve_relevant_chunks(query, text_chunks, text_embeddings)
        context = "\n".join(relevant_chunks)

        # Generate root cause analysis and solutions
        agent_result = query_generative_ai(context, reason_pred)

        st.info(f"The reason for failure is {reason_pred}")
        st.subheader("Consultant's Root Cause Analysis")
        st.write(agent_result)
