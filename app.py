import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.chat_models import ChatGooglePalm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Get GEMINI API key
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Initialize ChatGooglePalm model
chat_model = ChatGooglePalm(model="models/chat-bison-001", api_key=google_api_key)

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

# Load and preprocess PDF knowledge base
loader = PyPDFLoader(["PWguvM6DWT.pdf"])
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = text_splitter.split_documents(documents)

# Create FAISS VectorStore for knowledge retrieval
vectorstore = FAISS.from_documents(docs_split, embedding_model)
retriever = vectorstore.as_retriever()

# Prompt template
prompt_template = PromptTemplate(
    template="""
    You are a machine failure consultant assisting with root cause analysis and recommendations for maintenance. 
    Based on the provided context, analyze the failure reason and suggest actionable solutions.

    Context:
    {context}

    Failure Reason: {failure_reason}

    Provide a detailed analysis and solutions in a concise and professional tone:
    """,
    input_variables=["context", "failure_reason"]
)

# Initialize Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever, prompt=prompt_template)

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

        # Use Retrieval QA chain for root cause analysis
        query = f"Failure reason: {reason_pred}. Provide root cause analysis and maintenance solutions."
        agent_result = qa_chain.run({"failure_reason": reason_pred, "context": retriever.retrieve(query)})

        st.info(f"The reason for failure is {reason_pred}")
        st.subheader("Consultant's Root Cause Analysis")
        st.write(agent_result)
