import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from crewai import Agent, Task, Crew, Process, LLM, Knowledge
from crewai.knowledge.source import PDFKnowledgeSource
import os

# Get GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

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

# Initialize the agent
pdf_source = PDFKnowledgeSource(file_paths=["machine_failures.pdf", "maintenance_guidelines.pdf"])
knowledge = Knowledge(collection_name="machine_failure_knowledge", sources=[pdf_source])
gemini_llm = LLM(model="gemini/gemini-1.5-pro-002", api_key=GEMINI_API_KEY, temperature=0)

agent = Agent(
    role="Machine Failure Consultant",
    goal="Provide root cause analysis and solution recommendations for machine failures.",
    backstory="You are a highly skilled consultant specializing in diagnosing and solving machine failures based on expert knowledge.",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm,
)

task = Task(
    description="Analyze the provided failure reason and recommend a root cause analysis and solutions: {failure_reason}",
    expected_output="A detailed root cause analysis and possible solutions.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[pdf_source],
)

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

        # Use the agent for root cause analysis
        failure_reason_text = f"Considering the failure reason '{reason_pred}', provide a root cause analysis to fix it and make the machine healthier."
        agent_result = crew.kickoff(inputs={"failure_reason": failure_reason_text})
        

        st.info(f"The reason for failure is {reason_pred}")
        st.subheader("Consultant's Root Cause Analysis")
        st.write(agent_result)
