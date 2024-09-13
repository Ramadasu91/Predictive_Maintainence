import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt


# Load your datasets df1 and df2
df1 = pd.read_csv("Machine_failure.csv")
df2 = pd.read_csv("Reason.csv")

# Preprocessing: Label Encoding for Type column
le = LabelEncoder()
df1['Type'] = le.fit_transform(df1['Type'])
df2['Type'] = le.transform(df2['Type'])

# Create a mapping for Type to encoded values
type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Streamlit App
st.set_page_config(page_title="Skavch Predictive Maintainence Engine", page_icon="📊", layout="wide")

# Add an image to the header
st.image("bg1.jpg", use_column_width=True)

st.title("Skavch Predictive Maintainence Engine")

# Input Form
st.header("Enter Machine Parameters")
type_input_raw = st.selectbox('Type', ['L', 'M', 'H'])
type_input = type_mapping[type_input_raw]
air_temp_input = st.number_input('Air temperature [K]')
process_temp_input = st.number_input('Process temperature [K]')
rot_speed_input = st.number_input('Rotational speed [rpm]')
torque_input = st.number_input('Torque [Nm]')
tool_wear_input = st.number_input('Tool wear [min]')

# Predict Machine Failure
if st.button('Predict Machine Failure'):
    input_data = [[type_input, air_temp_input, process_temp_input, rot_speed_input, torque_input, tool_wear_input]]
    X = df1.drop('Machine failure', axis=1)
    y = df1['Machine failure']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model1 = RandomForestClassifier()
    model1.fit(X_train, y_train)
    
    y_pred = model1.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    st.subheader("Model Performance Metrics for Machine Failure Prediction")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    
    # ROC curve
    y_prob = model1.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]

    # Predict Machine Failure with the optimal threshold (moved before the plot)
    failure_prob = model1.predict_proba(input_data)[0][1]
    machine_failure_pred = 1 if failure_prob >= optimal_threshold else 0

   # Plot ROC Curve with swapped axes and modified colors
    plt.figure()

    # Fill regions with updated colors across the entire y-axis without labels in the legend
    plt.axvspan(0.0, 0.68, color='lightgreen', alpha=0.3)
    plt.axvspan(0.69, 0.87, color='lightyellow', alpha=0.3)
    plt.axvspan(0.88, 1.0, color='lightcoral', alpha=0.3)


    # Directly plot the predicted failure probability as a blue dot (no threshold reference)
    plt.scatter(failure_prob, failure_prob, color='blue', marker='o', s=100, label=f'Predicted Probability = {failure_prob:.2f}')

    # Add text labels for the regions with vertical orientation
    plt.text(0.325, 0.5, 'Healthy', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightcoral', alpha=0.5), rotation=90)
    plt.text(0.765, 0.5, 'Degradation', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightyellow', alpha=0.5), rotation=90)
    plt.text(0.935, 0.5, 'Failure', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightgreen', alpha=0.5), rotation=90)

    # Configure plot limits and labels with swapped axes
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Condition')
    plt.ylabel('Probabilities')
    plt.title('Receiver Operating Characteristic (ROC) ')
    plt.legend(loc='upper left')

    # Display the ROC curve in Streamlit
    st.pyplot(plt)

    st.subheader("Reason Analysis")
    st.image("Shap_Analysis.png")


    # Machine Failure Result
    if machine_failure_pred == 0:
        st.success(f"No Failure detected with probability of {1 - failure_prob:.2f}")
    else:
        st.error(f"Failure detected with probability of {failure_prob:.2f}")
        
        X2 = df2.drop('reason', axis=1)
        y2 = df2['reason']
        
        model2 = RandomForestClassifier()
        model2.fit(X2, y2)
        
        reason_pred = model2.predict(input_data)[0]
        reason_prob = model2.predict_proba(input_data)[0][1]
        
        st.info(f"The reason for failure is {reason_pred}")