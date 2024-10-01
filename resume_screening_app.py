import streamlit as st
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Load dataset
data = pd.read_csv('C:/Users/Vivek Vasani/Desktop/resume_screening_app/UpdatedResumeDataSet.csv')

# Print the DataFrame columns to verify
print("Columns in DataFrame:", data.columns.tolist())

# Preprocessing function for resumes
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Preprocess the resume data
data['Cleaned_Resume'] = data['Resume'].apply(preprocess_text)

# Vectorize the text data
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['Cleaned_Resume']).toarray()
y = data['Category']  # Make sure this matches the column name exactly

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=sorted(data['Category'].unique()))
print(report)

# Streamlit Web App
st.title("AI-based Resume Screening Tool")

st.write("Upload a resume (in plain text format) to check its suitability:")

uploaded_file = st.file_uploader("Choose a file", type="txt")

if uploaded_file is not None:
    resume_text = uploaded_file.read().decode("utf-8")
    st.write("Uploaded Resume:")
    st.write(resume_text)

    # Preprocess and vectorize the uploaded resume
    cleaned_resume = preprocess_text(resume_text)
    resume_vector = vectorizer.transform([cleaned_resume]).toarray()

    # Make prediction
    prediction = model.predict(resume_vector)
    prediction_proba = model.predict_proba(resume_vector)

    # Display the result
    if prediction[0] == 'Suitable':  # Adjust this condition based on your categories
        st.success("The resume is suitable for the job!")
    else:
        st.error("The resume is unsuitable for the job.")

    # Get the index of the predicted category
    classes = model.classes_  # Get the class labels
    prediction_index = np.where(classes == prediction[0])[0][0]  # Find index of the predicted class

    # Confidence score logic
    if 0 <= prediction_index < prediction_proba.shape[1]:
        st.write(f"Confidence: {prediction_proba[0][prediction_index] * 100:.2f}%")
    else:
        st.write("Prediction index out of bounds")
