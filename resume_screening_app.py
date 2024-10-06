import streamlit as st
import pandas as pd
import numpy as np
import spacy
import PyPDF2  # Added library for PDF processing
import re  # For regular expressions
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

st.write("Upload a resume (in plain text or PDF format) to check its suitability:")

# Updated to accept both txt and pdf formats
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to extract skills from the resume text
def extract_skills(text):
    # Define a list of common skills
    skill_keywords = [
        'Python', 'Java', 'C++', 'SQL', 'JavaScript', 'HTML', 'CSS', 'Machine Learning', 
        'Data Science', 'Deep Learning', 'NLP', 'Django', 'Flask', 'React', 'Node.js', 'Git', 'AWS',
        'Docker', 'Kubernetes', 'TensorFlow', 'PyTorch', 'Excel', 'Tableau', 'Power BI', 'Figma'
    ]
    
    # Use regex to find skill keywords in the text (case insensitive)
    found_skills = []
    for skill in skill_keywords:
        if re.search(r'\b' + skill + r'\b', text, re.IGNORECASE):
            found_skills.append(skill)
    
    return found_skills

if uploaded_file is not None:
    # Check if the uploaded file is a PDF or TXT
    if uploaded_file.name.endswith(".txt"):
        resume_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    
    #st.write("Uploaded Resume Text:")
    #st.write(resume_text)

    # Extract skills from the resume
    skills = extract_skills(resume_text)
    st.write("Extracted Skills:")
    st.write(", ".join(skills) if skills else "No skills found.")

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
