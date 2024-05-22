import streamlit as st
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Function to load user-uploaded dataset
@st.cache
def load_data(file):
    df = pd.read_csv(file)
    return df

# Sidebar for user inputs
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

    # Sidebar for user inputs
    st.sidebar.header('Choose Classifier')
    classifier_name = st.sidebar.selectbox('Classifier', ('Random Forest', 'Logistic Regression', 'SVM'))

    if classifier_name == 'Random Forest':
        n_estimators = st.sidebar.number_input('Number of estimators', 100, 5000, step=10)
        classifier = RandomForestClassifier(n_estimators=n_estimators)
    elif classifier_name == 'Logistic Regression':
        classifier = LogisticRegression()
    elif classifier_name == 'SVM':
        classifier = SVC()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f'Accuracy without ensemble: {accuracy}')

    # Now train an ensemble model with the selected classifier
    ensemble = BaggingClassifier(base_estimator=classifier, n_estimators=10, random_state=0)
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)

    st.write(f'Accuracy with ensemble: {accuracy_ensemble}')
