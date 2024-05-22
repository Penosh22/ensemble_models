import streamlit as st
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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
    st.write(df.head())

    # Sidebar for user inputs
    st.sidebar.header('Choose Target Column')
    target_column = st.sidebar.selectbox('Target Column', df.columns)

    # One-hot encode the feature columns
    features = df.drop(target_column, axis=1)
    features = pd.get_dummies(features)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, df[target_column], test_size=0.2, random_state=42)

    # Sidebar for user inputs
    st.sidebar.header('Choose Classifier')
    classifier_name = st.sidebar.selectbox('Classifier', ('Decision Tree', 'Logistic Regression', 'KNN'))

    if classifier_name == 'Decision Tree':
        classifier = DecisionTreeClassifier()
    elif classifier_name == 'Logistic Regression':
        classifier = LogisticRegression()
    elif classifier_name == 'KNN':
        classifier = KNeighborsClassifier()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f'Accuracy without ensemble: {accuracy}')

    # Now train an ensemble model with the selected classifier
    n_estimators = st.sidebar.number_input('Number of estimators', 100, 5000, step=10)
    ensemble = BaggingClassifier(base_estimator=classifier, n_estimators=n_estimators, random_state=0)
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)

    st.write(f'Accuracy with ensemble: {accuracy_ensemble}')
