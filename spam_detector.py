import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np


try:
    # This dataset uses a different encoding, so we add encoding='latin-1'
    df = pd.read_csv('Spam Mail.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'Spam Mail.csv' not found.")
    print("Please ensure the dataset file is in the correct directory.")
    exit()

# Display the first few rows of the dataframe
print("--- Data Head ---")
print(df.head())

# Display basic information about the dataset
print("\n--- Data Info ---")
df.info()


# Check for any missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())


# Separate the features (email text) and the target (spam/ham)
X = df['text']      # Use 'text' column for the email message
y = df['label_num'] # Use 'label_num' column for the 0/1 label

print("\n--- Using 'text' as features (X) and 'label_num' as target (y) ---")
print(df.head())


# Split the data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n--- Data Split ---")
print(f"Training set size: {len(X_train)} emails")
print(f"Testing set size: {len(X_test)} emails")


# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

# Fit the vectorizer on the training data and transform it
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data using the already fitted vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("\n--- Feature Extraction ---")
print(f"Shape of the training feature matrix: {X_train_tfidf.shape}")
print(f"Shape of the testing feature matrix: {X_test_tfidf.shape}")


# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_tfidf, y_train)

print("\n--- Model Training ---")
print("Logistic Regression model trained successfully.")


# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\n--- Confusion Matrix ---")
print("TN | FP")
print("FN | TP")
print(conf_matrix)


# Function to test custom email messages
def predict_email(message):
    message_tfidf = tfidf_vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    probability = model.predict_proba(message_tfidf)

    if prediction[0] == 1:
        return f"This email is predicted as SPAM with a {probability[0][1]:.2%} probability."
    else:
        return f"This email is predicted as HAM with a {probability[0][0]:.2%} probability."

# --- Test with custom examples ---
print("\n--- Testing with Custom Emails ---")
spam_example = "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/your-prize to claim now."
ham_example = "Hey, are we still on for the meeting tomorrow at 2 PM? Let me know."

print(f"Email 1: '{spam_example}'")
print(f"Prediction: {predict_email(spam_example)}\n")

print(f"Email 2: '{ham_example}'")
print(f"Prediction: {predict_email(ham_example)}")