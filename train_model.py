import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  # Import accuracy_score
import pickle

print("Starting the training process...")

# Load the dataset
try:
    df = pd.read_csv('Spam Mail.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'Spam Mail.csv' not found.")
    exit()

# Use the correct columns
X = df['text']
y = df['label_num']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the TF-IDF Vectorizer with optimization
print("Fitting the TF-IDF vectorizer with max_features=5000...")
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    max_features=5000 # The optimization
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test) # Transform test set as well

# Train the Logistic Regression model
print("Training the Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# --- NEW: Evaluate the optimized model ---
print("Evaluating the new model...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n---> New Model Accuracy: {accuracy:.4f} <---") # This is your new accuracy score!

# --- SAVE THE MODEL AND VECTORIZER ---
print("Saving the optimized model and vectorizer...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nTraining complete. New .pkl files have been created.")