# Spamurai 🗡️

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent spam detection tool that analyzes both raw text and the content of live web pages to classify them as spam or legitimate (ham). Built with Python, Scikit-learn, and a user-friendly Streamlit web interface.

---

## Live Demo - https://spamurai--spam-email-detector.streamlit.app/

---

## Key Features

-   **Dual Input Mode:** Analyzes both pasted text and live URLs.
-   **Live Web Scraping:** Uses `requests` and `BeautifulSoup` to fetch and parse web content in real-time.
-   **Efficient Machine Learning Model:** The `TfidfVectorizer` is optimized to use the top 5000 features, reducing memory usage and improving prediction speed.
-   **Performance Caching:** Implements Streamlit's `@st.cache_data` to instantly return results for previously analyzed URLs, avoiding redundant web requests.
-   **Interactive Data Visualization:** Uses `Altair` to generate charts that explain *why* a message was classified as spam by showing the most influential words.
-   **User Feedback Loop:** Allows users to report incorrect predictions. This feedback is logged to a `feedback_log.csv`, demonstrating a key concept in building iterative MLOps pipelines.
-   **Polished UI/UX:** A clean, modern interface with a custom sidebar, spinners for loading states, and custom result boxes.

---

## Technologies Used

-   **Backend & ML:** Python, Scikit-learn, Pandas, NumPy
-   **Web Framework:** Streamlit
-   **Web Scraping:** Requests, BeautifulSoup4, lxml
-   **Data Visualization:** Altair

---

## How It Works

The project is split into two main components:

1.  **`train_model.py`**: A one-time script that performs the following offline tasks:
    -   Loads the "Spam Mail" dataset.
    -   Splits the data into training and testing sets.
    -   Creates a `TfidfVectorizer` (optimized to 5000 features) and fits it on the training data.
    -   Trains a `LogisticRegression` classifier.
    -   Saves the trained vectorizer and model as `.pkl` files for use in the live app.

2.  **`app.py`**: The main Streamlit application script that:
    -   Loads the pre-trained `vectorizer.pkl` and `model.pkl`.
    -   Provides the web interface with text areas and buttons.
    -   Detects whether the user input is text or a URL.
    -   If it's a URL, it scrapes the web page content.
    -   Transforms the input text using the loaded vectorizer.
    -   Makes a prediction using the loaded model.
    -   Displays the result, confidence score, and visual explanations.

---

## Setup and Installation

To run this project locally, follow these steps:

**1. Clone the repository:**
git clone https://github.com/sarim-aliii/Spamurai--Spam-Email-Detector.git
cd Spamurai--Spam-Email-Detector

2. Create and activate a virtual environment (recommended):

**For macOS/Linux**
python3 -m venv venv
source venv/bin/activate

**For Windows**
python -m venv venv
.\venv\Scripts\activate


**3. Install the required dependencies:**
pip install -r requirements.txt


**4. Train the model:**
(You only need to do this once to generate the .pkl files)
python train_model.py


**5. Run the Streamlit app:**
streamlit run app.py

The application should now be open and running in your web browser!


**Future Improvements**
Database Integration: Replace the feedback_log.csv with a connection to a SQL or NoSQL database for more robust, persistent feedback storage.
Advanced Models: Experiment with more complex models like Naive Bayes, RandomForestClassifier, or even deep learning models like LSTMs for potentially higher accuracy.
Unit & Integration Tests: Add a testing suite (pytest) to ensure the reliability of individual functions and the overall application flow.


**License**
This project is licensed under the MIT License - see the LICENSE file for details.

Developed by Sarim Ali.