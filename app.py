import streamlit as st
import pickle
import pandas as pd
import altair as alt
import requests
from bs4 import BeautifulSoup
import re
import os



st.set_page_config(page_title="Spamurai", page_icon="🗡️")
st.sidebar.title("About Spamurai")
st.sidebar.info(
    """
    This project is an intelligent spam detector that analyzes both raw text
    and the content of live web pages to classify them as spam or legitimate (ham).
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("Key Features & Optimizations")
st.sidebar.markdown(
    """
    - **Dual Input Mode:** Analyzes both pasted text and live URLs.
    - **Web Scraping:** Uses `requests` and `BeautifulSoup` to fetch and parse web content.
    - **Efficient Model:** The text vectorizer is optimized to use the top 5000 features, reducing memory and improving speed.
    - **Caching:** Implements `@st.cache_data` to instantly return results for previously analyzed URLs.
    - **Interactive Visuals:** Uses `Altair` to chart the words that most influence a "spam" classification.
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")
st.sidebar.success(f"""**Model:** Logistic Regression""")
st.sidebar.success(f"""**Accuracy:** ~{0.9758 * 100:.1f}% (on test data)""")
st.sidebar.markdown("---")
st.sidebar.markdown("View the source code on [GitHub](https://github.com/sarim-aliii/Spamurai--Spam-Email-Detector)")



try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading vectorizer.pkl: {e}")
    st.info("Please ensure the file exists and is not corrupt. Try running train_model.py again.")
    st.stop()

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.info("Please ensure the file exists and is not corrupt. Try running train_model.py again.")
    st.stop()


def is_url(text):
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.match(text) is not None

@st.cache_data(show_spinner=False)
def get_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None

def log_feedback(feedback, text, prediction):
    try:
        file_exists = os.path.exists('feedback_log.csv')
        new_entry = pd.DataFrame([{'text': text, 'prediction': prediction, 'user_feedback': feedback}])
        new_entry.to_csv('feedback_log.csv', mode='a', header=not file_exists, index=False)
        st.toast("Thank you for your feedback!", icon="🙏")
    except Exception as e:
        st.error(f"Could not log feedback: {e}")

def result_box(text, box_type='success'):
    if box_type == 'success':
        border_color = "#28a745"
        icon = "✔️"
    else:
        border_color = "#dc3545"
        icon = "❌"
    box_html = f"""<div style="border: 2px solid {border_color}; border-radius: 5px; padding: 15px; margin-top: 20px; background-color: #262730;"><h4 style="color: white; margin: 0;">{icon} Result: {text}</h4></div>"""
    st.markdown(box_html, unsafe_allow_html=True)


def set_example_text(text):
    st.session_state.user_input = text

def clear_text():
    st.session_state.user_input = ""
    st.session_state.last_result = None


st.markdown("<h1 style='text-align: center;'>Spamurai 🗡️</h1>", unsafe_allow_html=True)
st.write("Enter a message or link below, or try one of our examples.")
spam_msg = "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/your-prize to claim now."
ham_msg = "Hey, are we still on for the meeting tomorrow at 2 PM? Let me know."
spam_link = "https://en.wikipedia.org/wiki/Email_spam"
ham_link = "https://docs.python.org/3/tutorial/index.html"
col1, col2 = st.columns(2)
with col1:
    st.button("Test a Spam Message", on_click=set_example_text, args=(spam_msg,), use_container_width=True)
with col2:
    st.button("Test a Ham Message", on_click=set_example_text, args=(ham_msg,), use_container_width=True)
col3, col4 = st.columns(2)
with col3:
    st.button("Test a Spam Link", on_click=set_example_text, args=(spam_link,), use_container_width=True)
with col4:
    st.button("Test a Ham Link", on_click=set_example_text, args=(ham_link,), use_container_width=True)
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
user_input = st.text_area("Enter text or link:", height=150, key="user_input")


check_button_col, clear_button_col = st.columns(2)
with check_button_col:
    if st.button("Check", use_container_width=True):
        if user_input:
            content_to_analyze = ""
            if is_url(user_input):
                with st.spinner('Spamurai is fetching content from the link...'):
                    content_to_analyze = get_text_from_url(user_input)
            else:
                content_to_analyze = user_input

            if content_to_analyze:
                with st.spinner('The Spamurai is analyzing the content...'):
                    input_tfidf = tfidf_vectorizer.transform([content_to_analyze])
                    prediction = model.predict(input_tfidf)[0]
                    probability = model.predict_proba(input_tfidf)[0]
                    
                    st.session_state.last_result = {'text': content_to_analyze, 'prediction': prediction}
                    
                    if prediction == 1:
                        result_box("This looks like SPAM!", box_type='error')
                        st.write("")
                        with st.expander("See details"):
                            st.write("Confidence Score:")
                            st.progress(probability[1])

                            feature_names = tfidf_vectorizer.get_feature_names_out()

                            coefficients = model.coef_[0]
                            coef_df = pd.DataFrame({'word': feature_names, 'spam_score': coefficients})

                            words_in_email = [word for word in coef_df['word'] if word in content_to_analyze.lower()]
                            top_spam_words = coef_df[coef_df['word'].isin(words_in_email)].sort_values('spam_score', ascending=False).head(5)

                            if not top_spam_words.empty:
                                st.warning("Top words that contributed to this classification:")
                                chart = alt.Chart(top_spam_words).mark_bar().encode(x=alt.X('spam_score:Q', title='Spam Contribution Score'),y=alt.Y('word:N', sort='-x', title='Word'),tooltip=['word', 'spam_score']).properties(title="Most Influential Spam Words")
                                st.altair_chart(chart, use_container_width=True)
                    else:
                        result_box("This looks like a legitimate email (HAM).", box_type='success')
                        st.write("")
                        with st.expander("See details"):
                            st.write("Confidence Score:")
                            st.progress(probability[0])
        else:
            st.warning("Please enter text or a URL to check.")

with clear_button_col:
    st.button("Clear", on_click=clear_text, use_container_width=True)


if 'last_result' in st.session_state and st.session_state.last_result:
    st.markdown("---")
    st.write("Was this prediction helpful?")
    feedback_col1, feedback_col2 = st.columns(2)
    with feedback_col1:
        st.button("✔️ Correct", on_click=log_feedback, args=('correct', st.session_state.last_result['text'], st.session_state.last_result['prediction']), use_container_width=True)
    with feedback_col2:
        st.button("❌ Incorrect", on_click=log_feedback, args=('incorrect', st.session_state.last_result['text'], st.session_state.last_result['prediction']), use_container_width=True)