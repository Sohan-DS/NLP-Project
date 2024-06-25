# Import necessary libraries
import streamlit as st
import spacy
import pickle
from spacy import displacy
from newspaper import Article

# Load the model from the .pkl file
def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Path to the .pkl file
model_path = "C:/Users/Sohan/datd_model.pkl"

# Load the NER model
nlp = load_model(model_path)

# Function to extract NER
def ner_analyser(text):
    doc = nlp(text)
    ent_html = displacy.render(doc, style="ent", jupyter=False)
    st.markdown(ent_html, unsafe_allow_html=True)

# UI code
st.title("Disaster Tweets NER")
url = st.text_input("Enter URL")
st.write("OR")
text = st.text_area("Enter paragraph")

# Button to analyze
if st.button("Analyze"):
    # Checking if both URL and Text are entered
    if text and url:
        st.write("Please enter either URL or Text to analyze")
    # Checking if only text is entered
    elif text:
        ner_analyser(text)
    # Checking if only URL is entered
    elif url:
        # Extracting text from URL
        try:
            article = Article(url)
            article.download()
            article.parse()
            url_text = article.text
            ner_analyser(url_text)
        except Exception as e:
            st.write("Please enter a valid URL")
    # Checking if nothing is entered
    else:
        st.write("Please enter URL or Text to analyze")