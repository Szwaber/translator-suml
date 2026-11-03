import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    mod = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de", device_map="auto")
    return tok, mod

with st.spinner("Loading model..."):
    tokenizer, model = load_model()

col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.image("united-states.png", width=60)
with col2:
    st.title("English to German Translator")
with col3:
    st.image("germany.png", width=60)


st.write("Welcome! This app is used for translating texts. Input desired text in english, and press translate, the "
         "application will translate it into German")

text = st.text_area(label="Input text in English")

button = st.button("Translate")

if button:
    if not text.strip():
        st.warning("Please enter some text to translate")
    else:
        with st.spinner(text='Translating...'):
            time.sleep(2)
            try:
                input_ids = tokenizer(text, return_tensors="pt").to(model.device)
                output = model.generate(**input_ids, max_length=512)
                translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

                st.text("Translated text:")
                st.success(translated_text)

            except Exception as e:
                st.error(f"An error occurred during translation: {e}")


st.write("aplication created by: s30380")
    


