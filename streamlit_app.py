import streamlit as st
import random
from perplexity import Perplexity, classify_perplexity
from transformers import BertTokenizerFast, GPT2LMHeadModel

from content import *

st.set_page_config(layout="centered")

p = Perplexity()

@st.experimental_singleton
def load_model():
    # code to load the model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = GPT2LMHeadModel.from_pretrained("ckiplab/gpt2-base-chinese")
    return tokenizer, model

tokenizer, model = load_model()

p.set_model(tokenizer, model)

random_list = [m1, m2, m3, m4, m5, m6, h1, h2, h3, h4, h5, h6, h7]
MAX_CHARS = 2000


st.title("ChatGPT Chinese Text Detector")
st.markdown("\n")

st.subheader("A tool for detecting machine-generated Chinese text")
st.markdown("\n")


text_area = st.empty()
col1, col2, col3, col4, col5  = st.columns(5)


text = text_area.text_area("Enter your Chinese text here:", \
    max_chars=MAX_CHARS, height=450)

if col3.button("Random Text Result"):
    random_text = "".join(random.choices(random_list))
    text_area.text_area("Enter your Chinese text here:", random_text, \
        max_chars=MAX_CHARS, height=450)
    if random_text:
        score = p.calculate(random_text)
        if score:
            col4.metric("Perplexity Score", score, "")
            result = classify_perplexity(score)
            if result == "AI":
                st.warning("This text is most likely written by AI (ChatGPT)", icon="⚠️")
            elif result == "Human":
                st.success("This text is most likely written by Human", icon="🐵")
            else:
                st.warning("It is hard to tell if this text is written by a human or AI", icon="❓")
        else:
            st.warning("Please try again")
    else:
        st.warning("Please add text above")

if col3.button('Get Result'):
    if text:
        if len(text) < 10:
            st.error("Text is too short. Please input more than 10 characters.")
        if len(text) > MAX_CHARS:
            st.error("Text is too long. Please input less than 2000 characters.")

        score = p.calculate(text)
        if score:
            col4.metric("Perplexity Score", score, "")
            result = classify_perplexity(score)
            if result == "AI":
                st.warning("This text is most likely written by AI (ChatGPT)", icon="⚠️")
            elif result == "Human":
                st.success("This text is most likely written by Human", icon="🐵")
            else:
                st.warning("It is hard to tell if this text is written by a human or AI", icon="❓")
        else:
            st.warning("Please try again")
    else:
        st.warning("Please add text above")

st.sidebar.header("Auther: ")
st.sidebar.subheader("RUI-LONG CHENG 鄭瑞龍")
st.sidebar.markdown("\n")
st.sidebar.header("Contact Information: ")
st.sidebar.markdown("Email: m61216051116@gmail.com")
st.sidebar.markdown("Follow me on [GitHub](https://github.com/RUI-LONG) or [Facebook](https://www.facebook.com/ruilongz/)")

st.sidebar.markdown("\n")
st.sidebar.markdown("\n")
st.sidebar.header("Refference: ")
st.sidebar.markdown("- [GPT Zreo](https://etedward-gptzero-main-zqgfwb.streamlit.app/)")
st.sidebar.markdown("- [OpenAI](https://openai-openai-detector.hf.space/)")