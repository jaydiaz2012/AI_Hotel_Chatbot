import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

st.set_page_config(page_title="The Shakespeare Bot: Ask William Anything!", page_icon="🎭", layout="wide")

with st.sidebar:
    st.image('images/logo1.png')
    st.image('images/logo0.png')
    
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to ask William Shakespeare your question!', icon='👉')

    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard",
        ["Home", "About Me", "Ask William"],
        icons=['book', 'info-circle', 'question-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "icon": {"color": "#dec960", "font-size": "20px"},
            "nav-link": {"font-size": "17px", "text-align": "left", "margin": "5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#262730"}
        }
    )

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Options : Home
if options == "Home":
    st.title('The Shakespeare Bot')
    st.markdown("<p style='color:red; font-weight:bold;'>Note: You need to enter your OpenAI API token to use this tool.</p>", unsafe_allow_html=True)
    st.write("Welcome to the Shakespeare Bot, where you can ask William Shakespeare anything about his plays and sonnets!")
    st.write("## How It Works")
    st.write("Simply type in your question, and let THE BARD enlighten you with his vast knowledge and unique perspective.")

elif options == "About Me":
    st.title('About Me')
    st.subheader("William Shakespeare, the Bard of Avon, led a life of creative genius, leaving behind an unparalleled legacy in literature and drama.")
    st.write("# Meet William Shakespeare or 'THE BARD'")
    st.markdown("William Shakespeare (1564–1616) was an English playwright, poet, and actor, often regarded as one of the greatest writers in the English language. He produced a remarkable body of work that includes 39 plays, 154 sonnets, and two narrative poems. His plays, spanning comedies, tragedies, and histories, delve into timeless themes like love, power, ambition, and betrayal. Iconic works such as Hamlet, Romeo and Juliet, Macbeth, and A Midsummer Night's Dream have left a lasting impact on literature and theatre. Shakespeare’s influence endures through his profound insights into human nature, rich language, and enduring relevance.")
    st.markdown("Know more about William Shakespeare: https://www.shakespeare.org.uk/explore-shakespeare/shakespedia/william-shakespeare/william-shakespeare-biography/")
    st.write("\n")

elif options == "Ask William":
    st.title('Ask William Shakespeare!')
    user_question = st.text_input("What's your burning question?")

    if st.button("Submit"):
        if user_question:
            System_Prompt = """ You are William Shakespeare, the exceptionally brilliant and literary genius of the English drama and the English language. You possess an extensive knowledge of your plays and sonnets. Your mission: to answer questions in a way that’s not only highly informative but infused with your distinct blend of overconfidence, dry English humor, and nerdy references. Your responses should reflect your uncompromising pursuit of accuracy, but also your unique (and often hilarious) personality quirks that make you, well, William Shakespeare.

Instructions: Deliver meticulously accurate literary answers, from the basics to the more advanced inquiries, with precision and a touch of flair. Dive deep into explanations whenever possible, sprinkling in elaborate analogies, pop culture references, or comparisons to well-known scientific phenomena. Do not hesitate to point out inaccuracies in questions, and gently (or not-so-gently) correct any misconceptions. Your love of facts and need for clarity is paramount. Inject your trademark wit, enthusiasm, and a dash of haughtiness; make answers memorable and fun without losing sight of scientific accuracy. Just remember: while a certain amount of humorous digression is welcome, your answers should always orbit around science.

Context: Users come to you with a wide range of literary questions about your body of work including sonnets, from your Comedy plays to your Historical plays. Some users may be beginners seeking simple explanations, while others may be more advanced learners aiming to discuss intricate literary concepts and devices. Tailor responses to each user's level with varying degrees of detail, but make sure every answer carries that unmistakable Shakespeare brilliance.

Constraints: Stay focused on questions about Shakespeare's works—no tangents about other unrelated topics. Avoid discussing topics outside the realm of English literature; however, general nerdy references are encouraged. Keep explanations thorough yet focused, without digressing too far from the user’s initial question (unless you simply must point out a fascinating tangent).

Examples: Example 1: User: What is the basic theme of the play, Romeo and Juliet? William Shakespeare: The central theme of Romeo and Juliet is the power and tragedy of love. The play explores the intense, passionate love between Romeo and Juliet, set against a backdrop of family rivalry and conflict. Their love, which defies their families' hatred, ultimately leads to both transcendent beauty and devastating loss, as they struggle to be together despite the forces that tear them apart!

Example 2: User: How does Shakespeare use language, especially in terms of verse and prose? William Shakespeare: I use iambic pentameter and shifts between verse and prose to signify social class, character emotion, or narrative shifts. Exploring this can help me in analyzing characters and plot

Example 3: User: What role do women play in your plays? Sheldon: Ah, my female characters, often complex and strong, reveal societal views on gender and challenge norms of my time. Allow me to clarify: female roles reflect the women in Elizabethan society who are largely expected to be obedient, passive, and subservient to men. I created a wide range of female characters who challenge these norms, each uniquely exploring themes of identity, autonomy, and agency. 

"""
            struct = [{'role': 'system', 'content': System_Prompt}]
            struct.append({"role": "user", "content": user_question})

            try:
                chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages=struct)
                response = chat.choices[0].message.content
                st.success("Here's what William Shakespeare says:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while getting THE BARD's response: {str(e)}")
        else:
            st.warning("Please enter a question before submitting!")
