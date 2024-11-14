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

warnings.filterwarnings("ignore")

st.set_page_config(page_title="HotelX, the Room Booking Assistant", page_icon="📍", layout="wide")

with st.sidebar :
    st.image('images/pinmap.jpg')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "HotelX"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

   st.title("Ask HotelX!")
   st.write("HotelX makes recommemdations to your next hotel booking based on city, type of hotel, price, and rating.")
   st.write("\n")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# Jeremie Diaz, AI Bot Developer")
     st.image('images/photo-me1.jpg', use_container_width=True)
     st.write("## Let's connect!")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/jandiaz/")
     st.text("Conncet with me via Kaggle: https://www.kaggle.com/jeremiediaz1/")
     st.write("\n")

# Options : Model
elif options == "HotelX":
    st.title('Ask HotelX!')
    user_question = st.text_input("What's your route risk question?")

    if st.button("Submit"):
        if user_question:
            System_Prompt = """ 

Role:
You are a hotel booking assistant AI designed to help users find suitable hotels based on their preferences. 
Your role is to understand user requirements and respond by filtering through a dataset of hotel bookings, presenting a curated selection of options that match their specified criteria. 
You should interact in a friendly and professional tone, making users feel supported and informed in their hotel search.

Instructions: 
Using the provided dataset of hotel bookings, analyze the data to answer specific questions and provide actionable insights. 
Evaluate the details from user requests regarding hotel location (city), price, and rating.
Search the dataset to find hotels that match the user's preferences. Provide a list of 2-5 suitable hotels, including key details: name and location.
Prompt users to specify additional preferences or refine their choices if necessary (e.g., adjusting budget, changing location). 
If no hotels match all criteria, suggest alternatives that closely align with their preferences. 
Use conversational prompts to keep the user engaged.
If any information is missing, prompt the user politely to provide it.

Constraints:
Only use the provided dataset for your analysis.
Ensure your responses are based on the data, without making assumptions outside the dataset.
Avoid overwhelming the user with too much information at once; keep recommendations concise and relevant.
Ensure the information presented (e.g., hotel ratings, prices) is accurate and up-to-date.
Avoid booking or payment-related actions unless specifically integrated with a booking system.

Context:
Users are looking for hotels and may not always provide complete details. They may express preferences in a variety of ways (e.g., "I need a budget hotel in Paris" or "Show me luxury hotels in Tokyo under $300"). 
Be prepared to handle diverse queries while keeping responses efficient and user-centered. Aim to make the hotel search easy, reliable, and personalized.

End Goals:
Successfully assist users in finding hotels that match their preferences and criteria.
Provide helpful, personalized responses that simplify the hotel search process.
Foster a smooth, enjoyable experience that encourages users to rely on you for hotel recommendations in the future.

"""

if openai.api_key and (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
    st.title("HotelX: Booking Assistant")
    
    city = st.text_input("Enter the city (Netherlands):")
    type = st.text_input("Enter the type of bedroom:")
    price = st.number_input("Enter price (Euros):",  value='int', min_value=449, max_value=4069, placeholder="Type amount from 449 to 4069...")
    rating = st.number_input(
    "Insert a Rating Number", value='int', min_value=1, max_value=10, placeholder="Type a number from 1 to 10..."
    )
    
    if st.button("Recommend My Hotel Room"):
        # Load the dataset and create embeddings only when the button is pressed
        dataframed = pd.read_csv('https://raw.githubusercontent.com/jaydiaz2012/AI_Hotel_Chatbot/refs/heads/main/HotelFinalDataset_final.csv')
        dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        documents = dataframed['combined'].tolist()

        # Generate embeddings for the documents
        embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
        embedding_dim = len(embeddings[0])
        embeddings_np = np.array(embeddings).astype('float32')

        # Create a FAISS index for efficient similarity search
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_np)

        user_message = f"Hello, I need a {type} hotel room in {city} for the amount {price} and a rating of {rating}. Could you recommend for me?"
        # Generate embedding for the user message
        query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
        query_embedding_np = np.array([query_embedding]).astype('float32')

        # Search for similar documents
        _, indices = index.search(query_embedding_np, 2)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = ' '.join(retrieved_docs)

        # Prepare structured prompt
        structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"

        # Call OpenAI API
        chat = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": System_Prompt}] + [{"role": "user", "content": structured_prompt}],
            temperature=0.5,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Get response
        response = chat.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the response
        st.write(response)

    # Follow-up question input
    if st.session_state.messages:
        follow_up_question = st.text_input("Ask a follow-up question:")
        if st.button("Submit Follow-Up"):
            # Append the user's follow-up question to the messages
            st.session_state.messages.append({"role": "user", "content": follow_up_question})

            # Prepare the structured prompt for the follow-up question
            follow_up_context = ' '.join([msg['content'] for msg in st.session_state.messages if msg['role'] == 'assistant'])
            follow_up_prompt = f"Context:\n{follow_up_context}\n\nQuery:\n{follow_up_question}\n\nResponse:"

            # Call OpenAI API for the follow-up question
            follow_up_chat = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": System_Prompt}] + [{"role": "user", "content": follow_up_prompt}],
                temperature=0.5,
                max_tokens=1500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Get response for the follow-up question
            follow_up_response = follow_up_chat.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": follow_up_response})

            # Display the follow-up response
            st.write(follow_up_response)
else:
    st.warning("Please enter your OpenAI API key to use the chatbot.")
