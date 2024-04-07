import streamlit as st
from langchain.chains import LLMChain
import os
from langchain.prompts import PromptTemplate
 
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a very friendly AI. Answer the question from the human
    
   
    chat_history: {chat_history}

    human: {question}
 
    AI:

    """
)

llm = ChatOpenAI(openai_api_key = st.secrets["openai_api_key"], temperature=0.1)
memory =ConversationBufferWindowMemory(memory_key="chat_history", k=5)
llm_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

st.set_page_config(
    page_title="ChatGPT test",
    layout="wide"
)

st.title("ChatGPT test")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello as a question"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response) 
    new_ai_message = {"role": "assistant", "content": ai_response} 
    st.session_state.messages.append(new_ai_message)      