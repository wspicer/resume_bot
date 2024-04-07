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
# Specify the tone file path
file_path_tone = 'spice_messages.txt'

# Specify the context file path
file_path_resume = 'will-resume-ai.txt'

tone_data = st.file_uploader(file_path_tone)
resume_data = st.file_uploader(file_path_resume)

# Read the contents of the tone file into a string
#with open(file_path_tone,  'r') as tone_file:
#    tone_data = tone_file.read()

# Read the contents of the context file into a string
#with open(file_path_resume, 'r') as resume_file:
#    resume_data = resume_file.read()


prompt = PromptTemplate(
    input_variables=["chat_history", "question", "context", "tone"],
    template = """Respond to this: {question} based only on the provided context:
    
    chat_history: {chat_history}

    <context>
    {context}
    </context>
    
    And answer in the tone of the person who sent these messages:

    START OF TONE DATA
    {tone}
    END OF TONE DATA
    
    AI:
    """
)

llm = ChatOpenAI(openai_api_key = st.secrets["openai_api_key"], temperature=0.1)
#memory =ConversationBufferWindowMemory(memory_key="chat_history", k=5)
#llm_chain = LLMChain(
#    llm=llm,
#    memory=memory,
#    prompt=prompt
#)

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

final_prompt = prompt.format(question=user_prompt, tone=tone_data, context=resume_data, chat_history="")

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm.predict(final_prompt) 
            st.write(ai_response) 
    new_ai_message = {"role": "assistant", "content": ai_response} 
    st.session_state.messages.append(new_ai_message)      