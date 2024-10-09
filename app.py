import streamlit as st
import os
import chardet
from io import StringIO
from ResearchBot.utils.common import *
from ResearchBot.variables.configs import Configs
from ResearchBot.components.response_synthesis import ResponseSynthesis
from ResearchBot.components.data_ingestion import DataIngestion

st.set_page_config(page_title="Research RAG and LLM - GPT4",
                    page_icon='‍🎓'
                    # layout='centered',
                    # initial_sidebar_state='collapsed')
)


def embed_data(papers):
    with st.spinner(text="Fetching the data"):
        data_ingestion = DataIngestion(papers=papers)
        data_ingestion.main()


@st.cache_resource(show_spinner=False)
def load_data_training():
    if not os.listdir(Configs.articles_dir) :
        st.write("No articles are uploaded. Please upload documents")
        
    with st.spinner(text="Loading and indexing the research docs – hang tight! This should take 1-2 minutes."):
        response_generator = ResponseSynthesis()
        return response_generator
######################################################
st.title("Research RAG and LLM - GPT4 🧑‍🎓")
######################################################
# If a file is uploaded, create embeddings
uploaded_file_training = st.file_uploader("Upload Input file training")
if uploaded_file_training or not os.path.exists(Configs.articles_dir):
    if uploaded_file_training:
        stringio = StringIO(uploaded_file_training.getvalue().decode("utf-8", errors="ignore"))
        papers = str(stringio.read())
        embed_data(papers=papers)
    else :
        embed_data(papers=Configs.papers)

    print('Will embed asap !')
    # embed_data()
    
response_generator = load_data_training()
######################################################
# If a file is uploaded, create embeddings
uploaded_file_toTest = st.file_uploader("Upload Input file to Test")
if uploaded_file_toTest or not os.path.exists(Configs.articles_dir):
    if uploaded_file_toTest:
        stringio = StringIO(uploaded_file_toTest.getvalue().decode("utf-8", errors="ignore"))
        papers = str(stringio.read()).rstrip('\r\n')
        embed_data(papers=papers)
    else:
        embed_data(papers=Configs.papers)

    print('Will embed asap !')
    # embed_data()

response_generator = load_data_training()

######################################################

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question !"}
    ]        

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = response_generator.chat(user_query=prompt)
            st.write(response.content)
            message = {"role": "assistant", "content": response.content}
            st.session_state.messages.append(message)

