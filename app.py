import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import nltk

# Load environment variables
load_dotenv()
nltk.download('punkt')

# Define working directory
working_dir = os.getcwd()

# Function to load documents
def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

# Function to set up vectorstore
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

# Function to create the conversation chain
def create_chain(vectorstore):
    llm = ChatGroq(
        model='llama-3.1-70b-versatile',
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain

# Streamlit page configuration
st.set_page_config(
    page_title="Chat With Groq",
    page_icon="📄",
    layout="centered"
)
st.title("🦙 Chat with Doc - LLAMA 3.1")

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader(label="Upload Your PDF File", type=["pdf"])
if uploaded_file:
    file_path = os.path.join(working_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    documents = load_documents(file_path)
    st.session_state.vectorstore = setup_vectorstore(documents)
    st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

if st.session_state.vectorstore is None:
    st.warning("Please upload a PDF to start the conversation.")
    st.stop()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response handling
user_input = st.chat_input("ASK LLAMA....")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    response = st.session_state.conversation_chain({"question": user_input})
    assistant_response = response["answer"]

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
