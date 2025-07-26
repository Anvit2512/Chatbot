import streamlit as st
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch # Although we use CPU, some transformers dependencies might need it

# --- Constants ---
DB_DIR = "chroma_db"
MODEL_NAME_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID_LLM = "google/flan-t5-base"

# --- Caching Functions for Performance ---

# Cache the function that loads and processes documents.
# This means it will only run once for a given set of uploaded files.
@st.cache_resource(show_spinner="Processing documents...")
def load_and_process_documents(_uploaded_files):
    docs = []
    temp_dir = "temp_docs"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for uploaded_file in _uploaded_files:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = TextLoader(temp_filepath)
        docs.extend(loader.load())

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME_EMBEDDINGS)

    # 4. Store in ChromaDB
    # The 'persist_directory' argument tells Chroma to save the DB to disk.
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return vectorstore

# Cache the function that creates the conversational chain.
# This will be created once per session based on the vectorstore.
@st.cache_resource
def get_conversational_chain(_vectorstore):
    # 1. Set up LLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_LLM)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID_LLM)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.2
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 2. Set up memory
    # This memory object will store the conversation history.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 3. Create the ConversationalRetrievalChain
    # This chain combines a retriever with conversational memory.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

# --- Streamlit App UI ---

def main():
    st.set_page_config(page_title="Chat with your Docs", page_icon="🤖")
    st.title("📄 Chat with your Documents")
    st.write("Upload your text documents, and this chatbot will answer questions based on them.")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # Sidebar for document uploading and processing
    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader(
            "Upload your .txt files here and click 'Process'",
            accept_multiple_files=True,
            type=['txt']
        )

        if st.button("Process Documents"):
            if uploaded_files:
                # Create and store the vectorstore
                vectorstore = load_and_process_documents(uploaded_files)
                # Create and store the conversational chain
                st.session_state.conversation = get_conversational_chain(vectorstore)
                # Store the names of processed files
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one document.")

        if st.session_state.processed_files:
            st.markdown("---")
            st.write("Active Documents:")
            for file_name in st.session_state.processed_files:
                st.write(f"- {file_name}")

    # Main chat interface
    if st.session_state.conversation:
        # Display previous chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input for new user question
        if user_question := st.chat_input("Ask a question about your documents:"):
            # Add user message to chat history and display it
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({'question': user_question})
                    ai_response = response['answer']
                    st.markdown(ai_response)

            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    else:
        st.info("Please upload and process your documents in the sidebar to start chatting.")


if __name__ == '__main__':
    main()