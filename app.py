import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chromadb

st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stChat {
            padding: 1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e6f3ff;
        }
        .bot-message {
            background-color: #f0f0f0;
        }
        .message-content {
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)


if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'retriever_chain' not in st.session_state:
    st.session_state.retriever_chain = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False


load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

def initialize_llm():
    return ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.6,
        max_retries=2,
        api_key=api_key
    )

def process_document(file_path, file_type):
    """Process a single document and return its content"""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type == "txt":
        loader = TextLoader(file_path)
        return loader.load()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def process_documents(uploaded_files):
    """Process uploaded PDF and text files and create retriever chain"""
    with st.spinner("Processing documents... Please wait."):
        all_documents = []
        
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            try:
                documents = process_document(temp_file_path, file_extension)
                all_documents.extend(documents)
            except Exception as e:
                st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                os.remove(temp_file_path)

        if not all_documents:
            raise ValueError("No documents were successfully processed")

      
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        content = text_splitter.split_documents(all_documents)

     
        embeddings = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")

        persist_directory = "chroma_db"
        os.makedirs(persist_directory, exist_ok=True)

        client = chromadb.PersistentClient(path=persist_directory)
        
        db = Chroma.from_documents(
            documents=content,
            embedding=embeddings,
            persist_directory=persist_directory,
            client=client
        )
        
        retriever = db.as_retriever(k=min(len(db), 10))

        llm = initialize_llm()
        prompt_template = PromptTemplate(
            input_variables=["context", "input"],
            template="""
            You are a chat bot answer the following question based only on the provided context.
            Think step by step before providing a detailed answer.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )
        
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
        st.session_state.retriever_chain = create_retrieval_chain(retriever, chain)
        st.session_state.processing_complete = True
        
        return len(content)

with st.sidebar:
    st.title("📚 Document Chat Assistant")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more PDF or text files to chat about their content"
    )
    
    if uploaded_files:
        try:
            num_chunks = process_documents(uploaded_files)
            st.success(f"✅ Successfully processed {len(uploaded_files)} documents into {num_chunks} chunks!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload your PDF or text documents using the uploader above
    2. Wait for the processing to complete
    3. Type your questions in the chat input below
    4. The assistant will answer based on the content of your documents
    """)

st.title("💬 Chat with your Documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
    if not st.session_state.processing_complete:
        st.error("Please upload documents first!")
    else:
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.retriever_chain.invoke({"input": prompt})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()


if st.sidebar.button("Clear Database"):
    try:
        import shutil
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
        st.session_state.processing_complete = False
        st.session_state.retriever_chain = None
        st.success("Database cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")


st.markdown(
    """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: white;'>
        <p>Created with ❤️ using Streamlit and LangChain</p>
    </div>
    """,
    unsafe_allow_html=True
)