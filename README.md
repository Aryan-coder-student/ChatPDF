
# 📚 Document Chat Assistant Documentation  

This application allows users to upload multiple documents (PDF or TXT files) and interact with them through a chat interface using **Retrieval-Augmented Generation (RAG)**. The assistant extracts relevant information from the uploaded documents and generates detailed responses to user queries.  

---

## **Features**  
1. **Multi-document Upload**: Upload multiple PDF and TXT files simultaneously.  
2. **Document Processing**: Automatically parses and preprocesses uploaded documents into manageable chunks for better retrieval.  
3. **RAG-based Interaction**: Combines a retrieval mechanism and a language model for generating context-aware responses.  
4. **Persistent Storage**: Uses `Chroma` for storing document embeddings and efficient retrieval.  
5. **Interactive Chat Interface**: Allows users to ask questions based on document content and get detailed answers.  
6. **Reset Options**: Clear chat history or delete the database to start afresh.  

---

## **Application Workflow**

### **1. Sidebar Interface**  
- **Document Upload**:  
   - Users can upload one or more files in PDF or TXT format.  
   - The app preprocesses documents and splits them into smaller chunks.  
   - Progress and success messages are displayed during processing.  

- **Instructions**:  
   - Guides users on how to use the assistant.  

- **Buttons**:  
   - **Clear Chat History**: Resets the chat interface.  
   - **Clear Database**: Deletes the stored embeddings and disables the chat until new documents are uploaded.  

---

### **2. Chat Interface**  
- Users can type questions related to the uploaded documents.  
- The assistant retrieves relevant information from the documents and generates answers using an LLM.  
- Both user and assistant messages are displayed in a structured, conversational format.  

---

## **Code Breakdown**

### **Document Upload and Processing**  
- **File Uploader**:  
   ```python
   uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
   ```  
   Allows users to upload documents.  

- **Processing Documents**:  
   - Uses `PyPDFLoader` for PDF files and `TextLoader` for TXT files.  
   - Splits the documents into smaller chunks using `RecursiveCharacterTextSplitter`.  
   - Converts chunks into embeddings using `HuggingFaceEmbeddings`.  
   - Stores embeddings in a persistent Chroma database.  

---

### **RAG-based Chat System**  
- **Retriever and LLM**:  
   - Uses a **Chroma retriever** for fetching relevant document passages.  
   - Leverages **ChatGroq** with the Llama 3.1 model for generating answers.  

- **Custom Prompt**:  
   The prompt ensures that answers are generated solely based on the provided context:  
   ```python
   template = """
   You are a chat bot answer the following question based only on the provided context.
   Think step by step before providing a detailed answer.
   <context>
   {context}
   </context>
   Question: {input}
   """  
   ```  

---

### **Session State Management**  
- Maintains user messages, retriever chains, and processing status across interactions using `st.session_state`.  

---

### **Chat Display**  
- Messages are displayed using a structured layout styled with custom HTML and CSS:  
   ```python
   st.markdown(
       """
       <style>
           .chat-message { ... }
       </style>
       """,
       unsafe_allow_html=True
   )
   ```  

---

### **Reset Functionality**  
- **Clear Chat History**:  
   ```python
   st.session_state.messages = []
   st.rerun()
   ```  

- **Clear Database**:  
   Deletes the Chroma database and resets the retriever chain.

---

## **Installation and Setup**

### **1. Prerequisites**
- Python 3.8 or higher  
- Install dependencies:  
  ```bash
  pip install streamlit langchain chromadb langchain_huggingface langchain_groq python-dotenv
  ```

---

### **2. Running the Application**
- Start the Streamlit server:  
  ```bash
  streamlit run app.py
  ```  

---

## **Conclusion**  
This assistant simplifies interacting with large amounts of document data by combining modern NLP techniques with an intuitive user interface. It is a powerful tool for extracting knowledge from documents and answering questions efficiently.  

---  
**Created with ❤️ using Streamlit and LangChain**.
