import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import ServerlessSpec
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

import os
from dotenv import load_dotenv
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG-Chatbot"
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

import pinecone


st.title("RAG Chatbot with LangChain and Pinecone")
st.markdown("This is a demo of RAG Chatbot with LangChain and Pinecone.")
st.markdown("You can ask questions about the document.")

# File Upload Section
uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.write("Processing the uploaded files...")

    # Load and process each uploaded PDF file
    docs = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary location
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load the PDF file
        loader = PyMuPDFLoader(uploaded_file.name)
        docs.extend(loader.load())

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks= text_splitter.split_documents(docs)
    st.write(f"Total chunks created: {len(chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Initialize Pinecone vector store
    pn= Pinecone(pinecone_api_key)
    
    if pn.has_index("rag-index"):
        st.write("Using existing index 'rag-index'")
        index = pn.Index("rag-index")
    else:
        pn.create_index("rag-index", dimension=768, metric="cosine", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        index= pn.Index("rag-index")

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    import uuid
    uuids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    vectorstore.add_documents(documents=chunks, ids=uuids)
    st.write("Documents processed and stored in Pinecone.")

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.8})

    # Define RAG Chain
    from langchain import hub
    prompt= hub.pull("rlm/rag-prompt")

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # User Input Section
    user_input = st.text_input("Ask a question about the document:")
    if user_input:
        st.write("Generating response...")
        response = rag_chain.invoke(user_input)
        st.write("Response:")
        st.write(response)
st.markdown("This is a demo of RAG Chatbot with LangChain and Pinecone.")



