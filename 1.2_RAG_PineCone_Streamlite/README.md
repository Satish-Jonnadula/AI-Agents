
## Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.8 or higher:** [Download Python](https://www.python.org/downloads/)
2.  **Pinecone Account:**
    -   Sign up for a free Pinecone account at [pinecone.io](https://www.pinecone.io/).
    -   Obtain your **Pinecone API Key** and **Environment/Region** (e.g., `us-east-1`).
3.  **Google Cloud Project & API Key:**
    -   Create a Google Cloud Project.
    -   Enable the "Vertex AI API" (which includes Generative AI models) for your project.
    -   Create an **API Key** with permissions to access the Generative AI models. [Google Cloud AI Platform Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/api-quickstart)
4.  **(Optional) LangSmith Account:**
    -   For enhanced tracing and debugging, sign up at [smith.langchain.com](https://smith.langchain.com/).
    -   Obtain your **LangChain API Key**.

## Setup Instructions

1.  **Clone the Repository (or create your project folder):**
    If you have this code in a Git repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
    Otherwise, create a project directory and place `app.py` inside it.

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    langchain
    langchain-core
    langchain-google-genai
    pinecone-client
    langchain-pinecone
    python-dotenv
    PyMuPDF
    langchainhub
    # Add any other specific versions if needed, e.g., pinecone-client==3.0.0
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the root of your project directory. Copy the content from `.env.example` (if you have one) or create it from scratch:
    ```env
    # .env

    # Pinecone Configuration
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    # The Pinecone index name, dimension, metric, cloud, and region are defined in app.py
    # but you could also move them here if you prefer.

    # Google Generative AI Configuration
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

    # LangSmith Configuration (Optional)
    LANGCHAIN_API_KEY="YOUR_LANGCHAIN_API_KEY"
    # LANGCHAIN_TRACING_V2="true" # Already set in app.py
    # LANGCHAIN_PROJECT="RAG-Chatbot-Project" # Already set in app.py, customize if needed
    ```
    Replace `"YOUR_PINECONE_API_KEY"`, `"YOUR_GOOGLE_API_KEY"`, and (if using) `"YOUR_LANGCHAIN_API_KEY"` with your actual keys.

## Running the Application

Once you have completed the setup:

1.  **Ensure your virtual environment is activated.**
2.  **Navigate to your project directory** in the terminal.
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  Streamlit will typically open the application automatically in your default web browser (usually at `http://localhost:8501`). If not, open the URL displayed in your terminal.

## How to Use

1.  **Upload PDFs:** Use the file uploader to select one or more PDF documents from your computer.
2.  **Processing:** The application will display status messages as it loads, chunks, embeds, and stores the document content in Pinecone. This might take a few moments, especially for large documents or on the first run when the Pinecone index is being created.
3.  **Ask Questions:** Once processing is complete, an input field will appear. Type your question about the content of the uploaded documents and press Enter.
4.  **View Answers:** The chatbot will generate an answer based on the information retrieved from your documents and display it.

## Code Overview (`app.py`)

The `app.py` script is structured as follows:

1.  **Imports and Configuration:**
    -   Imports necessary libraries from Streamlit, LangChain, Pinecone, Google, etc.
    -   Loads environment variables using `dotenv`.
    -   Sets API keys and LangSmith tracing parameters.
    -   Defines constants for Pinecone index configuration (name, dimension, metric, etc.).
2.  **Helper Functions:**
    -   `format_docs(docs)`: A utility to combine retrieved document chunks into a single string for the LLM prompt.
3.  **Streamlit UI Setup:**
    -   Sets the page title, main title, and introductory markdown.
    -   Initializes Streamlit `session_state` variables to persist the RAG chain and processed file list across reruns.
4.  **File Upload and Processing Logic:**
    -   Uses `st.file_uploader` for PDF uploads.
    -   **Document Loading:** If new files are uploaded, they are temporarily saved and loaded using `PyMuPDFLoader`.
    -   **Text Splitting (Chunking):** `RecursiveCharacterTextSplitter` breaks loaded documents into smaller chunks.
    -   **Embedding Model Initialization:** `GoogleGenerativeAIEmbeddings` is initialized.
    -   **Pinecone Setup:**
        -   Connects to Pinecone.
        -   Checks if the specified index exists; if not, creates it with the correct dimension and metric.
        -   `PineconeVectorStore.from_documents` is used to embed the chunks and upsert them into the Pinecone index.
    -   **Retriever Creation:** `vectorstore.as_retriever()` creates a retriever configured for similarity search with a score threshold.
    -   **RAG Chain Definition:**
        -   Pulls a RAG prompt template from `langchain_hub`.
        -   Initializes the `ChatGoogleGenerativeAI` model.
        -   Constructs the RAG chain using LangChain Expression Language (LCEL), linking the retriever, prompt, LLM, and output parser.
    -   The constructed `rag_chain` is stored in `st.session_state`.
    -   Temporary PDF files are cleaned up.
5.  **Q&A Section:**
    -   If the `rag_chain` is ready, it displays a text input for user queries.
    -   When a query is submitted, `rag_chain.invoke(user_query)` is called to get a response.
    -   The response is displayed to the user.

## Customization and Troubleshooting

-   **Pinecone Index:** The Pinecone index name (`rag-pdf-index`), embedding dimension (`768` for `models/embedding-001`), metric (`cosine`), cloud, and region are defined as constants in `app.py`. You can modify these if needed.
    -   **Important:** The `EMBEDDING_DIMENSION` *must* match the output dimension of your chosen embedding model.
-   **Embedding Model:** Currently uses `models/embedding-001`. You can switch to other Google embedding models or models from other providers (e.g., OpenAI, Hugging Face) by changing the `GoogleGenerativeAIEmbeddings` line and updating the `EMBEDDING_DIMENSION` accordingly.
-   **LLM:** Uses `gemini-1.5-flash`. You can change this to other Gemini models or LLMs from other providers supported by LangChain.
-   **Chunking Strategy:** Adjust `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` to see how it affects retrieval quality.
-   **Retriever Settings:** Modify `k` (number of chunks to retrieve) and `score_threshold` in `retriever.as_retriever()` to fine-tune retrieval.
-   **Error: API Key Not Found:** Ensure your `.env` file is correctly named, in the project root, and contains the correct API keys. Also, verify that `load_dotenv()` is called early in your script.
-   **Error: Pinecone Index Dimension Mismatch:** Double-check that the `dimension` specified when creating the Pinecone index matches the output dimension of your embedding model.
-   **LangSmith Tracing:** If you've set up LangSmith, you can visit your project on the LangSmith dashboard to see detailed traces of your RAG chain executions, which is very helpful for debugging.

## Contributing

Feel free to fork this project, make improvements, and submit pull requests.

## License

This project is open-source. (You can specify a license like MIT if you wish).
