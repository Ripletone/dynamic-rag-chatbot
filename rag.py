import streamlit as st
import os
import google.generativeai as genai
from langchain.chains.combine_documents import create_stuff_documents_chain
# Ensure MessagesPlaceholder is imported
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
import shutil
import re
from urllib.parse import urlparse

# New imports for Agents and DuckDuckGo search
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool

# CRITICAL FIX: Use nest_asyncio to handle event loop conflicts in Streamlit
# This allows asynchronous operations to run smoothly within Streamlit's environment.
import nest_asyncio
nest_asyncio.apply()


# --- Configuration Constants ---
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "default_rag_collection"

# --- API Key Handling ---
# Access API key securely via Streamlit secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("`GOOGLE_API_KEY` not found in `.streamlit/secrets.toml`. Please add it to your secrets file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- Streamlit Session State Initialization ---
def initialize_session_state():
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "current_content_source" not in st.session_state:
        st.session_state.current_content_source = None
    if "current_collection_name" not in st.session_state:
        st.session_state.current_collection_name = DEFAULT_COLLECTION_NAME
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "url_input_key" not in st.session_state:
        st.session_state.url_input_key = 0
    if "uploaded_file_key" not in st.session_state:
        st.session_state.uploaded_file_key = 0
    
    # Initialize configuration parameters in session state
    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = 0.0
    if "top_k_retrieval" not in st.session_state:
        st.session_state.top_k_retrieval = 4
    if "selected_search_type" not in st.session_state:
        st.session_state.selected_search_type = "Similarity"
    if "fetch_k_mmr" not in st.session_state:
        st.session_state.fetch_k_mmr = 20
    if "lambda_mult_mmr" not in st.session_state:
        st.session_state.lambda_mult_mmr = 0.5

initialize_session_state()

# --- LangChain Imports (Conditional, as they might be imported based on need) ---
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
# THIS IS THE CRITICAL LINE: CORRECTED IMPORT PATH FOR DOCUMENT LOADERS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader


# --- Helper Functions for Naming and Data Processing ---

def clean_collection_name(name: str) -> str:
    """
    Cleans a string to be a valid ChromaDB collection name.
    ChromaDB collection names must be 3-63 characters, use [a-zA-Z0-9._-],
    and start/end with an alphanumeric character.
    """
    if not isinstance(name, str) or not name.strip():
        return DEFAULT_COLLECTION_NAME

    # 1. Replace invalid characters with an underscore
    cleaned_name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)

    # 2. Remove leading/trailing non-alphanumeric characters (if any were introduced)
    cleaned_name = re.sub(r'^[^a-zA-Z0-9]+', '', cleaned_name)
    cleaned_name = re.sub(r'[^a-zA-Z0-9]+$', '', cleaned_name)

    # 3. Ensure minimum length
    if len(cleaned_name) < 3:
        cleaned_name = "collection_" + cleaned_name.lower()

    # 4. Ensure it starts and ends with an alphanumeric character (more robust check)
    if not cleaned_name:
        return DEFAULT_COLLECTION_NAME
    if not cleaned_name[0].isalnum():
        cleaned_name = 'c' + cleaned_name
    if not cleaned_name[-1].isalnum():
        cleaned_name = cleaned_name + 'c'

    # 5. Truncate if it's too long (ChromaDB limit is 63 characters)
    cleaned_name = cleaned_name[:63] # Changed from 512 to 63

    # Final fallback if cleaning results in an unrecoverable state or still invalid
    if not cleaned_name.strip() or not cleaned_name[0].isalnum() or not cleaned_name[-1].isalnum():
        return DEFAULT_COLLECTION_NAME

    return cleaned_name.lower()

def get_url_collection_name(url: str) -> str:
    """Generates a consistent collection name from a URL."""
    parsed_url = urlparse(url)
    # Use the domain and path, then hash for uniqueness and clean
    base_name = f"{parsed_url.netloc}{parsed_url.path}".strip('/')
    if not base_name:
        base_name = "web_content"

    # Hash the full URL to ensure uniqueness for different paths/queries on the same domain
    unique_suffix = str(abs(hash(url)))

    # Combine a cleaned version of the base name with the unique hash
    combined_name = f"{base_name}_{unique_suffix}"
    return clean_collection_name(combined_name)


# --- Caching and Document Handling Functions ---

@st.cache_resource
def _load_embedding_model():
    """Internal function to load the embedding model, to be cached."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})

def get_embedding_function():
    """Wrapper to load and cache the embedding model with UI feedback."""
    with st.spinner("Loading embedding model..."):
        embeddings = _load_embedding_model()
    st.toast("Embedding model loaded!", icon="✅")
    return embeddings


# New internal cached function for vector store
@st.cache_resource(hash_funcs={Chroma: id})
def _load_or_create_vector_store(_text_chunks, _embedding_function, collection_name_for_cache):
    """
    Internal function to create or load a Chroma vector store.
    This function contains the core logic and is what gets cached.
    """
    persist_directory = CHROMA_DB_PATH
    os.makedirs(persist_directory, exist_ok=True)

    if _text_chunks:
        # Create/Update the vector store
        try:
            vector_db = Chroma.from_documents(
                documents=_text_chunks,
                embedding=_embedding_function,
                collection_name=collection_name_for_cache,
                persist_directory=persist_directory
            )
            return vector_db
        except Exception as e:
            st.error(f"Error creating/updating knowledge base: {e}")
            return None
    else:
        # Attempt to load an existing vector store
        try:
            collection_path = os.path.join(persist_directory, collection_name_for_cache)
            if not os.path.exists(collection_path):
                return None

            vector_db = Chroma(
                collection_name=collection_name_for_cache, 
                embedding_function=_embedding_function,
                persist_directory=persist_directory
            )
            count = vector_db._collection.count()
            if count > 0:
                return vector_db
            else:
                return None 
        except Exception as e:
            st.warning(f"ChromaDB load failed for {collection_name_for_cache}: {e}") 
            return None


# Wrapper for get_vector_store to handle UI feedback
def get_vector_store(text_chunks, embedding_function, collection_name=DEFAULT_COLLECTION_NAME):
    """
    Wrapper function that calls the cached vector store logic and handles Streamlit UI feedback.
    """
    cleaned_collection_name = clean_collection_name(collection_name)
    st.session_state.current_collection_name = cleaned_collection_name

    vector_db = None

    if text_chunks:
        st.warning(f"Creating/Updating knowledge base collection '{cleaned_collection_name}'. This might take a moment...")
        vector_db = _load_or_create_vector_store(
            text_chunks,
            embedding_function,
            cleaned_collection_name
        )
        if vector_db:
            st.toast(f"Knowledge base collection '{cleaned_collection_name}' created/updated successfully with {len(text_chunks)} chunks!", icon="✨")
        # Error message is handled inside _load_or_create_vector_store if it returns None

    else:
        # This branch is for attempting to load an existing DB on app startup
        st.toast(f"Attempting to load existing knowledge base collection '{cleaned_collection_name}' from disk...", icon="⏳")
        vector_db = _load_or_create_vector_store(
            [], 
            embedding_function,
            cleaned_collection_name
        )
        if vector_db:
            st.toast(f"Knowledge base collection '{cleaned_collection_name}' loaded from disk with {vector_db._collection.count()} items!", icon="📚")
        else:
            st.warning(f"Could not load existing knowledge base collection '{cleaned_collection_name}'. It might not exist or is empty. Please upload a document or load a URL.")

    return vector_db


@st.cache_data(show_spinner=False)
def process_document_to_chunks(uploaded_file, chunk_size, chunk_overlap):
    """Loads and splits an uploaded document into text chunks."""
    if uploaded_file is None:
        return []

    temp_file_dir = "./temp_uploaded_files"
    os.makedirs(temp_file_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_file_dir, uploaded_file.name)

    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"Loading document: {uploaded_file.name}")

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        loader = None

        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
        elif file_extension == ".md":
            loader = UnstructuredMarkdownLoader(temp_file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)
        else:
            st.error(f"Unsupported file type: **{file_extension}**. Please upload a PDF, TXT, MD, or DOCX file.")
            return []

        documents = loader.load()
        if not documents:
            st.warning(f"No content found in {uploaded_file.name}.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = text_splitter.split_documents(documents)

        st.toast(f"Document '{uploaded_file.name}' processed into {len(text_chunks)} chunks!", icon="📄")
        return text_chunks

    except Exception as e:
        st.error(f"Error processing document '{uploaded_file.name}': {e}")
        return []
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@st.cache_data(show_spinner=False)
def process_url_to_chunks(url_input, chunk_size, chunk_overlap):
    """Loads and splits content from a URL into text chunks."""
    if not url_input:
        return []

    with st.status(f"Loading content from URL: {url_input}...", expanded=True) as status_message:
        try:
            status_message.write("1. Fetching content from URL...")
            loader = WebBaseLoader(url_input)
            documents = loader.load()
            if not documents:
                raise ValueError("No content extracted from URL.")
            status_message.write(f"2. Extracted {len(documents)} document(s).")

            status_message.write("3. Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            text_chunks = text_splitter.split_documents(documents)
            if not text_chunks:
                raise ValueError("No text chunks generated from content.")
            status_message.write(f"4. Created {len(text_chunks)} text chunks.")
            status_message.update(label=f"Content from '{url_input}' processed!", state="complete", expanded=False)
            return text_chunks
        except Exception as e:
            status_message.update(label=f"Error loading URL: {e}", state="error", expanded=True)
            st.error(f"Error loading URL: {e}")
            return []

# MODIFIED: Now an agent for flexible tool use
def get_llm_agent(
    vector_db,
    llm_temperature_param: float,
    search_type_param: str,
    k_param: int,
    fetch_k_param: int,
    lambda_mult_param: float
):
    """
    Constructs and returns an LLM Agent capable of using a RAG retriever and a web search tool.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=llm_temperature_param,
        google_api_key=GOOGLE_API_KEY
    )

    # --- 1. Define the Tools ---
    tools = []

    # A. Knowledge Base Retriever Tool (if vector_db exists)
    if vector_db:
        # Create retriever based on selected search type
        if search_type_param == "Similarity":
            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k_param})
        elif search_type_param == "MMR":
            retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k_param, "fetch_k": fetch_k_param, "lambda_mult": lambda_mult_param}
            )
        else:
            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k_param}) # Fallback

        knowledge_base_tool = create_retriever_tool(
            retriever,
            name="KnowledgeBase_Search", # Tool name for the agent to use
            description="Searches and returns information from the user-provided documents in the knowledge base. Use this for questions specifically about uploaded content.",
        )
        tools.append(knowledge_base_tool)

    # B. Web Search Tool (DuckDuckGo)
    web_search = DuckDuckGoSearchResults(max_results=5) # Limit results to avoid overwhelming LLM
    web_search_tool = Tool(
        name="Web_Search", # Tool name for the agent to use
        description="Useful for when you need to answer questions about current events, facts, or anything not covered in the provided documents. Prioritize the KnowledgeBase_Search tool if the question is likely about uploaded content.",
        func=web_search.run,
    )
    tools.append(web_search_tool)

    if not tools:
        st.warning("No tools are available for the agent. Please load a document or ensure tools are correctly defined.")
        return None
    
    # Removed st.sidebar.subheader("Agent Tools (for Debugging)") and loop to clean up layout


    # --- 2. Define the Agent's Prompt ---
    # The agent's prompt guides its reasoning and tool selection.
    # It needs to know about chat history, tools, and agent_scratchpad.
    agent_prompt = ChatPromptTemplate.from_messages([
        # System message for the agent's persona and general instructions
        ("system", """You are a helpful AI assistant. Your primary goal is to answer user questions accurately and comprehensively.
        
        Always prioritize 'KnowledgeBase_Search' if the question can be answered from the documents.
        If you use 'Web_Search', aim to summarize relevant information and indicate that it came from the web (e.g., "According to a web search...").
        If you use 'KnowledgeBase_Search', indicate that it came from the knowledge base (e.g., "From the knowledge base...").
        If you cannot find an answer using any tool, state that you don't know.
        """),
        
        # Placeholder for chat history (list of HumanMessage/AIMessage)
        MessagesPlaceholder(variable_name="chat_history"),
        
        # IMPORTANT: Removed MessagesPlaceholder(variable_name="tools")
        # create_tool_calling_agent handles tool injection automatically.
        
        # Human message for the current input
        ("human", "{input}"),
        
        # Placeholder for the agent's internal thoughts, tool calls, and tool outputs
        MessagesPlaceholder(variable_name="agent_scratchpad") 
    ])

    # --- 3. Create the Agent ---
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Set to True for debugging agent's thought process in terminal
        handle_parsing_errors=True # Important for robust agent behavior
    )
    return agent_executor


def generate_answer_with_memory(
    query,
    chat_history,
    vector_db # Removed individual params here, will get them from session state
):
    """Generates an answer using the LLM agent, including conversational memory and tool use."""

    # Retrieve configuration parameters from session state
    llm_temperature_param = st.session_state.llm_temperature
    search_type_param = st.session_state.selected_search_type
    k_param = st.session_state.top_k_retrieval
    fetch_k_param = st.session_state.fetch_k_mmr
    lambda_mult_param = st.session_state.lambda_mult_mmr

    # Get the agent executor
    agent_executor = get_llm_agent(
        vector_db,
        llm_temperature_param,
        search_type_param,
        k_param,
        fetch_k_param,
        lambda_mult_param
    )

    if not agent_executor:
        st.warning("Agent could not be initialized. Please ensure necessary tools are available.")
        return "An error occurred, agent not initialized.", []

    processed_query = str(query).strip()
    if not processed_query:
        return "I need a question to answer!", []

    formatted_chat_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_chat_history.append(HumanMessage(content=str(msg["content"])))
        elif msg["role"] == "assistant":
            formatted_chat_history.append(AIMessage(content=str(msg["content"])))

    try:
        # Invoke the agent executor
        # The agent will decide whether to use KnowledgeBase_Search or Web_Search
        response = agent_executor.invoke({
            "input": processed_query,
            "chat_history": formatted_chat_history
            # No need to pass 'tools' or 'agent_scratchpad' here; create_tool_calling_agent handles them internally
        })

        response_text = response.get("output", "No answer generated by agent.")
        # The agent's output will ideally contain the answer and potentially mention sources.
        # Extracting "source_docs" directly from agent output is more complex and usually
        # requires parsing the agent's final output string or custom callbacks.
        # For now, we'll indicate if it's from web or KB based on how the agent responds.
        
        # A more robust source display would involve inspecting agent_executor.iter()
        # or using callbacks to capture tool usage. For simplicity, we'll rely on
        # the agent's prompt to make it cite sources in its final output.
        source_docs_for_display = [] # Agent will ideally put source info in response_text
        return response_text, source_docs_for_display # Sources will be part of response_text

    except Exception as e:
        st.error(f"Error generating answer with agent: {e}")
        # When verbose=True, the traceback will be printed in the console
        return "An error occurred while trying to generate an answer. Please try again.", []


# --- Main Streamlit App Layout and Logic ---

st.set_page_config(page_title="Dynamic RAG Chatbot with Memory", layout="wide")

st.markdown(
    """
    <style>
    /* Force body and general text color/font */
    body, p, div, span, h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: #000000 !important; /* Changed to BLACK */
        font-family: sans-serif !important; /* You can change this to a specific font like "Roboto, sans-serif" */
        font-weight: normal !important; /* Changed to NORMAL (unbolded) */
    }

    /* Keep the original styling for Streamlit-generated code blocks and preformatted text */
    code, pre {
        font-family: monospace !important; /* Ensure code blocks remain monospace */
    }

    /* Styling for the URL input text box */
    div[data-testid="stTextInput"] input {
        border: 1px solid #4CAF50;
        border-radius: 8px;
        padding: 10px 15px;
        background-color: #f8fcf8;
        color: #333 !important; /* Ensure input text color is still dark, not black */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: sans-serif !important; /* Ensure input text font is sensible */
        font-weight: normal !important; /* Ensure input text is not bold unless intended */
    }

    /* Styling for the chat input text box */
    div[data-testid="stChatInput"] input {
        border: 1px solid #007bff;
        border-radius: 20px;
        padding: 12px 20px;
        background-color: #eaf6ff;
        color: #333 !important; /* Ensure chat input text color is still dark, not black */
        box_shadow: 0 2px 5px rgba(0,0,0,0.15);
        font-family: sans-serif !important; /* Ensure chat input text font is sensible */
        font-weight: normal !important; /* Ensure input text is not bold unless intended */
    }

    /* Styling for the Sources expander */
    .stExpander {
        border: 1px solid #a8dadc;
        border-radius: 10px;
        padding: 5px 15px;
        margin-top: 20px;
        background-color: #f1f8f9;
        box_shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Optional: Style for the chat message boxes themselves */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        box_shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .stChatMessage.st-chat-message-user {
        background-color: #e0f2f7;
        text-align: right;
    }
    .stChatMessage.st-chat-message-assistant {
        background-color: #e6ffe6;
        text-align: left;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Dynamic RAG Chatbot with Memory")
st.write("Upload a document or load a URL to create a knowledge base, then ask questions about it, or ask general questions!")


# Sidebar for controls
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload a Document", type=["pdf", "txt", "md", "docx"], key="file_uploader")

# --- Document Processing Settings (using expander and columns) ---
with st.sidebar.expander("Document Processing Settings", expanded=True):
    st.markdown("Adjust how documents are split into chunks for retrieval.")
    col1, col2 = st.columns(2)

    with col1:
        # Store in session state
        st.session_state.chunk_size = st.slider(
            "Chunk Size",
            min_value=100, max_value=2000, value=1000, step=50,
            help="Size of text segments (characters)."
        )
    with col2:
        # Store in session state
        st.session_state.chunk_overlap = st.slider(
            "Overlap",
            min_value=0, max_value=500, value=200, step=25,
            help="Overlapping characters between chunks."
        )

# --- Language Model Settings (using expander) ---
with st.sidebar.expander("Language Model Settings", expanded=True):
    st.markdown("Configure the behavior of the AI model.")
    # Store in session state
    st.session_state.llm_temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Controls creativity. Lower = more deterministic; higher = more creative."
    )

# --- Retrieval Settings (New Expander) ---
with st.sidebar.expander("Retrieval Settings", expanded=True):
    st.markdown("Control how relevant documents are retrieved from the knowledge base (used by KnowledgeBase_Search tool).")

    # Top-K Slider (applies to both search types) - Store in session state
    st.session_state.top_k_retrieval = st.slider(
        "Number of Chunks (k)",
        min_value=1,
        max_value=10,
        value=4, # Default to 4 chunks
        step=1,
        help="Number of top similar document chunks to retrieve."
    )

    # Search Type Selectbox - Store in session state
    st.session_state.selected_search_type = st.selectbox(
        "Search Type",
        options=["Similarity", "MMR"],
        index=0, # Default to Similarity
        help="Similarity: Retrieves most similar chunks. MMR: Maximizes relevance to query AND diversity among results."
    )

    # Conditional MMR Parameters - Store in session state
    if st.session_state.selected_search_type == "MMR":
        st.markdown("MMR Parameters:")
        mmr_col1, mmr_col2 = st.columns(2)
        with mmr_col1:
            st.session_state.fetch_k_mmr = st.number_input(
                "Fetch K (MMR)",
                min_value=st.session_state.top_k_retrieval, # Must be at least 'k'
                max_value=50,
                value=max(st.session_state.top_k_retrieval, 20), # Default to top_k or 20, whichever is larger
                step=1,
                help="Number of initial documents to fetch for MMR before re-ranking for diversity. Should be >= 'k'."
            )
        with mmr_col2:
            st.session_state.lambda_mult_mmr = st.slider(
                "Lambda (MMR)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Diversity vs. Relevance trade-off for MMR. 0.0 = maximum diversity, 1.0 = maximum relevance (similar to similarity search)."
            )
    else:
        # Ensure these are set to default or reasonable values if MMR is not selected
        # This prevents NameErrors if MMR parameters are accessed when Similarity is chosen
        if "fetch_k_mmr" not in st.session_state:
             st.session_state.fetch_k_mmr = 20
        if "lambda_mult_mmr" not in st.session_state:
            st.session_state.lambda_mult_mmr = 0.5


# --- Load from URL (using columns for input and button) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Load from URL")
url_col, button_col = st.sidebar.columns([3, 1])

with url_col:
    # Use st.session_state.url_input_key to ensure input clears on refresh
    url_input = st.text_input("Enter URL", label_visibility="collapsed", placeholder="Enter a URL to load...", key=f"url_input_{st.session_state.url_input_key}")

with button_col:
    load_url_button = st.button("Load", key="load_url_button", use_container_width=True)


# --- Initial Load/Check for Existing DB ---
# This block attempts to load an existing DB when the app first starts or reruns
if st.session_state.vector_db is None and st.session_state.current_content_source is None:
    embedding_function = get_embedding_function()
    st.session_state.vector_db = get_vector_store(
        text_chunks=[], # Pass empty chunks to trigger loading from disk
        embedding_function=embedding_function,
        collection_name=DEFAULT_COLLECTION_NAME
    )
    if st.session_state.vector_db:
        # If successfully loaded, update current_content_source
        st.session_state.current_content_source = f"Pre-existing knowledge base ({st.session_state.current_collection_name})"


# --- Handle Document Upload (triggered by file_uploader or Load File button) ---
# If a file is uploaded AND it's a new file (not already processed)
if uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_filename", ""):
    st.session_state.messages = [] # Clear chat on new document
    st.session_state.last_uploaded_filename = uploaded_file.name # Store for next rerun check

    with st.spinner(f"Processing document '{uploaded_file.name}'..."):
        text_chunks = process_document_to_chunks(uploaded_file, st.session_state.chunk_size, st.session_state.chunk_overlap)

    if text_chunks:
        embedding_function = get_embedding_function()
        file_collection_name = clean_collection_name(os.path.splitext(uploaded_file.name)[0])
        st.session_state.vector_db = get_vector_store(text_chunks, embedding_function, collection_name=file_collection_name)

        if st.session_state.vector_db:
            st.session_state.current_content_source = uploaded_file.name
            st.toast(f"Knowledge base ready for '{uploaded_file.name}'! You can now ask questions.", icon="🎉")
            st.rerun() # Rerun to update UI and prevent reprocessing on subsequent renders
    else:
        st.session_state.vector_db = None
        st.session_state.current_content_source = None
        st.session_state.uploaded_file_key += 1 # Increment key to clear uploader
        st.rerun() # Rerun to clear input if processing failed


# --- Handle URL Loading (triggered by Load URL button) ---
# Check if button was clicked AND url_input is not empty AND it's a new URL
if load_url_button and url_input and url_input != st.session_state.get("last_loaded_url", ""):
    st.session_state.messages = [] # Clear chat on new document
    st.session_state.last_loaded_url = url_input # Store for next rerun check

    text_chunks = process_url_to_chunks(url_input, st.session_state.chunk_size, st.session_state.chunk_overlap)

    if text_chunks:
        embedding_function = get_embedding_function()
        url_collection_name = get_url_collection_name(url_input)
        st.session_state.vector_db = get_vector_store(text_chunks, embedding_function, collection_name=url_collection_name)

        if st.session_state.vector_db:
            st.session_state.current_content_source = url_input
            st.toast(f"Knowledge base ready for '{url_input}'! You can now ask questions.", icon="🎉")
            st.session_state.url_input_key += 1 # Increment key to clear url input
            st.rerun() # Rerun to update UI and prevent reprocessing on subsequent renders
    else:
        st.session_state.vector_db = None
        st.session_state.current_content_source = None
        st.session_state.url_input_key += 1 # Increment key to clear url input
        st.rerun() # Rerun to clear input if processing failed


# --- Display Current Knowledge Base Status ---
if st.session_state.vector_db is None:
    st.toast("Please upload a document or load a URL to begin.", icon="⬆️")
elif st.session_state.current_content_source:
    st.write(f"Knowledge base active for: **{st.session_state.current_content_source}** (Collection: `{st.session_state.current_collection_name}`)")
else:
    st.toast("No content loaded yet. Please upload a document or load a URL to begin.", icon="🤷‍♀️")


# --- Chat Interface ---
# Initialize messages if empty (should happen on first run or clear chat)
if not st.session_state.messages:
    initial_ai_response = "Hi there! I'm your RAG chatbot. I can answer questions about the document or URL you provide, and now also search the web for general knowledge!"
    st.session_state.messages = [{"role": "assistant", "content": initial_ai_response}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking... (The agent is deciding whether to use the knowledge base or web search...)"):
        # The agent now handles whether vector_db is None or not internally
        response_text, source_docs = generate_answer_with_memory(
            query,
            st.session_state.messages[:-1], # Pass all but the current user message for chat history
            st.session_state.vector_db # Now generate_answer_with_memory gets params from session state
        )

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.write(response_text)

        # Removed the automatic source display for now, as agent will embed source info in its output
        # If you want structured source display, you'd need more advanced parsing of agent.invoke() output
        # or custom callbacks to trace tool usage.

# --- Sidebar: Clear Chat and Reset RAG ---
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat and Reset RAG Data"):
    st.session_state.messages = []
    st.session_state.vector_db = None
    st.session_state.current_content_source = None
    st.session_state.current_collection_name = DEFAULT_COLLECTION_NAME

    st.session_state.url_input_key += 1
    st.session_state.uploaded_file_key += 1
    st.session_state.last_uploaded_filename = ""
    st.session_state.last_loaded_url = ""

    if os.path.exists(CHROMA_DB_PATH):
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            st.toast("ChromaDB cleared from disk!", icon="🗑️")
        except Exception as e:
            st.error(f"Error clearing ChromaDB directory: {e}")

    st.rerun()
