Dynamic RAG Chatbot with Memory and Web Search
This is an interactive Streamlit application that demonstrates a Retrieval Augmented Generation (RAG) chatbot. It can answer questions based on custom documents/URLs you provide, and can also perform real-time web searches for general knowledge queries.

🌟 Features
Dynamic Knowledge Base: Upload PDF, TXT, Markdown, or DOCX files, or provide URLs to build a custom knowledge base on the fly.

Intelligent Agent: Leverages Google's Gemini Pro model through LangChain agents to intelligently decide whether to retrieve information from the custom knowledge base or perform a web search.

Conversational Memory: Maintains chat history for more natural and context-aware interactions across multiple turns.

Customizable Settings: Adjust document chunking parameters, LLM temperature, and retrieval methods (Similarity / Maximal Marginal Relevance - MMR).

Web Search Capability: Utilizes DuckDuckGo for real-time information retrieval on topics not covered in the provided documents.

User-Friendly Interface: Built with Streamlit for an intuitive and responsive web experience.

🚀 Technologies Used
Python: The core programming language for the entire application.

Streamlit: For creating the interactive and visually appealing web user interface.

LangChain: A powerful framework for developing applications with Large Language Models, handling:

Agents: For intelligent decision-making and tool use.

Tools: KnowledgeBase_Search (for local documents) and Web_Search (powered by DuckDuckGo).

Document Loaders: PyPDFLoader, WebBaseLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader.

Text Splitters: RecursiveCharacterTextSplitter.

Chat Models: Integration with Google Gemini.

Google Generative AI (google-generativeai): The underlying Python client for interacting with the Google Gemini API.

HuggingFace Embeddings (sentence-transformers): Used to convert text into numerical vector embeddings for semantic search (model: all-MiniLM-L6-v2).

ChromaDB: A lightweight, open-source vector database used to store and query the generated text embeddings locally.

nest_asyncio: Used to resolve asyncio event loop conflicts, ensuring smooth asynchronous operations within the Streamlit environment.

duckduckgo-search: Provides the functionality for the web search tool.

💻 Setup and Local Installation
Follow these steps to set up and run the chatbot on your local machine.

Clone the repository:

git clone https://github.com/YourGitHubUsername/dynamic-rag-chatbot.git
cd dynamic-rag-chatbot

(Replace YourGitHubUsername and dynamic-rag-chatbot with your actual GitHub username and repository name.)

Create and activate a Python virtual environment:
It's highly recommended to use a virtual environment to manage project dependencies.

python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows (use PowerShell or Git Bash)

Install dependencies:
This command will install all the necessary Python libraries listed in requirements.txt.

pip install -r requirements.txt

Troubleshooting torch or chromadb: If you encounter specific compilation issues, especially with torch or chromadb, you might try installing torch first for CPU-only:

pip install --upgrade torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
# Then run the main requirements.txt install again
pip install -r requirements.txt

Set up your Google API Key:
The application requires a Google API Key to access the Gemini model.

Get an API key from Google AI Studio.

Create a .streamlit directory in your dynamic-rag-chatbot project folder if it doesn't already exist:

mkdir -p .streamlit

Inside the newly created .streamlit directory, create a file named secrets.toml.

Add your API key to secrets.toml like this:

GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"

Important: Replace YOUR_ACTUAL_GOOGLE_API_KEY_HERE with your actual API key. This file is automatically ignored by Git (due to .gitignore) to keep your key secure.

Run the Streamlit application:
Ensure your virtual environment is active (from step 2) and you are in the dynamic-rag-chatbot directory.

streamlit run rag.py

A new tab should automatically open in your default web browser, displaying the chatbot interface.

💡 Usage
Upload Documents: Use the "Upload a Document" section in the left sidebar to upload PDF, TXT, Markdown, or DOCX files. The content will be processed and added to the chatbot's knowledge base.

Load from URL: Alternatively, paste a URL into the "Enter URL" field in the sidebar and click "Load." The webpage content will be fetched and added to the knowledge base.

Chat: Once content is loaded, ask questions in the chat input at the bottom. The chatbot will intelligently use its knowledge base (for document-specific questions) or perform a web search (for general knowledge) to provide answers.

Clear Data: The "Clear Chat and Reset RAG Data" button in the sidebar will clear the chat history and delete the local ChromaDB vector store, allowing you to start fresh.

Adjust Settings: Use the sliders and selectors in the sidebar to fine-tune the document processing, language model temperature, and retrieval methods.

✨ Live Demo (Optional)
You can deploy this application to Streamlit Cloud for a live, shareable demo.
[Link to your Streamlit Cloud deployed app here once deployed, e.g., https://yourusername-dynamic-rag-chatbot.streamlit.app/]

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or find issues, please feel free to open a pull request or an issue on the GitHub repository.

📄 License
This project is licensed under the MIT License.
(You can change this to your preferred license. The MIT License is a common and permissive open-source license.)