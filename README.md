# 🤖 Multi-AI Assistant (RAG + Memory + Routing)

A smart conversational AI assistant that can answer questions across multiple domains like **movies, nutrition, technology, and general topics** using:

- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 Context-aware conversation (chat history)
- 💾 Redis-based memory (session storage)
- 🎯 Domain-based query routing
- ⚡ Fast LLM inference using Groq

---

## 🚀 Features

- ✅ Multi-domain support (movies, nutrition, tech, general)
- ✅ Intelligent query routing
- ✅ RAG (Vector DB using Chroma)
- ✅ Context-aware conversations
- ✅ Redis session memory
- ✅ Greeting & small-talk handling
- ✅ Spelling correction using LLM
- ✅ Streamlit UI (chat interface)

---

## 🏗️ Project Structure

mini_chatbot/
│
├── app.py # Streamlit UI
├── chains/
│ ├── rag_chain.py # Core LLM + RAG logic
│ ├── router.py # Domain classifier
│
├── utils/
│ └── redis_memory.py # Chat memory (Redis)
│
├── data/
│ ├── raw/ # Raw datasets
│ ├── processed/ # Cleaned datasets
│
├── scripts/
│ ├── ingest.py # Load data into vector DB
│ ├── preprocess.py # Data cleaning
│ ├── query_all.py # Testing scripts
│
├── vector_db/ # Chroma vector database
├── requirements.txt
├── .env # API keys (NOT pushed)
└── README.md