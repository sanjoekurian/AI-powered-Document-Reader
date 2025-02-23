# 🦙 Chat with Doc - LLAMA 3.1

This is a Streamlit-based application that allows users to chat with their PDF documents using LangChain, FAISS, and the LLAMA 3.1 model.

## 🚀 Features
- Upload a PDF document and extract information via chat.
- Uses FAISS for vector-based retrieval.
- Employs LangChain with LLAMA 3.1 for responses.

## 🛠 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/chat-with-doc-llama.git
cd chat-with-doc-llama
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
- Rename `.env.example` to `.env`
- Add your API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

## 📜 License
This project is licensed under the MIT License.
