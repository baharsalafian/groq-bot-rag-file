import os
import streamlit as st
from groq import Groq
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader

# -----------------------
# Initialize session state
# -----------------------
if "docs" not in st.session_state:
    st.session_state.docs = []

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# Groq API
# -----------------------
groq_api_key = "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"
client = Groq(api_key=groq_api_key)

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Personalization")
model = st.sidebar.selectbox(
    'Choose a model', ['Llama3-8b-8192', 'Llama3-70b-8192','Mixtral-8x7b-32768','Gemma-7b-It']
)

st.title("💬 RAG Multi-Turn Chat with Groq's LLM")

# -----------------------
# Document upload
# -----------------------
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
if uploaded_file is not None and not st.session_state.docs:
    with open("temp_uploaded_file", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.type == "text/plain" or uploaded_file.name.endswith(".txt"):
        loader = TextLoader("temp_uploaded_file", encoding="utf-8")
    elif uploaded_file.type == "application/pdf" or uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader("temp_uploaded_file")
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                "application/msword"] or uploaded_file.name.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader("temp_uploaded_file")
    else:
        st.error("Unsupported file type!")
        loader = None

    if loader:
        loaded_docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        for d in loaded_docs:
            st.session_state.docs.extend(text_splitter.split_text(d.page_content))
        st.success(f"Document split into {len(st.session_state.docs)} chunks!")

# -----------------------
# Render chat messages
# -----------------------
for message in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["response"])

# -----------------------
# User input box (always at the bottom)
# -----------------------
if prompt := st.chat_input("Enter your question:"):
    # Build context from docs (first 3 chunks as naive example)
    if st.session_state.docs:
        context_text = "\n".join(st.session_state.docs[:3])
        final_prompt = f"Answer the question based on the context below:\n\nContext:\n{context_text}\n\nQuestion: {prompt}"
    else:
        final_prompt = prompt

    # Append user message immediately
    st.session_state.history.append({"query": prompt, "response": ""})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get Groq response
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}],
        model=model,
    )
    response = chat_completion.choices[0].message.content

    # Update the last history entry with response
    st.session_state.history[-1]["response"] = response
    with st.chat_message("assistant"):
        st.markdown(response)
