import streamlit as st
from workflow.graph import create_app
import time 
import re

# ================================
# Initialize LangGraph App
# ================================
app = create_app()

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="NexaAI Assistant",
    page_icon="🤖",
    layout="wide",
)

# ================================
# Premium CSS (ChatGPT Style)
# ================================
st.markdown(
    """
    <style>
    body { background-color: #0f172a; }
    .chat-container { max-width: 900px; margin: auto; }
    .user-msg {
        background: #2563eb;
        color: white;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 8px 0;
        text-align: right;
    }
    .bot-msg {
        background: #111827;
        color: #e5e7eb;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 8px 0;
        border: 1px solid #334155;
        white-space: pre-line;
    }
    .header-title {
        font-size: 45px;
        font-weight: bold;
        color: #22c55e;
        text-align: center;
        margin: 0 auto;
        
    }
    .subtitle {
        color: #9ca3af;
        font-size: 24px;
        text-align: center;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================
# Sidebar Settings Panel
# ================================
with st.sidebar:
    st.title("⚙️ NexaAI Settings")
    retrieval_toggle = st.toggle("Enable Vector Retrieval", True)
    web_toggle = st.toggle("Enable Web Search Fallback", True)
    st.markdown("---")
    st.write("Built with LangGraph + Streamlit")

# ================================
# Header
# ================================
st.markdown("<div class='header-title'>🤖 NexaAI Knowledge Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask anything about NexaAI docs, policies, or knowledge base.</div>", unsafe_allow_html=True)
st.markdown("---")

# ================================
# Session State
# ================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store AI result safely
if "result" not in st.session_state:
    st.session_state.result = None

# ================================
# Chat Container
# ================================
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ================================
# Chat Input Box
# ================================
question = st.chat_input("Type your question...")

if question:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": question})

    # AI Thinking
    with st.spinner("NexaAI is thinking..."):
        time.sleep(0.5)

        state_input = {
            "question": question,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer": "",
            "need_retrieval": retrieval_toggle,
            "need_web_search": web_toggle,
        }

        # Invoke LangGraph
        st.session_state.result = app.invoke(state_input)

        def clean_markdown(text):
            
            return re.sub(r"^\s*[\*\-\•]\s*", "", text, flags=re.MULTILINE)

        # After AI response
        answer = (
            st.session_state.result.get("answer")
            or st.session_state.result.get("output")
            or "No response generated."
            )

        answer = clean_markdown(answer)



        
        context = st.session_state.result.get("context", "")

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Refresh UI
    st.rerun()

# ================================
# Debug Panel
# ================================
with st.expander("🧠 Retrieved Context / Debug Info"):
    if st.session_state.result:
        st.write("### Context:")
        st.write(st.session_state.result.get("context", "No context"))

        st.write("### Full LangGraph State:")
        st.json(st.session_state.result)
    else:
        st.write("No debug data yet. Ask a question first.")



