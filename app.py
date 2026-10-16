# app.py

import streamlit as st
from dotenv import load_dotenv
import os, re
from openai import OpenAI


# ────────────────────────────────────────────────────────────────────────────────
# 1. Load environment variables from .env (in project root)
#
#    Your .env file should contain:
#       AI_API_KEY=or_sk_XXXXXXXXXXXXXXXXXXXXXX
# ────────────────────────────────────────────────────────────────────────────────

# Local development: read from .env via python-dotenv.
# Streamlit Cloud: fall back to st.secrets if the env var is not set locally.
load_dotenv()
AI_API_KEY = os.getenv("AI_API_KEY")

if not AI_API_KEY:
    try:
        AI_API_KEY = st.secrets["AI_API_KEY"]
    except Exception:
        pass

if not AI_API_KEY:
    raise RuntimeError("Please set AI_API_KEY in .env or in Streamlit Secrets")

client = OpenAI(
    api_key=AI_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# ────────────────────────────────────────────────────────────────────────────────
# 2. Detailed system prompt for a medical‐knowledgeable assistant
# ────────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a medical expert and healthcare assistant.  
Your primary goals:  
  1. If the user explicitly asks for a definition or explanation (e.g., “What is cough?”, “Explain fever.”, “Define hypertension.”), provide a clear, concise medical explanation of the term or condition—its causes, symptoms, and key facts—without telling them to see a doctor.  
  2. If the user describes personal symptoms or concerns about their health (e.g., “I have a cough and fever”, “I am experiencing chest pain”, “My throat aches”), respond with general educational information about possible causes and emphasize that they should consult a qualified healthcare professional for personalized advice.  
  3. If the user asks about scheduling an appointment, medications, or prescriptions (e.g., “How do I book an appointment?”, “What medication should I take for a cold?”), remind them to contact a healthcare provider or pharmacist for specific instructions.  
  4. Always include a disclaimer that you are an AI assistant and your responses are for informational purposes only, not a substitute for professional medical advice.  
  5. Provide explanations in accessible, non‐technical terms suitable for a layperson, but include accurate medical terminology where helpful.  
  6. Do not provide dosage recommendations or diagnose specific conditions—always refer such queries to a healthcare professional.
"""

# ────────────────────────────────────────────────────────────────────────────────
# 3. Helper function to call the AI model via OpenRouter’s Chat Completions API
#    (defaults to OpenRouter's free router, which auto-selects an available free model)
# ────────────────────────────────────────────────────────────────────────────────
def ask_ai(
    prompt: str,
    model: str = "openrouter/free",
    temperature: float = 0.7
) -> str:
    """
    Send a user question (with SYSTEM_PROMPT) to the AI model via OpenRouter’s Chat API.
    Returns the assistant’s reply.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error]: {e}"

# ────────────────────────────────────────────────────────────────────────────────
# 4. Business logic: “healthcare_chatbot” with refined conditions
# ────────────────────────────────────────────────────────────────────────────────
def healthcare_chatbot(user_input: str) -> str:
    """
    1) If the user’s input begins with keywords indicating a request for definition/explanation
       (e.g., “what is”, “explain”, “define”), call the AI model to provide a medical explanation.
    2) If the user describes symptoms in first person (e.g., “I have a cough”), provide general
       info about possible causes and urge them to see a professional.
    3) If the user asks about appointments, medications, or prescriptions, give a reminder to contact
       a provider or pharmacist.
    4) Otherwise, forward to the AI model for an informative response.
    """
    prompt = user_input.strip()
    if not prompt:
        return "⚠️ Please enter a question."

    lower = prompt.lower()

    # 1) Definition/explanation requests
    if re.match(r"^\s*(what is|explain|define)\b", lower):
        # e.g., "What is cough?" or "Explain fever."
        return ask_ai(prompt)

    # 2) Personal symptom descriptions (first-person)
    if re.search(r"\b(i have|i am|i’m|i feel)\b", lower) and re.search(r"\b(symptom|pain|ache|fever|cough|cold|headache|nausea)\b", lower):
        # e.g., "I have a cough", "I feel a sharp pain"
        # Provide general info about symptom, then refer to professional
        symptom = re.search(r"\b(symptom|pain|ache|fever|cough|cold|headache|nausea)\b", lower).group(1)
        general_info = ask_ai(f"What are possible causes and general information about {symptom}?")
        return (
            f"{general_info}\n\n"
            "⚠️ Disclaimer: This information is educational only and not a substitute for professional medical advice. "
            "Please consult a qualified healthcare professional for personalized guidance."
        )

    # 3) Appointment / medication / prescription queries
    if re.search(r"\b(appointment|doctor|physician|clinic)\b", lower):
        return "📅 For appointment scheduling or physician referrals, please contact your healthcare provider directly."

    if re.search(r"\b(medication|prescription|drugs|pills)\b", lower):
        return "💊 For medication or prescription concerns, please consult your doctor or pharmacist."

    # 4) All other queries: forward to the AI model for an informative response
    return ask_ai(prompt)

# ────────────────────────────────────────────────────────────────────────────────
# 5. Streamlit UI — Blue/Teal Medical Theme
# ────────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary-teal: #007A7A;
    --primary-teal-dark: #005f5f;
    --teal-bg: #e6f4f4;
    --teal-border: #b8d8d8;
    --user-msg-bg: #e3f2fd;
    --user-msg-border: #90caf9;
    --assistant-msg-bg: #f1f8ff;
    --assistant-msg-border: #cce5ff;
    --sidebar-bg: #f8fafc;
    --text-main: #1f2937;
    --text-muted: #6b7280;
}

html, body, [class*="st"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 800px;
}

.app-header {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}
.app-title {
    font-size: 2.25rem;
    font-weight: 700;
    color: var(--primary-teal-dark);
    margin: 0;
    letter-spacing: -0.5px;
}
.app-subtitle {
    font-size: 1rem;
    color: var(--text-muted);
    margin: 0.25rem 0 0 0;
    font-weight: 500;
}

disclaimer-banner {
    background: var(--teal-bg);
    border: 1px solid var(--teal-border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 1.5rem 0;
    font-size: 0.875rem;
    color: var(--text-main);
    text-align: center;
}
disclaimer-banner b {
    color: var(--primary-teal-dark);
}

.welcome-msg {
    text-align: center;
    color: var(--text-muted);
    padding: 2rem 1rem;
    font-size: 1rem;
}

.stChatInputContainer {
    padding-top: 0.5rem;
}

section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
}
.sidebar-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--primary-teal-dark);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.75rem;
}

.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    border: 1px solid var(--teal-border) !important;
    color: var(--text-main) !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.75rem !important;
    font-size: 0.85rem !important;
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 0.5rem !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--teal-bg) !important;
    border-color: var(--primary-teal) !important;
    color: var(--primary-teal-dark) !important;
}

.stButton > button[kind="primary"] {
    background: var(--primary-teal) !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--primary-teal-dark) !important;
}

.stSpinner > div {
    border-color: var(--primary-teal) transparent transparent transparent !important;
}
</style>"""


def apply_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>💡 Try asking</div>", unsafe_allow_html=True)
        sample_queries = [
            "What is hypertension?",
            "Explain diabetes.",
            "I have a fever and cough.",
            "How do I book an appointment?",
            "What medication for a cold?",
        ]
        for query in sample_queries:
            if st.button(query, key=f"sample_{hash(query)}", use_container_width=True):
                st.session_state.pending_query = query
        st.markdown("---")
        if st.button("🗑️ Clear Chat", key="clear_chat", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()
        st.markdown("---")
        st.markdown("<div class='sidebar-title'>ℹ️ About</div>", unsafe_allow_html=True)
        st.caption(
            "🤖 Built with Streamlit + OpenRouter. "
            "👨‍⚕️ A personal project by Vedant Roy."
        )


def render_chat_history():
    if "messages" not in st.session_state or not st.session_state.messages:
        st.markdown(
            "<div class='welcome-msg'>"
            "👋 👨‍⚕️ Hello! I'm your healthcare assistant. "
            "Ask me about medical terms, symptoms, or general health questions."
            "</div>",
            unsafe_allow_html=True,
        )
        return
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])


def main():
    st.set_page_config(
        page_title="Healthcare Assistant Chatbot",
        page_icon="👨‍⚕️",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    st.markdown(
        "<div class='app-header'>"
        "<h1 class='app-title'>👨‍⚕️ Healthcare Assistant Chatbot</h1>"
        "<p class='app-subtitle'>by Vedant Roy</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='disclaimer-banner'>"
        "<b>⚠️ Disclaimer:</b> This chatbot is for demonstration purposes only. "
        "Any medical information provided is not a substitute for professional advice. "
        "Always consult a healthcare professional for personalized guidance."
        "</div>",
        unsafe_allow_html=True,
    )
    render_sidebar()
    render_chat_history()
    prompt = None
    if st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None
    user_input = st.chat_input("Type your health question here…")
    final_input = prompt if prompt else user_input
    if final_input:
        st.session_state.messages.append({"role": "user", "content": final_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(final_input)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI is thinking…"):
                answer = healthcare_chatbot(final_input)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    main()
