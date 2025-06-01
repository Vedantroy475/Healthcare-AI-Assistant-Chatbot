# app.py

import streamlit as st
from dotenv import load_dotenv
import os, re
from openai import OpenAI


# ────────────────────────────────────────────────────────────────────────────────
# 1. Load environment variables from .env (in project root)
#
#    Your .env file should contain:
#       Deepseek_API_KEY=or_sk_XXXXXXXXXXXXXXXXXXXXXX
# ────────────────────────────────────────────────────────────────────────────────
# No need to load .env at runtime on Streamlit Cloud
# Instead, read from st.secrets (fallback to os.getenv locally)

if st.runtime.exists():  
    # When running on Streamlit Cloud, st.secrets will have our key
    Deepseek_API_KEY = st.secrets["Deepseek_API_KEY"]
else:
    # Local development fallback: read from .env
    load_dotenv()
    Deepseek_API_KEY = os.getenv("Deepseek_API_KEY")

if not Deepseek_API_KEY:
    raise RuntimeError("Please set Deepseek_API_KEY in .env or in Streamlit Secrets")

client = OpenAI(
    api_key=Deepseek_API_KEY,
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
# 3. Helper function to call Deepseek R1 0528 Qwen3 8B (free) via OpenRouter’s Chat Completions API
# ────────────────────────────────────────────────────────────────────────────────
def ask_deepseek(
    prompt: str,
    model: str = "deepseek/deepseek-r1-0528-qwen3-8b:free",
    temperature: float = 0.7
) -> str:
    """
    Send a user question (with SYSTEM_PROMPT) to Deepseek R1 0528 Qwen3 8B (free) via OpenRouter’s Chat API.
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
       (e.g., “what is”, “explain”, “define”), call Deepseek to provide a medical explanation.
    2) If the user describes symptoms in first person (e.g., “I have a cough”), provide general
       info about possible causes and urge them to see a professional.
    3) If the user asks about appointments, medications, or prescriptions, give a reminder to contact
       a provider or pharmacist.
    4) Otherwise, forward to Deepseek for an informative response.
    """
    prompt = user_input.strip()
    if not prompt:
        return "⚠️ Please enter a question."

    lower = prompt.lower()

    # 1) Definition/explanation requests
    if re.match(r"^\s*(what is|explain|define)\b", lower):
        # e.g., "What is cough?" or "Explain fever."
        return ask_deepseek(prompt)

    # 2) Personal symptom descriptions (first-person)
    if re.search(r"\b(i have|i am|i’m|i feel)\b", lower) and re.search(r"\b(symptom|pain|ache|fever|cough|cold|headache|nausea)\b", lower):
        # e.g., "I have a cough", "I feel a sharp pain"
        # Provide general info about symptom, then refer to professional
        symptom = re.search(r"\b(symptom|pain|ache|fever|cough|cold|headache|nausea)\b", lower).group(1)
        general_info = ask_deepseek(f"What are possible causes and general information about {symptom}?")
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

    # 4) All other queries: forward to Deepseek for an informative response
    return ask_deepseek(prompt)

# ────────────────────────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", layout="centered")
    st.title("Healthcare Assistant Chatbot by Vedant Roy")
    st.markdown(
        "<small><b>Disclaimer:</b> This chatbot is for demonstration purposes only. "
        "Any medical information provided is not a substitute for professional advice. "
        "Always consult a healthcare professional for personalized guidance.</small>",
        unsafe_allow_html=True,
    )

    user_input = st.text_area("How can I assist you today?", height=150)

    if st.button("Submit"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a message before submitting.")
        else:
            with st.spinner("Deepseek is thinking..."):
                answer = healthcare_chatbot(user_input)
            st.markdown(f"**Assistant:** {answer}")

if __name__ == "__main__":
    main()
