# 🚑 AI-Powered Healthcare Assistant Chatbot

This repository contains a **personal project** demonstrating how to build a simple, yet powerful, AI-powered healthcare assistant using Python, Streamlit, and OpenRouter-hosted large language models. The chatbot can answer basic medical questions, explain common health terms, and provide general advice—all for educational purposes.

---

## 🎯 Project Overview

This chatbot aims to assist users with:
- 🩺 **Symptom Checking**
- 💊 **Medication Information**
- 🧠 **Basic Mental Health Support**
- 📚 **General Health Queries**

- **What it does:**  
  - Provides definitions and explanations for medical terms (e.g., “What is cough?”, “Explain hypertension.”)  
  - Offers general information about symptoms when described in first person (e.g., “I have a fever.”)  
  - Reminds users to consult professionals for appointment scheduling or medication queries  
  - Combines rule-based logic for simple keyword detection with AI-generated responses for deeper explanations  

- **Why I built it:**  
  - To learn how to integrate an LLM via OpenRouter’s API into a Streamlit app  
  - To explore basic natural language processing techniques (keyword matching, regex) for handling health-related queries  
  - To create a standalone, locally runnable chatbot that showcases how AI can assist with educational healthcare content  

- **Key Technologies:**  
  - **Streamlit**: For a clean, interactive web UI  
  - **OpenAI Python SDK (pointing to OpenRouter)**: To call any OpenRouter-hosted model (“free” tier)  
  - **Python 3.x** + standard libraries (`os`, `re`, `dotenv`) for configuration and logic  

---

## 📁 Project Structure

```

healthcare-chatbot/
├── app.py               # Main Streamlit application
├── requirements.txt     # All Python dependencies
├── .gitignore           # Ignore patterns (e.g., .env)
└── README.md            # (You are here!)

````

- **app.py**  
  - Contains the entire chatbot logic: loading environment variables, defining a system prompt, handling user input, applying keyword rules, and calling the LLM.
  - Hosts a Streamlit UI that displays a text area for user questions and shows the AI’s responses.

- **requirements.txt**  
  - Lists all packages needed to run the project:
    ```
    python-dotenv
    openai
    streamlit
    ```

- **.gitignore**  
  - Ensures that your local `.env` file (with personal API keys) is not committed to GitHub.

---

## 🚀 How the Code Works

### 1. Configuration & Authentication

- **Environment Variables**  
  - The app expects a file named `.env` (in the project root) containing:
    ```
    AI_API_KEY=or_sk_XXXXXXXXXXXXXXXXXXXXXX
    ```
  - We use `python-dotenv` (`load_dotenv()`) to load that key into `os.getenv("AI_API_KEY")`.  
  - On Streamlit Community Cloud (if you choose to deploy), you would add the same key under **Settings → Secrets** instead of using a local `.env`.

- **OpenRouter / LLM Client**  
  - We instantiate an `OpenAI` client (from the official OpenAI Python library) but point it to OpenRouter’s API:
    ```python
    client = OpenAI(
        api_key=AI_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    ```
  - This allows us to call any OpenRouter-hosted model (in this case, `openrouter/free`).

### 2. System Prompt & Helper Function

- **SYSTEM_PROMPT**  
  ```python
  SYSTEM_PROMPT = """
  You are a medical expert and healthcare assistant.
  Your primary goals:
    1. If the user explicitly asks for a definition or explanation (e.g., “What is cough?”, “Explain fever.”, “Define hypertension.”), 
       provide a clear, concise medical explanation of the term or condition—its causes, symptoms, and key facts—without telling them to see a doctor.
    2. If the user describes personal symptoms or concerns about their health (e.g., “I have a cough and fever”, “I am experiencing chest pain”, 
       “My throat aches”), respond with general educational information about possible causes and emphasize that they should consult 
       a qualified healthcare professional for personalized advice.
    3. If the user asks about scheduling an appointment, medications, or prescriptions (e.g., “How do I book an appointment?”, 
       “What medication should I take for a cold?”), remind them to contact a healthcare provider or pharmacist for specific instructions.
    4. Always include a disclaimer that you are an AI assistant and your responses are for informational purposes only, 
       not a substitute for professional medical advice.
    5. Provide explanations in accessible, non-technical terms suitable for a layperson, but include accurate medical terminology where helpful.
    6. Do not provide dosage recommendations or diagnose specific conditions—always refer such queries to a healthcare professional.
  """


* This prompt is prepended to every AI call to ensure the model behaves as a knowledgeable, responsible medical assistant.

* **ask_ai(prompt, …)**

  ```python
  def ask_ai(prompt: str, model="openrouter/free", ...):
      response = client.chat.completions.create(
          model=model,
          messages=[
              {"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user",   "content": prompt}
          ],
          temperature=temperature,
          max_tokens=max_tokens
      )
      return response.choices[0].message.content.strip()
  ```

  * Sends a two-message chat: the `SYSTEM_PROMPT` + the user’s query.
  * Returns the LLM’s reply as plain text.

### 3. Business Logic & Keyword Rules

```python
def healthcare_chatbot(user_input: str) -> str:
    prompt = user_input.strip()
    if not prompt:
        return "⚠️ Please enter a question."

    lower = prompt.lower()

    # 1) Definition / Explanation Requests
    if re.match(r"^\s*(what is|explain|define)\b", lower):
        return ask_ai(prompt)
    if re.search(r"\b(i have|i am|i’m|i feel)\b", lower) and re.search(r"\b(symptom|pain|ache|fever|cough|cold|headache|nausea)\b", lower):
        symptom = re.search(r"\b(symptom|pain|ache|fever|cough|cold|headache|nausea)\b", lower).group(1)
        general_info = ask_ai(f"What are possible causes and general information about {symptom}?")
        return (
            f"{general_info}\n\n"
            "⚠️ Disclaimer: This information is educational only and not a substitute for professional medical advice. "
            "Please consult a qualified healthcare professional for personalized guidance."
        )

    # 3) Appointment / Medication Queries
    if re.search(r"\b(appointment|doctor|physician|clinic)\b", lower):
        return "📅 For appointment scheduling or physician referrals, please contact your healthcare provider directly."
    if re.search(r"\b(medication|prescription|drugs|pills)\b", lower):
        return "💊 For medication or prescription concerns, please consult your doctor or pharmacist."

    # 4) Fallback: Forward to the AI model
    return ask_ai(prompt)
```

1. **Definition/explanation**:

   * Detects queries starting with “what is”, “explain”, or “define”.
   * Forwards those directly to `ask_ai(…)`.

2. **Personal symptom descriptions**:

   * Checks if the user says “I have” (or “I’m/I am/I feel”) plus a symptom keyword (e.g., cough, fever).
   * Extracts that symptom (e.g., `“cough”`) and asks the LLM for “possible causes and general information about cough.”
   * Adds a mandatory disclaimer at the end.

3. **Appointment / medication**:

   * Any mention of “appointment”, “doctor”, “clinic” → returns a static reminder to contact a provider.
   * Any mention of “medication”, “prescription”, “drugs” → returns a static reminder to consult a doctor or pharmacist.

4. **Fallback**:

   * If none of the above apply, ask the LLM with the original prompt for a more general answer.

### 4. Streamlit Front-End

```python
def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", layout="centered")
    st.title("Healthcare Assistant Chatbot by [Your Name]")
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
            with st.spinner("AI is thinking..."):
                answer = healthcare_chatbot(user_input)
            st.markdown(f"**Assistant:** {answer}")
```

* **st.title** and **st.markdown**: show a title and a small disclaimer.
* **st.text\_area**: multi-line input for the user’s question.
* **st.button**: triggers the `healthcare_chatbot(...)` function when clicked.
* **st.spinner**: shows an “AI is thinking…” message while waiting.
* **st.markdown**: displays the returned `answer` below the text area.

---

## ⚙️ Installation & Setup (Run Locally on PC)

### 🔑 Prerequisites

* **Python 3.8+** installed on your system
* A **virtual environment** (optional, but recommended)
* A free **OpenRouter API key** (`openrouter/free`)

  * Sign up at [openrouter.ai](https://openrouter.ai), create an API key (it will start with `or_sk_…`)

### 📥 Step-by-Step Guide

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<YOUR_USERNAME>/healthcare-chatbot.git
   cd healthcare-chatbot
   ```

2. **Create & Activate a Virtual Environment** (recommended)

   * macOS / Linux:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * Windows (Command Prompt):

     ```bat
     python -m venv venv
     venv\Scripts\activate
     ```
   * Windows (PowerShell):

     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` File**
   In the project root, create a file named `.env` with exactly:

   ```
   AI_API_KEY=or_sk_XXXXXXXXXXXXXXXXXXXXXX
   ```

   (Replace `or_sk_…` with your actual OpenRouter API key.)

5. **Verify `.gitignore`**
   Ensure your `.gitignore` contains at least:

   ```
   .env
   __pycache__/
   *.pyc
   ```

   This prevents your API key from being committed.

6. **Run the Chatbot Locally**

   ```bash
   streamlit run app.py
   ```

   * Streamlit will start a local server (default at `http://localhost:8501`).
   * Open your browser and visit that URL.

7. **Interact with the Bot**

   * Type queries like “What is cough?” or “I have a headache and nausea” into the text area.
   * Click **Submit** and wait a few seconds for the AI model to respond.

---

## 📋 Requirements (requirements.txt)

```plaintext
python-dotenv
openai
streamlit
```

> **Note:**
>
> * `python-dotenv`: Loads environment variables from `.env`.
> * `openai`: Official OpenAI Python SDK (pointed at OpenRouter).
> * `streamlit`: For the interactive web app interface.

---

## 📊 Sample Queries

* **“What are the symptoms of COVID-19?”**
  → AI generates a concise definition of COVID-19 and its symptoms.

* **“Explain hypertension.”**
  → AI provides a medical overview of high blood pressure, risk factors, management tips.

* **“I have a fever and cough.”**
  → AI describes possible causes (e.g., viral infection, common cold, flu) and gives a disclaimer to see a doctor.

* **“How do I book an appointment?”**
  → Static reply reminding the user to contact their healthcare provider.

* **“What medication should I take for a cold?”**
  → Static reply advising to consult a pharmacist or doctor for prescriptions.

---

## 🔍 How to Customize

1. **Change the Model**

   * In `ask_ai(…)`, you can swap the model string from
     `"openrouter/free"` to any other OpenRouter-hosted model (e.g., `google/gemma-3n-e4b-it:free`), provided you have token quota.

2. **Adjust the System Prompt**

   * Modify `SYSTEM_PROMPT` at the top of `app.py` to tweak the assistant’s tone, level of detail, or add additional constraints (e.g., focus on mental health).

3. **Add More Keywords**

   * In `healthcare_chatbot(…)`, extend the `re.search(...)` patterns to capture terms like “anxiety”, “depression”, “diabetes”, etc., to trigger either static replies or targeted AI calls.

---

## 📈 Future Enhancements

* **Voice-Enabled Interface**: Integrate Web Speech API (JavaScript) or `streamlit-webrtc` for speech-to-text and text-to-speech, so users can talk to the bot.
* **EHR Integration**: Pull real patient data (with proper authentication) to provide personalized information.
* **Multilingual Support**: Detect user language and forward to a multilingual LLM for responses in Spanish, Hindi, etc.
* **Enhanced Symptom Checker**: Build a decision-tree logic or integrate a medical knowledge graph to give more detailed differential diagnoses.
* **Deployment to Streamlit Community Cloud**: Follow [Streamlit’s deployment guide](https://docs.streamlit.io/) and add `AI_API_KEY` as a secret in the web UI so the app runs in the cloud.

---

## 🤝 Acknowledgments

* **OpenRouter** for hosting free AI models
* **Streamlit** team for an easy-to-use Python web framework
* **Python community** for maintaining open-source libraries

---

## 📬 🌐 **Connect with Me:**

**Your Name** – [vedantroy3@gmail.com](mailto:vedantroy3@gmail.com)

- [LinkedIn](https://www.linkedin.com/in/vedant-roy-b58117227/) 💼
- [GitHub](https://github.com/Vedantroy475) 💻

---

Happy coding! 🚀
This chatbot is entirely a **personal project**, built to showcase how AI can be integrated into a lightweight web app for educational healthcare assistance. Feel free to fork, modify, and experiment!
