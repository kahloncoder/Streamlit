import streamlit as st
#import google.generativeai as genai # Keep genai import if you might use its specific features later, otherwise optional
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
# import json # Not used in the final code, can be removed
# import os # REMOVED - No longer needed for API key

# --- Configuration and Setup ---
st.set_page_config(page_title="Multi-Persona AI Advisor", page_icon="🤖", layout="wide")

# --- Personas Dictionary ---
PERSONAS = {
    "Machine Learning Engineer": {
        "description": "An expert in AI, machine learning, and data science.",
        "expertise": """
        You are a senior machine learning engineer with 15 years of experience.
        Provide in-depth, technical advice about:
        - Machine learning algorithms (theory and implementation)
        - Deep learning architectures (CNNs, RNNs, Transformers, etc.)
        - Data preprocessing and feature engineering techniques
        - Model evaluation, tuning, and optimization strategies
        - MLOps (deployment, monitoring, scaling)
        - Current research trends and breakthroughs
        Use precise technical language, provide code examples (Python preferred) where applicable,
        and offer practical, implementable solutions grounded in best practices.
        """,
        "conversation_style": "Technical, precise, analytical, provide code examples and references"
    },
    "Full Stack Developer": {
        "description": "An experienced full-stack web and mobile developer.",
        "expertise": """
        You are a senior full-stack developer with expertise in modern web and potentially mobile technologies.
        Provide comprehensive advice about:
        - Frontend frameworks (React, Vue, Angular, Svelte, etc.) and state management
        - Backend technologies (Node.js/Express, Python/Django/Flask, Ruby/Rails, Go, Java/Spring)
        - Database design (SQL, NoSQL) and ORMs/ODMs
        - API design (REST, GraphQL)
        - Cloud deployment (AWS, GCP, Azure), serverless, containers (Docker, Kubernetes)
        - System architecture and microservices patterns
        - Testing methodologies (unit, integration, e2e)
        - CI/CD pipelines and DevOps practices
        - Web security best practices
        Offer practical solutions, architecture diagrams (using markdown/mermaid if possible),
        code snippets, and strategic implementation advice.
        """,
        "conversation_style": "Practical, solution-oriented, technical depth, explain trade-offs"
    },
    "Medical Doctor": {
        "description": "A compassionate and knowledgeable healthcare professional (for informational purposes only).",
        "expertise": """
        You are a board-certified physician providing general health information for educational purposes.
        Provide insights into:
        - Common medical conditions and their general symptoms/treatments
        - Preventative health measures and wellness recommendations
        - Explanation of medical terminology and procedures in simple terms
        - Interpretation of general health news and studies
        ALWAYS EMPHASIZE:
        - This is NOT medical advice.
        - Information provided is for general knowledge only.
        - Users MUST consult a qualified healthcare professional for diagnosis and treatment.
        - Do not ask for or store personally identifiable health information.
        Explain medical concepts clearly, compassionately, and accurately based on current evidence.
        Avoid definitive diagnoses or treatment plans.
        """,
        "conversation_style": "Empathetic, clear, scientifically accurate, cautious, strongly disclaimer-focused"
    },
    "Financial Advisor": {
        "description": "An experienced financial planning and investment expert (for informational purposes only).",
        "expertise": """
        You are a certified financial planner providing general financial education and information.
        Provide insights into:
        - Personal finance concepts (budgeting, saving, debt management)
        - Investment principles (asset allocation, diversification, risk tolerance)
        - Retirement planning strategies (401k, IRA, pensions)
        - General tax concepts (optimization strategies are complex and individual)
        - Risk management and insurance basics
        ALWAYS EMPHASIZE:
        - This is NOT financial advice.
        - Information is for educational purposes only.
        - Investment involves risk, including potential loss of principal.
        - Users MUST consult a qualified and licensed financial advisor before making any financial decisions.
        - Do not ask for or store sensitive personal financial data.
        Offer clear explanations of financial concepts, discuss general strategies, and use data/examples where appropriate.
        Avoid specific investment recommendations.
        """,
        "conversation_style": "Professional, objective, data-informed, cautious, strongly disclaimer-focused"
    },
    "Career Coach": {
        "description": "A professional career development and guidance expert.",
        "expertise": """
        You are an experienced career coach specializing in professional development and guidance.
        Provide comprehensive advice including:
        - Career path exploration and planning
        - Resume/CV building and optimization strategies
        - Cover letter writing tips
        - Interview preparation techniques (behavioral, technical)
        - Skill development and learning strategies
        - Professional networking tactics
        - Salary negotiation guidance (general principles)
        - Career transition planning and execution
        Offer motivational support, actionable steps, and strategic insights based on common best practices in career development.
        """,
        "conversation_style": "Motivational, supportive, strategic, actionable, encouraging"
    }
}

# --- Securely Access API Key using Streamlit Secrets ---
try:
    # Attempt to get the API key from Streamlit secrets
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not GOOGLE_API_KEY: # Check if the key exists but is empty
         st.error("🚨 Google API Key found in secrets, but it appears to be empty. Please check your Streamlit Cloud secret configuration.")
         st.stop() # Stop execution if key is empty
except KeyError:
    # Handle the case where the secret key doesn't exist at all
    st.error("🚨 Google API Key not found! Please add `GOOGLE_API_KEY = 'YOUR_KEY'` to your Streamlit secrets.")
    st.info("Refer to Streamlit documentation on how to add secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management")
    st.stop() # Stop execution if key is missing

# --- Initialize Gemini LLM ---
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        # Safety settings can be adjusted if needed, e.g.:
        # safety_settings={
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        # }
        # Adding this might help if system prompts cause issues, sometimes needed for Gemini via Langchain
        convert_system_message_to_human=True
    )
    # Optional: Perform a quick test call to ensure the key is valid (can add cost)
    # llm.invoke("Hello!")

except Exception as e:
    st.error(f"😥 Failed to initialize the AI model. Please verify your API key in Streamlit secrets and ensure it's valid for the Gemini API. Error: {e}")
    st.stop() # Stop execution if LLM fails to initialize

# --- Helper Function for Prompts ---
def create_persona_prompt(persona_name):
    """
    Create a dynamic Langchain prompt template for the selected persona.
    """
    persona_info = PERSONAS[persona_name]
    # Updated template structure for better clarity with Langchain ConversationChain
    template = f"""
You are role-playing as a '{persona_name}'.
{persona_info['expertise']}

Conversation Style: {persona_info['conversation_style']}

Current conversation:
{{history}}
Human: {{input}}
AI ({persona_name}):"""

    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    return prompt_template

# --- Main Application Logic ---
def main():
    st.title("🤖 Multi-Persona AI Advisor")
    st.caption("Select an advisor persona from the sidebar and start chatting!")

    # --- Persona Selection Sidebar ---
    with st.sidebar:
        st.header("Choose Your Advisor")
        selected_persona = st.selectbox(
            "Select Persona:",
            list(PERSONAS.keys()),
            key="persona_select" # Add a key for stability
        )

        # Display Persona Description
        st.markdown("---")
        st.subheader(f"About: {selected_persona}")
        st.write(PERSONAS[selected_persona]["description"])
        st.markdown("---")
        st.info("⚠️ AI responses are informational and may not always be accurate or complete. Use caution, especially with medical or financial topics.")

    # --- Session State Initialization ---
    # Use unique keys based on persona to potentially store separate histories later if desired
    # For now, we'll use one memory store, but changing persona effectively restarts context
    # due to the prompt changing. A more advanced version might manage separate memory objects.
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=False) # return_messages=False is often better for raw history string
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Clear history if persona changes (optional, but often makes sense)
    if 'current_persona' not in st.session_state or st.session_state.current_persona != selected_persona:
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.current_persona = selected_persona
        st.rerun() #

    # --- Chat Interface ---
    # Display existing chat messages
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # --- Handle User Input ---
    if prompt := st.chat_input(f"Ask {selected_persona}..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # --- Generate AI Response ---
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner(f"Thinking as {selected_persona}..."):
                try:
                    # Create persona-specific prompt and conversation chain *each time*
                    # This ensures the correct persona context is used for the current turn
                    persona_prompt = create_persona_prompt(selected_persona)

                    conversation = ConversationChain(
                        llm=llm,
                        prompt=persona_prompt,
                        memory=st.session_state.memory, # Use the shared memory
                        verbose=False # Set to True for debugging Langchain steps
                    )

                    # Generate response
                    response = conversation.predict(input=prompt)

                    # Display the response
                    message_placeholder.markdown(response)

                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
                    response = f"Sorry, I encountered an error trying to respond as {selected_persona}."
                    message_placeholder.markdown(response)


        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Run the App ---
if __name__ == "__main__":
    main()