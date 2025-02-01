import streamlit as st
from pathlib import Path
from typing import Any
from openai import OpenAI

# # Set up minimal logging to stdout
# logging.basicConfig(
#     level=logging.ERROR,
#     format="[%(asctime)s] %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
# logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY: str = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

def load_custom_css() -> None:
    """
    Apply custom CSS styling to enhance the appearance of the Streamlit app.

    :return: None.
    """
    custom_css: str = """
    <style>
      body {
          background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
          font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      }
      .stButton > button {
          background-color: #4CAF50;
          color: white;
          border: none;
          padding: 10px 20px;
          text-align: center;
          text-decoration: none;
          font-size: 16px;
          margin: 4px 2px;
          cursor: pointer;
          border-radius: 12px;
          transition: background-color 0.3s;
      }
      .stButton > button:hover {
          background-color: #45a049;
      }
      .card {
          background-color: #fff;
          padding: 20px;
          margin: 20px 0;
          border-radius: 15px;
          box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
      }
      .header {
          text-align: center;
          padding: 10px;
          font-size: 36px;
          font-weight: bold;
          color: #333;
      }
      .concept-card {
          background-color: #2C2C2C;
          padding: 15px 20px;
          border-radius: 10px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          margin-bottom: 20px;
      }
      .description-input textarea {
          border: 2px solid #4CAF50;
          border-radius: 5px;
          padding: 10px;
          font-size: 16px;
      }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def safe_chat_completion_create(**kwargs: Any) -> tuple[Any, str | None]:
    """
    Execute the chat API call with the provided parameters.

    :param kwargs: Arbitrary keyword arguments to pass to the OpenAI API.
    :return: A tuple with the response object (if successful) and an error string (or None).
    """
    try:
        response = client.chat.completions.create(**kwargs)  # ✅ Corrected method
        return response, None
    except Exception as e:
        # logger.error(f"OpenAI API call failed: {e}")
        return None, f"OpenAI API call failed: {e}"


def generate_random_concept() -> str:
    """
    Generate a random concept or word using GPT-4o-mini.

    :return: A string representing the generated concept.
    """
    params: dict = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI that comes up with a guessable concept."
            },
            {
                "role": "user",
                "content": "Generate a single  concept or word that is guessable but not too obscure. Please avoid cliché and overused words like something with 'droom', 'zon' or similar terms. It should be accessible and it should be in Dutch."
            }
        ],
        "temperature": 1
    }
    response, error = safe_chat_completion_create(**params)
    if error:
        st.error(error)
        return "Error! Could not generate concept."
    concept: str = response.choices[0].message.content.strip()
    return concept


def verify_guess(original_concept: str, guess: str) -> bool:
    """
    Use GPT-4o to verify if the guessed concept matches the original concept.

    :param original_concept: The concept that should be guessed.
    :param guess: The user's guess or the LLM's guess of the concept.
    :return: True if the guess is correct, False otherwise.
    """
    prompt: str = (
        f"Original concept: '{original_concept}'\n"
        f"Guess: '{guess}'\n"
        "Are they the same concept? Answer only with yes or no."
    )
    params: dict = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful AI that responds with either 'yes' or 'no'."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    response, error = safe_chat_completion_create(**params)
    if error:
        st.error(error)
        return False
    answer: str = response.choices[0].message.content.strip().lower()
    return "yes" in answer


def check_if_cheating(concept: str, user_description: str) -> bool:
    """
    Use GPT-4o-mini to determine if the user is cheating by revealing the concept name or synonyms in their description.

    :param concept: The concept the user is supposed to describe.
    :param user_description: The user's description.
    :return: True if the user is cheating, otherwise False.
    """
    prompt: str = (
        f"Concept: '{concept}'\n"
        f"User description: '{user_description}'\n"
        "Determine if the user's description reveals the concept. If the description includes the exact concept respond with CHEATING. Otherwise, respond with NOT CHEATING. "
        "Please respond with exactly one word (in all caps): either CHEATING or NOT CHEATING."
    )
    params: dict = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an AI that checks if the user is cheating by revealing the concept directly. Respond only with CHEATING or NOT CHEATING."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    response, error = safe_chat_completion_create(**params)
    if error:
        st.error(error)
        return False
    verdict: str = response.choices[0].message.content.strip().upper()
    return verdict == "CHEATING"


def describe_concept(concept: str) -> str:
    """
    Let GPT-4o describe a concept for the user to guess.

    :param concept: The concept to be described.
    :return: A string containing the description of the concept.
    """
    params: dict = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a playful AI giving cryptic hints."},
            {"role": "user", "content": f"Describe the concept '{concept}' without revealing its name."}
        ],
        "temperature": 0.7
    }
    response, error = safe_chat_completion_create(**params)
    if error:
        st.error(error)
        return "Error describing concept. See logs."
    description: str = response.choices[0].message.content.strip()
    return description


def store_concept_locally(concept: str) -> None:
    """
    Store the concept in a local file using an OS-independent path.

    :param concept: The concept to store locally.
    :return: None.
    """
    folder: Path = Path("concept_storage")
    folder.mkdir(exist_ok=True)
    filepath: Path = folder.joinpath("current_concept.txt")
    filepath.write_text(concept)


def swap_roles(current_role: str) -> str:
    """
    Swap roles from 'User describes, LLM guesses' to 'LLM describes, User guesses', or vice versa.

    :param current_role: The current role mode string.
    :return: The next role mode string.
    """
    return "LLM_DESCRIBES" if current_role == "USER_DESCRIBES" else "USER_DESCRIBES"


def run_app(state: Any, config: Any, store: Any) -> dict:
    """
    Execute the app using GPT-4o, ensuring full error handling.

    :param state: The current conversation state.
    :param config: The runtime configuration.
    :param store: The store for managing long-term state.
    :return: A dictionary indicating success or containing any errors encountered.
    """
    load_custom_css()
    st.markdown("<div class='header'>Guess the Concept</div>", unsafe_allow_html=True)

    if "concept" not in st.session_state:
        concept: str = generate_random_concept()
        st.session_state["concept"] = concept
        store_concept_locally(concept)
    else:
        concept = st.session_state["concept"]

    if "role_mode" not in st.session_state:
        st.session_state["role_mode"] = "USER_DESCRIBES"

    if st.button("Play Again"):
        st.session_state["concept"] = generate_random_concept()
        store_concept_locally(st.session_state["concept"])
        st.session_state["role_mode"] = "USER_DESCRIBES"
        st.markdown("<meta http-equiv='refresh' content='0'>", unsafe_allow_html=True)
        st.stop()

    st.markdown(f"<div class='concept-card'>Current Concept: <strong>{concept}</strong></div>", unsafe_allow_html=True)

    if st.session_state["role_mode"] == "USER_DESCRIBES":
        with st.form("guess_form", clear_on_submit=False):
            user_description: str = st.text_area("Be as cryptic or revealing as you wish...", key="user_desc", help="Type your description here.", height=150)
            submitted: bool = st.form_submit_button("Submit")
        st.markdown(
            """
            <script>
            document.addEventListener('DOMContentLoaded', function() {
              const textAreas = document.getElementsByTagName('textarea');
              if (textAreas.length > 0) {
                textAreas[0].addEventListener('keydown', function(e) {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const btn = document.querySelector('button[type="submit"]');
                    if (btn) { btn.click(); }
                  }
                });
              }
            });
            </script>
            """,
            unsafe_allow_html=True
        )
        if submitted:
            if check_if_cheating(st.session_state["concept"], user_description):
                st.error("Cheating! Please rewrite the description without revealing the concept directly.")
            else:
                params: dict = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a guesser AI, you only know the user's new description."},
                        {"role": "user", "content": f"Guess this concept based on: {user_description}"}
                    ],
                    "temperature": 0.7
                }
                response, error = safe_chat_completion_create(**params)
                if error:
                    st.error(error)
                elif response:
                    llm_guess: str = response.choices[0].message.content.strip()
                    st.markdown(f"<div class='concept-card'>LLM's guess: <strong>{llm_guess}</strong></div>", unsafe_allow_html=True)
                    if verify_guess(st.session_state["concept"], llm_guess):
                        st.success("OpenAI verified: The LLM's guess is correct!")
                    else:
                        st.error("OpenAI says: The LLM's guess is not correct.")

    if st.button("Swap Roles"):
        st.session_state["role_mode"] = swap_roles(st.session_state["role_mode"])
        st.markdown("<meta http-equiv='refresh' content='0'>", unsafe_allow_html=True)
        st.stop()

    return {"message": "App ran successfully"}


if __name__ == "__main__":
    run_app(None, None, None)
