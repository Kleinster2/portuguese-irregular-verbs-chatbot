import os
import gradio as gr
import openai

# Use OpenAI v1+ client
client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment

SYSTEM_PROMPT = (
    "You are a Brazilian Portuguese irregular verb trainer. "
    "The user is a complete beginner and knows no Portuguese. "
    "Always interact in English. "
    "Your job is to present fill-in-the-blank sentences in Brazilian Portuguese where an irregular verb is missing. "
    "Only use the following irregular verbs in your exercises (in any of the present, preterite, or imperfect indicative forms): ser, estar, ir, ter, fazer, dizer, ver, vir, poder, querer, saber, trazer, dar, pôr. "
    "Do NOT use any verbs outside this list. Never use regular verbs. "
    "The answer must always be a single, simple, irregular verb form (never a compound verb or a phrase, and never a regular verb). "
    "Do NOT use compound tenses, periphrastic forms, or multi-word answers (e.g., 'tinha feito', 'teria ido', etc.). "
    "Before presenting a question, check if a compound form (e.g., 'tinha feito', 'teria ido', 'foi feito') or a compound tense (e.g., 'had done', 'has gone', 'been', 'past perfect', etc.) would be a better or more natural answer for the blank than a single-word irregular verb. "
    "If so, do NOT use this sentence. Only generate questions where the single-word irregular verb is the best and most natural answer for the blank. "
    "If you ever use a regular verb or a compound verb by mistake, apologize and immediately generate a new exercise using a single, simple, irregular verb. "
    "For each question, show a full sentence in Portuguese with the verb omitted and replaced by a blank (____). "
    "Briefly explain in English the meaning of the sentence, the missing verb, and the pronoun/subject. "
    "Ask the user to type the correct conjugated verb in the blank. "
    "After the user's answer, ALWAYS do the following before presenting a new question: "
    "1. Clearly state if the answer is correct or incorrect. If correct, start the feedback with '✅ Correct!'. If incorrect, start with '❌ Incorrect.' "
    "2. Show the correct answer in the context of the sentence. "
    "3. Briefly explain the conjugation and meaning in English. "
    "4. Only then, present the next fill-in-the-blank question. "
    "Be encouraging and educational, and do not use Portuguese unless showing the verb, the answer, or the sentence. "
    "Always use proclisis (pronoun before the verb, e.g., 'me disseram', 'se lembra') in all exercises and explanations. "
    "Do NOT use enclisis (pronoun after the verb, e.g., 'disseram-me', 'lembra-se'), as it is considered very formal and uncommon in Brazilian Portuguese. "
    "Never ask general questions about verbs or conjugations—always use fill-in-the-blank sentences. "
    "Do not repeat questions already asked in the same session. "
    "5. If the user's answer is correct except for missing accent marks (for example, 've' instead of 'vê', or 'estao' instead of 'estão'), do NOT mark it as fully incorrect. "
    "Instead, respond with: "
    "- 'Almost correct!' (or a similar encouraging phrase) "
    "- Show the correct answer with accents." 
    "- Briefly explain the importance of accents in Portuguese and encourage the user to try to include them." 
    "- Do NOT penalize the user harshly for missing accents; focus on encouragement and education." 
    "Example: If the correct answer is 'vê' and the user types 've', respond: 'Almost correct! The correct answer is 'vê' (with an accent on 'e'). In Portuguese, accents are important because they change the meaning and pronunciation of words. Try to include the accent next time!'"
    "6. If the user's answer is a valid conjugation of the verb but in the wrong tense (for example, 'pode' instead of 'pôde'), explain the difference in tense, show the correct answer, and encourage the user to pay attention to tense as well as accents. "
    "Example: If the correct answer is 'pôde' (past) and the user types 'pode' (present), respond: 'Not quite! 'Pode' is the present tense, but here you need the past tense: 'pôde' (with an accent). In Portuguese, tense and accents are important for meaning. Keep practicing!'"
    "7. If more than one conjugation (e.g., 'sabia' and 'soube', or 'dizem' and 'disseram') could be correct for the blank depending on context, accept all valid answers. Always highlight all acceptable answers, explain the difference in nuance (such as present vs. past, or imperfect vs. preterite), and clarify when each would be used. Encourage the user and make the distinction explicit. "
    "Example: If the sentence is 'Eles ____ que vêm nos visitar no próximo mês.' and the user types 'disseram', respond: 'Correct! Both 'dizem' (present, 'they say') and 'disseram' (preterite, 'they said') are valid answers depending on context. 'Disseram' means a completed action in the past, while 'dizem' means an ongoing or habitual action. In Portuguese, tense changes the nuance. Great job!'"
    "8. Whenever the correct answer uses the imperfect tense (pretérito imperfeito), give a brief explanation of why the imperfect is used in this context (e.g., for habitual actions, ongoing states, or repeated events in the past). Compare it to the preterite or other tenses if relevant, and explain how the meaning would change. Encourage the user to notice these nuances in Portuguese. "
    "Example: If the sentence is 'Durante as férias, eu ____ à casa dos meus avós todos os dias.' and the user types 'fui', respond: 'Not quite! 'Fui' is the preterite (a single completed action), but here the imperfect 'ia' is used because it describes a habitual action in the past ('I used to go'). In Portuguese, the imperfect tense is for repeated or ongoing actions in the past. Great job trying—keep an eye out for these distinctions!'"
)

# Keep chat history in memory for session continuity
chat_history = []
asked_questions = set()

# Helper to build the prompt for the LLM
def build_prompt(history, user_message):
    prompt = SYSTEM_PROMPT + "\n\n"
    for turn in history:
        prompt += f"User: {turn['user']}\nTrainer: {turn['bot']}\n"
    prompt += f"User: {user_message}\nTrainer:"
    return prompt

# Gradio handler
TRIGGER_MESSAGE = "Please start the session with a fill-in-the-blank sentence."

def llm_chatbot(user_message, history):
    global chat_history, asked_questions
    if not os.getenv("OPENAI_API_KEY"):
        print("[DEBUG] OPENAI_API_KEY not set in environment.")
        error_msg = "Error: OPENAI_API_KEY not set in environment. Please set your OpenAI API key and restart the app."
        error_history = [{"role": "assistant", "content": error_msg}]
        return error_msg, error_history

    # Reconstruct chat_history from Gradio's history format (preserve all prior turns)
    chat_history = []
    for pair in history:
        if isinstance(pair, dict) and "role" in pair and "content" in pair:
            if pair["role"] == "user":
                chat_history.append({"user": pair["content"]})
            elif pair["role"] == "assistant":
                # If last is user, pair it; else, append new
                if chat_history and "user" in chat_history[-1] and "bot" not in chat_history[-1]:
                    chat_history[-1]["bot"] = pair["content"]
                else:
                    chat_history.append({"bot": pair["content"]})
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            chat_history.append({"user": pair[0], "bot": pair[1]})

    # Now, handle the new user_message
    if user_message:
        # If last is assistant, append user's answer
        if chat_history and "bot" in chat_history[-1] and "user" not in chat_history[-1]:
            chat_history[-1]["user"] = user_message
        else:
            chat_history.append({"user": user_message})

    # Guarantee at least system prompt and user message for the initial call
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # If there is no chat history, this is the first call: trigger with TRIGGER_MESSAGE
    if not history or len(history) == 0:
        messages.append({"role": "user", "content": TRIGGER_MESSAGE})
    else:
        # Reconstruct alternating assistant/user turns
        for turn in chat_history:
            if "bot" in turn:
                messages.append({"role": "assistant", "content": turn["bot"]})
            if "user" in turn:
                messages.append({"role": "user", "content": turn["user"]})

    try:
        print("[DEBUG] OpenAI API call - model: gpt-4o")
        print(f"[DEBUG] OpenAI API call - messages: {messages}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )

        print(f"[DEBUG] OpenAI API raw response: {response}")
        import re
        # --- Build Gradio chatbot history for full session display ---
        # Get the LLM's reply
        if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message'):
            bot_reply = response.choices[0].message.content.strip()
        else:
            bot_reply = "[Error: No response from LLM]"

        # Build new Gradio history
        gradio_history = []
        if not history or len(history) == 0:
            # Initial question: show as ("", exercise)
            gradio_history.append(("", bot_reply))
        else:
            # Copy prior history (as (user, assistant) tuples)
            for pair in history:
                if isinstance(pair, dict):
                    # Convert OpenAI-style dicts to tuples
                    if pair.get("role") == "user":
                        last_user = pair["content"]
                    elif pair.get("role") == "assistant":
                        last_assistant = pair["content"]
                        gradio_history.append((last_user if 'last_user' in locals() else "", last_assistant))
                        last_user = ""
                elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                    gradio_history.append((pair[0], pair[1]))
            # Add the new user answer and LLM reply
            if user_message:
                gradio_history.append((user_message, bot_reply))

        # Continue with the rest of the logic (filtering, etc.)
        # Helper: List of common regular verbs (expandable)
        REGULAR_VERBS_PT = [
            "amar", "andar", "aprender", "beber", "cantar", "chegar", "comprar", "comer", "dançar", "decidir", "estudar", "falar", "gostar", "jogar", "lavar", "levar", "morar", "nadar", "olhar", "parar", "partir", "pensar", "perguntar", "procurar", "receber", "trabalhar", "usar", "vender", "viajar", "visitar", "entender", "abrir", "assistir", "existir", "dividir", "obedecer", "permitir", "prometer", "responder", "servir", "subir", "unir", "viver"
        ]
        def contains_regular_verb(text):
            for verb in REGULAR_VERBS_PT:
                # Look for the infinitive or a blank with the verb name nearby
                if re.search(rf'\b{verb}\b', text, re.IGNORECASE):
                    return verb
            return None

        def extract_expected_answer(text):
            # Try to extract the answer from the explanation
            match = re.search(r"(?:correct answer|missing verb) (?:is|are|:)\s*['\"]?([\wÀ-ÿ]+)['\"]?", text, re.IGNORECASE)
            if match:
                return match.group(1)
            # Fallback: look for a verb at the end of the explanation
            lines = text.split('\n')
            for line in lines:
                if 'verb' in line.lower() or 'answer' in line.lower():
                    words = re.findall(r'\b([\wÀ-ÿ]+)\b', line)
                    if words:
                        return words[-1]
            return None

        def contains_compound_verb(text):
            answer = extract_expected_answer(text)
            print(f"[DEBUG] Extracted answer for compound check: {answer}")
            if answer and ' ' in answer.strip():
                return True
            return False

        # Heuristic: If the explanation contains signs of compound ambiguity, treat as ambiguous
        def explanation_has_compound_ambiguity(text):
            # Only flag as ambiguous if the LLM explanation or translation suggests a compound is better/more natural
            preferred_compound_keywords = [
                "should have", "would have", "had already", "has been", "have been", "had been", "would be better", "the best answer is", "the most natural answer is", "preferred answer", "more natural answer", "would be more correct", "most appropriate answer", "a better answer would be", "more appropriate answer"
            ]
            lowered = text.lower()
            for kw in preferred_compound_keywords:
                if kw in lowered:
                    return True
            return False

        retry_count = 0
        max_retries = 10  # Higher cap, no user-facing error
        valid_exercise = False
        while retry_count < max_retries:
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message'):
                bot_reply = response.choices[0].message.content.strip()
                regular_found = contains_regular_verb(bot_reply)
                compound_found = contains_compound_verb(bot_reply)
                ambiguous_explanation = explanation_has_compound_ambiguity(bot_reply)
                if regular_found or compound_found or ambiguous_explanation:
                    print(f"[DEBUG] Regular, compound, or ambiguous answer detected in LLM response: regular={regular_found}, compound={compound_found}, ambiguous={ambiguous_explanation}")
                    # Ask the LLM again for a new exercise
                    retry_prompt = "You used a "
                    if regular_found:
                        retry_prompt += f"regular verb ('{regular_found}')"
                    if compound_found:
                        retry_prompt += (" and " if regular_found else "") + "a compound verb or phrase (the answer must be a single, simple, irregular verb form)"
                    if ambiguous_explanation:
                        retry_prompt += (" and " if (regular_found or compound_found) else "") + "an ambiguous or compound-possible answer (the answer must be a single, simple, irregular verb with no possible compound alternative)"
                    retry_prompt += ". Please apologize and immediately generate a new fill-in-the-blank exercise using a single, simple, irregular verb only, where no compound form could also be correct."
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages + [{"role": "user", "content": retry_prompt}],
                        max_tokens=200,
                        temperature=0.7,
                    )
                    retry_count += 1
                    continue
                # If we get here, the exercise is valid
                chat_history[-1]["bot"] = bot_reply
                valid_exercise = True
                break
            else:
                print(f"[DEBUG] OpenAI API response missing expected fields: {response}")
                retry_count += 1
                continue
        # Only show a question if valid_exercise is True
        if not valid_exercise:
            print(f"[DEBUG] LLM failed to generate a valid exercise after {max_retries} retries. No message shown to user.")
            return "", []

        # Remove the trigger message from history after the first LLM response
        chat_history = [turn for turn in chat_history if turn.get("user") != TRIGGER_MESSAGE]

        # Convert back to Gradio format (OpenAI-style dictionaries)
        gradio_history = []
        for turn in chat_history:
            if "user" in turn:
                gradio_history.append({"role": "user", "content": turn["user"]})
            if "bot" in turn:
                gradio_history.append({"role": "assistant", "content": turn["bot"]})
        # If for any reason the history is empty but we have a bot_reply, add it as the first message
        if not gradio_history and bot_reply:
            gradio_history = [{"role": "assistant", "content": bot_reply}]
        print(f"[DEBUG] llm_chatbot() returning: '', {gradio_history}")
        return "", gradio_history
    except Exception as e:
        print(f"[DEBUG] llm_chatbot() exception: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return error in OpenAI-style format for chat UI
        error_msg = f"Error accessing OpenAI API: {str(e)}"
        error_history = [{"role": "assistant", "content": error_msg}]
        return error_msg, error_history

def reset_chat():
    global chat_history, asked_questions
    chat_history = []
    asked_questions = set()
    initial_message = "Vamos começar! Conjugue o verbo 'ser' para a pessoa 'eu'."
    return "", [["", initial_message]]

def reset_ui():
    reset_msg, reset_history = reset_chat()
    return reset_msg, reset_history

# On app load, trigger the LLM to generate the first exercise
def initial_llm():
    # Generate the first LLM message and store it as history
    # Use a special message to trigger the first question
    initial_user = "Please start the session with a fill-in-the-blank sentence."
    print("[DEBUG] Calling llm_chatbot from initial_llm()...")
    msg, history = llm_chatbot(initial_user, [])
    print(f"[DEBUG] initial_llm() msg: {msg}")
    print(f"[DEBUG] initial_llm() history: {history}")
    # If there's an error or no valid exercise, do not show any error to the user—just return empty history
    if not history or len(history) == 0:
        print("[DEBUG] initial_llm() fallback: history is empty after initialization. No message shown to user.")
        return "", []
    return "", history

with gr.Blocks() as demo:
    gr.Markdown("# Brazilian Portuguese Irregular Verb Trainer\nPractice filling in the blanks with the correct verb form!")
    chatbot = gr.Chatbot(type="messages")
    with gr.Row():
        msg = gr.Textbox(placeholder="Type the correct verb form here...", label="Your Answer")
        submit = gr.Button("Submit")
    restart = gr.Button("Restart")

    demo.load(initial_llm, outputs=[msg, chatbot])
    submit.click(llm_chatbot, [msg, chatbot], [msg, chatbot])
    restart.click(reset_ui, outputs=[msg, chatbot])

def main():
    demo.launch()

if __name__ == "__main__":
    main()
