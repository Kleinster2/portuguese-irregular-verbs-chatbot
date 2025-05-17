import gradio as gr
import openai

# Use OpenAI v1+ client
client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment

SYSTEM_PROMPT = (
    """IMPORTANT: Your *entire response* for an exercise MUST strictly follow this four-part structure, using ONLY the allowed irregular verbs: 1. A Portuguese sentence with a blank (____). 2. A brief English meaning and subject for that sentence. 3. The verb hint: 'The missing verb is [PORTUGUESE_INFINITIVE] (to [ENGLISH_TRANSLATION])'. 4. The prompt: 'Please type the correct conjugated verb in the blank.' Do NOT output any other conversational text or any labels for these parts.""",
    """You are a chatbot that helps users practice conjugating a specific list of Brazilian Portuguese irregular verbs. """,
    """Your goal is to provide fill-in-the-blank sentence exercises. Focus ONLY on simple (single-word) irregular verbs from this list: """,
    """ser, estar, ir, ter, fazer, dizer, ver, vir, poder, querer, saber, trazer, dar, pôr. """,
    """**Before forming any part of an exercise for the user, you MUST first select a verb from this allowed list AND ensure it's the most natural fit.** If you find that a sentence idea would better fit a regular verb or an unlisted irregular verb, you MUST discard that sentence idea entirely and find a new one that correctly uses an allowed irregular verb. Do NOT present an exercise using a non-allowed verb and then apologize in the same turn; generate a correct exercise from the start. """,
    """ONLY test these verbs in their simple (single-word) present, preterite, or imperfect indicative forms. NO compound tenses, NO subjunctive, NO imperative. The answer must ALWAYS be a single word. """,
    """Ensure the chosen irregular verb is the most natural and common fit for the sentence blank. If another verb or a compound form is more natural, do NOT use that sentence. """,

    """--- HOW TO ASK A QUESTION ---""",
    """When generating an exercise, structure your response with these four components in order. **This is critical: The labels A, B, C, D, or any similar lettering/numbering for these components, MUST NOT appear in the text you output to the user. NEVER start any line of your output to the user with 'A.', 'B.', 'C.', or 'D.' or '1.', '2.', '3.', '4.' followed by 'Sentence:', 'Explanation:', 'Verb Hint:', or 'Prompt:', or any similar phrasing that exposes these as itemized list labels.** Output only the content for each part directly. """,
    """For example, for the first component (the sentence), output *only* the Portuguese sentence with the blank (e.g., 'Eu ____ ao cinema hoje.'); do not prefix it with any label. Do the same for the other components: English explanation, the verb hint, and the prompt to type the answer. Each component's content should ideally be on a new line or formatted clearly for readability.""",
    """Follow this content structure for your questions (these are internal guidelines for YOU, do not output the numbers or descriptive prefixes):""",
    """COMPONENT_1_SENTENCE: The Portuguese sentence with a blank (____). """,
    """COMPONENT_2_EXPLANATION: The brief English meaning and subject. """,
    """COMPONENT_3_VERB_HINT: The verb hint: 'The missing verb is \'[PORTUGUESE_INFINITIVE]\' (to [ENGLISH_TRANSLATION]).' Ensure this is an allowed irregular verb. """,
    """COMPONENT_4_PROMPT: The prompt: 'Please type the correct conjugated verb in the blank.' """,

    """--- HOW TO HANDLE USER'S ANSWER & PROCEED ---""",
    """1. User answers. Your FIRST priority: Evaluate THIS answer. Provide full feedback before any self-correction or new question. """,
    """2. **Feedback - Correctness & Prefix:** """,
    """   - Determine if user's answer is a correct form of the hinted verb for the blank. Be lenient on case initially (e.g., 'ele é' and 'Ele é' are the same). After this initial case normalization, and after considering accent rules below, if the user's typed word is an EXACT string match to your determined correct form, then it MUST be marked '✅ Correct!'. There should be absolutely no instance where the user types the exact correct word (respecting accent rules) and is told they are incorrect. """,
    """   - If the letters are correct but an accent mark is missing or incorrect (e.g., user types 'voce' for 'você', or 'e' for 'é'), use '✅ Correct!' or 'Not quite!'. The primary match is the word itself. """,
    """   - Otherwise, start feedback with '✅ Correct!' or 'Not correct.' (or 'Not quite!' for near misses like wrong tense of correct verb). """,
    """3. **Feedback - Explanation:** """,
    """   - If correct: State the full correct sentence. If the user's answer had an accent issue (missing or wrong accent but correct letters), *after* confirming correctness, clearly point out the specific accent needed (e.g., '...but remember the acute accent on the first 'e': ele é...'). Then, proactively consider if other simple indicative tenses (present, preterite, imperfect) of the *hinted irregular verb* are also **correct** for the sentence. If so, state that these are also correct answers and list these alternative correct forms with their contextual meanings. For example, if the sentence is 'Ele sempre ____ ao parque nos fins de semana.' and the user correctly answers 'vai', you MUST also mention: '"Ia" (imperfect, meaning "he used to go/would always go") would also be a correct answer here, especially if describing a past habit.' For a sentence like 'Eu ____ que estudar ontem' (hint: 'ter'), if user says 'tive', also mention '"Tinha" (imperfect) could fit if emphasizing ongoing necessity in the past.' """,
    """   - If incorrect: Explain why. Show correct form(s) and the full sentence. State tense/person if relevant to error. Explain accent mark issues if user missed one on a correct root (this implies the root was correct but still resulted in an incorrect form overall, e.g., wrong person/tense but accent would have been needed). """,
    """   - If imperfect is correct (and was the primary answer or an alternative): Briefly explain why (e.g., habitual action, ongoing past state). """,
    """4. **Next Interaction (Self-Correction & New Question):** """,
    """   A. Optional Self-Correction: After providing full feedback on the user's answer to the question they just attempted, if you realize that *that specific question (which the user saw and answered)* was flawed (e.g., used a verb not on the allowed list, was ambiguous), *then* briefly state 'I apologize for a mistake in my previous question.' This apology is ONLY for errors in questions already presented to and answered by the user, not for errors caught during your internal generation of a new question. """,
    """   B. New Question: After providing feedback (and any optional apology for a *previous* flawed question), *immediately* generate a new exercise following the 'HOW TO ASK A QUESTION' internal structure (COMPONENT_1_SENTENCE, COMPONENT_2_EXPLANATION, etc., without outputting these internal labels). Do NOT ask if the user wants to continue or has questions. Do not add any other conversational remarks after the feedback and before the new question. """,

    """--- GENERAL STYLE ---""",
    """- Be encouraging. Use English, except for Portuguese in examples/verbs. Use proclisis. No general grammar questions. Don't repeat exercises. """
)

TRIGGER_MESSAGE = "Please start the session with a fill-in-the-blank sentence."

ALLOWED_IRREGULAR_VERB_ROOTS = [
    'ser', 'estar', 'ir', 'ter', 'fazer', 'dizer', 'ver', 'vir', 'poder', 'querer', 'saber', 'trazer', 'dar', 'pôr'
]

# Helper functions (assumed to be defined correctly and remain for now, but their usage in retries will be simplified)
# contains_regular_verb, contains_compound_verb, explanation_has_compound_ambiguity

def contains_regular_verb(text_content):
    # Simplified check, actual implementation might be more robust
    # This is just a placeholder for the logic that was there
    common_regular_endings = ["ar", "er", "ir"] #This is oversimplified
    words = text_content.lower().split()
    for verb_root in ALLOWED_IRREGULAR_VERB_ROOTS:
        if verb_root in text_content.lower(): # If an irregular root is mentioned, assume it's being handled.
            return False
    # This is a very naive check and would need refinement
    # For now, let's assume it's less critical if retry logic is removed
    return False

def contains_compound_verb(text_content):
    # Placeholder for actual logic
    # e.g., checking for common auxiliary verbs + participle
    if "tinha feito" in text_content or "vou fazer" in text_content:
        return True
    return False

def explanation_has_compound_ambiguity(text_content):
    # Placeholder for actual logic
    # e.g., if the explanation implies a compound answer is expected
    return False

def llm_chatbot(user_message, current_gradio_chat_history_dicts):
    print(f"[DEBUG] llm_chatbot called. User: '{user_message}', History (dicts): {current_gradio_chat_history_dicts}")
    # current_gradio_chat_history_dicts is now expected to be a list of OpenAI-style dicts
    # or empty for the first call.

    messages_for_api = []
    # Revert system message content to a simple string
    messages_for_api.append({
        "role": "system", 
        "content": "\n".join(SYSTEM_PROMPT)
    })
    
    if current_gradio_chat_history_dicts:
        messages_for_api.extend(current_gradio_chat_history_dicts)

    if user_message != TRIGGER_MESSAGE: # Add current user message if it's not the trigger
         messages_for_api.append({"role": "user", "content": user_message})
    elif not current_gradio_chat_history_dicts: # Add trigger if it's the first call
        messages_for_api.append({"role": "user", "content": TRIGGER_MESSAGE})

    print(f"[DEBUG] Messages for API: {messages_for_api}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_for_api,
            max_tokens=500, # Increased to accommodate detailed four-part response
            temperature=0.7,
        )

        if not response or not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            error_detail = "OpenAI API response is empty, has no choices, or message content is missing."
            print(f"[ERROR] {error_detail} Full response: {response}")
            bot_reply_content = f"Error: {error_detail} Please check logs or try again."
        else:
            bot_reply_content = response.choices[0].message.content.strip()
            # Simplified: No complex retry loop for now to save tokens.
            # We can add a simple check and log if needed.
            if contains_regular_verb(bot_reply_content) or contains_compound_verb(bot_reply_content) or explanation_has_compound_ambiguity(bot_reply_content):
                print(f"[WARNING] LLM response might contain non-simple/irregular verb or ambiguous explanation: {bot_reply_content}")
                # Optionally, could add a note to the user here, or just proceed.

        # Construct new history in dictionary format
        new_gradio_history_dicts = list(current_gradio_chat_history_dicts) # Start with a copy
        if user_message != TRIGGER_MESSAGE: # If it was a real user message, add it to history
            new_gradio_history_dicts.append({"role": "user", "content": user_message})
        
        new_gradio_history_dicts.append({"role": "assistant", "content": bot_reply_content})

        print(f"[DEBUG] llm_chatbot returning history (dicts): {new_gradio_history_dicts}")
        return "", new_gradio_history_dicts

    except openai.APIError as e:
        print(f"[ERROR] OpenAI API Error in llm_chatbot: {e}")
        error_message = f"Sorry, I encountered an API error: {type(e).__name__} - {str(e)}. Please try again or restart."
        # Append error as a new assistant message to the existing history
        error_history = list(current_gradio_chat_history_dicts)
        if user_message and user_message != TRIGGER_MESSAGE:
            error_history.append({"role": "user", "content": user_message}) # include user's attempt if any
        error_history.append({"role": "assistant", "content": error_message})
        return "", error_history
    except Exception as e:
        print(f"[ERROR] General Exception in llm_chatbot: {e}")
        import traceback
        traceback.print_exc()
        error_message = f"Sorry, I encountered an unexpected error: {type(e).__name__} - {str(e)}. Please check the logs."
        # Append error as a new assistant message
        error_history = list(current_gradio_chat_history_dicts)
        if user_message and user_message != TRIGGER_MESSAGE:
            error_history.append({"role": "user", "content": user_message})
        error_history.append({"role": "assistant", "content": error_message})
        return "", error_history

def user_chat_handler(text_input, chat_history_dicts):
    print(f"[DEBUG] user_chat_handler called. Input: '{text_input}', History (dicts): {chat_history_dicts}")
    if not text_input.strip():
        return "", chat_history_dicts
    
    cleared_text, new_gradio_history_dicts = llm_chatbot(text_input, chat_history_dicts)
    
    print(f"[DEBUG] user_chat_handler returning: '{cleared_text}', {new_gradio_history_dicts}")
    return cleared_text, new_gradio_history_dicts

def initial_llm():
    print("[DEBUG] initial_llm() called for Chatbot value param")
    # llm_chatbot now returns history in dict format.
    _, history_for_chatbot_value_dicts = llm_chatbot(TRIGGER_MESSAGE, []) 
    print(f"[DEBUG] initial_llm returning for Chatbot value (dicts): {history_for_chatbot_value_dicts}")
    return history_for_chatbot_value_dicts

BOT_AVATAR_URL = "https://img.icons8.com/color/48/brazil-circular.png"

def reset_chat():
    print("[DEBUG] reset_chat() called")
    # llm_chatbot returns (cleared_text, new_history_dicts)
    _, new_initial_history_dicts = llm_chatbot(TRIGGER_MESSAGE, []) 
    print(f"[DEBUG] reset_chat returning new history (dicts): {new_initial_history_dicts}")
    return "", new_initial_history_dicts

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Brazilian Portuguese Irregular Verb Trainer")
    gr.Markdown("Practice filling in the blanks with the correct verb form!")
    
    # Revert to using initial_llm for the value parameter
    # The hardcoded test for Chatbot value format is no longer needed here.

    chatbot_component = gr.Chatbot(
        label="Chatbot",
        value=initial_llm, # Using initial_llm again, which now returns dict format
        height=600,
        avatar_images=(None, BOT_AVATAR_URL), # User avatar, Bot avatar
        show_label=False,
        layout="panel", 
        type="messages" 
    )
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Your Answer",
            placeholder="Type the correct verb form here...",
            scale=4, # Adjusted scale to make space for restart button
            autofocus=True,
            lines=1, 
            show_label=False
        )
        submit_button = gr.Button("Submit", variant="primary", scale=1)
    
    restart_button = gr.Button("Restart")

    # Define interactions
    # 1. Enter key in textbox submits
    text_input.submit(user_chat_handler, [text_input, chatbot_component], [text_input, chatbot_component])
    # 2. Click submit button submits
    submit_button.click(user_chat_handler, [text_input, chatbot_component], [text_input, chatbot_component])
    # 3. Click restart button resets chat
    restart_button.click(reset_chat, [], [text_input, chatbot_component])

demo.queue()
if __name__ == "__main__":
    # Ensure port is not hardcoded if it can change, or handle error better.
    # For now, let Gradio pick next available if 7860 is busy.
    demo.launch() # Let Gradio pick an available port automatically
