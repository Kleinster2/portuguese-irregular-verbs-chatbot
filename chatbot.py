import os
import gradio as gr
import openai
import re

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

TRIGGER_MESSAGE = "Please start the session with a fill-in-the-blank sentence."

# --- Helper function definitions --- START ---
ALLOWED_IRREGULAR_VERB_ROOTS = [
    "ser", "estar", "ir", "ter", "fazer", "dizer", "ver", "vir",
    "poder", "querer", "saber", "trazer", "dar", "pôr"
]
VERB_IDENTIFICATION_PATTERN = re.compile(r"The missing verb is '(\w+)' \(to ", re.IGNORECASE)
PROPOSED_ANSWER_PATTERN = re.compile(r"The correct answer (?:is|would be) '([^']+)'", re.IGNORECASE)
COMPOUND_INDICATOR_WORDS = [
    "tinha", "havia", "teria", "houvera", "hei de", "vou", "vai", "vamos", "vão", 
    "sido", "estado", "ido", "tido", "feito", "dito", "visto", "vindo",
    "podido", "querido", "sabido", "trazido", "dado", "posto",
    "past perfect", "pluperfect", "future perfect", "compound tense",
    "would have been", "could have been", "passive voice"
]

def contains_regular_verb(bot_response_text):
    match = VERB_IDENTIFICATION_PATTERN.search(bot_response_text)
    if match:
        identified_verb_root = match.group(1).lower()
        if identified_verb_root not in ALLOWED_IRREGULAR_VERB_ROOTS:
            return identified_verb_root
    if re.search(r"apologize(?:d|s)? for using a regular verb", bot_response_text, re.IGNORECASE):
        return "explicit_apology_for_regular"
    return None

def contains_compound_verb(bot_response_text):
    match = PROPOSED_ANSWER_PATTERN.search(bot_response_text)
    if match:
        answer = match.group(1)
        if ' ' in answer.strip():
            return answer
    if re.search(r"apologize(?:d|s)? for using a compound verb", bot_response_text, re.IGNORECASE):
        return "explicit_apology_for_compound"
    return None

def explanation_has_compound_ambiguity(bot_response_text):
    if re.search(r"(better|more natural|more appropriate|also correct)(?: answer| fit| form| option)? (?:would be|is|might be) [\w\s]*(" + '|'.join(COMPOUND_INDICATOR_WORDS) + r")", bot_response_text, re.IGNORECASE):
        return True
    if re.search(r"should not have used this sentence because a compound.*? more natural", bot_response_text, re.IGNORECASE):
        return True
    return None
# --- Helper function definitions --- END ---

# --- Core Chatbot Logic --- START ---
def llm_chatbot(user_message, current_gradio_chat_history):
    if not os.getenv("OPENAI_API_KEY"):
        return "", [{"role": "assistant", "content": "Error: OPENAI_API_KEY not set."}]

    messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]
    if current_gradio_chat_history:
        messages_for_api.extend(current_gradio_chat_history)
    
    if user_message != TRIGGER_MESSAGE or current_gradio_chat_history:
        messages_for_api.append({"role": "user", "content": user_message})
    elif user_message == TRIGGER_MESSAGE and not current_gradio_chat_history:
        messages_for_api.append({"role": "user", "content": TRIGGER_MESSAGE})

    print(f"[DEBUG] OpenAI API call - messages: {messages_for_api}")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_for_api,
            max_tokens=300,
            temperature=0.7,
        )
        bot_reply_content = response.choices[0].message.content.strip() if response.choices else "[Error: No response from LLM]"

        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            regular_found = contains_regular_verb(bot_reply_content)
            compound_found = contains_compound_verb(bot_reply_content)
            ambiguous_explanation = explanation_has_compound_ambiguity(bot_reply_content)
            if not (regular_found or compound_found or ambiguous_explanation):
                break
            
            print(f"[DEBUG] Invalid exercise. Retrying ({retry_count+1}/{max_retries})...")
            retry_prompt = "You previously generated an exercise that was not suitable. Please apologize and generate a new, different exercise strictly using one of the allowed single-word irregular verbs."
            
            retry_api_messages = messages_for_api + [{"role": "assistant", "content": bot_reply_content}, {"role": "user", "content": retry_prompt}]
            
            new_response = client.chat.completions.create(
                model="gpt-4o", messages=retry_api_messages, max_tokens=300, temperature=0.7 + (retry_count * 0.05)
            )
            bot_reply_content = new_response.choices[0].message.content.strip() if new_response.choices else "[Error: No response on retry]"
            retry_count += 1
        
        new_gradio_chat_history = list(current_gradio_chat_history)
        if user_message == TRIGGER_MESSAGE and not current_gradio_chat_history:
            new_gradio_chat_history = [{"role": "assistant", "content": bot_reply_content}]
        else:
            new_gradio_chat_history.append({"role": "user", "content": user_message})
            new_gradio_chat_history.append({"role": "assistant", "content": bot_reply_content})
        
        print(f"[DEBUG] llm_chatbot returning: '', {new_gradio_chat_history}")
        return "", new_gradio_chat_history

    except Exception as e:
        print(f"[DEBUG] llm_chatbot exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return "", [{"role": "assistant", "content": f"Error: {str(e)}"}]

def initial_llm():
    print("[DEBUG] initial_llm() called for Chatbot value param")
    _, initial_history_for_gradio = llm_chatbot(TRIGGER_MESSAGE, [])
    print(f"[DEBUG] initial_llm() returning for Gradio: {initial_history_for_gradio}")
    return initial_history_for_gradio

def reset_chat():
    print("[DEBUG] reset_chat() called")
    _, new_initial_history = llm_chatbot(TRIGGER_MESSAGE, [])
    return "", new_initial_history
# --- Core Chatbot Logic --- END ---

# --- Gradio Interface --- START ---
BOT_AVATAR_URL = "https://img.icons8.com/color/48/brazil-circular.png"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Brazilian Portuguese Irregular Verb Trainer")
    gr.Markdown("Practice filling in the blanks with the correct verb form!")

    chatbot_component = gr.Chatbot(
        value=initial_llm(), # initial_llm returns List[Dict[role, content]]
        label="Chatbot",
        bubble_full_width=False,
        avatar_images=(None, BOT_AVATAR_URL),
        show_label=False,
        layout="panel", # Using panel layout often works better with type="messages"
        type="messages" # Explicitly set type to messages
    )

    with gr.Row():
        user_input_textbox = gr.Textbox(
            placeholder="Type the correct verb form here...", 
            label="Your Answer",
            show_label=False,
            scale=3
        )
        submit_button = gr.Button("Submit", scale=1, variant="primary")
    
    restart_button = gr.Button("Restart")

    submit_button.click(
        fn=llm_chatbot, 
        inputs=[user_input_textbox, chatbot_component], 
        outputs=[user_input_textbox, chatbot_component]
    )
    restart_button.click(
        fn=reset_chat, 
        inputs=None, 
        outputs=[user_input_textbox, chatbot_component]
    )

demo.queue()
if __name__ == "__main__":
    demo.launch()
