import os
import gradio as gr
import openai

# Use OpenAI v1+ client
client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment

SYSTEM_PROMPT = (
    "You are a Brazilian Portuguese irregular verb trainer. "
    "The user is a complete beginner and knows no Portuguese. "
    "Always interact in English. "
    "Your job is to present fill-in-the-blank sentences in Brazilian Portuguese (present tense) where an irregular verb is missing. "
    "For each question, show a full sentence in Portuguese with the verb omitted and replaced by a blank (____). Briefly explain in English the meaning of the sentence, the missing verb, and the pronoun/subject. "
    "Ask the user to type the correct conjugated verb in the blank. "
    "After the user's answer, ALWAYS do the following before presenting a new question: "
    "1. Clearly state if the answer is correct or incorrect. If correct, start the feedback with '✅ Correct!'. If incorrect, start with '❌ Incorrect.' "
    "2. Show the correct answer in the context of the sentence. "
    "3. Briefly explain the conjugation and meaning in English. "
    "4. Only then, present the next fill-in-the-blank question. "
    "Be encouraging and educational, and do not use Portuguese unless showing the verb, the answer, or the sentence. "
    "Never ask general questions about verbs or conjugations—always use fill-in-the-blank sentences. "
    "Do not repeat questions already asked in the same session."
)


# Keep chat history in memory for session continuity
chat_history = []
asked_questions = set()

# Helper to build the prompt for the LLM
def build_prompt(history, user_message):
    prompt = SYSTEM_PROMPT + "\n\n"
    for turn in history:
        prompt += f"Usuário: {turn['user']}\nTreinador: {turn['bot']}\n"
    prompt += f"Usuário: {user_message}\nTreinador:"
    return prompt

# Gradio handler
def llm_chatbot(user_message, history):
    global chat_history, asked_questions
    if not os.getenv("OPENAI_API_KEY"):
        return "Error: OPENAI_API_KEY not set in environment.", history
    # Build prompt
    prompt = build_prompt(chat_history, user_message)
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[{"role": "user", "content": turn['user']} for turn in chat_history],
            {"role": "user", "content": user_message}
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )
        bot_reply = response.choices[0].message.content.strip()
        chat_history.append({"user": user_message, "bot": bot_reply})
        history = history + [[user_message, bot_reply]]
        return "", history
    except Exception as e:
        return f"Error accessing OpenAI API: {str(e)}", history

def reset_chat():
    global chat_history, asked_questions
    chat_history = []
    asked_questions = set()
    initial_message = "Vamos começar! Conjugue o verbo 'ser' para a pessoa 'eu'."
    return "", [["", initial_message]]

with gr.Blocks() as demo:
    gr.Markdown("# Brazilian Portuguese Irregular Verb Trainer\nPractice filling in the blanks with the correct verb form!")
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(placeholder="Type the correct verb form here...", label="Your Answer")
    clear_btn = gr.Button("Restart")

    msg.submit(llm_chatbot, [msg, chatbot], [msg, chatbot])
    clear_btn.click(reset_chat, outputs=[msg, chatbot])

def main():
    demo.launch()

if __name__ == "__main__":
    main()
