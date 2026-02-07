#Libraries
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import os
import gradio as gr

#extract inormation from Profile.df
reader = PdfReader("Profile.pdf")
resume = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        resume += text

#extract additional information from summary
with open("summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

#default prompts
name = "Vyshnav Prasanth"

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and resume pdf which you can use to answer questions. \
Resume consist of Education section, Links, Experience, Skills, Projects, Achievements \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
Don't assume any facts or information. \
Please ensure you dont mention anyinfo like it was mentioned in summary instead just respond as if {name} is responding\
If you don't know the answer, say so." 
system_prompt += f"\n\n## Summary:\n{summary}\n\n## Resume:\n{resume}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

#load api key from env
load_dotenv(override=True)
api_key=os.getenv("GEMINI_API_KEY")
client=genai.Client(api_key=api_key)

#core module
def chat(message, history):
    """
    message: str
    history: list of (user, assistant) tuples from Gradio
    """

    if history is None:
        history = []

    # Prepare Gemini contents
    contents = []

    # System instruction (first message)
    contents.append({
        "role": "user",
        "parts": [{"text": system_prompt}]
    })

    #only past 'max_turn' conversation need to be sent to the model every time
    max_turn=4
    # Add previous conversation only if it exists
    for entry in history[-max_turn:]:
        if len(entry) == 2:
            user_msg, bot_msg = entry
            contents.append({"role": "user", "parts": [{"text": user_msg}]})
            contents.append({"role": "model", "parts": [{"text": bot_msg}]})

    # Add current user message
    contents.append({"role": "user", "parts": [{"text": message}]})

    # Generate response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
    )
    return response.text

demo=gr.ChatInterface(fn=chat)
# Get port from environment variable (Render provides this) or default to 7860
port = int(os.getenv("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
