#Libraries
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import os
import gradio as gr
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

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


class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {name} and is representing {name} on their website. \
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
The Agent has been provided with context on {name} in the form of their summary and Resume. Here's the information:"

evaluator_system_prompt += f"\n\n## Summary:\n{summary}\n\n## Resume:\n{resume}\n\n"
evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."


def evaluator_user_prompt(reply, message, history):
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt

def evaluate(reply, message, history) -> Evaluation:
    # Build a plain-text prompt for the evaluator model
    user_prompt = evaluator_user_prompt(reply, message, history)

    evaluator_messages = [
        {
            "role": "user",
            "parts": [
                {
                    "text": user_prompt,
                }
            ],
        }
    ]

    # Generate structured evaluation response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=evaluator_messages,
        config=GenerateContentConfig(
            system_instruction=evaluator_system_prompt,
            response_mime_type="application/json",
            response_schema=Evaluation,
        ),
    )
    # `response.text` should now be a JSON representation of `Evaluation`
    return Evaluation.model_validate_json(response.text)

def conversation_setup(message, history):
    content = []
    #only past 'max_turn' conversation need to be sent to the model every time
    max_turn = 4
    for entry in history[-max_turn:]:
        if len(entry) == 2:
            user_msg, bot_msg = entry
            content.append({"role": "user", "parts": [{"text": user_msg}]})
            content.append({"role": "model", "parts": [{"text": bot_msg}]})
    content.append({"role": "user", "parts": [{"text": message}]})
    return content

def rerun(reply, message, history, feedback):
    updated_system_prompt = system_prompt + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
    updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
    updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"

    # Rebuild the conversation history in the same format used for the main model
    updated_message=conversation_setup(message,history)
    # Generate response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=updated_message,
        config=GenerateContentConfig(
            system_instruction=updated_system_prompt
        ),
    )
    return response.text

#core module
def chat(message, history):
    """
    message: str
    history: list of (user, assistant) tuples from Gradio
    """
    if "patent" in message:
        prompt = system_prompt + "\n\nEverything in your reply needs to be in pig latin - \
              it is mandatory that you respond only and entirely in pig latin"
    else:
        prompt = system_prompt
    if history is None:
        history = []
    # Prepare Gemini contents for ceonversation
    content = conversation_setup(message, history)
    # Generate response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=content,
        config=GenerateContentConfig(
            system_instruction=prompt
        ),
    )
    # Response sent to evaluator model for evaluation
    reply = response.text
    evaluation = evaluate(reply, message, history)
    
    if evaluation.is_acceptable:
        print("Passed evaluation - returning reply")
    else:
        print("Failed evaluation - retrying")
        print(evaluation.feedback)
        reply = rerun(reply, message, history, evaluation.feedback)   
    return reply

demo=gr.ChatInterface(fn=chat)
# Get port from environment variable (Render provides this) or default to 7860
port = int(os.getenv("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port) #to run on localhost
