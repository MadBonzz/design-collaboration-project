import  json
from llama_cpp import Llama

llm = Llama(
    model_path="C:/Users/shour/.cache/lm-studio/models/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf", chat_format="chatml-function-calling", n_ctx=2048
)

user_response = """
Question: "Could you give a quick summary of who you are and what you do?"
Answer: "I’m a freelance UI/UX designer from India with a passion for crafting intuitive and user-friendly digital experiences. Over the past five years, I’ve collaborated with startups and businesses worldwide to create impactful interfaces for web and mobile applications."

Question: "What projects or areas are you currently focused on?"
Answer: "Currently, I’m focusing on designing for SaaS platforms and e-commerce applications, aiming to improve accessibility and streamline user flows. I’m also experimenting with motion design to add interactivity to my projects."

Question: "Could you tell me about some of your past projects or ventures?"
Answer: "I designed a productivity app for a tech startup that was featured on the App Store’s ‘Best New Apps’ list. I’ve also worked on a social networking app tailored for small communities, which received excellent user feedback for its clean and functional design."

Question: "Have you had any viral success or any particular projects that really took off?"
Answer: "Yes, a Dribbble post showcasing a redesign concept for an Indian rail booking app went viral, garnering over 50,000 likes and sparking discussions about improving UX for public services. It led to several offers for collaboration."

Question: "Are you open to collaboration? What kind of partnerships or collaborators are you interested in working with?"
Answer: "Definitely! I’m looking to partner with developers, entrepreneurs, or product managers who are passionate about building innovative digital products. I’d especially love to work on projects that prioritize simplicity and enhance user experiences globally."
"""

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "Based on the set of questions and answers provided by the user, create a very short summary of the user with the following details: tldr, today, past ventures, viral success, looking for collaborators. Base your output on the answers provided by the user.",
        },
        {"role": "user", "content": f"{user_response}"},
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "UserDetail",
                "parameters": {
                    "type": "object",
                    "title": "UserDetail",
                    "properties": {
                        "tldr": {
                            "title": "tldr",
                            "type": "string",
                            "description": "a short summary of the user, like what he does and what is his expertise.",
                        },
                        "today": {
                            "title": "today",
                            "type": "string",
                            "description": "Summary of what the user is working on currently in two or three points.",
                        },
                        "past-ventures": {
                            "title": "past ventures",
                            "type": "string",
                            "description": "Summary of the user's past ventures in bullet points.",
                        },
                        "viral-success": {
                            "title": "viral success",
                            "type": "string",
                            "description": "Summary of what the user is proud about in two or three points.",
                        },
                        "looking-for-collaborators": {
                            "title": "viral success",
                            "type": "string",
                            "description": "Summary of what the user is looking for in collaborators in two or three points.",
                        },
                    },
                    "required": [
                        "tldr",
                        "today",
                        "past-ventures",
                        "viral-success",
                        "looking-for-collaborators",
                    ],
                },
            },
        }
    ],
    tool_choice={"type": "function", "function": {"name": "UserDetail"}},
    temperature=1,
)

parsed_response = json.loads(response["choices"][0]['message']["function_call"]["arguments"])
with open('user_details_5.json', 'w') as f:
    json.dump(parsed_response, f, indent=4)