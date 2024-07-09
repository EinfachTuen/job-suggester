from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o")

# Import prompt data
with open(os.path.join('promptData', 'questions.txt'), 'r') as file:
    questions = file.read()
with open(os.path.join('promptData', 'system-prompt.txt'), 'r') as file:
    system_prompt = file.read()

language = "De_de"

# Initialize chat history with the system prompt
messages = [SystemMessage(content=system_prompt)]


# Define function to generate response
def generate_response(user_input):
    # Add user input to messages
    messages.append(HumanMessage(content=user_input))

    # Trim messages if necessary
    trimmed_messages = trim_messages(
        messages,
        max_tokens=500000,
        strategy="last",
        token_counter=model,
    )

    # Build prompt with trimmed messages
    prompt = ChatPromptTemplate.from_messages([
        (msg.type, msg.content) for msg in trimmed_messages
    ])

    # Generate response
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({
        "userinput": user_input,
        "language": language,
        "questions": questions
    })

    # Add response to messages
    messages.append(AIMessage(content=str(response)))

    return response


print("-----------------------------------")
print("Chatbot is ready. Type 'exit' to end the conversation.")
print("-----------------------------------")
print("")
StartMessage = "Assistant: Hallo ich bin ein Online Karrierberater. Wie kann ich dir helfen?" " Brauchst du Hilfe bei der Orientierung? Oder bist du auf der Suche nach einem neuen Karriereweg?"
print(StartMessage)
messages.append(AIMessage(content=str(StartMessage)))

# Start conversation loop
while True:
    print("")
    user_input = input("You: ")
    print("")
    if user_input.lower() == 'exit':
        print("Ending the chat. Goodbye!")
        break

    # Generate and display response
    response = generate_response(user_input)
    print(f"Assistant: {response}")
    print("-----------------------------------")