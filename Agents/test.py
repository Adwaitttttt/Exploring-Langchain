import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

# Define tools
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

def text(a: int) -> str:
    """Return the English spelling of a number"""
    return f"The English spelling of {a} number"

tools = [
    StructuredTool.from_function(multiply),
    StructuredTool.from_function(text)
]

# Memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    return_messages=True
)

# Agent (this one supports multi-input and Gemini)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # memory=memory,
    verbose=True
)

# Chat loop
while True:
    user_input = input("You: ").strip()
    if not user_input:
        print("⚠️ Please enter a message.")
        continue
    if user_input.lower() == 'q':
        print("Exiting chat.")
        break
    try:
        response = agent.run(user_input)
        print("AI:", response)
    except Exception as e:
        print("❌ Error occurred:", e)
