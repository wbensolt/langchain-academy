from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

# Define LLM with bound tools
#llm = ChatOpenAI(model="gpt-4o")
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Charge les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupère ta clé d'API Groq
api_key = os.getenv("GROQ_API_KEY")

# Initialisation du modèle ChatGroq avec ton modèle préféré
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=api_key
)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="""
You are a helpful assistant performing arithmetic operations.

When calling tools, always pass parameters as integers, not strings.

For example, call add with parameters: {"a": 8, "b": 5} (integers), NOT {"a": "8", "b": "5"} (strings).
""")
# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()
