from langchain_community.chat_models import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Outil simple
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int

    Returns:
        int: product of a and b
    """
    return a * b

# LLM

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

"""LLM_MODEL = "llama3.2:latest"
llm = ChatOllama(model=LLM_MODEL, temperature=0)"""

# Node LLM (sans bind_tools)
def tool_calling_llm(state: MessagesState):
    # Ici, tu traites state["messages"] et appelles le LLM
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Construction du graphe
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()
