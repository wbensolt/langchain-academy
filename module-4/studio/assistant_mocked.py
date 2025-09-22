from pydantic import BaseModel, Field
from typing import List, Annotated
from typing_extensions import TypedDict
import operator
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langgraph.types import Send
from langgraph.graph import START, END, StateGraph

# --- Mock LLM (juste pour simuler le comportement) ---
class MockLLM:
    def invoke(self, messages):
        content = "Réponse simulée par LLM"
        msg = AIMessage(content=content)
        msg.name = "expert"
        return msg

llm = MockLLM()

# --- Schemas ---
class Analyst(BaseModel):
    affiliation: str
    name: str
    role: str
    description: str
    @property
    def persona(self) -> str:
        return f"{self.name} ({self.role}) - {self.affiliation}: {self.description}"

class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str
    retry_count: int

class InterviewState(TypedDict):
    analyst: Analyst
    messages: List
    interview: str
    sections: list
    max_num_turns: int
    context: list

# --- Nœuds mocks ---
def create_analysts(state: ResearchGraphState):
    USE_MOCK_DATA = True
    retry_count = state.get("retry_count", 0)
    if USE_MOCK_DATA:
        print("📋 MOCK: Création des analystes")
        mock_analysts = [
            Analyst(name="Dr. Sarah Chen", role="AI Ethics Researcher",
                    affiliation="Stanford University", description="Expert en éthique de l'IA"),
            Analyst(name="Prof. James Wilson", role="ML Security Specialist",
                    affiliation="MIT", description="Spécialiste sécurité ML")
        ]
        return {"analysts": mock_analysts, "retry_count": retry_count + 1}
    return {"analysts": []}

def human_feedback(state: ResearchGraphState):
    feedback = (state.get("human_analyst_feedback") or "").lower()
    if feedback == "approve":
        return {"next": "launch_interviews"}
    else:
        return {"next": "create_analysts", "human_analyst_feedback": None}

def route_after_feedback(state: ResearchGraphState):
    return state.get("next", "create_analysts")

def launch_interviews(state: ResearchGraphState):
    analysts = state.get("analysts", [])
    topic = state.get("topic", "")
    if not analysts:
        return {"next": "create_analysts"}
    return [Send("conduct_interview", {
        "analyst": analyst,
        "messages": [HumanMessage(content=f"Research topic: {topic}")],
        "max_num_turns": 2
    }) for analyst in analysts]

# --- Nœuds interview mocks ---
def conduct_interview(state: InterviewState):
    print(f"📋 MOCK: Interview pour {state['analyst'].name}")
    mock_transcript = f"Interview simulée pour {state['analyst'].name} sur le sujet."
    return {"interview": mock_transcript, "sections": [f"Section simulée pour {state['analyst'].name}"]}

def write_section(state: InterviewState):
    print(f"📋 MOCK: Écriture de section pour {state['analyst'].name}")
    return {"sections": [f"Résumé simulé pour {state['analyst'].name}"]}

# --- Rapport mocks ---
def write_report(state: ResearchGraphState):
    print("📋 MOCK: Rapport final")
    sections = "\n".join(state.get("sections", []))
    return {"content": f"## Insights\n{sections}\n\n## Sources\n[1] Source simulée"}

def write_introduction(state: ResearchGraphState):
    print("📋 MOCK: Introduction générée")
    return {"introduction": "## Introduction\nCeci est une introduction simulée."}

def write_conclusion(state: ResearchGraphState):
    print("📋 MOCK: Conclusion générée")
    return {"conclusion": "## Conclusion\nCeci est une conclusion simulée."}

def finalize_report(state: ResearchGraphState):
    final_report = state["introduction"] + "\n\n---\n\n" + state["content"] + "\n\n---\n\n" + state["conclusion"]
    return {"final_report": final_report}

# --- Construction du graphe ---
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("conduct_interview", conduct_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "conduct_interview")
interview_builder.add_edge("conduct_interview", "write_section")
interview_builder.add_edge("write_section", END)

builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("launch_interviews", launch_interviews)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", route_after_feedback, ["create_analysts", "launch_interviews"])
builder.add_edge("launch_interviews", "conduct_interview")
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_report","write_introduction","write_conclusion"], "finalize_report")
builder.add_edge("finalize_report", END)

# --- Compilation avec interruption avant feedback ---
graph = builder.compile(interrupt_before=['human_feedback'])
