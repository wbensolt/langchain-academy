from typing import List, Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add

# ----- Types communs -----
class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]

# ----- Failure Analysis Sub-graph -----
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]

def get_failures(state: FailureAnalysisState):
    cleaned_logs = state["cleaned_logs"]
    
    # Filter out any non-dict items and check for grade
    failures = []
    for log in cleaned_logs:
        # Check if log is a dictionary
        if isinstance(log, dict):
            # Check if it has a grade that is not None
            if log.get("grade") is not None:
                failures.append(log)
    
    return {"failures": failures}

def fa_generate_summary(state: FailureAnalysisState):
    failures = state["failures"]
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {
        "fa_summary": fa_summary,
        "processed_logs": [f"failure-analysis-on-log-{f['id']}" for f in failures]
    }

fa_builder = StateGraph(FailureAnalysisState, output=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", fa_generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)
failure_analysis_graph = fa_builder.compile()

# ----- Question Summarization Sub-graph -----
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def qs_generate_summary(state: QuestionSummarizationState):
    cleaned_logs = state["cleaned_logs"]
    
    # Filter out non-dict items
    valid_logs = [log for log in cleaned_logs if isinstance(log, dict)]
    
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {
        "qs_summary": summary,
        "processed_logs": [f"summary-on-log-{log.get('id', 'unknown')}" for log in valid_logs]
    }

def send_to_slack(state: QuestionSummarizationState):
    qs_summary = state["qs_summary"]
    report = f"Slack report: {qs_summary}"
    return {"report": report}

qs_builder = StateGraph(QuestionSummarizationState, output=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", qs_generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)
question_summarization_graph = qs_builder.compile()

# ----- Entry Graph -----
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str
    report: str
    processed_logs: Annotated[List[str], add]  # Use Annotated to handle multiple updates

def clean_logs(state: EntryGraphState):
    raw_logs_input = state["raw_logs"]
    
    # Handle the case where LangGraph Studio wraps the input
    if isinstance(raw_logs_input, dict) and "raw_logs" in raw_logs_input:
        print("Detected wrapped input from LangGraph Studio, extracting actual raw_logs")
        raw_logs = raw_logs_input["raw_logs"]
    else:
        raw_logs = raw_logs_input
    
    print(f"Processing {len(raw_logs)} logs")
    
    # Ensure we're working with proper Log dictionaries
    cleaned_logs = []
    for log in raw_logs:
        if isinstance(log, dict):
            # Ensure all required fields are present with defaults
            cleaned_log = {
                "id": log.get("id", "unknown"),
                "question": log.get("question", ""),
                "docs": log.get("docs"),
                "answer": log.get("answer", ""),
                "grade": log.get("grade"),
                "grader": log.get("grader"),
                "feedback": log.get("feedback")
            }
            cleaned_logs.append(cleaned_log)
        else:
            print(f"Warning: Skipping non-dict item in raw_logs: {log}")
    
    print(f"Cleaned {len(cleaned_logs)} logs successfully")
    return {"cleaned_logs": cleaned_logs}

entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)

# Mapping correct pour passer les logs au sous-graphes
entry_builder.add_node(
    "failure_analysis",
    failure_analysis_graph,
    input_mapping={"cleaned_logs": lambda state: state["cleaned_logs"]}
)
entry_builder.add_node(
    "question_summarization",
    question_summarization_graph,
    input_mapping={"cleaned_logs": lambda state: state["cleaned_logs"]}
)

entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()