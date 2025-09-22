from research_assistant import create_analysts
import json

state = {
    "topic": "AI Ethics in Europe",
    "max_analysts": 2,
    "human_analyst_feedback": ""
}

result = create_analysts(state)
print(result)