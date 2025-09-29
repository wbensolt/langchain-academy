import json
import operator
import re
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import ChatOpenAI

from langgraph.constants import Send
# from langgraph.graph import Send
from langgraph.graph import END, MessagesState, START, StateGraph

### LLM

from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Charge les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupère ta clé d'API Groq
api_key = os.getenv("GROQ_API_KEY")
api_key2 = os.getenv("GOOGLE_API_KEY")
api_key3 = os.getenv("SK_API_KEY")

# Initialisation du modèle ChatGroq avec ton modèle préféré
# llm = ChatGroq(
#     model="meta-llama/llama-4-scout-17b-16e-instruct",
#     temperature=0,
#     api_key=api_key
# ) 
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",#llama-3.1-8b-instant",#
    temperature=0,
    api_key=api_key,
    max_retries=5,           # Plus de retries
    timeout=60.0,            # Timeout plus long
    max_tokens=1000,         # Limiter les tokens
    request_timeout=30.0,    # Timeout de requête
)

def trace_node(func):
    def wrapper(state, *args, **kwargs):
        node_name = func.__name__
        print(f"\n➡️ Entering node: {node_name}")
        # Affiche les clés importantes de l'état
        keys_to_show = ['analyst', 'analysts', 'completed_interviews', 'human_analyst_feedback']
        for key in keys_to_show:
            if key in state:
                print(f"   {key}: {state[key]}")
        result = func(state, *args, **kwargs)
        print(f"✅ Exiting node: {node_name}\n")
        return result
    return wrapper

# from langchain_mistralai import ChatMistralAI

# mistral_key = os.getenv("MISTRAL_API_KEY")
# llm = ChatMistralAI(
#     model_name=os.getenv("MODEL_NAME", "mistral-small-latest"),
#     temperature=0,
#     timeout=180,
#     # metadata={"component": "marketing_llm"},
# )

### Schema 

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    next: str # Next step in the process

class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report
    retry_count: int 
    next: str # Next step in the process
    # messages: List

### Nodes and edges

analyst_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

from groq import APIError
from langchain.schema import SystemMessage

# def create_analysts(state: GenerateAnalystsState):
#     """Create analysts safely, prevent any tool calls."""

#     topic = state['topic']
#     max_analysts = state['max_analysts']
#     human_analyst_feedback = state.get('human_analyst_feedback', '')

#     structured_llm = llm.with_structured_output(Perspectives)

#     system_message = analyst_instructions.format(
#         topic=topic,
#         human_analyst_feedback=human_analyst_feedback,
#         max_analysts=max_analysts
#     ) + "\nImportant: RETURN ONLY JSON. DO NOT CALL TOOLS. No HumanMessage."

#     try:
#         # Use only SystemMessage to reduce tool-calling behavior
#         analysts = structured_llm.invoke([SystemMessage(content=system_message)])
#         return {"analysts": analysts.analysts}

#     except APIError as e:
#         print(f"Groq APIError: {e}")
#         return {"analysts": []}

#     except Exception as e:
#         print(f"Unexpected error generating analysts: {e}")
#         return {"analysts": []}

@trace_node
def create_analysts(state: ResearchGraphState):
    """Create analysts safely, prevent any tool calls."""
    
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    retry_count = state.get('retry_count', 0)

    USE_MOCK_DATA = False
    
    if USE_MOCK_DATA:
        print(f"📋 MODE TEST: Création d'analystes (tentative {retry_count + 1})")
        
        # Varier les analystes selon le retry_count pour tester le feedback
        if retry_count == 0:
            mock_analysts = [
                Analyst(
                    name="Dr. Sarah Chen",
                    role="AI Ethics Researcher", 
                    affiliation="Stanford University",
                    description="Expert en éthique de l'IA, se concentre sur les biais algorithmiques."
                ),
                Analyst(
                    name="Prof. James Wilson",
                    role="Machine Learning Security Specialist",
                    affiliation="MIT",
                    description="Spécialiste de la sécurité des modèles de ML."
                )
            ]
        else:
            # Analystes différents pour les tentatives suivantes
            mock_analysts = [
                Analyst(
                    name="Dr. Maria Rodriguez",
                    role="AI Governance Expert",
                    affiliation="Oxford University", 
                    description="Experte en gouvernance et régulation de l'IA."
                ),
                Analyst(
                    name="Dr. Kevin Liu",
                    role="Algorithmic Fairness Researcher",
                    affiliation="Berkeley",
                    description="Recherche sur l'équité algorithmique et la discrimination."
                )
            ]
        
        return {"analysts": mock_analysts, "retry_count": retry_count + 1}
    
    else:
        # Mode réel avec LLM
        try:
            structured_llm = llm.with_structured_output(Perspectives)
            system_message = analyst_instructions.format(
                topic=topic,
                human_analyst_feedback=human_analyst_feedback,
                max_analysts=max_analysts
            ) + "\nImportant: RETURN ONLY JSON. DO NOT CALL TOOLS."

            analysts = structured_llm.invoke([SystemMessage(content=system_message)])
            print("💬 Réponse brute Mistral:", analysts)
            return {"analysts": analysts.analysts, "retry_count": retry_count + 1}

        except Exception as e:
            print(f"❌ Erreur LLM: {e}")
            return {"analysts": [], "retry_count": retry_count + 1}

# @trace_node
# def create_analysts(state: ResearchGraphState):
#     """Create analysts safely, prevent any tool calls."""
    
#     topic = state['topic']
#     max_analysts = state['max_analysts']
#     human_analyst_feedback = state.get('human_analyst_feedback', '')
#     retry_count = state.get('retry_count', 0)

#     USE_MOCK_DATA = False
    
#     if USE_MOCK_DATA:
#         print(f"📋 MODE TEST: Création d'analystes (tentative {retry_count + 1})")
#         # Mock data existante
#         mock_analysts = [
#             Analyst(
#                 name="Dr. Sarah Chen",
#                 role="AI Ethics Researcher", 
#                 affiliation="Stanford University",
#                 description="Expert en éthique de l'IA, se concentre sur les biais algorithmiques."
#             ),
#             Analyst(
#                 name="Prof. James Wilson",
#                 role="Machine Learning Security Specialist",
#                 affiliation="MIT",
#                 description="Spécialiste de la sécurité des modèles de ML."
#             )
#         ]
#         return {"analysts": mock_analysts, "retry_count": retry_count + 1}
    
#     else:
#         system_message = analyst_instructions.format(
#             topic=topic,
#             human_analyst_feedback=human_analyst_feedback,
#             max_analysts=max_analysts
#         ) + (
#             "\n⚠️ Very important:\n"
#             "Return ONLY valid JSON following this schema exactly:\n"
#             "{\n"
#             '  "analysts": [\n'
#             '    {\n'
#             '      "name": "string",\n'
#             '      "theme": "string",\n'
#             '      "expertise": "string",\n'
#             '      "background": "string",\n'
#             '      "approach": "string"\n'
#             '    }\n'
#             '  ]\n'
#             "}\n"
#             "Do NOT add any extra text, explanation, or comments."
#         )

#         structured_llm = llm.with_structured_output(Perspectives)
#         response = structured_llm.invoke([SystemMessage(content=system_message)])

#         if response is None:
#             print("⚠️ Mistral n'a pas renvoyé de JSON structuré. Peut-être problème de prompt.")
#             return {"analysts": [], "retry_count": retry_count + 1}

#         return {"analysts": response.analysts, "retry_count": retry_count + 1}

@trace_node
def human_feedback(state: ResearchGraphState):
    feedback = (state.get("human_analyst_feedback") or "").strip().lower()

    if feedback == "approve":
        print("DEBUG human_feedback: Approbation -> launch_interviews")
        return {"next": "launch_interviews"}  # ⚡ retour partiel
    else:
        print("DEBUG human_feedback: Refus -> create_analysts + reset feedback")
        return {
            "next": "create_analysts",
            "human_analyst_feedback": None
        }

@trace_node
def route_after_feedback(state: ResearchGraphState):
    next_step = state.get("next")
    if next_step:
        print(f"DEBUG route_after_feedback: direction = {next_step}")
        return next_step
    print("DEBUG route_after_feedback: aucun 'next' trouvé -> défaut create_analysts")
    return "create_analysts"

@trace_node
def launch_interviews(state: dict):
    """
    Nœud qui initialise et lance les interviews.
    """
    analysts = state.get("analysts", [])
    topic = state.get("topic", "")

    if not analysts:
        # Aucun analyste restant, fin des interviews
        return {"next_step": "all_interviews_done"}

    # Prendre le premier analyste
    analyst = analysts.pop(0)
    state["analysts"] = analysts  # mettre à jour la liste
    state["analyst"] = analyst     # mettre analyst dans l'état local pour le prochain nœud

    # Envoyer le nœud conduct_interview avec le topic seulement (analyst déjà dans l'état)
    return {
        "next_send": Send(
            node="conduct_interview",
            arg={"topic": topic}
        ),
        "state": state
    }





# def check_approval(state: ResearchGraphState):
#     """Nœud séparé pour vérifier l'approbation APRÈS l'interruption"""
#     human_feedback = state.get("human_analyst_feedback", "").strip().lower()
#     analysts = state.get("analysts", [])
    
#     print(f"DEBUG: Checking approval - feedback='{human_feedback}'")
    
#     if not analysts:
#         return {"next": "create_analysts"}
    
#     if human_feedback == "approve":
#         return {"next": "launch_interviews"}
#     else:
#         return {"next": "create_analysts"}

# def route_after_approval(state: ResearchGraphState):
#     """Router basé sur la décision de check_approval"""
#     return state["next"]

# def launch_interviews(state: ResearchGraphState):
#     """Nœud dédié au lancement des interviews"""
#     analysts = state["analysts"]
#     topic = state["topic"]
    
#     print(f"Launching {len(analysts)} interviews")
#     return [Send("conduct_interview", {
#         "analyst": analyst,
#         "messages": [HumanMessage(content=f"Research topic: {topic}")],
#         "max_num_turns": 2
#     }) for analyst in analysts]

# Generate analyst question
question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

@trace_node
def ask_question(state: dict, topic: str):
    """
    Nœud qui génère et envoie une question à l'analyste.
    """
    analyst = state.get("analyst")
    if not analyst:
        raise ValueError("Missing 'analyst' in local state for this interview")

    # Exemple de génération de question
    question = f"Dear {analyst.name}, what are your thoughts on {topic}?"

    # Ajouter la question à l’interview
    state["interview"].append({"question": question})

    # Ici, tu peux décider si tu veux continuer les tours ou passer au prochain analyste
    max_turns = 2
    if len(state["interview"]) >= max_turns:
        # Interview terminée, lancer la suivante
        return launch_interviews(state)
    else:
        # Continuer l'interview avec une autre question
        return {
            "next_send": Send(
                node="ask_question",
                arg={"topic": topic}
            ),
            "state": state
        }
    
# Search query writing
search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")

@trace_node
def search_web(state: InterviewState):
    """Retrieve docs from web search with robust structured output handling."""
    from langchain_core.messages import get_buffer_string
    from groq import APIError
    from langchain_community.tools.tavily_search import TavilySearchResults

    # Convert messages list to simple text
    conversation_text = get_buffer_string(state['messages'])

    # Prepare structured LLM
    structured_llm = llm.with_structured_output(SearchQuery)

    # Prepare system + human messages
    prompt_messages = [
        SystemMessage(content=str(search_instructions.content)),
        HumanMessage(content=conversation_text)
    ]

    try:
        # Invoke LLM with structured output
        search_query = structured_llm.invoke(prompt_messages)
    except APIError as e:
        print("Groq structured output failed:", e)
        if hasattr(e, "failed_generation"):
            print("Failed generation details:", e.failed_generation)
        # Fallback: empty search query
        search_query = SearchQuery(search_query="")

    # Perform web search with Tavily
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(search_query.search_query or "")

    # Inspect what search_docs contient
    # print(search_docs)

    # Formatte correctement les documents
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{getattr(doc, "url", "")}"/>\n{getattr(doc, "content", str(doc))}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

@trace_node
def search_wikipedia(state: InterviewState):
    from langchain_core.messages import get_buffer_string
    from groq import APIError

    conversation_text = get_buffer_string(state['messages'])
    structured_llm = llm.with_structured_output(SearchQuery)
    prompt = [SystemMessage(content=str(search_instructions.content)),
              HumanMessage(content=conversation_text)]

    try:
        search_query = structured_llm.invoke(prompt)
    except APIError as e:
        print("⚠️ Groq structured output failed:", e)
        search_query = SearchQuery(search_query="")

    q = (search_query.search_query or "").strip() or state["analyst"].description
    docs = WikipediaLoader(query=q, load_max_docs=2).load()

    if not docs:
        return {"context": [f"<Document>No results for {q}</Document>"]}

    formatted = "\n\n---\n\n".join(
        f'<Document source="{d.metadata.get("source","")}" page="{d.metadata.get("page","")}"/>\n{d.page_content}\n</Document>'
        for d in docs
    )
    return {"context": [formatted]}



# Generate expert answer
answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""

# @trace_node
# def generate_answer_(state: InterviewState):
    
#     """ Node to answer a question """

#     # Get state
#     analyst = state.get("analyst")
#     if not analyst:
#         raise ValueError("Missing 'analyst' in local state for this interview")
#     messages = state["messages"]
#     context = state["context"]

#     # Answer question
#     system_message = answer_instructions.format(goals=analyst.persona, context=context)
#     answer = llm.invoke([SystemMessage(content=system_message)]+messages)
            
#     # Name the message as coming from the expert
#     answer.name = "expert"
    
#     # Append it to state
#     return {"messages": [answer]}

@trace_node
def generate_answer(state: InterviewState):
    """Node to answer a question safely for Mistral (no assistant last message)"""
    
    analyst = state.get("analyst")
    if not analyst:
        raise ValueError("Missing 'analyst' in local state for this interview")
    
    # Filtrer uniquement les HumanMessage
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    
    # Préparer le message système
    system_message = answer_instructions.format(goals=analyst.persona, context=state["context"])
    
    # Appel au LLM
    answer = llm.invoke([SystemMessage(content=system_message)] + human_messages)
    
    # Nommer le message comme venant de l'expert
    answer.name = "expert"
    
    return {"messages": [answer]}

@trace_node
def save_interview(state: InterviewState):
    
    """ Save interviews """

    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_buffer_string(messages)
    
    # Save to interviews key
    return {"interview": interview}

@trace_node
def route_messages(state: InterviewState, 
                   name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

# Write a summary (section of the final report) of the interview
section_writer_instructions = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

@trace_node
def write_section(state: InterviewState):

    """ Node to write a section """

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state.get("analyst")
    if not analyst:
        raise ValueError("Missing 'analyst' in local state for this interview")
   
    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
                
    # Append it to state
    return {"sections": [section.content]}

# @trace_node
# def generate_question_(state: InterviewState):
#     """Node to generate a question with debug info"""
    
#     print(f"DEBUG generate_question: state keys = {list(state.keys())}")
#     print(f"DEBUG generate_question: analyst in state = {'analyst' in state}")
    
#     if "analyst" not in state:
#         print("❌ ERROR: 'analyst' key missing in InterviewState")
#         print(f"State content: {state}")
#         raise ValueError("Missing 'analyst' in InterviewState. Check how Send() is configured.")
    
#     analyst = state["analyst"]
#     messages = state.get("messages", [])
    
#     # Generate question 
#     system_message = question_instructions.format(goals=analyst.persona)
#     question = llm.invoke([SystemMessage(content=system_message)] + messages)
        
#     return {"messages": [question]}

@trace_node
def generate_question(state: InterviewState):
    """Node to generate a question safely for Mistral"""
    
    if "analyst" not in state:
        raise ValueError("Missing 'analyst' in InterviewState")
    
    analyst = state["analyst"]
    
    # Filtrer uniquement les HumanMessage
    human_messages = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    
    # Préparer le message système
    system_message = question_instructions.format(goals=analyst.persona)
    
    question = llm.invoke([SystemMessage(content=system_message)] + human_messages)
    
    return {"messages": [question]}

@trace_node
def continue_interviews(state: dict):
    """
    Après avoir sauvegardé une interview, retourne vers launch_interviews
    pour le prochain analyst ou finit si tous terminés.
    """
    return launch_interviews(state)
# Add nodes and edges 
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# def initiate_all_interviews(state: ResearchGraphState):

#     """ Conditional edge to initiate all interviews via Send() API or return to create_analysts """    

#     # Check if human feedback
#     human_analyst_feedback=state.get('human_analyst_feedback','approve')
#     if human_analyst_feedback.lower() != 'approve':
#         # Return to create_analysts
#         return "create_analysts"

#     # Otherwise kick off interviews in parallel via Send() API
#     else:
#         topic = state["topic"]
#         return [Send("conduct_interview", {"analyst": analyst,
#                                            "messages": [HumanMessage(
#                                                content=f"So you said you were writing an article on {topic}?"
#                                            )
#                                                        ]}) for analyst in state["analysts"]]

# 

# 
@trace_node
def conduct_interview_(state: ResearchGraphState):
    """Nœud qui gère UN SEUL interview à la fois"""
    analysts = state.get("analysts", [])
    completed_interviews = state.get("completed_interviews", 0)
    
    if completed_interviews >= len(analysts):
        # Tous les interviews sont terminés
        return "write_report"
    
    # Get current analyst
    current_analyst = analysts[completed_interviews]
    
    # Préparer l'état pour le sous-graphe d'interview
    interview_state = {
        "analyst": current_analyst,
        "messages": [HumanMessage(content=f"Research topic: {state['topic']}")],
        "max_num_turns": 2,
        "context": [],
        "interview": "",
        "sections": []
    }
    
    # Compiler le résultat du sous-graphe
    interview_result = interview_builder.compile().invoke(interview_state)
    
    # Ajouter la section au rapport principal
    new_sections = state.get("sections", []) + interview_result.get("sections", [])
    
    # Marquer cet interview comme terminé
    return {
        "sections": new_sections,
        "completed_interviews": completed_interviews + 1
    }

@trace_node
def conduct_interview(state: ResearchGraphState):
    """Nœud qui gère UN SEUL interview à la fois"""
    analysts = state.get("analysts", [])
    completed_interviews = state.get("completed_interviews", 0)
    
    if completed_interviews >= len(analysts):
        # Tous les interviews sont terminés
        return "write_report"
    
    # Analyste actuel
    current_analyst = analysts[completed_interviews]
    
    # Préparer l'état pour le sous-graphe d'interview
    interview_state = {
        "analyst": current_analyst,
        "messages": [HumanMessage(content=f"Research topic: {state['topic']}")],
        "max_num_turns": 2,
        "context": [],
        "interview": "",
        "sections": []
    }
    
    # Lancer le sous-graphe d'interview (ask_question → save_interview → write_section)
    interview_result = interview_builder.compile().invoke(interview_state)
    
    # Ajouter la section au rapport principal
    new_sections = state.get("sections", []) + interview_result.get("sections", [])
    
    # Mettre à jour le compteur pour passer au prochain analyste
    return {
        "sections": new_sections,
        "completed_interviews": completed_interviews + 1
    }

@trace_node
def initiate_all_interviews(state: ResearchGraphState):
    """Fonction conditionnelle qui décide simplement du prochain nœud"""
    
    feedback = (state.get("human_analyst_feedback") or "").strip().lower()
    analysts = state.get("analysts", [])
    
    print(f"DEBUG: Feedback = '{feedback}', Analysts count = {len(analysts)}")
    
    if not analysts or feedback != "approve":
        return "create_analysts"
    else:
        return "conduct_interview"

    
report_writer_instructions = """You are a technical writer creating a report on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""

@trace_node
def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
    return {"content": report.content}

intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

@trace_node
def write_introduction(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}

@trace_node
def write_conclusion(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
    return {"conclusion": conclusion.content}

@trace_node
def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

# Add nodes and edges 
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("launch_interviews", launch_interviews)
builder.add_node("continue_interviews", continue_interviews)
builder.add_node("conduct_interview", conduct_interview)
builder.add_node("write_report",write_report)
builder.add_node("write_introduction",write_introduction)
builder.add_node("write_conclusion",write_conclusion)
builder.add_node("finalize_report",finalize_report)

# Logic
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, 
                             ["create_analysts", "conduct_interview"])
builder.add_conditional_edges(
    "conduct_interview",
    lambda state: "conduct_interview" if state.get("completed_interviews", 0) < len(state.get("analysts", [])) else "write_report",
    ["conduct_interview", "write_report"]
)
builder.add_edge("write_report", "write_introduction")
builder.add_edge("write_report", "write_conclusion")
builder.add_edge(["write_introduction", "write_conclusion"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile
# Compiler avec interruption
# graph = builder.compile(interrupt_before=['human_feedback'])

from langgraph.checkpoint.memory import MemorySaver

# Crée un checkpointer
# checkpointer = MemorySaver()

# Compile ton graph AVEC checkpointer
graph = builder.compile(
    interrupt_before=['human_feedback'],
    # checkpointer=checkpointer
)

