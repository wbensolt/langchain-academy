from research_assistant import graph, Analyst
from typing import List, TypedDict, Optional
# Test
initial_state = {
    "topic": "AI Safety Research",
    "max_analysts": 2,
    "human_analyst_feedback": "",  # Vide pour l'interruption
    "analysts": [Analyst(name="Test", role="Test", affiliation="Test", description="Test")],
    "sections": [],
    "retry_count": 0,
    "introduction": "",
    "content": "",
    "conclusion": "",
    "final_report": "",
    "next": "create_analysts"
}

result = graph.invoke(initial_state, {"configurable": {"thread_id": "test1"}})
print("Should interrupt at human_feedback")

import uuid
from pprint import pprint
import json

# Importez votre graph ici
# from your_module import graph, Analyst

# def test_graph_with_interruption():
#     """Test complet du graph avec interruption et feedback en boucle"""
    
#     print("=" * 60)
#     print("TEST DU GRAPH AVEC INTERRUPTION")
#     print("=" * 60)
    
#     # 1. État initial
#     initial_state = {
#         "topic": "Intelligence Artificielle et Éthique",
#         "max_analysts": 2,
#         "human_analyst_feedback": None,
#         "analysts": [],
#         "sections": [],
#         "retry_count": 0,
#         "introduction": "",
#         "content": "",
#         "conclusion": "",
#         "final_report": ""
#     }
    
#     # Configuration avec thread ID unique
#     thread_id = str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}
    
#     print(f"Thread ID: {thread_id}")
#     print(f"État initial:")
#     pprint(initial_state)
#     print("\n" + "-" * 60)
    
#     try:
#         # 2. Première exécution - interruption attendue à human_feedback
#         print("ÉTAPE 1: Lancement initial (devrait s'interrompre)")
#         print("-" * 60)
        
#         first_result = None
#         for step in graph.stream(initial_state, config):
#             print(f"Étape: {step}")
#             first_result = step
        
#         print(f"\nRésultat après interruption: {first_result}")
        
#         # 3. Vérifier l'état actuel
#         print("\n" + "-" * 60)
#         print("ÉTAPE 2: Vérification de l'état après interruption")
#         print("-" * 60)
        
#         current_state = graph.get_state(config)
#         print(f"État actuel: {current_state}")
#         print(f"Next nodes: {current_state.next}")
        
#         # Afficher les analystes créés
#         def afficher_analystes():
#             if current_state.values.get("analysts"):
#                 print("\nAnalystes créés:")
#                 for i, analyst in enumerate(current_state.values["analysts"]):
#                     print(f"  {i+1}. {analyst.name} - {analyst.role} ({analyst.affiliation})")
#                     print(f"     Description: {analyst.description}")
        
#         afficher_analystes()
        
#         # 4. Boucle de feedback utilisateur
#         while True:
#             print("\n" + "-" * 60)
#             print("ÉTAPE 3: Feedback utilisateur")
#             print("-" * 60)
            
#             user_feedback = input("Voulez-vous approuver ces analystes? (approve/other): ").strip().lower()
#             graph.update_state(config, {"human_analyst_feedback": user_feedback})

#             if user_feedback == "approve":
#                 print("\n✅ Analystes approuvés, passage à la rédaction...")
#                 break  # Passe à la suite
#             else:
#                 print("\n❌ Feedback non approuvé, recréation d'une nouvelle équipe d'analystes...")
#                 # Relancer le sous-graphe de création d'analystes
#                 for step in graph.stream(None, config):
#                     print(f"Étape recréation: {step}")
                
#                 current_state = graph.get_state(config)
#                 afficher_analystes()
        
#         # 5. Continuer l'exécution après feedback approuvé
#         print("\n" + "-" * 60)
#         print("ÉTAPE 4: Continuation après feedback")
#         print("-" * 60)
        
#         final_results = []
#         for step in graph.stream(None, config):
#             print(f"Étape finale: {step}")
#             final_results.append(step)
        
#         # 6. Afficher les résultats finaux
#         print("\n" + "=" * 60)
#         print("RÉSULTATS FINAUX")
#         print("=" * 60)
        
#         final_state = graph.get_state(config)
#         print(f"État final:")
#         pprint(final_state.values)
        
#         if final_state.values.get("final_report"):
#             print("\n" + "-" * 40)
#             print("RAPPORT FINAL:")
#             print("-" * 40)
#             print(final_state.values["final_report"])
        
#         return final_state.values
        
#     except Exception as e:
#         print(f"ERREUR: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# def test_graph_without_interruption():
#     """Test du graph avec feedback pré-défini (pas d'interruption)"""
    
#     print("\n" + "=" * 60)
#     print("TEST DU GRAPH SANS INTERRUPTION")
#     print("=" * 60)
    
#     # État avec feedback pré-défini
#     state_with_feedback = {
#         "topic": "Machine Learning et Créativité",
#         "max_analysts": 2,
#         "human_analyst_feedback": "approve",  # Feedback pré-défini
#         "analysts": [],
#         "sections": [],
#         "retry_count": 0,
#         "introduction": "",
#         "content": "",
#         "conclusion": "",
#         "final_report": ""
#     }
    
#     thread_id = str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}
    
#     print(f"Thread ID: {thread_id}")
#     print("Exécution complète sans interruption...")
    
#     try:
#         results = []
#         for step in graph.stream(state_with_feedback, config):
#             print(f"Étape: {list(step.keys())}")
#             results.append(step)
        
#         final_state = graph.get_state(config)
        
#         print("\nRésultat final:")
#         if final_state.values.get("final_report"):
#             print(final_state.values["final_report"])
#         else:
#             pprint(final_state.values)
        
#         return final_state.values
        
#     except Exception as e:
#         print(f"ERREUR: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

def debug_graph_structure():
    """Affiche la structure du graph pour debugging"""
    
    print("\n" + "=" * 60)
    print("STRUCTURE DU GRAPH")
    print("=" * 60)
    
    try:
        # Essayer d'obtenir des informations sur le graph
        print("Nœuds du graph:")
        if hasattr(graph, 'nodes'):
            for node_name in graph.nodes:
                print(f"  - {node_name}")
        
        print("\nConfiguration d'interruption:")
        if hasattr(graph, 'interrupt_before'):
            print(f"  interrupt_before: {graph.interrupt_before}")
        if hasattr(graph, 'interrupt_after'):
            print(f"  interrupt_after: {graph.interrupt_after}")
            
    except Exception as e:
        print(f"Impossible d'analyser la structure: {e}")

def save_results_to_file(results, filename="graph_test_results.json"):
    """Sauvegarde les résultats dans un fichier"""
    
    if results:
        try:
            # Convertir les objets Analyst en dict pour la sérialisation
            serializable_results = {}
            for key, value in results.items():
                if key == "analysts" and value:
                    serializable_results[key] = [
                        {
                            "name": analyst.name,
                            "role": analyst.role, 
                            "affiliation": analyst.affiliation,
                            "description": analyst.description
                        } for analyst in value
                    ]
                else:
                    serializable_results[key] = value
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"\nRésultats sauvegardés dans {filename}")
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")

import uuid
from pprint import pprint
import asyncio
from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage

# Assumes graph is imported
# from your_module import graph

class StateTracker:
    def __init__(self):
        self.continue_the_loop = True
        self.as_node = None

    def update_as_node(self, chunk):
        if chunk.event == "messages/metadata":
            metadata = next(iter(chunk.data.values())).get("metadata", {})
            node = metadata.get("langgraph_node", None)
            priority_nodes = ["human_feedback"]
            if node in priority_nodes:
                if self.as_node is None or priority_nodes.index(node) < priority_nodes.index(self.as_node):
                    self.as_node = node

    def set_conversation_ended(self):
        self.continue_the_loop = False

def parse_messages(chunk):
    if chunk.event == "messages/complete":
        return chunk.data[-1].get("content")
    elif chunk.event == "messages/partial":
        if chunk.data and len(chunk.data) > 0:
            return chunk.data[-1].get("content", "")
    return None

async def run_graph_streaming(initial_state, config, client, placeholder=None):
    """
    Execute le graphe avec streaming et interruptions.
    placeholder : Streamlit ou None si console
    """
    # Création du thread sans arguments supplémentaires
    thread = await client.threads.create()

    # Initial run
    stream = client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id="research_assistant",  # Ici on précise le graph/assistant
        input={"messages": [HumanMessage(content=initial_state.get("topic", ""))]},  # On passe le topic
        stream_mode="messages",
        interrupt_before=["human_feedback"],
    )

    messages = []
    full_response = ""
    state_tracker = StateTracker()

    async for chunk in stream:
        content = parse_messages(chunk)
        if content and content.strip():
            if chunk.event == "messages/partial":
                if messages:
                    messages[-1] = content
                else:
                    messages.append(content)
            elif chunk.event == "messages/complete":
                messages.append(content)
            full_response = "\n\n".join(messages)

            if placeholder:
                placeholder.markdown(full_response)
            else:
                print(f"\r{full_response}", end="")

        state_tracker.update_as_node(chunk)

    state_tracker.set_conversation_ended()
    return thread["thread_id"], messages, state_tracker, full_response


def test_graph_streaming_version():
    """Test complet du graphe avec feedback géré dans le graphe"""
    initial_state = {
        "topic": "Intelligence Artificielle et Éthique",
        "max_analysts": 2,
        "human_analyst_feedback": None,
        "analysts": [],
        "sections": [],
        "retry_count": 0,
        "introduction": "",
        "content": "",
        "conclusion": "",
        "final_report": ""
    }

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"Thread ID: {thread_id}")
    pprint(initial_state)

    loop = asyncio.get_event_loop()
    client = get_client(url="http://127.0.0.1:2024")

    thread_id, messages, state_tracker, full_response = loop.run_until_complete(
        run_graph_streaming(initial_state, config, client)
    )

    print("\n--- Résultat final ---")
    pprint(messages)
    print(f"Continue Loop? {state_tracker.continue_the_loop}")
    print(f"Current Node: {state_tracker.as_node}")

    # Après ce run, LangGraph aura géré automatiquement :
    # - la création des analystes
    # - le feedback utilisateur via interruptions "await_user_clarification"
    # - la continuation jusqu'à la génération du rapport final


def main():
    """Fonction principale du script de test"""
    
    print("SCRIPT DE TEST LANGGRAPH")
    print("=" * 60)
    
    # Menu interactif
    while True:
        print("\nOptions disponibles:")
        print("1. Test avec interruption (recommandé)")
        print("2. Test sans interruption")
        print("3. test_graph_streaming_version")
        print("4. Quitter")
        
        choice = input("\nVotre choix (1-4): ").strip()
        
        # if choice == "1":
        #     results = test_graph_with_interruption()
        #     if results:
        #         save_results_to_file(results)
                
        # elif choice == "2":
        #     results = test_graph_without_interruption()
        #     if results:
        #         save_results_to_file(results)
                
        if choice == "3":
            test_graph_streaming_version()
            
        elif choice == "4":
            print("Au revoir!")
            break
            
        else:
            print("Choix invalide, veuillez réessayer.")

if __name__ == "__main__":
    # Vérifier que le graph est importé
    try:
        # Vous devez remplacer cette ligne par l'import de votre graph
        # from your_research_module import graph
        print("ATTENTION: Vous devez importer votre graph dans ce script!")
        print("Remplacez la ligne 'from your_research_module import graph'")
        print("par l'import correct de votre module.")
        
        # Pour tester, décommentez cette ligne si votre graph est disponible:
        main()
        
    except ImportError as e:
        print(f"Erreur d'import: {e}")
        print("Assurez-vous que votre module contenant le graph est importable.")