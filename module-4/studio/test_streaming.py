import asyncio
from pprint import pprint
from langgraph_sdk import get_client

class AdvancedGraphRunner:
    """Runner avancé qui gère correctement les cycles de feedback."""
    
    def __init__(self, client):
        self.client = client
        
    async def run_complete_workflow(self, topic: str, assistant_id: str = "research_assistant"): #assistant_mocked
        """Exécute le workflow complet avec gestion des cycles feedback."""
        
        print(f"🚀 WORKFLOW COMPLET - Sujet: {topic}")
        print("=" * 70)
        
        # Créer un thread persistant
        thread = await self.client.threads.create()
        thread_id = thread["thread_id"]
        print(f"📍 Thread ID: {thread_id}")
        
        # État initial
        initial_input = {
            "topic": topic,
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
        
        # Variables de contrôle
        max_feedback_cycles = 3
        current_cycle = 0
        workflow_completed = False
        
        # Première exécution
        print("🔄 PHASE 1: Démarrage initial...")
        await self.execute_until_interruption(thread_id, assistant_id, initial_input)
        
        # Boucle de feedback
        while current_cycle < max_feedback_cycles and not workflow_completed:
            current_cycle += 1
            print(f"\n🔁 CYCLE DE FEEDBACK {current_cycle}/{max_feedback_cycles}")
            print("-" * 50)
            
            # Vérifier et afficher l'état actuel
            current_state = await self.get_and_display_state(thread_id)
            
            # Si pas d'analystes, problème de création
            if not current_state.get("analysts"):
                print("❌ Erreur: Aucun analyste créé")
                break
            
            # Demander le feedback utilisateur
            feedback = await self.get_user_feedback(current_state, current_cycle)
            
            # Mettre à jour l'état avec le feedback
            print(f"📤 Envoi du feedback: '{feedback}'")
            await self.client.threads.update_state(
                thread_id,
                {"human_analyst_feedback": feedback}
            )
            
            # Continuer l'exécution
            result = await self.continue_execution(thread_id, assistant_id)
            
            # Analyser le résultat
            if result == "completed":
                workflow_completed = True
                print("✅ WORKFLOW TERMINÉ AVEC SUCCÈS!")
                break
            elif result == "back_to_feedback":
                print("🔄 Retour au cycle de feedback (nouveaux analystes)")
                continue
            elif result == "error":
                print("❌ Erreur dans l'exécution")
                break
            else:
                print(f"⚠️  Résultat inattendu: {result}")
                break
        
        # Afficher les résultats finaux
        await self.display_final_results(thread_id)
        
        return thread_id
    
    async def execute_until_interruption(self, thread_id: str, assistant_id: str, input_data: dict):
        """Exécute jusqu'à interruption avec monitoring détaillé."""
        
        try:
            stream = self.client.runs.stream(
                thread_id=thread_id,
                assistant_id=assistant_id,
                input=input_data,
                stream_mode="values"
            )
            
            step_count = 0
            async for chunk in stream:
                if chunk.event == "values":
                    step_count += 1
                    data = chunk.data
                    node_keys = list(data.keys())
                    print(f"  📊 Étape {step_count}: {node_keys}")
                    
                    # Afficher des détails importants
                    if "analysts" in data and data["analysts"]:
                        print(f"    👥 {len(data['analysts'])} analystes créés")
                    
                    if "retry_count" in data:
                        print(f"    🔢 Tentative: {data['retry_count']}")
                
                elif chunk.event == "error":
                    print(f"    ❌ Erreur: {chunk.data}")
            
            print(f"🔒 Interruption détectée après {step_count} étapes")
            
        except Exception as e:
            print(f"❌ Erreur d'exécution: {e}")
    
    async def continue_execution(self, thread_id: str, assistant_id: str):
        """Continue l'exécution et détermine le statut de fin."""
        
        print("🔄 Reprise de l'exécution...")
        
        try:
            stream = self.client.runs.stream(
                thread_id=thread_id,
                assistant_id=assistant_id,
                input=None,  # Continuer avec l'état existant
                stream_mode="values"
            )
            
            final_data = None
            step_count = 0
            
            async for chunk in stream:
                if chunk.event == "values":
                    step_count += 1
                    final_data = chunk.data
                    node_keys = list(final_data.keys())
                    print(f"  📊 Étape {step_count}: {node_keys}")
                    
                    # Vérifier la progression
                    if final_data.get("final_report"):
                        print("  ✅ Rapport final généré!")
                    
                    if final_data.get("sections"):
                        print(f"  📝 {len(final_data['sections'])} sections créées")
            
            # Analyser l'état final pour déterminer le statut
            if not final_data:
                return "error"
            
            # Vérification: workflow complet?
            if (final_data.get("final_report") and 
                final_data.get("sections") and 
                len(final_data.get("sections", [])) > 0):
                return "completed"
            
            # Vérification: retour au feedback?
            if (final_data.get("human_analyst_feedback") is None and 
                final_data.get("analysts")):
                return "back_to_feedback"
            
            # État incertain
            return "unknown"
            
        except Exception as e:
            print(f"❌ Erreur continuation: {e}")
            return "error"
    
    async def get_and_display_state(self, thread_id: str):
        """Récupère et affiche l'état actuel."""
        
        try:
            state = await self.client.threads.get_state(thread_id)
            values = state.get("values", {})
            
            print("📋 ÉTAT ACTUEL:")
            print(f"  🎯 Sujet: {values.get('topic', 'N/A')}")
            print(f"  🔢 Tentatives: {values.get('retry_count', 0)}")
            print(f"  💬 Feedback: {values.get('human_analyst_feedback', 'None')}")
            
            analysts = values.get("analysts", [])
            if analysts:
                print(f"  👥 Analystes ({len(analysts)}):")
                for i, analyst in enumerate(analysts, 1):
                    name = analyst.get('name', 'N/A') if isinstance(analyst, dict) else getattr(analyst, 'name', 'N/A')
                    role = analyst.get('role', 'N/A') if isinstance(analyst, dict) else getattr(analyst, 'role', 'N/A')
                    print(f"    {i}. {name} - {role}")
            else:
                print("  👥 Analystes: Aucun")
            
            return values
            
        except Exception as e:
            print(f"❌ Erreur récupération état: {e}")
            return {}
    
    async def get_user_feedback(self, current_state: dict, cycle_number: int):
        """Interface utilisateur améliorée pour le feedback."""
        
        print("\n" + "=" * 60)
        print(f"💬 FEEDBACK REQUIS - Cycle {cycle_number}")
        print("=" * 60)
        
        analysts = current_state.get("analysts", [])
        if analysts:
            print("Les analystes suivants ont été créés:")
            for i, analyst in enumerate(analysts, 1):
                name = analyst.get('name', 'N/A') if isinstance(analyst, dict) else getattr(analyst, 'name', 'N/A')
                role = analyst.get('role', 'N/A') if isinstance(analyst, dict) else getattr(analyst, 'role', 'N/A')
                affiliation = analyst.get('affiliation', 'N/A') if isinstance(analyst, dict) else getattr(analyst, 'affiliation', 'N/A')
                description = analyst.get('description', 'N/A') if isinstance(analyst, dict) else getattr(analyst, 'description', 'N/A')
                
                print(f"\n{i}. {name}")
                print(f"   Rôle: {role}")
                print(f"   Affiliation: {affiliation}")
                print(f"   Description: {description}")
        
        print("\n" + "-" * 60)
        print("OPTIONS:")
        print("  ✅ Tapez 'approve' pour continuer avec ces analystes")
        print("  🔄 Tapez autre chose pour créer de nouveaux analystes")
        print("-" * 60)
        
        # Input asynchrone
        try:
            loop = asyncio.get_event_loop()
            feedback = await loop.run_in_executor(None, input, "Votre décision: ")
            return feedback.strip()
        except:
            return input("Votre décision: ").strip()
    
    async def display_final_results(self, thread_id: str):
        """Affiche les résultats finaux de manière détaillée."""
        
        print("\n" + "=" * 70)
        print("📋 RÉSULTATS FINAUX")
        print("=" * 70)
        
        try:
            state = await self.client.threads.get_state(thread_id)
            values = state.get("values", {})
            
            # Statistiques générales
            print(f"🎯 Sujet traité: {values.get('topic', 'N/A')}")
            print(f"🔢 Nombre de tentatives: {values.get('retry_count', 0)}")
            print(f"👥 Analystes finaux: {len(values.get('analysts', []))}")
            print(f"📝 Sections créées: {len(values.get('sections', []))}")
            
            # Rapport final
            final_report = values.get("final_report", "")
            if final_report:
                print("\n📄 RAPPORT FINAL:")
                print("-" * 50)
                print(final_report)
                print("-" * 50)
                print("✅ Workflow terminé avec succès!")
            else:
                print("\n⚠️  WORKFLOW INCOMPLET")
                print("Éléments disponibles:")
                if values.get("introduction"):
                    print("  ✅ Introduction générée")
                if values.get("content"):
                    print("  ✅ Contenu principal généré")
                if values.get("conclusion"):
                    print("  ✅ Conclusion générée")
                if values.get("sections"):
                    print(f"  ✅ {len(values['sections'])} sections d'interview")
                
                # Afficher l'état brut pour diagnostic
                print("\n🔍 ÉTAT COMPLET POUR DIAGNOSTIC:")
                pprint(values)
                
        except Exception as e:
            print(f"❌ Erreur affichage résultats: {e}")

# Fonction principale de test
async def test_complete_workflow():
    """Test du workflow complet."""
    
    print("🧪 TEST DU WORKFLOW COMPLET")
    print("=" * 70)
    
    try:
        # Connexion au client
        client = get_client(url="http://127.0.0.1:2024")
        print("✅ Connexion au serveur LangGraph établie")
        
        # Créer le runner
        runner = AdvancedGraphRunner(client)
        
        # Sujet de recherche
        topic = "Intelligence Artificielle et Éthique dans la Société"
        
        # Lancer le workflow
        thread_id = await runner.run_complete_workflow(topic)
        print(f"\n🏁 Workflow terminé - Thread ID: {thread_id}")
        
    except KeyboardInterrupt:
        print("\n🛑 Interruption utilisateur")
    except Exception as e:
        print(f"❌ Erreur globale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complete_workflow())