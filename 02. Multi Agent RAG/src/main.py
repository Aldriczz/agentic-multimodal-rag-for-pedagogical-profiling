"""Main entry point for the pedagogical profiling multi-agent RAG workflow."""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from src.state import GraphState
from src.graph import compile_graph

class PedagogicalProfilingRAG:
    """Main orchestrator for the pedagogical profiling multi-agent RAG system.
    
    This system performs comprehensive educational analysis and recommendations by:
    1. Routing user intent
    2. Building learner profiles
    3. Retrieving relevant resources
    4. Running parallel expert analyses (Application, UI/UX, Curriculum, Pedagogy, Reviews)
    5. Aggregating evaluations
    6. Synthesizing recommendations with justification
    7. Optimizing output quality with potential iteration loops
    """
    
    def __init__(self):
        """Initialize the pedagogical profiling system with LLM and compiled graph."""
        api_key = os.getenv("LLM_API_KEY")
        if api_key is None:
            raise ValueError("LLM_API_KEY environment variable not set")
        
        api_key = SecretStr(api_key)
        
        model = os.getenv("LLM_MODEL", "gpt-4")
        
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.7,
        )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
        
        self.graph = compile_graph(self.llm, self.embeddings)
    
    async def process_request(
        self,
        user_input: str,
        max_iterations: int = 3,
    ) -> dict:
        initial_state: GraphState = {
            "user_input": user_input,
            "learner_profile": {},
            "retrieved_docs": [],
            "application_analysis_evaluation": {},
            "ui_ux_analysis_evaluation": {},
            "curriculum_analysis_evaluation": {},
            "pedagogical_analysis_evaluation": {},
            "review_analysis_evaluation": {},
            "aggregated_evaluation": {},
            "recommendation": "",
            "justification": "",
            "quality_assessment": {},
            "optimization_status": "unsatisfactory",
            "recommended_apps": [],
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "agent_messages": [],
            "error_message": None,
        }
        
        try:
            result = await self.graph.ainvoke(initial_state)
            return result
        except Exception as e:
            print(f"Error processing request: {e}")
            initial_state["error_message"] = str(e)
            return initial_state
    
    def process_request_sync(
        self,
        user_input: str,
        max_iterations: int = 3,
    ) -> dict:
        return asyncio.run(self.process_request(user_input, max_iterations))

async def main():
    load_dotenv()
    
    rag = PedagogicalProfilingRAG()
    
    user_input = """
    I'm a high school student struggling with advanced mathematics, particularly calculus.
    I learn best through interactive visualizations and hands-on examples. I have about 
    1-2 hours per week available for learning. I'm interested in EdTech tools that can help 
    me improve my understanding of derivatives and integrals.
    """
    
    print("=" * 80)
    print("Processing Pedagogical Profiling Request")
    print("=" * 80)
    print(f"User Input: {user_input}\n")
    
    result = await rag.process_request(user_input, max_iterations=3)

    print("\n" + "=" * 80)
    print("Retrieved Docs")
    print("=" * 80)
    print(result.get("retrieved_docs", "No documents retrieved"))

    print("\n" + "=" * 80)
    print("Application Analysis Evaluation")
    print("=" * 80)
    print(result.get("application_analysis_evaluation", "No evaluation available"))

    print("\n" + "=" * 80)
    print("UI/UX Analysis Evaluation")
    print("=" * 80)
    print(result.get("ui_ux_analysis_evaluation", "No evaluation available"))

    print("\n" + "=" * 80)
    print("Curriculum Analysis Evaluation")
    print("=" * 80)
    print(result.get("curriculum_analysis_evaluation", "No evaluation available"))

    print("\n" + "=" * 80)
    print("Pedagogical Analysis Evaluation")
    print("=" * 80)
    print(result.get("pedagogical_analysis_evaluation", "No evaluation available"))

    print("\n" + "=" * 80)
    print("User Review Analysis Evaluation")
    print("=" * 80)
    print(result.get("review_analysis_evaluation", "No evaluation available"))

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print(result.get("recommendation", "No recommendation generated"))
    
    print("\n" + "=" * 80)
    print("JUSTIFICATION")
    print("=" * 80)
    print(result.get("justification", "No justification available"))
    
    print("\n" + "=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)
    assessment = result.get("quality_assessment", {})
    print(f"Status: {result.get('optimization_status', 'unknown')}")
    print(f"Details: {assessment}")
    
    if result.get("error_message"):
        print(f"\nError: {result['error_message']}")


if __name__ == "__main__":
    asyncio.run(main())
