"""LangGraph workflow definition for the pedagogical profiling multi-agent RAG system."""

import time
from typing import Literal, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAIEmbeddings
from langchain_core.language_models import BaseLanguageModel

from src.state import GraphState
from src.agents import create_agents


def create_workflow(llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings) -> StateGraph:
    """Create the LangGraph workflow for pedagogical profiling and recommendations.
    
    The workflow follows this structure:
    1. Router -> Profiling -> Retrieval (Sequential initialization)
    2. Retrieval -> 4 Parallel Analysis Agents (Fan-out)
    3. All Analysis Agents -> Aggregator (Fan-in)
    4. Aggregator -> Recommendation -> Justification -> Optimizer (Sequential finalization)
    5. Optimizer -> END or back to ROUTER (Conditional loop)
    """
    
    # Initialize agents
    agents = create_agents(llm, embedding)
    
    # Create graph
    workflow = StateGraph(GraphState)
    
    # ==================== NODE DEFINITIONS ====================
    
    async def router_node(state: GraphState) -> dict[str, Any]:
        """Router Agent: Determines user intent and initializes the flow."""
        result = await agents["router"].invoke(state)
        
        # Initialize missing state fields
        if "iteration_count" not in state:
            result["iteration_count"] = 0
        if "max_iterations" not in state:
            result["max_iterations"] = 3
        if "evaluations" not in state:
            result["evaluations"] = {}
            
        return result
    
    
    async def profiling_node(state: GraphState) -> dict[str, Any]:
        """Learner Profiling Agent: Builds learner profile."""
        result = await agents["profiling"].invoke(state)
        
        return result
    
    
    async def retrieval_node(state: GraphState) -> dict[str, Any]:
        """Retrieval Agent: Retrieves documents from VectorDB."""
        result = await agents["retrieval"].invoke(state)
        
        return result
    
    
    # ========== PARALLEL ANALYSIS AGENTS (Fan-out) ==========
    
    async def application_analysis_node(state: GraphState) -> dict[str, Any]:
        """Application Analysis Agent: EdTech Analyst evaluation."""
        # Agent stores its result in state key 'application_analysis_evaluation'
        result = await agents["application_analysis"].invoke(state)
        return result
    
    
    async def ui_ux_analysis_node(state: GraphState) -> dict[str, Any]:
        """UI/UX Analysis Agent: Expert Designer evaluation."""
        # Agent stores its result in state key 'ui_ux_analysis_evaluation'
        result = await agents["ui_ux_analysis"].invoke(state)
        return result
    
    
    async def curriculum_analysis_node(state: GraphState) -> dict[str, Any]:
        """Curriculum Alignment Agent: Curriculum Maker evaluation."""
        # Agent stores its result in state key 'curriculum_analysis_evaluation'
        result = await agents["curriculum_analysis"].invoke(state)
        return result
    
    
    async def pedagogical_analysis_node(state: GraphState) -> dict[str, Any]:
        """Pedagogical Expert Agent: Learning science evaluation."""
        # Agent stores its result in state key 'pedagogical_analysis_evaluation'
        result = await agents["pedagogical_analysis"].invoke(state)
        return result
    
    
    # ========== FAN-IN AND SEQUENTIAL FINALIZATION ==========
    
    async def aggregator_node(state: GraphState) -> dict[str, Any]:
        """Evaluation Aggregator: Collects and synthesizes parallel evaluations."""
        result = await agents["aggregator"].invoke(state)
        
        return result
    
    
    async def recommendation_node(state: GraphState) -> dict[str, Any]:
        """Recommendation Synthesizer: Generates final recommendation."""
        result = await agents["recommendation"].invoke(state)
        
        return result
    
    
    async def justification_node(state: GraphState) -> dict[str, Any]:
        """Justification Synthesizer: Provides detailed reasoning."""
        result = await agents["justification"].invoke(state)
        
        return result
    
    
    async def optimizer_node(state: GraphState) -> dict[str, Any]:
        """Evaluator-Optimizer Agent: Quality assurance and optimization."""
        result = await agents["optimizer"].invoke(state)
        
        return result
    
    
    # ==================== CONDITIONAL ROUTING ====================
    
    def should_optimize(state: GraphState) -> Literal["end", "router"]:
        """Determine if output is satisfactory or needs optimization.
        
        Returns:
            "end": Output is satisfactory, proceed to END
            "router": Output needs improvement, loop back to ROUTER
        """
        optimization_status = state.get("optimization_status", "unsatisfactory")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        # If satisfactory, we're done
        if optimization_status == "satisfactory":
            return "end"
        
        # If we've hit max iterations, end anyway (avoid infinite loops)
        if iteration_count >= max_iterations:
            return "end"
        
        # Otherwise, optimize
        return "router"
    
    
    # ==================== ADD NODES TO GRAPH ====================
    
    # Sequential initialization
    workflow.add_node("router", router_node)
    workflow.add_node("profiling", profiling_node)
    workflow.add_node("retrieval", retrieval_node)
    
    # Parallel analysis agents
    workflow.add_node("application_analysis", application_analysis_node)
    workflow.add_node("ui_ux_analysis", ui_ux_analysis_node)
    workflow.add_node("curriculum_analysis", curriculum_analysis_node)
    workflow.add_node("pedagogical_analysis", pedagogical_analysis_node)
    
    # Sequential finalization
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("justification", justification_node)
    workflow.add_node("optimizer", optimizer_node)
    
    
    # ==================== ADD EDGES ====================
    
    # Sequential initialization flow
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "profiling")
    workflow.add_edge("profiling", "retrieval")
    
    # Fan-out: Retrieval to all 4 analysis agents (parallel)
    workflow.add_edge("retrieval", "application_analysis")
    workflow.add_edge("retrieval", "ui_ux_analysis")
    workflow.add_edge("retrieval", "curriculum_analysis")
    workflow.add_edge("retrieval", "pedagogical_analysis")
    
    # Fan-in: All analysis agents to aggregator
    workflow.add_edge("application_analysis", "aggregator")
    workflow.add_edge("ui_ux_analysis", "aggregator")
    workflow.add_edge("curriculum_analysis", "aggregator")
    workflow.add_edge("pedagogical_analysis", "aggregator")
    
    # Sequential finalization
    workflow.add_edge("aggregator", "recommendation")
    workflow.add_edge("recommendation", "justification")
    workflow.add_edge("justification", "optimizer")
    
    # Conditional loop: Optimizer either ends or loops back to router
    workflow.add_conditional_edges(
        "optimizer",
        should_optimize,
        {
            "end": END,
            "router": "router",
        }
    )
    
    return workflow


def compile_graph(llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings) -> StateGraph:
    """Compile the workflow into an executable graph.
    
    Args:
        llm: Language model instance (langchain_openai or langchain_community)
        embedding: Embedding model instance (langchain_openai or langchain_community)
    
    Returns:
        Compiled LangGraph workflow ready for execution
    """
    workflow = create_workflow(llm, embedding)
    return workflow.compile()

