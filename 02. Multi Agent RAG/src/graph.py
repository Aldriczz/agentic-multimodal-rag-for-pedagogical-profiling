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
    
    agents = create_agents(llm, embedding)
    
    workflow = StateGraph(GraphState)
    
    async def router_node(state: GraphState) -> dict[str, Any]:
        """Router Agent: Determines user intent and initializes the flow."""
        result = await agents["router"].invoke(state)
        
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
    
    
    
    async def application_analysis_node(state: GraphState) -> dict[str, Any]:
        """Application Analysis Agent: EdTech Analyst evaluation."""
        result = await agents["application_analysis"].invoke(state)
        return result
    
    
    async def ui_ux_analysis_node(state: GraphState) -> dict[str, Any]:
        """UI/UX Analysis Agent: Expert Designer evaluation."""
        result = await agents["ui_ux_analysis"].invoke(state)
        return result
    
    
    async def curriculum_analysis_node(state: GraphState) -> dict[str, Any]:
        """Curriculum Alignment Agent: Curriculum Maker evaluation."""
        result = await agents["curriculum_analysis"].invoke(state)
        return result
    
    
    async def pedagogical_analysis_node(state: GraphState) -> dict[str, Any]:
        """Pedagogical Expert Agent: Learning science evaluation."""
        result = await agents["pedagogical_analysis"].invoke(state)
        return result
    
    async def review_analysis_node(state: GraphState) -> dict[str, Any]:
        result = await agents["review_analysis"].invoke(state)
        return result    
    
    
    async def aggregator_node(state: GraphState) -> dict[str, Any]:
        result = await agents["aggregator"].invoke(state)
        
        return result
    
    
    async def recommendation_node(state: GraphState) -> dict[str, Any]:
        result = await agents["recommendation"].invoke(state)
        
        return result
    
    
    async def justification_node(state: GraphState) -> dict[str, Any]:
        result = await agents["justification"].invoke(state)
        
        return result
    
    
    async def optimizer_node(state: GraphState) -> dict[str, Any]:
        result = await agents["optimizer"].invoke(state)
        
        return result
    
    def should_optimize(state: GraphState) -> Literal["end", "router"]:
        optimization_status = state.get("optimization_status", "unsatisfactory")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        if optimization_status == "satisfactory":
            return "end"
        
        if iteration_count >= max_iterations:
            return "end"
        
        return "router"
    
    workflow.add_node("router", router_node)
    workflow.add_node("profiling", profiling_node)
    workflow.add_node("retrieval", retrieval_node)
    
    workflow.add_node("application_analysis", application_analysis_node)
    workflow.add_node("ui_ux_analysis", ui_ux_analysis_node)
    workflow.add_node("curriculum_analysis", curriculum_analysis_node)
    workflow.add_node("pedagogical_analysis", pedagogical_analysis_node)
    workflow.add_node("review_analysis", review_analysis_node)
    
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("justification", justification_node)
    workflow.add_node("optimizer", optimizer_node)
    
    
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "profiling")
    workflow.add_edge("profiling", "retrieval")
    
    workflow.add_edge("retrieval", "application_analysis")
    workflow.add_edge("retrieval", "ui_ux_analysis")
    workflow.add_edge("retrieval", "curriculum_analysis")
    workflow.add_edge("retrieval", "pedagogical_analysis")
    workflow.add_edge("retrieval", "review_analysis")
    
    workflow.add_edge("application_analysis", "aggregator")
    workflow.add_edge("ui_ux_analysis", "aggregator")
    workflow.add_edge("curriculum_analysis", "aggregator")
    workflow.add_edge("pedagogical_analysis", "aggregator")
    workflow.add_edge("review_analysis", "aggregator")
    workflow.add_edge("aggregator", "recommendation")
    workflow.add_edge("recommendation", "justification")
    workflow.add_edge("justification", "optimizer")
    
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

