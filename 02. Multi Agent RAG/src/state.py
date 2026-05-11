"""State management for the multi-agent RAG workflow."""

from typing import TypedDict, Any, Optional, Annotated
import operator


class GraphState(TypedDict, total=False):
    user_input: str
    """The original user query or request."""
    
    learner_profile: dict[str, Any]
    """Data from the Learner Profiling Agent containing learner background, preferences, and needs."""
    
    retrieved_docs: list[dict[str, Any]]
    """Documents and resources retrieved from the VectorDB relevant to the user query."""
    
    application_analysis_evaluation: dict[str, Any]
    ui_ux_analysis_evaluation: dict[str, Any]
    curriculum_analysis_evaluation: dict[str, Any]
    pedagogical_analysis_evaluation: dict[str, Any]
    review_analysis_evaluation: dict[str, Any]
    """A dictionary storing outputs from the four analysis agents:
    - 'application_analysis': EdTech Analyst evaluation
    - 'ui_ux_analysis': UI/UX Expert evaluation
    - 'curriculum_analysis': Curriculum Alignment evaluation
    - 'pedagogical_analysis': Pedagogical Expert evaluation
    - 'review_analysis': User Review analysis
    """
    
    aggregated_evaluation: dict[str, Any]
    """Aggregated findings from all four analysis agents."""
    
    recommendation: str
    """The final synthesized recommendation summary (1-2 sentences)."""
    
    recommended_apps: list[dict[str, Any]]
    """Structured list of up to 5 recommended apps, each with name, app_id, and reason."""
    
    justification: str
    """Detailed reasoning and justification for the recommendation."""
    
    quality_assessment: dict[str, Any]
    """Quality metrics and assessment from the Evaluator-Optimizer Agent."""
    
    optimization_status: str
    """Status from evaluator: 'satisfactory' or 'unsatisfactory'."""
    
    iteration_count: int
    """Track number of optimization iterations to prevent infinite loops."""
    
    max_iterations: int
    """Maximum allowed iterations (default: 3)."""
    
    agent_messages: Annotated[list[dict[str, Any]], operator.add]
    """Log of messages from all agents in the workflow."""
    
    error_message: Optional[str]
    """Error message if any step fails."""
