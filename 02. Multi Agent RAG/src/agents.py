"""Agent definitions for the multi-agent RAG workflow."""

import asyncio
import os
import json
from typing import Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from src.state import GraphState


class BaseAgent:
    """Base class for all RAG agents."""
    
    def __init__(
        self,
        llm: BaseLanguageModel[Any],
        embedding: OpenAIEmbeddings,
        name: str,
        system_prompt: str,
    ):
        self.llm = llm
        self.embedding = embedding
        self.name = name
        self.system_prompt = system_prompt
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Process the state and return updates. Override in subclasses."""
        raise NotImplementedError(f"invoke() not implemented for {self.name}")
    
    def _add_message(self, state: GraphState, role: str, content: str) -> None:
        """Helper to add a message to the agent messages log."""
        if "agent_messages" not in state:
            state["agent_messages"] = []
        state["agent_messages"].append({
            "agent": self.name,
            "role": role,
            "content": content,
        })


class RouterAgent(BaseAgent):
    """Router Agent: Determines user intent and initializes the flow."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Router Agent",
            system_prompt="""You are an Intent Router for an educational technology recommendation system.
Your role is to analyze the user's input and understand:
1. What type of educational need they have
2. Their current learning context
3. Any specific requirements or constraints

Respond with a JSON containing:
{
    "intent": "profile_request|content_recommendation|tool_suggestion|course_planning",
    "context": "brief description of the learning context",
    "key_requirements": ["requirement1", "requirement2"]
}"""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Analyze user input to understand intent and initialize flow."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Analyze this educational query: {state.get('user_input', '')}")
        ]
        
        response = await self.llm.ainvoke(messages)
        self._add_message(state, "output", f"Intent analysis: {response.content}")
        
        return {
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": response.content,
                }
            ]
        }


class LearnerProfilingAgent(BaseAgent):
    """Learner Profiling Agent: Acts as an Expert Education Counselor."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Learner Profiling Agent",
            system_prompt="""You are an Expert Education Counselor specialized in understanding learners.
Build a comprehensive learner profile by analyzing:
1. Learning goals and objectives
2. Current skill level and experience
3. Learning preferences (visual, auditory, kinesthetic, reading/writing)
4. Available time and resources
5. Any learning challenges or disabilities
6. Educational background

Return a JSON profile with these fields:
{
    "learning_goals": [],
    "proficiency_level": "beginner|intermediate|advanced",
    "learning_style": {
        "visual": 0.0-1.0,
        "auditory": 0.0-1.0,
        "kinesthetic": 0.0-1.0,
        "reading_writing": 0.0-1.0
    },
    "time_availability": "hours_per_week",
    "learning_challenges": [],
    "educational_background": "",
    "motivation_level": 0.0-1.0
}"""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Build a learner profile from the user input."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Create a learner profile based on: {state.get('user_input', '')}")
        ]
        
        response = await self.llm.ainvoke(messages)
        self._add_message(state, "output", f"Learner profile created")
        
        return {
            "learner_profile": {
                "analysis": response.content,
                "timestamp": None,
            },
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": response.content,
                }
            ]
        }


class RetrievalAgent(BaseAgent):
    """Retrieval Agent: Connects to Pinecone VectorDB to pull relevant educational resources."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Retrieval Agent",
            system_prompt="""You are a Document Retrieval Specialist for a mobile e-learning app recommendation system.
The database contains real Google Play Store educational apps with their descriptions, user reviews, and screenshot captions.
Generate concise, diverse search queries to find the most relevant apps.

Return ONLY valid JSON — no markdown, no extra text:
{
    "search_queries": ["query1", "query2", "query3", "query4", "query5"]
}

Keep each query short (5-10 words) and focused on finding mobile learning apps.
Variety is key: include subject keywords, feature keywords, and audience keywords."""
        )
        
        self.pinecone_api_key = os.getenv("PINECONE_TES")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX")

        if self.pinecone_index_name is None or self.pinecone_api_key is None:
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME environment variables DNE")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.pinecone_index_name)
    
    def _search_pinecone(
        self,
        query_text: str,
        namespace: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search Pinecone using the Integrated Embedding (Records) API.

        The index was populated via index.upsert_records(), which uses Pinecone's
        built-in embedding model. It MUST be queried with index.search() — NOT
        index.query(vector=...), which is for standard dense indexes only.

        Args:
            query_text: Plain-text search query
            namespace: Pinecone namespace to search (app-info | reviews | image-captions)
            top_k: Number of top results to return per query

        Returns:
            List of document dicts with id, relevance_score, and metadata fields
        """
        try:
            print(f"[Pinecone] Searching namespace='{namespace}' | query={query_text[:70]!r}")
            results = self.index.search(
                namespace=namespace,
                query={
                    "inputs": {"text": query_text},
                    "top_k": top_k,
                },
            )

            hits = results.result.hits
            print(f"[Pinecone] Hits in namespace='{namespace}': {len(hits)}")

            documents = []
            for hit in hits:
                doc = {
                    "id": hit._id,
                    "relevance_score": hit._score,
                    "namespace": namespace,
                }
                fields = hit.fields if hasattr(hit, "fields") and hit.fields else {}
                doc.update(fields)
                documents.append(doc)

            return documents
        except Exception as e:
            print(f"[Pinecone] Error searching namespace='{namespace}': {e}")
            return []

    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Retrieve relevant documents from Pinecone using integrated text search.

        Uses index.search() with text queries across all namespaces
        (app-info, reviews, image-captions) — matching the upsert_records()
        ingestion method used by the data acquisition pipeline.
        """
        learner_profile = state.get("learner_profile", {})
        user_input = state.get("user_input", "")

        # Step 1: Use LLM to generate diverse text search queries
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=f"Generate search queries for this learner profile: {learner_profile}\nUser query: {user_input}"
            ),
        ]
        response = await self.llm.ainvoke(messages)

        try:
            response_json = json.loads(response.content)
            search_queries = response_json.get("search_queries", [user_input])
        except (json.JSONDecodeError, AttributeError):
            # Fallback: use raw user input if LLM response isn't valid JSON
            search_queries = [user_input]

        # Cap at 5 queries to keep latency reasonable
        search_queries = search_queries[:5]
        print(f"[RetrievalAgent] Search queries: {search_queries}")

        # Step 2: Search all namespaces for each query and deduplicate by id
        # app-info   → app name, description, developer, ratings, install count, price
        # image-captions → screenshot visual descriptions (multimodal)
        # reviews    → user sentiment (skipped by default to reduce API calls)
        namespaces_to_search = ["app-info", "image-captions"]
        all_documents: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for query in search_queries:
            for namespace in namespaces_to_search:
                docs = self._search_pinecone(query, namespace, top_k=5)
                for doc in docs:
                    doc_id = doc.get("id", "")
                    if doc_id not in seen_ids:
                        all_documents.append(doc)
                        seen_ids.add(doc_id)

        # Step 3: Sort by relevance score, keep top 15
        all_documents = sorted(
            all_documents,
            key=lambda x: x.get("relevance_score", 0),
            reverse=True,
        )[:15]

        # Step 4: Hard stop — do NOT continue with empty context.
        # Downstream agents would hallucinate without real Pinecone data.
        if not all_documents:
            try:
                stats = self.index.describe_index_stats()
                vector_count = stats.total_vector_count
            except Exception:
                vector_count = "unknown"
            raise RuntimeError(
                f"Retrieval failed: Pinecone returned 0 documents across all namespaces and queries. "
                f"Index '{self.pinecone_index_name}' reports {vector_count} total vectors. "
                f"If 0, run the ingestion pipeline first: "
                f"cd '01. Data Acquisition' && python main.py"
            )

        self._add_message(
            state, "output",
            f"Retrieved {len(all_documents)} documents from Pinecone ({', '.join(namespaces_to_search)} namespaces)"
        )

        return {
            "retrieved_docs": all_documents,
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": (
                        f"Retrieved {len(all_documents)} educational app resources from Pinecone. "
                        f"Queries used: {search_queries}"
                    ),
                }
            ],
        }


class ApplicationAnalysisAgent(BaseAgent):
    """Application Analysis Agent: EdTech Analyst evaluates apps and tools."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Application Analysis Agent",
            system_prompt="""You are an EdTech Analyst expert in evaluating educational applications and tools.
Analyze the retrieved resources and learner profile to evaluate:
1. Feature alignment with learner needs
2. Ease of integration into learning flow
3. Cost-effectiveness and accessibility
4. Integration with other tools
5. Data privacy and security

Return a JSON analysis:
{
    "suitable_applications": [
        {
            "name": "app_name",
            "score": 0.0-1.0,
            "alignment": "brief explanation",
            "pros": [],
            "cons": [],
            "recommendation": "recommended|conditional|not_recommended"
        }
    ],
    "overall_assessment": "summary of application ecosystem fit"
}"""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Analyze applications and tools from retrieved resources."""
        retrieved_docs = state.get("retrieved_docs", [])
        learner_profile = state.get("learner_profile", {})
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Analyze these resources for learner: {learner_profile}\nResources: {retrieved_docs}")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        return {
            "application_analysis_evaluation": {
                "agent": self.name,
                "analysis": response.content,
            },
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": response.content,
                }
            ]
        }


class UIUXAnalysisAgent(BaseAgent):
    """UI/UX Analysis Agent: Expert Designer evaluates usability and accessibility."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="UI/UX Analysis Agent",
            system_prompt="""You are an Expert UI/UX Designer specializing in educational interfaces.
Evaluate the pedagogical tools and applications based on:
1. User interface clarity and intuitive design
2. Accessibility for diverse learners (WCAG compliance)
3. Mobile responsiveness
4. Learning experience optimization
5. Cognitive load management

Return a JSON analysis:
{
    "usability_score": 0.0-1.0,
    "accessibility_score": 0.0-1.0,
    "design_evaluation": {
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    },
    "learner_fit": "high|medium|low",
    "overall_recommendation": "strong_recommend|recommend|conditional|not_recommended"
}"""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Evaluate UI/UX aspects of retrieved resources."""
        retrieved_docs = state.get("retrieved_docs", [])
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Evaluate UI/UX for these resources: {retrieved_docs}")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        return {
            "ui_ux_analysis_evaluation": {
                "agent": self.name,
                "analysis": response.content,
            },
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": response.content,
                }
            ]
        }


class CurriculumAlignmentAgent(BaseAgent):
    """Curriculum Alignment Agent: Ensures alignment with educational standards."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Curriculum Alignment Agent",
            system_prompt="""You are a Curriculum Maker specializing in instructional design.
Evaluate curriculum and content alignment:
1. Alignment with learning objectives
2. Scaffolding and progression logic
3. Formative and summative assessment integration
4. Interdisciplinary connections
5. Real-world application relevance

Return a JSON analysis:
{
    "curriculum_alignment": 0.0-1.0,
    "learning_objectives_match": 0.0-1.0,
    "assessment_quality": 0.0-1.0,
    "content_evaluation": {
        "strengths": [],
        "gaps": [],
        "suggestions": []
    },
    "pedagogical_soundness": 0.0-1.0,
    "recommendation": "strong_align|align|partial_align|misaligned"
}"""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Analyze curriculum alignment of retrieved resources."""
        retrieved_docs = state.get("retrieved_docs", [])
        learner_profile = state.get("learner_profile", {})
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Analyze curriculum alignment for learner goals: {learner_profile}\nResources: {retrieved_docs}")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        return {
            "curriculum_analysis_evaluation": {
                "agent": self.name,
                "analysis": response.content,
            },
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": response.content,
                }
            ]
        }


class PedagogicalExpertAgent(BaseAgent):
    """Pedagogical Expert Agent: Evaluates pedagogical soundness and effectiveness."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Pedagogical Expert Agent",
            system_prompt="""You are a Pedagogical Expert specializing in learning science and instructional design.
Evaluate pedagogical effectiveness:
1. Alignment with evidence-based teaching practices
2. Engagement and motivation strategies
3. Personalization and differentiation capabilities
4. Feedback mechanisms and learning support
5. Metacognitive skill development
6. Social and collaborative learning opportunities

Return a JSON analysis:
{
    "pedagogical_effectiveness": 0.0-1.0,
    "evidence_base": "strong|moderate|weak",
    "engagement_potential": 0.0-1.0,
    "personalization_capability": 0.0-1.0,
    "supports": [],
    "limitations": [],
    "learning_impact_prediction": "high|moderate|low",
    "overall_rating": 0.0-1.0
}"""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Evaluate pedagogical aspects of retrieved resources."""
        retrieved_docs = state.get("retrieved_docs", [])
        learner_profile = state.get("learner_profile", {})
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Evaluate pedagogy for learner: {learner_profile}\nResources: {retrieved_docs}")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        return {
            "pedagogical_analysis_evaluation": {
                "agent": self.name,
                "analysis": response.content,
            },
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": response.content,
                }
            ]
        }


class EvaluationAggregatorAgent(BaseAgent):
    """Evaluation Aggregator: Collects and synthesizes outputs from parallel agents."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Evaluation Aggregator",
            system_prompt="""You are an Evaluation Aggregator responsible for synthesizing multiple expert analyses.
Your task is to:
1. Collect all four evaluation perspectives
2. Identify areas of consensus and disagreement
3. Highlight key insights from each expert
4. Create a unified assessment framework

Return a JSON aggregation:
{
    "consensus_score": 0.0-1.0,
    "key_strengths": [],
    "key_concerns": [],
    "recommendations_summary": [],
    "expert_perspectives": {
        "application_analysis": { "score": 0.0-1.0, "weight": 0.25 },
        "ui_ux_analysis": { "score": 0.0-1.0, "weight": 0.20 },
        "curriculum_analysis": { "score": 0.0-1.0, "weight": 0.25 },
        "pedagogical_analysis": { "score": 0.0-1.0, "weight": 0.30 }
    },
    "weighted_score": 0.0-1.0
}"""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Aggregate evaluations from all parallel agents."""
        # Read from the correct individual state keys populated by each analysis agent
        evaluations = {
            "application_analysis": state.get("application_analysis_evaluation", {}),
            "ui_ux_analysis":       state.get("ui_ux_analysis_evaluation", {}),
            "curriculum_analysis":  state.get("curriculum_analysis_evaluation", {}),
            "pedagogical_analysis": state.get("pedagogical_analysis_evaluation", {}),
        }
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Aggregate these expert evaluations: {evaluations}")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        return {
            "aggregated_evaluation": {
                "synthesis": response.content,
                "timestamp": None,
            },
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": response.content,
                }
            ]
        }


class RecommendationSynthesizerAgent(BaseAgent):
    """Recommendation Synthesizer: Generates the final recommendation."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Recommendation Synthesizer",
            system_prompt="""You are a Recommendation Synthesizer for personalized learning pathways.
Generate a clear, actionable recommendation that:
1. Names SPECIFIC learning applications/tools by name (e.g., "Codecademy", "Khan Academy", "Duolingo", "Coursera")
2. Directly addresses the learner's goals
3. Explains why this app/tool matches their profile

CRITICAL CONSTRAINT: Your entire response MUST be EXACTLY 3 sentences maximum and 50 words maximum total.
Count every word. Do not exceed this limit under any circumstance.
Example: "I recommend Codecademy for Python fundamentals based on your beginner level. Their interactive coding environment matches your kinesthetic learning style. Start with their free Python course today."
(That's exactly 3 sentences, ~40 words)"""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Generate the final recommendation."""
        aggregated_eval = state.get("aggregated_evaluation", {})
        learner_profile = state.get("learner_profile", {})
        retrieved_docs = state.get("retrieved_docs", [])
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Generate a recommendation based on:
Learner Profile: {learner_profile}
Aggregated Evaluation: {aggregated_eval}
Available Resources: {retrieved_docs}""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Truncate to ensure word limit compliance
        recommendation = self._truncate_to_words(response.content, max_words=50, max_sentences=3)
        
        return {
            "recommendation": recommendation,
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": recommendation,
                }
            ]
        }
    
    def _truncate_to_words(self, text: str, max_words: int = 50, max_sentences: int = 3) -> str:
        """Truncate text to max words and sentences."""
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentences = sentences[:max_sentences]  # Limit sentences
        
        # Rejoin and count words
        truncated = '.'.join(sentence for sentence in sentences[:max_sentences]) + ('.' if sentences else '')
        words = truncated.split()
        if len(words) > max_words:
            truncated = ' '.join(words[:max_words]) + '.'
        
        return truncated


class JustificationSynthesizerAgent(BaseAgent):
    """Justification Synthesizer: Provides detailed reasoning for the recommendation."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Justification Synthesizer",
            system_prompt="""You are a Justification Expert responsible for transparent reasoning.
Provide justification for the recommendation that explains why the specific learning application is suitable.

Focus on:
1. How the app addresses the learner's specific needs
2. Why it matches their learning style/goals
3. Clear rationale for why THIS tool is best

CRITICAL CONSTRAINT: Your entire response MUST be EXACTLY 3 sentences maximum and 50 words maximum total.
Count every word. Do not exceed this limit.
Example: "Codecademy excels for your kinesthetic learning style. Its hands-on coding projects match your need for practical experience. Python fundamentals is the perfect starting point for your career transition."
(That's exactly 3 sentences, ~35 words)"""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Generate detailed justification for the recommendation."""
        recommendation = state.get("recommendation", "")
        aggregated_eval = state.get("aggregated_evaluation", {})
        learner_profile = state.get("learner_profile", {})
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Justify this recommendation:
Recommendation: {recommendation}
Learner Profile: {learner_profile}
Evaluation Summary: {aggregated_eval}""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Truncate to ensure word limit compliance
        justification = self._truncate_to_words(response.content, max_words=50, max_sentences=3)
        
        return {
            "justification": justification,
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": justification,
                }
            ]
        }
    
    def _truncate_to_words(self, text: str, max_words: int = 50, max_sentences: int = 3) -> str:
        """Truncate text to max words and sentences."""
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentences = sentences[:max_sentences]  # Limit sentences
        
        # Rejoin and count words
        truncated = '.'.join(sentence for sentence in sentences[:max_sentences]) + ('.' if sentences else '')
        words = truncated.split()
        if len(words) > max_words:
            truncated = ' '.join(words[:max_words]) + '.'
        
        return truncated


class EvaluatorOptimizerAgent(BaseAgent):
    """Evaluator-Optimizer Agent: Reviews output quality and decides on optimization."""
    
    def __init__(self, llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings):
        super().__init__(
            llm=llm,
            embedding=embedding,
            name="Evaluator-Optimizer",
            system_prompt="""You are a Quality Assurance and Optimization Expert.
Evaluate the final recommendations and justifications for:
1. Completeness and comprehensiveness
2. Learner-centeredness and personalization
3. Clarity and actionability
4. Alignment with all expert perspectives
5. Potential gaps or areas for refinement

Return a concise quality assessment.

CRITICAL CONSTRAINT: Your assessment must be EXACTLY 3 sentences maximum, 50 words maximum total.
Be extremely concise. Focus on: status (satisfactory/unsatisfactory) and 1-2 key findings.
Example format: "Assessment satisfactory. Recommendation is clear and personalized. Aligns well with learner goals."

If unsatisfactory, briefly state what needs refinement."""
        )
    
    async def invoke(self, state: GraphState) -> dict[str, Any]:
        """Evaluate and optimize the final output."""
        recommendation = state.get("recommendation", "")
        justification = state.get("justification", "")
        learner_profile = state.get("learner_profile", {})
        iteration_count = state.get("iteration_count", 0)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Evaluate this output:
Recommendation: {recommendation}
Justification: {justification}
Learner Profile: {learner_profile}
Iteration: {iteration_count}""")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Truncate to enforce word limit compliance
        quality_assessment = self._truncate_to_words(response.content, max_words=50, max_sentences=3)
        
        # Parse status from response (simplified - in production use proper JSON parsing)
        status = "satisfactory" if "satisfactory" in response.content.lower() else "unsatisfactory"
        
        return {
            "quality_assessment": {
                "evaluation": quality_assessment,
                "status": status,
            },
            "optimization_status": status,
            "iteration_count": iteration_count + 1,  # increment so max_iterations guard works
            "agent_messages": [
                {
                    "agent": self.name,
                    "role": "output",
                    "content": quality_assessment,
                }
            ]
        }
    
    def _truncate_to_words(self, text: str, max_words: int = 50, max_sentences: int = 3) -> str:
        """Truncate text to max words and sentences."""
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentences = sentences[:max_sentences]  # Limit sentences
        
        # Rejoin and count words
        truncated = '.'.join(sentence for sentence in sentences[:max_sentences]) + ('.' if sentences else '')
        words = truncated.split()
        if len(words) > max_words:
            truncated = ' '.join(words[:max_words]) + '.'
        
        return truncated


def create_agents(llm: BaseLanguageModel[Any], embedding: OpenAIEmbeddings) -> dict[str, BaseAgent]:
    """Create and return all agents for the workflow."""
    return {
        "router": RouterAgent(llm, embedding),
        "profiling": LearnerProfilingAgent(llm, embedding),
        "retrieval": RetrievalAgent(llm, embedding),
        "application_analysis": ApplicationAnalysisAgent(llm, embedding),
        "ui_ux_analysis": UIUXAnalysisAgent(llm, embedding),
        "curriculum_analysis": CurriculumAlignmentAgent(llm, embedding),
        "pedagogical_analysis": PedagogicalExpertAgent(llm, embedding),
        "aggregator": EvaluationAggregatorAgent(llm, embedding),
        "recommendation": RecommendationSynthesizerAgent(llm, embedding),
        "justification": JustificationSynthesizerAgent(llm, embedding),
        "optimizer": EvaluatorOptimizerAgent(llm, embedding),
    }
