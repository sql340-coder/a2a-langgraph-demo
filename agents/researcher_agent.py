"""
Researcher Agent - Specialized in information gathering and analysis.
Uses LangGraph to build a research workflow with tool calling capabilities.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field


class ResearchState(BaseModel):
    """State for the researcher agent"""
    topic: str = ""
    query: str = ""
    search_results: List[str] = []
    analysis: str = ""
    final_report: str = ""
    status: str = "idle"
    error: Optional[str] = None


class ResearcherAgent:
    """
    Researcher Agent that specializes in information gathering and analysis.
    
    This agent uses LangGraph to orchestrate a multi-step research process:
    1. Understand the research question
    2. Generate search queries
    3. Simulate searching (in demo, use LLM to generate results)
    4. Analyze findings
    5. Create comprehensive report
    
    Example usage:
        >>> researcher = ResearcherAgent()
        >>> result = await researcher.research("What are the latest AI trends?")
        >>> print(result["final_report"])
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.name = "researcher_agent"
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        
        # Build research workflow
        self.workflow = self._build_research_workflow()
        self.app = self.workflow.compile()
    
    def _build_research_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for research"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("understand", self._understand_query)
        workflow.add_node("generate_queries", self._generate_search_queries)
        workflow.add_node("simulate_search", self._simulate_search)
        workflow.add_node("analyze", self._analyze_results)
        workflow.add_node("report", self._create_report)
        
        # Define flow
        workflow.set_entry_point("understand")
        workflow.add_edge("understand", "generate_queries")
        workflow.add_edge("generate_queries", "simulate_search")
        workflow.add_edge("simulate_search", "analyze")
        workflow.add_edge("analyze", "report")
        workflow.add_edge("report", END)
        
        return workflow
    
    async def _understand_query(self, state: ResearchState) -> dict:
        """Understand and clarify the research question"""
        print(f"🔍 Understanding query: {state.topic}")
        
        prompt = f"""
        Analyze this research request and break it down into key aspects:
        
        Topic: {state.topic}
        
        Provide a structured analysis with:
        1. Main question being asked
        2. Key sub-topics to explore
        3. Important keywords for searching
        
        Keep your response concise and focused.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            state.analysis = str(response.content)
            state.status = "understood"
            
            print("✅ Query understood")
            return {"analysis": state.analysis, "status": "understood"}
            
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {"error": str(e), "status": "error"}
    
    async def _generate_search_queries(self, state: ResearchState) -> dict:
        """Generate search queries based on the research topic"""
        print(f"📝 Generating search queries...")
        
        prompt = f"""
        Based on this research analysis, generate 5 specific search queries:
        
        Analysis: {state.analysis}
        
        Return ONLY a JSON array of query strings, like:
        ["query1", "query2", "query3"]
        
        Make each query specific and likely to return useful results.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            
            # Parse the JSON array from response
            import json
            content = str(response.content)
            # Extract JSON array
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != -1:
                queries = json.loads(content[start:end])
                state.query = ", ".join(queries[:3])  # Use top 3 queries
            else:
                state.query = content[:200]
            
            print(f"✅ Generated {len(state.search_results)} search queries")
            return {"query": state.query, "status": "queries_generated"}
            
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {"error": str(e), "status": "error"}
    
    async def _simulate_search(self, state: ResearchState) -> dict:
        """Simulate searching and gathering information"""
        print(f"🌐 Simulating search for: {state.topic}")
        
        # In a real implementation, this would call actual search APIs
        # For demo purposes, we'll use LLM to generate realistic search results
        
        prompt = f"""
        Act as if you searched the web for: "{state.topic}"
        
        Provide 5 realistic search result snippets that would be relevant.
        Format each result with:
        - Title
        - URL (fake but realistic)
        - Brief description (2-3 sentences)
        
        Return as a numbered list. Make them look like real search results.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            
            # Parse simulated results
            content = str(response.content)
            results = [line.strip() for line in content.split('\n') if line.strip()]
            state.search_results = results[:5]  # Take top 5 results
            
            print(f"✅ Found {len(state.search_results)} search results")
            return {"search_results": state.search_results, "status": "searched"}
            
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {"error": str(e), "status": "error"}
    
    async def _analyze_results(self, state: ResearchState) -> dict:
        """Analyze the search results and extract key insights"""
        print(f"🧠 Analyzing {len(state.search_results)} research results...")
        
        results_text = "\n\n".join([f"{i+1}. {r}" for i, r in enumerate(state.search_results)])
        
        prompt = f"""
        Analyze these research findings and provide key insights:
        
        {results_text}
        
        Provide a structured analysis with:
        1. Main themes identified
        2. Key facts and statistics
        3. Contradictions or gaps in information
        4. Most important conclusions
        
        Keep it concise but comprehensive.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            state.analysis = str(response.content)
            
            print("✅ Analysis complete")
            return {"analysis": state.analysis, "status": "analyzed"}
            
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {"error": str(e), "status": "error"}
    
    async def _create_report(self, state: ResearchState) -> dict:
        """Create a comprehensive research report"""
        print(f"📄 Creating final report...")
        
        prompt = f"""
        Create a professional research report based on these findings:
        
        Topic: {state.topic}
        
        Analysis: {state.analysis}
        
        Write a well-structured report with:
        - Executive Summary (2-3 sentences)
        - Key Findings (bullet points)
        - Detailed Analysis (paragraphs)
        - Conclusion and Recommendations
        
        Make it informative and engaging. Target length: 500-800 words.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            state.final_report = str(response.content)
            
            print("✅ Report created successfully!")
            return {"final_report": state.final_report, "status": "completed"}
            
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {"error": str(e), "status": "error"}
    
    async def research(self, topic: str) -> Dict[str, Any]:
        """
        Main method to conduct research on a topic
        
        Args:
            topic: The research topic or question
            
        Returns:
            Dictionary containing the final report and metadata
        """
        # Initialize state
        state = ResearchState(topic=topic, status="initiated")
        
        print(f"\n{'='*60}")
        print(f"🔬 Starting Research Agent")
        print(f"Topic: {topic}")
        print(f"{'='*60}\n")
        
        # Run workflow
        result = await self.app.ainvoke(state.dict())
        
        # Format output
        output = {
            "topic": topic,
            "status": result.get("status", "error"),
            "report": result.get("final_report", ""),
            "analysis": result.get("analysis", ""),
            "search_results_count": len(result.get("search_results", [])),
            "timestamp": datetime.now().isoformat()
        }
        
        if result.get("error"):
            output["error"] = result["error"]
        
        return output
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming A2A message"""
        msg_type = message.get("type", "research")
        
        if msg_type == "research":
            topic = message.get("content", {}).get("topic", "")
            if not topic:
                topic = message.get("parameters", {}).get("topic", "")
            
            if topic:
                result = await self.research(topic)
                return {
                    "type": "research_result",
                    "status": "success",
                    "result": result
                }
            else:
                return {
                    "type": "error",
                    "status": "error",
                    "message": "No research topic provided"
                }
        
        elif msg_type == "chat":
            user_message = message.get("content", {}).get("message", "")
            response = f"I'm a researcher agent. I can help you research topics! Try sending me a 'research' message with a topic."
            return {"type": "response", "content": response}
        
        else:
            return {"type": "error", "message": f"Unknown message type: {msg_type}"}


# Demo usage
async def demo_researcher():
    """Demo the researcher agent"""
    print("\n🚀 Researcher Agent Demo\n")
    
    researcher = ResearcherAgent()
    
    # Example 1: Simple research
    result = await researcher.research("Latest developments in artificial intelligence 2024")
    
    if result["status"] == "completed":
        print("\n📊 Research Results:")
        print(f"Topic: {result['topic']}")
        print(f"Status: {result['status']}")
        print(f"\nReport Preview (first 500 chars):")
        print(result['report'][:500] + "...")
    
    return result


if __name__ == "__main__":
    asyncio.run(demo_researcher())
