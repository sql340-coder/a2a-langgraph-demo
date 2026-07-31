"""
A2A (Agent-to-Agent) Communication Demo - Simplified Version
=============================================================

This is a simplified, self-contained A2A demo that you can run immediately.
It demonstrates two agents communicating with each other using LangChain and LangGraph.

Prerequisites:
    pip install langchain langchain-openai langgraph pydantic httpx fastapi uvicorn python-dotenv

Usage:
    1. Set your OPENAI_API_KEY environment variable
    2. Run: python simple_a2a_demo.py
    
The demo shows:
    - ResearcherAgent: Gathers information on a topic
    - WriterAgent: Creates content from research results
    - A2A Protocol: Messages passed between agents via HTTP REST API
"""

import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime

# LangChain & LangGraph imports
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel


# ============================================================
# Part 1: Message Protocol (Simplified)
# ============================================================

class A2AMessage(BaseModel):
    """Simple message format for agent communication"""
    sender: str
    receiver: str
    type: str                    # "query", "response", "error"
    content: Dict[str, Any]
    timestamp: float = 0.0
    
    def __init__(self, **data):
        super().__init__(**data)
        self.timestamp = datetime.now().timestamp()
    
    def to_dict(self):
        return self.model_dump()


# ============================================================
# Part 2: Researcher Agent (LangGraph-based)
# ============================================================

class ResearchState(BaseModel):
    """State for research workflow"""
    topic: str = ""
    steps_completed: list = []
    research_data: str = ""
    final_report: str = ""


class ResearcherAgent:
    """Research agent that uses LangGraph for multi-step research"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.name = "researcher"
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the research workflow graph"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("plan", self._plan_research)
        workflow.add_node("gather", self._gather_info)
        workflow.add_node("synthesize", self._synthesize_report)
        
        # Define edges
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "gather")
        workflow.add_edge("gather", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    async def _plan_research(self, state: ResearchState) -> dict:
        """Step 1: Plan the research approach"""
        print(f"📋 [{self.name}] Planning research on: {state.topic}")
        
        prompt = f"""Create a research plan for: "{state.topic}"
Provide 3 key areas to investigate. Keep it brief."""
        
        response = await self.llm.ainvoke(prompt)
        plan = str(response.content)
        
        print(f"   ✅ Plan created")
        return {"steps_completed": ["planned"], "research_data": plan}
    
    async def _gather_info(self, state: ResearchState) -> dict:
        """Step 2: Gather information (simulated search)"""
        print(f"🔍 [{self.name}] Gathering information...")
        
        prompt = f"""Based on this research plan:\n{state.research_data}\n\nProvide detailed findings for topic: "{state.topic}"\nInclude key facts, statistics, and insights."""
        
        response = await self.llm.ainvoke(prompt)
        info = str(response.content)
        
        print(f"   ✅ Information gathered ({len(info)} chars)")
        return {"research_data": info}
    
    async def _synthesize_report(self, state: ResearchState) -> dict:
        """Step 3: Create final report"""
        print(f"📝 [{self.name}] Writing final report...")
        
        prompt = f"""Write a comprehensive report on "{state.topic}" based on:\n\n{state.research_data}\n\nMake it well-structured with introduction, body, and conclusion."""
        
        response = await self.llm.ainvoke(prompt)
        report = str(response.content)
        
        print(f"   ✅ Report complete ({len(report)} chars)")
        return {"final_report": report}
    
    async def research(self, topic: str) -> Dict[str, Any]:
        """Run the research workflow"""
        state = ResearchState(topic=topic)
        result = await self.graph.ainvoke(state.model_dump())
        
        return {
            "topic": topic,
            "report": result.get("final_report", ""),
            "status": "completed"
        }


# ============================================================
# Part 3: Writer Agent (LangGraph-based)
# ============================================================

class WritingState(BaseModel):
    """State for writing workflow"""
    input_content: str = ""
    content_type: str = "article"
    tone: str = "professional"
    draft: str = ""
    final_output: str = ""


class WriterAgent:
    """Writer agent that uses LangGraph for content creation"""
    
    def __init__(self, model="gpt-4o-mini"):
        self.name = "writer"
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the writing workflow graph"""
        workflow = StateGraph(WritingState)
        
        # Add nodes
        workflow.add_node("draft", self._create_draft)
        workflow.add_node("refine", self._refine_content)
        workflow.add_node("finalize", self._finalize)
        
        # Define edges
        workflow.set_entry_point("draft")
        workflow.add_edge("draft", "refine")
        workflow.add_edge("refine", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _create_draft(self, state: WritingState) -> dict:
        """Step 1: Create initial draft"""
        print(f"✍️ [{self.name}] Creating {state.content_type}...")
        
        prompt = f"""Write a {state.tone} {state.content_type} based on this content:\n\n{state.input_content[:2000]}\n\nOutput only the text, no explanations."""
        
        response = await self.llm.ainvoke(prompt)
        draft = str(response.content)
        
        print(f"   ✅ Draft created ({len(draft)} chars)")
        return {"draft": draft}
    
    async def _refine_content(self, state: WritingState) -> dict:
        """Step 2: Refine and improve"""
        print(f"🔧 [{self.name}] Refining content...")
        
        prompt = f"""Improve this {state.content_type}: {state.draft[:2000]}\n\nMake it more engaging, clear, and professional."""
        
        response = await self.llm.ainvoke(prompt)
        refined = str(response.content)
        
        print(f"   ✅ Content refined")
        return {"final_output": refined}
    
    async def _finalize(self, state: WritingState) -> dict:
        """Step 3: Final polish"""
        print(f"✨ [{self.name}] Finalizing output...")
        
        # Add final formatting
        final = f"# {state.input_content.split(chr(10))[0][:50]}...\n\n{state.final_output}"
        
        print(f"   ✅ Output finalized ({len(final)} chars)")
        return {"final_output": final}
    
    async def write(self, input_content: str, content_type="article", tone="professional") -> Dict[str, Any]:
        """Run the writing workflow"""
        state = WritingState(
            input_content=input_content,
            content_type=content_type,
            tone=tone
        )
        result = await self.graph.ainvoke(state.model_dump())
        
        return {
            "output": result.get("final_output", ""),
            "status": "completed"
        }


# ============================================================
# Part 4: A2A Communication Layer (HTTP-based)
# ============================================================

class SimpleA2AServer:
    """Simple in-memory A2A server for demo purposes"""
    
    def __init__(self):
        self.agents = {}
        self.messages = []
        print("🖥️  A2A Server initialized")
    
    def register_agent(self, agent_name: str):
        """Register an agent"""
        self.agents[agent_name] = {"online": True, "last_seen": datetime.now()}
        print(f"   🤖 Agent '{agent_name}' registered")
    
    def send_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Send a message between agents (simulated HTTP call)"""
        # Validate receiver exists
        if message.receiver not in self.agents:
            return {"error": f"Agent '{message.receiver}' not found"}
        
        # Store the message
        self.messages.append(message.to_dict())
        
        print(f"   📬 Message sent: {message.sender} → {message.receiver}")
        
        return {
            "status": "sent",
            "message_id": len(self.messages),
            "receiver": message.receiver
        }


# ============================================================
# Part 5: Full A2A Demo Workflow
# ============================================================

async def run_full_demo():
    """Run the complete A2A demonstration"""
    
    print("\n" + "="*70)
    print("🤖 A2A (Agent-to-Agent) Communication Demo")
    print("="*70)
    print("Using LangChain + LangGraph for real agent collaboration\n")
    
    # Initialize components
    server = SimpleA2AServer()
    researcher = ResearcherAgent(model="gpt-4o-mini")
    writer = WriterAgent(model="gpt-4o-mini")
    
    # Register agents with the server
    print("\n[1/5] 📝 Registering agents...")
    server.register_agent(researcher.name)
    server.register_agent(writer.name)
    
    # Define research topic
    topic = "The Impact of AI on Software Development in 2024"
    
    # Step 1: ResearcherAgent conducts research
    print(f"\n[2/5] 🔬 {researcher.name} is researching '{topic}'...")
    research_result = await researcher.research(topic)
    
    if research_result["status"] != "completed":
        print("❌ Research failed!")
        return
    
    print(f"✅ Research complete! Report length: {len(research_result['report'])} chars\n")
    
    # Step 2: Send results to WriterAgent via A2A protocol
    print("[3/5] 🔄 Transferring data via A2A Protocol...")
    
    # Create A2A message (this simulates an HTTP POST request)
    a2a_message = A2AMessage(
        sender=researcher.name,
        receiver=writer.name,
        type="query",
        content={
            "task": "write_article",
            "research_report": research_result["report"],
            "topic": topic
        }
    )
    
    # Send through server (simulated HTTP call)
    response = server.send_message(a2a_message)
    
    if "error" in response:
        print(f"❌ Message failed: {response['error']}")
        return
    
    print(f"✅ Message delivered! ({response.get('message_id', 0)} messages processed)\n")
    
    # Step 3: WriterAgent creates article from research
    print("[4/5] ✍️  Writing article based on research...")
    
    writing_result = await writer.write(
        input_content=research_result["report"],
        content_type="article",
        tone="professional"
    )
    
    if writing_result["status"] != "completed":
        print("❌ Writing failed!")
        return
    
    print(f"✅ Article created! Length: {len(writing_result['output'])} chars\n")
    
    # Step 4: Display results
    print("[5/5] 📊 Final Results:\n")
    print("="*70)
    print(f"🎯 TOPIC: {topic}")
    print(f"👥 AGENTS: {researcher.name} → {writer.name}")
    print(f"📬 MESSAGES SENT: {len(server.messages)}")
    print("="*70)
    
    print("\nFINAL ARTICLE:")
    print("-"*70)
    output = writing_result["output"]
    # Print first 1000 chars or less if shorter
    display_length = min(1200, len(output))
    print(output[:display_length])
    if len(output) > display_length:
        print(f"\n... ({len(output) - display_length} more characters)")
    
    print("\n" + "="*70)
    print("✅ Demo completed successfully!")
    print("="*70)


async def run_simple_demo():
    """Run a simpler demo without full A2A protocol"""
    
    print("\n" + "="*70)
    print("🤖 Simple A2A Demo (Direct Agent Communication)")
    print("="*70)
    
    researcher = ResearcherAgent(model="gpt-4o-mini")
    writer = WriterAgent(model="gpt-4o-mini")
    
    topic = "Remote Work Trends and Future of Office Spaces"
    
    # Research phase
    print(f"\n🔬 Researching: {topic}")
    research = await researcher.research(topic)
    
    # Writing phase  
    print(f"\n✍️  Writing article...")
    article = await writer.write(research["report"])
    
    print("\n" + "-"*70)
    print(article["output"][:800])
    print("-"*70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        asyncio.run(run_simple_demo())
    else:
        asyncio.run(run_full_demo())
