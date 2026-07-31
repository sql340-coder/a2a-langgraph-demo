"""
Base Agent class for A2A communication.
Provides common functionality for all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel


class AgentState(BaseModel):
    """Common state structure for all agents"""
    agent_name: str
    messages: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {}
    status: str = "idle"
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Abstract base class for A2A agents
    
    All agents must implement the process_message method to handle incoming messages.
    The base class provides common functionality like LLM integration and message handling.
    
    Example implementation:
        class ResearcherAgent(BaseAgent):
            def __init__(self):
                super().__init__("researcher_agent")
            
            async def process_message(self, message: dict) -> dict:
                # Implement your agent logic here
                return {"result": "Research completed"}
    """
    
    def __init__(
        self, 
        name: str,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        self.name = name
        self.llm_model = llm_model
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )
        
        # Agent state
        self.state = AgentState(agent_name=name)
        self.message_history: List[Dict[str, Any]] = []
        
        # Build the agent graph
        self.graph = self._build_graph()
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        This method must be implemented by all concrete agents.
        
        Args:
            message: The incoming message dictionary
            
        Returns:
            Response dictionary with results
        """
        pass
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for this agent"""
        workflow = StateGraph(AgentState)
        
        # Add nodes (will be overridden by subclasses)
        workflow.add_node("process", self._process_node)
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        
        return workflow.compile()
    
    async def _process_node(self, state: AgentState) -> Dict[str, Any]:
        """Process a node in the graph - calls process_message"""
        try:
            # Get last message from history
            if state.messages:
                last_message = state.messages[-1]
                response = await self.process_message(last_message)
                
                # Update state with response
                state.messages.append({
                    "role": "assistant",
                    "content": str(response),
                    "timestamp": asyncio.get_event_loop().time()
                })
                state.status = "completed"
                
                return {
                    "messages": state.messages,
                    "status": "completed"
                }
            else:
                state.status = "idle"
                return {"status": "idle"}
                
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {
                "error": str(e),
                "status": "error"
            }
    
    async def invoke(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent with a message"""
        # Add message to history
        self.message_history.append({
            "role": "user",
            "content": message,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Update state
        self.state.messages.append({
            "role": "user",
            "content": message
        })
        
        # Run through graph
        result = await self.graph.ainvoke(self.state.dict())
        
        return result
    
    async def chat(self, user_input: str) -> str:
        """Simple chat interface"""
        response = await self.invoke({"type": "chat", "message": user_input})
        return str(response.get("result", "No response"))
