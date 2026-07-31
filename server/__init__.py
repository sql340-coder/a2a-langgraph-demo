"""
A2A Server Implementation
Provides HTTP endpoints for agent-to-agent communication.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from shared.message_protocol import (
    AgentMessage, 
    A2ARequest, 
    A2AResponse, 
    MessageType,
    AgentStatus
)


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class A2AServer:
    """
    A2A Server that manages agent communication
    
    This server provides REST endpoints for agents to communicate with each other.
    It handles message routing, status tracking, and basic load balancing.
    
    Example usage:
        >>> server = A2AServer()
        >>> await server.start()
        >>> # Agents can now send messages via HTTP endpoints
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.app = FastAPI(
            title="A2A Communication Server",
            description="Agent-to-Agent communication server using LangChain & LangGraph",
            version="1.0.0"
        )
        
        # In-memory storage for demo purposes
        self.agents: Dict[str, AgentStatus] = {}
        self.message_queue: Dict[str, list] = {}  # agent_name -> [messages]
        self.active_requests: Dict[str, A2ARequest] = {}
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "A2A Communication Server",
                "version": "1.0.0",
                "endpoints": {
                    "register_agent": "/api/v1/agents/register",
                    "send_message": "/api/v1/messages/send",
                    "get_status": "/api/v1/status/{agent_name}",
                    "health_check": "/health"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "registered_agents": len(self.agents)
            }
        
        @self.app.post("/api/v1/agents/register")
        async def register_agent(request: AgentStatus):
            """Register a new agent"""
            self.agents[request.agent_name] = request
            self.message_queue[request.agent_name] = []
            
            # Send heartbeat
            await self._send_heartbeat(request.agent_name)
            
            return {
                "message": f"Agent '{request.agent_name}' registered successfully",
                "agent": request
            }
        
        @self.app.post("/api/v1/messages/send")
        async def send_message(request: A2ARequest, background_tasks: BackgroundTasks):
            """Send a message from one agent to another"""
            # Validate receiver exists
            if request.target_agent not in self.agents:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Target agent '{request.target_agent}' not found"
                )
            
            # Create and store message
            message = request.to_message()
            
            # Add to queue with correlation ID for response tracking
            if request.target_agent not in self.message_queue:
                self.message_queue[request.target_agent] = []
            
            self.message_queue[request.target_agent].append(message)
            self.active_requests[request.request_id] = request
            
            # Update sender status
            if request.sender in self.agents:
                self.agents[request.sender].current_task = f"Sending to {request.target_agent}"
            
            return {
                "message": "Message sent successfully",
                "request_id": request.request_id,
                "receiver": request.target_agent
            }
        
        @self.app.post("/api/v1/messages/respond")
        async def send_response(response: A2AResponse):
            """Send a response back to the original requester"""
            # Find the original request
            original_request = self.active_requests.get(response.request_id)
            if not original_request:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Original request '{response.request_id}' not found"
                )
            
            # Create response message
            message = response.to_message()
            message.sender = "system"  # Will be overridden by actual agent
            
            return {
                "message": "Response sent successfully",
                "request_id": response.request_id
            }
        
        @self.app.get("/api/v1/status/{agent_name}")
        async def get_agent_status(agent_name: str):
            """Get status of a specific agent"""
            if agent_name not in self.agents:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Agent '{agent_name}' not found"
                )
            
            agent_status = self.agents[agent_name]
            queued_messages = len(self.message_queue.get(agent_name, []))
            
            return {
                "agent_name": agent_status.agent_name,
                "is_online": agent_status.is_online,
                "current_task": agent_status.current_task,
                "queued_messages": queued_messages,
                "last_heartbeat": datetime.fromtimestamp(agent_status.last_heartbeat).isoformat()
            }
        
        @self.app.get("/api/v1/agents")
        async def list_agents():
            """List all registered agents"""
            return {
                "agents": [
                    {
                        "name": name,
                        "status": status.is_online,
                        "task": status.current_task
                    }
                    for name, status in self.agents.items()
                ]
            }
    
    async def _send_heartbeat(self, agent_name: str):
        """Send heartbeat to update agent status"""
        if agent_name in self.agents:
            self.agents[agent_name].last_heartbeat = datetime.now().timestamp()
    
    async def process_message_queue(self, agent_name: str):
        """Process messages in an agent's queue (to be called by the agent)"""
        if agent_name not in self.message_queue:
            return []
        
        messages = self.message_queue[agent_name].copy()
        self.message_queue[agent_name] = []  # Clear processed messages
        
        for message in messages:
            # Update agent status
            if agent_name in self.agents:
                self.agents[agent_name].current_task = f"Processing {message.message_type.value}"
            
            yield message
    
    async def start(self):
        """Start the server"""
        import uvicorn
        
        print(f"🚀 Starting A2A Server on {self.config.host}:{self.config.port}")
        print("📋 Available endpoints:")
        print("   POST /api/v1/agents/register - Register an agent")
        print("   POST /api/v1/messages/send   - Send a message")
        print("   GET  /api/v1/status/{name}    - Get agent status")
        print("   GET  /health                  - Health check")
        
        config = uvicorn.Config(
            self.app, 
            host=self.config.host, 
            port=self.config.port,
            log_level="info" if not self.config.debug else "debug"
        )
        server = uvicorn.Server(config)
        await server.serve()


# Convenience function for quick setup
def create_server(config: Optional[ServerConfig] = None) -> A2AServer:
    """Create and return an A2A server instance"""
    return A2AServer(config)
