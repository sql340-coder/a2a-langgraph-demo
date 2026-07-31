"""
A2A Client Implementation
Provides methods for agents to communicate via the A2A server.
"""

import asyncio
from typing import Dict, Any, Optional, Callable
import httpx
import json
from datetime import datetime

from shared.message_protocol import (
    AgentMessage, 
    A2ARequest, 
    A2AResponse, 
    MessageType,
    AgentStatus
)


class A2AClient:
    """
    A2A Client for agent communication
    
    This client provides methods for agents to send and receive messages
    through the A2A server. It handles connection management, message
    formatting, and response processing.
    
    Example usage:
        >>> client = A2AClient(server_url="http://localhost:8000")
        >>> await client.register("researcher_agent", is_online=True)
        >>> request = A2ARequest(
        ...     sender="user",
        ...     target_agent="researcher_agent",
        ...     task_type="research",
        ...     parameters={"topic": "AI trends"}
        ... )
        >>> response = await client.send_request(request)
    """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.agent_name: Optional[str] = None
        self.client = httpx.AsyncClient(base_url=server_url, timeout=30.0)
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.is_connected = False
    
    async def connect(self):
        """Connect to the A2A server"""
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                self.is_connected = True
                print(f"✅ Connected to A2A Server at {self.server_url}")
                return True
            else:
                raise ConnectionError(f"Server returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the A2A server"""
        await self.client.aclose()
        self.is_connected = False
    
    async def register(self, agent_name: str, **kwargs) -> bool:
        """Register this agent with the server"""
        if not self.is_connected:
            await self.connect()
        
        try:
            status = AgentStatus(
                agent_name=agent_name,
                is_online=True,
                current_task="registered",
                queue_size=0
            )
            
            response = await self.client.post(
                "/api/v1/agents/register",
                json=status.dict()
            )
            
            if response.status_code == 200:
                self.agent_name = agent_name
                print(f"🤖 Agent '{agent_name}' registered successfully")
                return True
            else:
                print(f"❌ Registration failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False
    
    async def send_request(self, request: A2ARequest) -> dict:
        """Send a request to another agent"""
        if not self.is_connected:
            await self.connect()
        
        try:
            response = await self.client.post(
                "/api/v1/messages/send",
                json=request.dict()
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"📤 Message sent to '{request.target_agent}'")
                return result
            else:
                raise Exception(f"Failed to send message: {response.text}")
                
        except Exception as e:
            print(f"❌ Send error: {e}")
            raise
    
    async def send_response(self, response: A2AResponse) -> dict:
        """Send a response to complete a request"""
        try:
            response_data = response.dict()
            # Override sender with actual agent name if set
            if self.agent_name:
                response_data["sender"] = self.agent_name
            
            resp = await self.client.post(
                "/api/v1/messages/respond",
                json=response_data
            )
            
            if resp.status_code == 200:
                return resp.json()
            else:
                raise Exception(f"Failed to send response: {resp.text}")
                
        except Exception as e:
            print(f"❌ Response error: {e}")
            raise
    
    async def get_agent_status(self, agent_name: str) -> dict:
        """Get the status of a specific agent"""
        try:
            response = await self.client.get(f"/api/v1/status/{agent_name}")
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get status: {response.text}")
                
        except Exception as e:
            print(f"❌ Status check error: {e}")
            raise
    
    async def list_agents(self) -> list:
        """List all registered agents"""
        try:
            response = await self.client.get("/api/v1/agents")
            
            if response.status_code == 200:
                data = response.json()
                return data.get("agents", [])
            else:
                raise Exception(f"Failed to list agents: {response.text}")
                
        except Exception as e:
            print(f"❌ List agents error: {e}")
            raise
    
    async def send_message(
        self, 
        receiver: str, 
        message_type: MessageType, 
        content: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """Send a direct message to another agent"""
        
        # Create A2A request for the message
        request = A2ARequest(
            sender=self.agent_name or "unknown",
            target_agent=receiver,
            task_type=message_type.value,
            parameters={
                "message_content": content,
                "correlation_id": correlation_id
            }
        )
        
        result = await self.send_request(request)
        
        # Create message object for tracking
        message = AgentMessage(
            sender=self.agent_name or "unknown",
            receiver=receiver,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id
        )
        
        return message
    
    async def process_incoming_messages(self, handler: Callable):
        """
        Process incoming messages for this agent
        
        This method should be called in a loop to continuously check for new messages.
        The handler function will be called with each message as it arrives.
        
        Args:
            handler: Async function that takes an AgentMessage and returns a response
        """
        print(f"👂 Listening for messages as '{self.agent_name}'...")
        
        while True:
            try:
                # Get current status
                status = await self.get_agent_status(self.agent_name)
                
                if status.get("queued_messages", 0) > 0:
                    print(f"📨 {status['queued_messages']} message(s) waiting")
                    
                    # In a real implementation, you'd poll for new messages
                    # For demo purposes, we'll simulate processing
                    
                await asyncio.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                print(f"❌ Error processing messages: {e}")
                await asyncio.sleep(5)


# Convenience function for quick client creation
def create_client(server_url: str = "http://localhost:8000") -> A2AClient:
    """Create and return an A2A client instance"""
    return A2AClient(server_url)
