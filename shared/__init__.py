"""
A2A Communication Protocol Definition
Defines the message format and communication protocol for agent-to-agent interactions.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import time


class MessageType(str, Enum):
    """Type of A2A message"""
    QUERY = "query"              # 查询请求
    RESPONSE = "response"        # 响应
    TOOL_CALL = "tool_call"      # 工具调用
    TOOL_RESULT = "tool_result"  # 工具执行结果
    ERROR = "error"              # 错误消息
    STATUS = "status"            # 状态更新
    HEARTBEAT = "heartbeat"      # 心跳检测


class AgentMessage(BaseModel):
    """
    Standard message format for A2A communication
    
    Examples:
        >>> msg = AgentMessage(
        ...     sender="researcher_agent",
        ...     receiver="writer_agent",
        ...     message_type="query",
        ...     content={"task": "Research AI trends"}
        ... )
        >>> print(msg.message_id)  # Has auto-generated ID
    """
    message_id: str = Field(default_factory=lambda: f"msg_{int(time.time() * 1000)}")
    sender: str                              # Agent名称
    receiver: str                            # 目标Agent
    message_type: MessageType                # 消息类型
    content: Dict[str, Any]                  # 消息内容
    timestamp: float = Field(default_factory=time.time)
    correlation_id: Optional[str] = None     # 关联ID，用于匹配请求和响应
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据
    
    def to_dict(self) -> dict:
        """Convert message to dictionary"""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentMessage':
        """Create message from dictionary"""
        if isinstance(data.get("message_type"), str):
            data["message_type"] = MessageType(data["message_type"])
        return cls(**data)


class AgentStatus(BaseModel):
    """Agent status information"""
    agent_name: str
    is_online: bool = True
    current_task: Optional[str] = None
    queue_size: int = 0
    last_heartbeat: float = Field(default_factory=time.time)


class A2ARequest(BaseModel):
    """Standard request format for A2A communication"""
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time() * 1000)}")
    sender: str
    target_agent: str
    task_type: str                           # 任务类型，如 "research", "write"
    parameters: Dict[str, Any]               # 任务参数
    timeout: int = 300                       # 超时时间（秒）
    metadata: Optional[Dict[str, Any]] = None
    
    def to_message(self) -> AgentMessage:
        """Convert request to message"""
        return AgentMessage(
            sender=self.sender,
            receiver=self.target_agent,
            message_type=MessageType.QUERY,
            content={
                "request_id": self.request_id,
                "task_type": self.task_type,
                "parameters": self.parameters
            },
            correlation_id=self.request_id
        )


class A2AResponse(BaseModel):
    """Standard response format for A2A communication"""
    request_id: str
    status: str                              # "success", "error", "in_progress"
    result: Optional[Dict[str, Any]] = None  # 结果数据
    error_message: Optional[str] = None      # 错误信息
    metadata: Optional[Dict[str, Any]] = None
    
    def to_message(self) -> AgentMessage:
        """Convert response to message"""
        return AgentMessage(
            sender="system",  # Will be overridden by actual agent
            receiver="system",  # Will be overridden by actual agent
            message_type=MessageType.RESPONSE if self.status == "success" else MessageType.ERROR,
            content={
                "request_id": self.request_id,
                "status": self.status,
                "result": self.result,
                "error_message": self.error_message
            },
            correlation_id=self.request_id
        )


# 消息模板示例
MESSAGE_TEMPLATES = {
    "research_task": {
        "task_type": "research",
        "parameters": {
            "topic": str,
            "depth": str,      # "basic", "intermediate", "deep"
            "sources": int     # 期望的信息源数量
        }
    },
    "writing_task": {
        "task_type": "write", 
        "parameters": {
            "content_type": str,   # "article", "report", "summary"
            "tone": str,           # "formal", "casual", "technical"
            "length": str          # "short", "medium", "long"
        }
    }
}
