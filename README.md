# 🤖 A2A (Agent-to-Agent) Communication Demo with LangChain & LangGraph

一个完整的 Agent-to-Agent 通信示例，展示如何使用 **LangChain** 和 **LangGraph** 实现两个真实 agent 之间的交互。

## ✨ 特性

- 🔬 **ResearcherAgent**: 使用 LangGraph 构建的多步骤研究工作流
- ✍️ **WriterAgent**: 基于研究结果创建专业内容的写作工作流  
- 📡 **A2A Protocol**: 标准化的 HTTP REST API 消息传递协议
- 🔄 **完整协作流程**: Researcher → Writer 的端到端 agent 协作
- 💻 **可执行代码**: 提供简化版和完整版两种运行方式

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-openai langgraph pydantic httpx fastapi uvicorn python-dotenv
```

### 2. 设置 API Key

```bash
export OPENAI_API_KEY="sk-your-key-here"
# 或创建 .env 文件
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 3. 运行 Demo

**简化版（推荐新手）** - 直接运行，无需启动服务器：
```bash
python simple_a2a_demo.py
```

**完整版** - 包含完整 HTTP Server/Client 架构：
```bash
python main.py --mode full
```

**仅客户端演示**:
```bash
python main.py --mode client
```

## 📁 项目结构

```
a2a-langgraph-demo/
├── simple_a2a_demo.py          # ⭐ 简化版单文件 demo（可直接运行）
├── main.py                     # 完整版入口（支持多种模式）
├── requirements.txt            # Python 依赖
├── .env.example               # API Key 配置模板
│
├── agents/                    # Agent 实现
│   ├── __init__.py           # BaseAgent 基类
│   ├── researcher_agent.py   # 🔬 Researcher Agent (LangGraph)
│   └── writer_agent.py       # ✍️ Writer Agent (LangGraph)
│
├── server/                    # A2A Server
│   └── __init__.py           # FastAPI REST API 服务器
│
├── client/                    # A2A Client  
│   └── __init__.py           # HTTP 客户端实现
│
└── shared/                    # 共享模块
    └── __init__.py           # 消息协议定义
```

## 🎯 Demo 工作流程

### 简化版 (`simple_a2a_demo.py`)

```
用户请求 → ResearcherAgent 研究 → WriterAgent 写作 → 输出结果
              (LangGraph)            (LangGraph)        ✨
```

**步骤**:
1. **ResearcherAgent** 使用 LangGraph 执行多步研究：
   - `plan` → 制定研究计划
   - `gather` → 收集信息（模拟搜索）
   - `synthesize` → 综合分析报告

2. **A2A 协议传递** 研究结果到 WriterAgent

3. **WriterAgent** 使用 LangGraph 创建内容：
   - `draft` → 生成初稿
   - `refine` → 润色改进
   - `finalize` → 最终定稿

### 完整版 (`main.py`)

```
HTTP Client → POST /messages/send → A2A Server → Researcher Agent
                                                    ↓
                                              Writer Agent ← GET /status
```

包含完整的 HTTP REST API 服务器和客户端，模拟真实的分布式 agent 通信。

## 💡 代码示例

### 核心 A2A 消息协议

```python
from pydantic import BaseModel
from typing import Dict, Any

class A2AMessage(BaseModel):
    sender: str                          # 发送者 agent 名称
    receiver: str                        # 接收者 agent 名称  
    type: str                            # "query", "response", "error"
    content: Dict[str, Any]              # 消息内容
    timestamp: float                     # 时间戳
```

### ResearcherAgent 使用 LangGraph

```python
from langgraph.graph import StateGraph, END

class ResearcherAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ResearchState)
        
        # 添加节点
        workflow.add_node("plan", self._plan_research)
        workflow.add_node("gather", self._gather_info) 
        workflow.add_node("synthesize", self._synthesize_report)
        
        # 定义流程
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "gather")
        workflow.add_edge("gather", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
```

### WriterAgent 使用 LangGraph

```python
class WriterAgent:
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WritingState)
        
        workflow.add_node("draft", self._create_draft)
        workflow.add_node("refine", self._refine_content)
        workflow.add_node("finalize", self._finalize)
        
        workflow.set_entry_point("draft")
        workflow.add_edge("draft", "refine")
        workflow.add_edge("refine", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
```

### A2A 通信示例

```python
# 创建消息
message = A2AMessage(
    sender="researcher",
    receiver="writer", 
    type="query",
    content={
        "task": "write_article",
        "research_report": research_result["report"]
    }
)

# 通过服务器发送（模拟 HTTP POST）
response = server.send_message(message)

# WriterAgent 接收并处理
result = await writer.process_message(message.content)
```

## 🔧 扩展建议

1. **添加更多 Agent**: 代码审查、数据分析、翻译等
2. **真实搜索集成**: 接入 Google Search API / Tavily
3. **持久化存储**: 使用 Redis/PostgreSQL 保存消息历史
4. **WebSocket 支持**: 实现实时双向通信
5. **认证授权**: 添加 JWT Token 验证
6. **监控日志**: 集成 Prometheus + Grafana

## 📚 技术栈

| 组件 | 用途 |
|------|------|
| [LangChain](https://python.langchain.com/) | LLM 抽象和工具调用 |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Agent 状态管理和工作流编排 |
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP API 服务器框架 |
| [Pydantic](https://docs.pydantic.dev/) | 数据验证和序列化 |
| [httpx](https://www.python-httpx.org/) | Async HTTP 客户端 |

## 📝 License

MIT License - 自由使用和修改
