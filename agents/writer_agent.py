"""
Writer Agent - Specialized in content creation and editing.
Uses LangGraph to build a writing workflow with multiple stages.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field


class WritingState(BaseModel):
    """State for the writer agent"""
    topic: str = ""
    input_content: str = ""
    content_type: str = "article"  # article, report, summary, etc.
    tone: str = "professional"     # formal, casual, technical, etc.
    draft: str = ""
    edited_draft: str = ""
    final_output: str = ""
    status: str = "idle"
    error: Optional[str] = None


class WriterAgent:
    """
    Writer Agent that specializes in content creation and editing.
    
    This agent uses LangGraph to orchestrate a multi-step writing process:
    1. Understand requirements (type, tone, length)
    2. Generate initial draft
    3. Review and edit
    4. Finalize output
    
    Can also receive research results from ResearcherAgent and transform them into polished content.
    
    Example usage:
        >>> writer = WriterAgent()
        >>> result = await writer.write(
        ...     topic="AI Ethics",
        ...     content_type="article",
        ...     tone="professional"
        ... )
        >>> print(result["final_output"])
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.name = "writer_agent"
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        
        # Build writing workflow
        self.workflow = self._build_writing_workflow()
        self.app = self.workflow.compile()
    
    def _build_writing_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for writing"""
        workflow = StateGraph(WritingState)
        
        # Add nodes
        workflow.add_node("understand", self._understand_requirements)
        workflow.add_node("draft", self._create_draft)
        workflow.add_node("review", self._review_and_edit)
        workflow.add_node("finalize", self._finalize_output)
        
        # Define flow
        workflow.set_entry_point("understand")
        workflow.add_edge("understand", "draft")
        workflow.add_edge("draft", "review")
        workflow.add_edge("review", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow
    
    async def _understand_requirements(self, state: WritingState) -> dict:
        """Understand the writing requirements"""
        print(f"📋 Understanding requirements...")
        print(f"   Topic: {state.topic}")
        print(f"   Type: {state.content_type}")
        print(f"   Tone: {state.tone}")
        
        # Validate and enhance requirements
        prompt = f"""
        Analyze these writing requirements and provide a brief plan:
        
        Topic: {state.topic}
        Content Type: {state.content_type}
        Tone: {state.tone}
        
        Provide:
        1. Key points to cover
        2. Suggested structure
        3. Target audience
        
        Keep it concise (3-5 bullet points).
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            state.input_content = str(response.content)
            
            print("✅ Requirements understood")
            return {"input_content": state.input_content, "status": "requirements_set"}
            
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {"error": str(e), "status": "error"}
    
    async def _create_draft(self, state: WritingState) -> dict:
        """Create the initial draft"""
        print(f"✍️ Creating draft...")
        
        if state.topic and not state.input_content:
            # Generate from topic
            prompt = f"""
            Write a {state.content_type} about "{state.topic}" in a {state.tone} tone.
            
            Requirements:
            - Professional quality
            - Well-structured with clear sections
            - Engaging and informative
            - Appropriate length for a {state.content_type}
            
            Start writing directly without preamble.
            """
        elif state.input_content:
            # Transform existing content
            prompt = f"""
            Transform the following content into a well-written {state.content_type}:
            
            Input: {state.input_content[:1000]}  # Limit input size
            
            Requirements:
            - Tone: {state.tone}
            - Improve clarity and flow
            - Add appropriate transitions
            - Maintain key information
            
            Output only the final text, no explanations.
            """
        else:
            return {"error": "No topic or input content provided", "status": "error"}
        
        try:
            response = await self.llm.ainvoke(prompt)
            state.draft = str(response.content)
            
            print(f"✅ Draft created ({len(state.draft)} characters)")
            return {"draft": state.draft, "status": "drafted"}
            
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {"error": str(e), "status": "error"}
    
    async def _review_and_edit(self, state: WritingState) -> dict:
        """Review and edit the draft"""
        print(f"🔍 Reviewing and editing...")
        
        prompt = f"""
        Review and improve this {state.content_type}:
        
        Current Draft:
        {state.draft[:2000]}  # Limit for context
        
        Please:
        1. Fix any grammar or spelling errors
        2. Improve sentence structure where needed
        3. Enhance clarity and flow
        4. Ensure consistent tone ({state.tone})
        5. Add missing transitions
        
        Return ONLY the improved version, no explanations.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            state.edited_draft = str(response.content)
            
            print("✅ Review complete")
            return {"edited_draft": state.edited_draft, "status": "reviewed"}
            
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {"error": str(e), "status": "error"}
    
    async def _finalize_output(self, state: WritingState) -> dict:
        """Finalize the output"""
        print(f"📝 Finalizing output...")
        
        # Add final polish if needed
        prompt = f"""
        Give this {state.content_type} a final polish:
        
        Current Version:
        {state.edited_draft[:2000]}
        
        Make any last improvements for:
        - Readability
        - Impact
        - Professionalism
        
        Return ONLY the final text.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            state.final_output = str(response.content)
            
            print("✅ Output finalized!")
            return {"final_output": state.final_output, "status": "completed"}
            
        except Exception as e:
            state.error = str(e)
            state.status = "error"
            return {"error": str(e), "status": "error"}
    
    async def write(
        self, 
        topic: str = "", 
        input_content: str = "",
        content_type: str = "article",
        tone: str = "professional"
    ) -> Dict[str, Any]:
        """
        Main method to create written content
        
        Args:
            topic: The topic to write about
            input_content: Optional existing content to transform
            content_type: Type of content (article, report, summary, etc.)
            tone: Writing tone (formal, casual, technical, etc.)
            
        Returns:
            Dictionary containing the final output and metadata
        """
        # Initialize state
        state = WritingState(
            topic=topic,
            input_content=input_content,
            content_type=content_type,
            tone=tone,
            status="initiated"
        )
        
        print(f"\n{'='*60}")
        print(f"✍️ Starting Writer Agent")
        print(f"Topic: {topic or 'Transforming existing content'}")
        print(f"Type: {content_type} | Tone: {tone}")
        print(f"{'='*60}\n")
        
        # Run workflow
        result = await self.app.ainvoke(state.dict())
        
        # Format output
        output = {
            "topic": topic,
            "content_type": content_type,
            "tone": tone,
            "status": result.get("status", "error"),
            "output": result.get("final_output", ""),
            "draft": result.get("draft", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        if result.get("error"):
            output["error"] = result["error"]
        
        return output
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming A2A message"""
        msg_type = message.get("type", "write")
        
        if msg_type == "write":
            content_params = message.get("content", {})
            params = message.get("parameters", {})
            
            topic = content_params.get("topic", "") or params.get("topic", "")
            input_content = content_params.get("input_content", "") or params.get("input_content", "")
            content_type = content_params.get("content_type", "article") or params.get("content_type", "article")
            tone = content_params.get("tone", "professional") or params.get("tone", "professional")
            
            result = await self.write(
                topic=topic,
                input_content=input_content,
                content_type=content_type,
                tone=tone
            )
            
            return {
                "type": "writing_result",
                "status": "success",
                "result": result
            }
        
        elif msg_type == "transform":
            # Transform existing research into article format
            research_data = message.get("content", {})
            report = research_data.get("final_report", "")
            
            if not report:
                report = str(research_data)
            
            result = await self.write(
                input_content=report,
                content_type="article",
                tone="professional"
            )
            
            return {
                "type": "transformation_result",
                "status": "success",
                "result": result
            }
        
        elif msg_type == "chat":
            user_message = message.get("content", {}).get("message", "")
            response = f"I'm a writer agent. I can help you create articles, reports, and other written content!"
            return {"type": "response", "content": response}
        
        else:
            return {"type": "error", "message": f"Unknown message type: {msg_type}"}


# Demo usage
async def demo_writer():
    """Demo the writer agent"""
    print("\n🚀 Writer Agent Demo\n")
    
    writer = WriterAgent()
    
    # Example 1: Write from topic
    result = await writer.write(
        topic="The Future of Remote Work in 2024",
        content_type="article",
        tone="professional"
    )
    
    if result["status"] == "completed":
        print("\n📄 Writing Results:")
        print(f"Topic: {result['topic']}")
        print(f"Type: {result['content_type']}")
        print(f"Tone: {result['tone']}")
        print(f"\nOutput Preview (first 500 chars):")
        print(result['output'][:500] + "...")
    
    return result


if __name__ == "__main__":
    asyncio.run(demo_writer())
