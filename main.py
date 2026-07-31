"""
Main entry point for A2A Demo
Demonstrates full agent-to-agent communication workflow.
"""

import asyncio
import argparse
from typing import Dict, Any

# Import our modules
from server.a2a_server import A2AServer, ServerConfig
from client.a2a_client import A2AClient
from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent


class A2ADemo:
    """
    Complete A2A demonstration showing agent collaboration
    
    This class orchestrates the full workflow:
    1. Start A2A server
    2. Register both agents
    3. Send research request to ResearcherAgent
    4. Pass results to WriterAgent for content creation
    5. Display final output
    """
    
    def __init__(self):
        self.server = None
        self.client = None
        self.researcher = None
        self.writer = None
    
    async def setup(self):
        """Setup all components"""
        print("🔧 Setting up A2A Demo...")
        
        # Initialize server
        config = ServerConfig(host="127.0.0.1", port=8000, debug=True)
        self.server = A2AServer(config)
        
        # Initialize client
        self.client = A2AClient(server_url="http://127.0.0.1:8000")
        
        # Connect to server
        connected = await self.client.connect()
        if not connected:
            raise ConnectionError("Failed to connect to A2A server")
        
        # Register agents
        await self.client.register("researcher_agent")
        await self.client.register("writer_agent")
        
        # Initialize agent instances
        self.researcher = ResearcherAgent(model="gpt-4o-mini")
        self.writer = WriterAgent(model="gpt-4o-mini")
        
        print("✅ Setup complete!\n")
    
    async def demo_research_only(self):
        """Demo: Research only workflow"""
        print("="*70)
        print("📚 DEMO 1: Research Agent Only")
        print("="*70)
        
        topic = "Latest developments in Large Language Models and AI"
        result = await self.researcher.research(topic)
        
        if result["status"] == "completed":
            print(f"\n🎯 Topic: {result['topic']}")
            print(f"📊 Status: {result['status']}")
            print(f"🔍 Search Results Found: {result.get('search_results_count', 0)}")
            
            print("\n" + "="*70)
            print("FINAL REPORT:")
            print("="*70)
            print(result['report'][:1500] + "...")
    
    async def demo_writing_only(self):
        """Demo: Writing only workflow"""
        print("\n" + "="*70)
        print("✍️ DEMO 2: Writer Agent Only")
        print("="*70)
        
        topic = "The Impact of Artificial Intelligence on Modern Healthcare"
        result = await self.writer.write(
            topic=topic,
            content_type="article",
            tone="professional"
        )
        
        if result["status"] == "completed":
            print(f"\n🎯 Topic: {result['topic']}")
            print(f"📄 Type: {result['content_type']} | Tone: {result['tone']}")
            
            print("\n" + "="*70)
            print("FINAL OUTPUT:")
            print("="*70)
            print(result['output'][:1500] + "...")
    
    async def demo_a2a_collaboration(self):
        """Demo: Full A2A collaboration workflow"""
        print("\n" + "="*70)
        print("🤖 DEMO 3: Agent-to-Agent Collaboration")
        print("="*70)
        
        topic = "Quantum Computing: Current State and Future Prospects"
        
        # Step 1: ResearcherAgent researches the topic
        print("\n[Step 1/4] 📚 ResearcherAgent is researching...")
        research_result = await self.researcher.research(topic)
        
        if research_result["status"] != "completed":
            print("❌ Research failed!")
            return
        
        print(f"\n✅ Research complete! Found {research_result.get('search_results_count', 0)} sources")
        
        # Step 2: Send results to WriterAgent via A2A protocol
        print("\n[Step 2/4] 🔄 Transferring data from Researcher → Writer...")
        
        # Simulate A2A message passing
        a2a_message = {
            "type": "transform",
            "content": {
                "final_report": research_result['report'],
                "topic": topic,
                "analysis": research_result.get('analysis', '')
            }
        }
        
        print(f"📤 Sending message to writer_agent...")
        await asyncio.sleep(1)  # Simulate network delay
        
        # Step 3: WriterAgent processes the research and creates article
        print("\n[Step 3/4] ✍️ WriterAgent is creating content...")
        writing_result = await self.writer.process_message(a2a_message)
        
        if writing_result["status"] != "success":
            print("❌ Writing failed!")
            return
        
        final_output = writing_result["result"]["output"]
        
        # Step 4: Display results
        print("\n[Step 4/4] 📊 Final Results:")
        print("="*70)
        print(f"🎯 Topic: {topic}")
        print(f"👥 Agents Used: ResearcherAgent → WriterAgent")
        print(f"⏱️  Total Time: ~{research_result.get('timestamp', '')} to {writing_result['result'].get('timestamp', '')}")
        
        print("\n" + "="*70)
        print("FINAL ARTICLE:")
        print("="*70)
        print(final_output[:2000])
        if len(final_output) > 2000:
            print(f"\n... (truncated, total length: {len(final_output)} chars)")
    
    async def demo_interactive(self):
        """Demo: Interactive mode with user input"""
        print("\n" + "="*70)
        print("💬 DEMO 4: Interactive A2A Chat")
        print("="*70)
        
        print("\nWelcome to the interactive A2A chat!")
        print("Type 'research <topic>' to research a topic")
        print("Type 'write <topic>' to write about a topic")
        print("Type 'full <topic>' for full research+writing workflow")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                topic = parts[1] if len(parts) > 1 else ""
                
                if not topic:
                    print("⚠️ Please provide a topic!\n")
                    continue
                
                if command == "research":
                    result = await self.researcher.research(topic)
                    if result["status"] == "completed":
                        print(f"\n✅ Research complete for: {topic}")
                        print(f"Report preview:\n{result['report'][:500]}...\n")
                
                elif command == "write":
                    result = await self.writer.write(
                        topic=topic,
                        content_type="article",
                        tone="professional"
                    )
                    if result["status"] == "completed":
                        print(f"\n✅ Article written on: {topic}")
                        print(f"Preview:\n{result['output'][:500]}...\n")
                
                elif command == "full":
                    # Full A2A workflow
                    research_result = await self.researcher.research(topic)
                    
                    if research_result["status"] == "completed":
                        a2a_message = {
                            "type": "transform",
                            "content": {"final_report": research_result['report']}
                        }
                        
                        writing_result = await self.writer.process_message(a2a_message)
                        
                        if writing_result["status"] == "success":
                            final_output = writing_result["result"]["output"]
                            print(f"\n✅ Complete article on '{topic}':")
                            print("="*70)
                            print(final_output[:1500] + "...")
                
                else:
                    print("⚠️ Unknown command. Use: research, write, full, or quit\n")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


async def run_server_only():
    """Run only the server"""
    config = ServerConfig(host="127.0.0.1", port=8000)
    server = A2AServer(config)
    
    print("🚀 Starting A2A Server...")
    print("Server URL: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop\n")
    
    await server.start()


async def run_client_demo():
    """Run client demo with pre-configured workflow"""
    # Setup
    demo = A2ADemo()
    await demo.setup()
    
    try:
        # Run full collaboration demo
        await demo.demo_a2a_collaboration()
        
    finally:
        # Cleanup
        if demo.client:
            await demo.client.disconnect()


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="A2A LangGraph Demo - Agent-to-Agent Communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full          # Full A2A collaboration demo
  python main.py --mode server        # Start only the server
  python main.py --mode client        # Run client demos
  python main.py --mode interactive   # Interactive chat mode
        """
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="full",
        choices=["full", "server", "client", "interactive"],
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🤖 A2A LangGraph Demo")
    print("="*70)
    print("Agent-to-Agent Communication with LangChain & LangGraph\n")
    
    if args.mode == "server":
        asyncio.run(run_server_only())
    
    elif args.mode == "client":
        asyncio.run(run_client_demo())
    
    elif args.mode == "interactive":
        # Setup first, then enter interactive mode
        demo = A2ADemo()
        asyncio.run(demo.setup())
        
        try:
            asyncio.run(demo.demo_interactive())
        finally:
            if demo.client:
                asyncio.run(demo.client.disconnect())
    
    else:  # full mode - run all demos sequentially
        async def run_all_demos():
            demo = A2ADemo()
            await demo.setup()
            
            try:
                await demo.demo_research_only()
                await demo.demo_writing_only()
                await demo.demo_a2a_collaboration()
                
                print("\n" + "="*70)
                print("✅ All demos completed successfully!")
                print("="*70)
                
            finally:
                if demo.client:
                    await demo.client.disconnect()
        
        asyncio.run(run_all_demos())


if __name__ == "__main__":
    main()
