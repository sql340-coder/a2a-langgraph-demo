#!/usr/bin/env python3
"""
Quick test script to verify the A2A demo code structure.
Run this to check if all imports work correctly before running the full demo.
"""

import sys
import importlib


def check_import(module_name, description):
    """Check if a module can be imported successfully."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        return False


def main():
    print("\n🔍 Checking A2A Demo Dependencies...\n")
    
    checks = [
        ("langchain", "LangChain"),
        ("langchain_openai", "LangChain OpenAI"),
        ("langgraph.graph", "LangGraph StateGraph"),
        ("pydantic", "Pydantic"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("httpx", "HTTPX"),
    ]
    
    results = []
    for module, desc in checks:
        results.append(check_import(module, desc))
    
    print()
    
    if all(results):
        print("🎉 All dependencies are installed! You can run the demo.")
        print("\nQuick start:")
        print("  python simple_a2a_demo.py")
        return 0
    else:
        missing = [desc for (module, desc), result in zip(checks, results) if not result]
        print(f"⚠️ Missing dependencies: {', '.join(missing)}")
        print("\nInstall them with:")
        print("  pip install langchain langchain-openai langgraph pydantic httpx fastapi uvicorn")
        return 1


if __name__ == "__main__":
    sys.exit(main())
