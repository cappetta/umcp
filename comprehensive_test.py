#!/usr/bin/env python3
"""
Comprehensive test script for Ultimate MCP Server
Tests specific tools and REST API endpoints
"""

import asyncio
import json

import aiohttp
from fastmcp import Client


def _to_plain(obj):
    """Lightweight normaliser for tool result payloads used by tests.

    This mirrors the behaviour expected by the tests: if an object contains
    a JSON string in a `.text` field we attempt to parse it, otherwise we
    return Python-native structures where possible.
    """
    # Strings and primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # dict/list passthrough
    if isinstance(obj, dict) or isinstance(obj, list):
        return obj
    # Fallback to string representation
    try:
        return str(obj)
    except Exception:
        return None


def extract_payload(result):
    """Extract a useful payload from a tool result.

    Accepts multiple shapes:
    - legacy: a list/sequence of content blocks where first element has `.text`
    - fastmcp.CallToolResult: object with `.structured_content`, `.data` or `.content`
    - plain dict/list/string

    Returns a parsed JSON object when possible, otherwise returns the
    best-effort Python value (string/dict/list) or None.
    """
    if result is None:
        return None

    # If already a plain dict/list, return it
    if isinstance(result, (dict, list)):
        return result

    # If result exposes structured_content or data (fastmcp.CallToolResult)
    structured = getattr(result, "structured_content", None)
    if structured:
        return structured
    data = getattr(result, "data", None)
    if data is not None:
        return data

    # If result has a .content attribute (CallToolResult from mcp.types)
    content = getattr(result, "content", None)
    if content is None and isinstance(result, (list, tuple)):
        content = result

    if content is not None:
        # iterate content blocks and try to extract JSON or text
        for item in content:
            # prefer .text if available
            text = getattr(item, "text", None)
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            if isinstance(text, str) and text.strip():
                try:
                    return json.loads(text)
                except Exception:
                    return text

            # try common fields
            for fld in ("data", "value", "json"):
                val = getattr(item, fld, None)
                if val is not None:
                    return val

            # if the item itself is a dict
            if isinstance(item, dict):
                return item

    # If it's a simple object with a text attribute
    text = getattr(result, "text", None)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if isinstance(text, str) and text.strip():
        try:
            return json.loads(text)
        except Exception:
            return text

    # As a last resort, stringise and return
    try:
        return str(result)
    except Exception:
        return None


async def test_mcp_interface():
    """Test the MCP interface functionality."""
    server_url = "http://127.0.0.1:8013/mcp"
    
    print("🔧 Testing MCP Interface")
    print("=" * 40)
    
    try:
        async with Client(server_url) as client:
            print("✅ MCP client connected")
            
            # Test core tools
            tools_to_test = [
                ("echo", {"message": "Hello MCP!"}),
                ("get_provider_status", {}),
                ("list_models", {}),
            ]
            
            for tool_name, params in tools_to_test:
                try:
                    result = await client.call_tool(tool_name, params)
                    if result:
                        print(f"✅ {tool_name}: OK")
                        # Show sample of result for key tools
                        if tool_name == "get_provider_status":
                            data = extract_payload(result)
                            # extract_payload returns parsed JSON or a dict-like object when possible
                            provider_count = len((data or {}).get('providers', {}))
                            print(f"   → {provider_count} providers configured")
                        elif tool_name == "list_models":
                            data = extract_payload(result)
                            total_models = 0
                            try:
                                total_models = sum(len(models) for models in (data or {}).get('models', {}).values())
                            except Exception:
                                # fallback for alternate shapes
                                if isinstance(data, dict) and data:
                                    for v in data.values():
                                        if isinstance(v, list):
                                            total_models += len(v)
                            print(f"   → {total_models} total models available")
                    else:
                        print(f"❌ {tool_name}: No response")
                except Exception as e:
                    print(f"❌ {tool_name}: {e}")
            
            # Test filesystem tools
            print("\n📁 Testing filesystem access...")
            try:
                dirs_result = await client.call_tool("list_allowed_directories", {})
                if dirs_result:
                    print("✅ Filesystem access configured")
            except Exception as e:
                print(f"❌ Filesystem access: {e}")
            
            # Test Python execution
            print("\n🐍 Testing Python sandbox...")
            try:
                python_result = await client.call_tool("execute_python", {
                    "code": "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')"
                })
                if python_result:
                    result_data = extract_payload(python_result)
                    if isinstance(result_data, dict) and result_data.get('success'):
                        print("✅ Python sandbox working")
                        print(f"   → {str(result_data.get('output', '')).strip()}")
                    else:
                        print("❌ Python sandbox failed")
            except Exception as e:
                print(f"❌ Python sandbox: {e}")
                
    except Exception as e:
        print(f"❌ MCP interface failed: {e}")


async def test_rest_api():
    """Test the REST API endpoints."""
    base_url = "http://127.0.0.1:8013"
    
    print("\n🌐 Testing REST API Endpoints")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        # Test discovery endpoint
        try:
            async with session.get(f"{base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Discovery endpoint: {data.get('type')}")
                    print(f"   → Transport: {data.get('transport')}")
                    print(f"   → Endpoint: {data.get('endpoint')}")
                else:
                    print(f"❌ Discovery endpoint: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Discovery endpoint: {e}")
        
        # Test health endpoint
        try:
            async with session.get(f"{base_url}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health endpoint: {data.get('status')}")
                else:
                    print(f"❌ Health endpoint: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Health endpoint: {e}")
        
        # Test OpenAPI docs
        try:
            async with session.get(f"{base_url}/api/docs") as response:
                if response.status == 200:
                    print("✅ Swagger UI accessible")
                else:
                    print(f"❌ Swagger UI: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Swagger UI: {e}")
        
        # Test cognitive states endpoint
        try:
            async with session.get(f"{base_url}/api/cognitive-states") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Cognitive states: {data.get('total', 0)} states")
                else:
                    print(f"❌ Cognitive states: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Cognitive states: {e}")
        
        # Test performance overview
        try:
            async with session.get(f"{base_url}/api/performance/overview") as response:
                if response.status == 200:
                    data = await response.json()
                    overview = data.get('overview', {})
                    print(f"✅ Performance overview: {overview.get('total_actions', 0)} actions")
                else:
                    print(f"❌ Performance overview: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Performance overview: {e}")
        
        # Test artifacts endpoint
        try:
            async with session.get(f"{base_url}/api/artifacts") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Artifacts: {data.get('total', 0)} artifacts")
                else:
                    print(f"❌ Artifacts: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Artifacts: {e}")


async def test_tool_completions():
    """Test actual completions with available providers."""
    server_url = "http://127.0.0.1:8013/mcp"
    
    print("\n🤖 Testing LLM Completions")
    print("=" * 40)
    
    try:
        async with Client(server_url) as client:
            # Get available providers first
            provider_result = await client.call_tool("get_provider_status", {})
            provider_data = extract_payload(provider_result) or {}

            candidates = []
            for name, status in (provider_data.get('providers', {}) or {}).items():
                if status.get('available'):
                    # Query models for this provider
                    try:
                        lm = await client.call_tool("list_models", {"provider": name})
                        lm_data = extract_payload(lm) or {}
                        models = (lm_data.get('models') or {}).get(name, [])
                        if models:
                            candidates.append((name, models[0]))
                    except Exception:
                        continue

            if not candidates:
                print("❌ No providers available for testing")
                return

            # Test with first available provider
            provider_name, model_info = candidates[0]
            model_id = model_info.get('id')

            print(f"🧪 Testing with {provider_name} / {model_id}")

            try:
                result = await client.call_tool("generate_completion", {
                    "prompt": "Count from 1 to 5",
                    "provider": provider_name,
                    "model": model_id,
                    "max_tokens": 50,
                })

                if result:
                    response_data = extract_payload(result)
                    if not isinstance(response_data, dict):
                        response_data = {"text": str(response_data)}
                    if response_data.get('success', True):
                        print("✅ Completion successful")
                        print(f"   → Response: {response_data.get('text', '')[:100]}...")
                        if 'usage' in response_data:
                            usage = response_data['usage']
                            print(f"   → Tokens: {usage.get('total_tokens', 'N/A')}")
                    else:
                        print(f"❌ Completion failed: {response_data.get('error')}")
                else:
                    print("❌ No completion response")

            except Exception as e:
                print(f"❌ Completion error: {e}")

    except Exception as e:
        print(f"❌ Completion test failed: {e}")


async def test_memory_system():
    """Test the memory and cognitive state system."""
    server_url = "http://127.0.0.1:8013/mcp"
    
    print("\n🧠 Testing Memory System")
    print("=" * 40)
    
    try:
        async with Client(server_url) as client:
            # Ensure a workflow exists for memory/cognitive state tests
            workflow_id = None
            try:
                wf_result = await client.call_tool("create_workflow", {"title": "test_workflow_from_client"})
                wf_data = extract_payload(wf_result)
                if isinstance(wf_data, dict):
                    workflow_id = wf_data.get("workflow_id") or wf_data.get("workflowId")
            except Exception:
                # Non-fatal: some servers may not expose create_workflow; fall back to None
                workflow_id = None

            # Test memory storage
            if not workflow_id:
                print("⚠️ Skipping memory tests: no workflow_id available (create_workflow not exposed)")
            else:
                try:
                    store_params = {
                        "memory_type": "test",
                        "content": "This is a test memory for the test client",
                        "importance": 7.5,
                        "tags": ["test", "client"],
                    }
                    store_params["workflow_id"] = workflow_id

                    memory_result = await client.call_tool("store_memory", store_params)

                    if memory_result:
                        memory_data = extract_payload(memory_result)
                        if isinstance(memory_data, dict) and memory_data.get('success'):
                            memory_id = memory_data.get('memory_id')
                            print(f"✅ Memory stored: {memory_id}")

                            # Test memory retrieval
                            try:
                                get_result = await client.call_tool("get_memory_by_id", {"memory_id": memory_id})
                                if get_result:
                                    print("✅ Memory retrieved successfully")
                            except Exception as e:
                                print(f"❌ Memory retrieval: {e}")
                        else:
                            print(f"❌ Memory storage failed: {memory_data.get('error')}")
                except Exception as e:
                    print(f"❌ Memory system: {e}")
                
            # Test cognitive state
            if not workflow_id:
                print("⚠️ Skipping cognitive state tests: no workflow_id available (create_workflow not exposed)")
            else:
                try:
                    # save_cognitive_state requires: workflow_id, title, working_memory_ids
                    state_params = {
                        "workflow_id": workflow_id,
                        "title": "test_state_from_client",
                        "working_memory_ids": [],
                        "focus_area_ids": [],
                        "context_action_ids": [],
                        "current_goals": [],
                    }

                    state_result = await client.call_tool("save_cognitive_state", state_params)

                    if state_result:
                        state_data = extract_payload(state_result)
                        if isinstance(state_data, dict) and state_data.get('success'):
                            print("✅ Cognitive state saved")
                        else:
                            print(f"❌ Cognitive state failed: {state_data.get('error') if isinstance(state_data, dict) else state_data}")

                except Exception as e:
                    print(f"❌ Cognitive state: {e}")
                
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")


async def main():
    """Run all comprehensive tests."""
    print("🚀 Ultimate MCP Server Comprehensive Test Suite")
    print("=" * 60)
    
    # Test MCP interface
    await test_mcp_interface()
    
    # Test REST API
    await test_rest_api()
    
    # Test completions
    await test_tool_completions()
    
    # Test memory system
    await test_memory_system()
    
    print("\n🎯 Comprehensive testing completed!")
    print("\nIf you see mostly ✅ symbols, your server is working correctly!")
    print("Any ❌ symbols indicate areas that may need attention.")


if __name__ == "__main__":
    asyncio.run(main()) 