#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
from typing import Any, Dict

# Try both client imports to be robust
Client = None  # type: ignore
try:
    from mcp.client import Client  # fastmcp client under "mcp"
except Exception:
    try:
        from fastmcp.client import Client  # alternate import path
    except Exception as e:
        print("Could not import MCP client. Install one of:\n  pip install fastmcp", file=sys.stderr)
        raise

def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

async def main():
    parser = argparse.ArgumentParser(description="UMCP crash course")
    parser.add_argument("--url", required=True, help="MCP HTTP URL (e.g., http://127.0.0.1:8013/mcp)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    parser.add_argument("--provider", default="openai", help="Provider to demo (default: openai)")
    parser.add_argument("--model", default=None, help="Optional explicit model id for completion")
    args = parser.parse_args()

    print(f"\nConnecting to MCP server: {args.url}\n")

    # Create client and connect
    base_client = Client(args.url, timeout=args.timeout)

    # Be compatible with different client APIs (fastmcp vs mcp)
    async def _connect(cli):
        # Prefer async context manager if available
        if hasattr(cli, "__aenter__") and hasattr(cli, "__aexit__"):
            try:
                connected = await cli.__aenter__()
                return connected, "ctx"
            except TypeError:
                # Some clients have sync __aenter__; ignore and try methods
                pass
        for meth in ("connect", "open", "start", "initialize"):
            fn = getattr(cli, meth, None)
            if callable(fn):
                maybe = fn()
                # method may be async or sync
                if asyncio.iscoroutine(maybe):
                    await maybe
                return cli, "method"
        # If neither ctx nor method exists, assume no-op connection needed
        return cli, "none"

    client, _conn_kind = await _connect(base_client)

    try:
        # 1) List available tools (basic connectivity smoke test)
        # List tools across API variants
        tools = None
        if hasattr(client, "list_tools"):
            tools = await client.list_tools()
        elif hasattr(client, "tools") and hasattr(client.tools, "list"):
            tools = await client.tools.list()
        else:
            raise RuntimeError("Client does not expose a list_tools API")
        tool_names = [t.name for t in tools]
        print("Available tools:")
        print("  " + ", ".join(sorted(tool_names)))
        print()

        # Helper to call a tool by name (consistent across fastmcp versions)
        async def call_tool(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
            # Try common call patterns
            if hasattr(client, "call_tool"):
                return await client.call_tool(name, **params)
            # fastmcp: client.tools.call(name, **params)
            if hasattr(client, "tools") and hasattr(client.tools, "call"):
                return await client.tools.call(name, **params)
            # fastmcp: client.mcp.call_tool(name, **params)
            if hasattr(client, "mcp") and hasattr(client.mcp, "call_tool"):
                return await client.mcp.call_tool(name, **params)
            raise RuntimeError("Client does not support tool invocation API")

        # 2) Provider status (which providers are configured/available)
        print("== get_provider_status ==")
        status = await call_tool("get_provider_status", {})
        print(pretty(status))
        print()

        # 3) List models for a provider (default: openai)
        print(f"== list_models (provider={args.provider}) ==")
        models = await call_tool("list_models", {"provider": args.provider})
        print(pretty(models))
        print()

        # Pick a model if not provided, try a sensible default from status/models
        model_id = args.model
        if not model_id:
            # Try to pick the first model id from the response
            try:
                model_list = models.get("models", {}).get(args.provider, [])
                if model_list and isinstance(model_list, list):
                    # list_models may return list of dicts with "id" field or list of strings
                    first = model_list[0]
                    if isinstance(first, dict) and "id" in first:
                        model_id = f"{args.provider}/{first['id']}"
                    else:
                        # If the list already contains full ids with provider prefix
                        model_id = first if isinstance(first, str) else None
            except Exception:
                pass

        # Fallback if still unknown
        if not model_id:
            # Reasonable default if openai configured
            model_id = f"{args.provider}/gpt-4.1-mini"

        # 4) Estimate cost for a hypothetical prompt
        prompt = "Explain the Model Context Protocol (MCP) in three bullet points."
        print("== estimate_cost ==")
        est = await call_tool("estimate_cost", {"prompt": prompt, "model": model_id})
        print(pretty(est))
        print()

        # 5) Recommend a model for the task
        print("== recommend_model ==")
        rec = await call_tool(
            "recommend_model",
            {
                "task_type": "explain_protocol",
                "expected_input_length": len(prompt),
                "expected_output_length": 300,
                "required_capabilities": ["instruction-following"],
                "priority": "balanced",
            },
        )
        print(pretty(rec))
        print()

        # 6) Generate a completion safely
        print("== generate_completion ==")
        comp = await call_tool(
            "generate_completion",
            {
                "provider": args.provider,
                "model": model_id.split("/", 1)[1] if "/" in model_id else model_id,
                "prompt": prompt,
                "temperature": 0.6,
                "max_tokens": 300,
            },
        )
        print(pretty(comp))
        print()

        text = comp.get("text") or comp.get("completion") or ""
        print("== Completion (first 400 chars) ==")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print()

        # 7) Multi-provider comparison (if both openai/anthropic present)
        print("== multi_completion (if providers available) ==")
        providers_to_try = []
        provs = status.get("providers", {}) if isinstance(status, dict) else {}
        for prov_name, st in provs.items():
            if isinstance(st, dict) and st.get("available") is True:
                providers_to_try.append(prov_name)
        # Keep to 2 providers to save time/cost
        providers_to_try = [p for p in providers_to_try if p in ("openai", "anthropic", "gemini")][:2]

        if len(providers_to_try) >= 2:
            mc = await call_tool(
                "multi_completion",
                {
                    "prompt": "List three practical benefits of using MCP for agent-tooling.",
                    "providers": [{"provider": p} for p in providers_to_try],
                    "temperature": 0.5,
                    "max_tokens": 200,
                },
            )
            print(pretty(mc))
            print()
        else:
            print("Skipping multi_completion (need at least two available providers).")
            print()

    finally:
        # Gracefully close using any supported method
        # Try context manager exit first
        if hasattr(base_client, "__aexit__"):
            try:
                await base_client.__aexit__(None, None, None)
                print("Closed MCP connection (ctx).")
                return
            except TypeError:
                pass
        async def _close(cli):
            for meth in ("close", "disconnect", "stop", "shutdown"):
                fn = getattr(cli, meth, None)
                if callable(fn):
                    maybe = fn()
                    if asyncio.iscoroutine(maybe):
                        await maybe
                    return
        await _close(client)
        print("Closed MCP connection.")

if __name__ == "__main__":
    asyncio.run(main())