import os
import re
import shlex
import subprocess
from openai import OpenAI
import json
import math
import datetime
import random
import itertools
import collections
import numpy as np
import pandas as pd
import scipy as sp
from typing import Any
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

MODEL_NAME = os.getenv("TEXT_MODEL")

def terminal(command: str) -> str:
    tokens = shlex.split(command)
    blocked = {"rm", "sudo", "dd", "chmod"}
    if tokens and tokens[0].lower() in blocked:
        msg = "Cannot execute 'rm, sudo, dd, chmod' commands since they are dangerous"
        print(msg)
        return msg

    print(f"Executing terminal command `{command}`")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            check=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return result.stdout if result.stdout else "(No output returned)"
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.stderr}"

# 1. Define the safe_import logic
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    A wrapper around __import__ that only allows specific libraries.
    """
    allowed_modules = {
        "math", "datetime", "random", "itertools", "collections", 
        "numpy", "pandas", "scipy", "json", "re"
    }
    
    # Check the root module name (e.g. 'numpy' from 'numpy.random')
    root_name = name.split('.')[0]
    
    if root_name in allowed_modules:
        return __import__(name, globals, locals, fromlist, level)
    
    raise ImportError(f"Importing '{name}' is restricted. Allowed: {list(allowed_modules)}")

# 2. Define the tool function
def python(code: str) -> str:
    """
    Execute Python code in a safe environment.
    """
    
    # Define what built-in functions the agent can use
    allowed_builtins = {
        "print": print,
        "range": range,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "sum": sum,
        "min": min,
        "max": max,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "sorted": sorted,
        "reversed": reversed,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        "all": all,
        "any": any,
        "bool": bool,
        "chr": chr,
        "ord": ord,
        "slice": slice,
        "type": type,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        # CRITICAL FIX: Allow imports via our safe wrapper
        "__import__": safe_import, 
    }

    # Define the environment variables (pre-loaded libraries)
    safe_globals = {
        "__builtins__": allowed_builtins,
        "math": math,
        "datetime": datetime,
        "random": random,
        "itertools": itertools,
        "collections": collections,
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "sp": sp,
        "scipy": sp,
    }

    local_vars = {}

    try:
        # Execute the code
        exec(code, safe_globals, local_vars)
        
        # Clean up internal variables before returning
        if "__builtins__" in local_vars:
            del local_vars["__builtins__"]
            
        return str(local_vars)
        
    except Exception as e:
        return f"Python execution error: {e}"

MAP_FN = {
    "terminal": terminal,
    "python": python,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": """
                Execute a shell command on the local machine and return the output.
                Use this whenever the user asks to:
                - list files
                - inspect folders
                - run system commands
                - check system state
                """,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command you wish to launch, e.g `ls`, `rm`, ...",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python code in a safe environment with access to built-in functions, math, datetime, random, itertools, collections, numpy, pandas, and scipy modules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to run",
                    },
                },
                "required": ["code"],
            },
        },
    },
]

DEFAULT_MAX_TOOL_DEPTH = int(os.getenv("MAX_TOOL_DEPTH", "12"))


def run_agent(messages, max_tool_depth=DEFAULT_MAX_TOOL_DEPTH):
    """
    Runs one full assistant turn including tool chaining.
    Streams tokens live.
    Prevents duplicate tool loops and infinite recursion.
    """

    last_tool_signature = None
    tool_depth = 0

    while True:
        if tool_depth > max_tool_depth:
            print("\n[Max tool depth reached — stopping]")
            return messages

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
            temperature=0.2,
            top_p=1.0,
            extra_body={
                "top_k": 3,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
        )

        assistant_message = {"role": "assistant", "content": ""}
        tool_calls_buffer = {}

        print("\nAssistant:", end=" ", flush=True)

        # --- STREAM LOOP ---
        first_token = True
        for chunk in stream:
            delta = chunk.choices[0].delta

            # Stream normal content
            if delta.content:
                if first_token:
                    print("\nAssistant:", end=" ", flush=True)
                    first_token = False
                print(delta.content, end="", flush=True)
                assistant_message["content"] += delta.content

            # Collect tool calls incrementally
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    tool_calls_buffer.setdefault(idx, {
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })

                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function.name:
                        tool_calls_buffer[idx]["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_buffer[idx]["function"]["arguments"] += tc.function.arguments

        print()

        # Convert dict -> ordered list
        tool_calls = list(tool_calls_buffer.values())

        if not tool_calls:
            fallback_calls = []
            for match in re.finditer(r"<tool_code>\s*(\{.*?\})\s*</tool_(?:call|code)>", assistant_message["content"], re.DOTALL):
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                name = payload.get("name")
                args = payload.get("arguments", {})
                if not name:
                    continue
                fallback_calls.append({
                    "id": f"fallback_{len(fallback_calls)}",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                })
            tool_calls = fallback_calls

        # If no tool calls → we're done
        if not tool_calls:
            messages.append(assistant_message)
            return messages

        # --- DUPLICATE TOOL CALL PROTECTION ---
        current_signature = json.dumps(tool_calls, sort_keys=True)

        if current_signature == last_tool_signature:
            print("\n[Duplicate tool call detected — stopping loop]")
            messages.append(assistant_message)
            return messages

        last_tool_signature = current_signature
        tool_depth += 1

        # Attach tool calls to assistant message
        assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        # Execute tools
        for tc in tool_calls:
            name = tc["function"]["name"]

            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            print(f"\n[Running tool: {name} with args {args}]")

            result = MAP_FN[name](**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": str(result),
            })

        # Loop continues → model sees tool output and responds again


def cli():
    print(f"Using model = {MODEL_NAME}")
    print("Type 'exit' or 'quit' to stop.\n")

    messages = [{
        "role": "system",
        "content": """
            You are a local CLI AI agent.

            You have access to tools and MUST use them when necessary.

            Available tools:
            - terminal: Execute shell commands and inspect files/folders
            - python: Execute Python code in a safe environment

            Important rules:
            - You do NOT have direct filesystem access.
            - To inspect files, list directories, or read file contents,
            you MUST use the `terminal` tool.
            - For code execution, use the python tool.
            - If the user refers to "this file" or "that file",
            infer the correct filename from conversation history.
            - Always use tools instead of saying you lack access.
            - Never claim you cannot access files — you can via the terminal tool.
            - After receiving tool output, explain the results clearly.
        """
    }]


    while True:
        user_input = input("> ")

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        messages = run_agent(messages)

if __name__ == "__main__":
    cli()
