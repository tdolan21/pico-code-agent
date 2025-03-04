import subprocess
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from colorama import init, Fore, Style

from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

import asyncio
import yaml
import json
import os

from workflow import (
    AgentConfig,
    ConciergeAgent,
    ProgressEvent,
    ToolRequestEvent,
    ToolApprovedEvent,
)

from tools import FunctionToolWithContext

load_dotenv(override=True)
init(autoreset=True)

def load_config(config_path: str = "./config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Process ANSI escape codes in colors
            if "colors" in config:
                for key, value in config["colors"].items():
                    # If the value is a string that looks like an ANSI escape code
                    if isinstance(value, str) and value.startswith("\\033"):
                        # Convert the raw string representation to actual escape code
                        config["colors"][key] = value.encode().decode('unicode_escape')
            return config
    else:
        # Create default config
        default_config = {
            "colors": {
                "system_prefix": '\033[38;5;51m',
                "gold": '\033[38;5;220m',
                "message": '\033[38;5;255m',
                "reset": '\033[0m',
                "agent_prefix": '\033[38;5;99m',
                "user_text": '\033[38;5;255m',
                "arrow": '\033[38;5;99m>>>\033[0m'
            },
            "llm": {
                "model": "gpt-4o",
                "temperature": 0.4
            },
            "workspace": {
                "base_dir": "./agents/workspace/",
                "output_dir": "./agents/workspace/"
            },
            "agent_prompts": {
                "search_agent": "You can search the code repository by using search_code(query, max_results).\nThis will return snippets from code files in the codebase that match your query.",
                "editor_agent": "You are a helpful assistant that can create and update code files and execute shell commands within the codebase directory.\nYou can:\n- create_file(filename, content)\n- update_file(filename, new_content, mode)\n- edit_code_lines(filename, instructions)\n- execute_shell_command(command)\n\nBe careful that all shell commands should be safe and restricted to the codebase directory only.",
                "analysis_agent": "You analyze code search results or existing code files and suggest improvements.",
                "completion_agent": "You assess if the code meets user requirements and is complete."
            }
        }
        
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
            
        return default_config
# Load the config
config = load_config()
colors = config["colors"]
llm_config = config["llm"]
workspace_config = config["workspace"]

# Use colors from config
SYSTEM_PREFIX_COLOR = colors["system_prefix"]
GOLD = colors["gold"]
MESSAGE_COLOR = colors["message"]
RESET = colors["reset"]
AGENT_PREFIX = colors["agent_prefix"]
USER_TEXT = colors["user_text"]
ARROW = colors["arrow"]


def get_coder_state() -> Dict:
    return {
        "search_history": [],
        "current_query": None,
        "search_results": [],
        "created_files": [],
        "document_verified": False,
        "analysis_notes": {},
        "preferences": {}
    }

class CodeRepository:
    def __init__(self, base_dir: str = workspace_config["output_dir"]):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def list_files(self) -> List[str]:
        return os.listdir(self.base_dir)

    def read_file(self, filename: str) -> str:
        filepath = os.path.join(self.base_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        return ""

    def write_file(self, filename: str, content: str):
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def search_files(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        results = []
        for filename in self.list_files():
            content = self.read_file(filename)
            if query.lower() in content.lower():
                results.append({
                    "filename": filename,
                    "snippet": content[:200],
                    "full_content": content
                })
                if len(results) >= max_results:
                    break
        return results


class CodeEditor:
    """Handles fine-grained code editing operations."""

    def __init__(self, repo: CodeRepository):
        self.repo = repo

    def create_file(self, filename: str, content: str):
        self.repo.write_file(filename, content)

    def get_file_content(self, filename: str) -> str:
        return self.repo.read_file(filename)

    def update_file(
        self,
        filename: str,
        new_content: str,
        mode: str = "replace"
    ) -> str:
        old_content = self.repo.read_file(filename)
        if mode == "replace":
            updated = new_content
        elif mode == "append":
            updated = old_content + "\n" + new_content
        elif mode == "prepend":
            updated = new_content + "\n" + old_content
        else:
            updated = new_content
        self.repo.write_file(filename, updated)
        return updated

    def edit_lines(
        self,
        filename: str,
        instructions: List[Dict[str, Any]]
    ) -> str:
        content = self.repo.read_file(filename)
        lines = content.split('\n')
        for instr in instructions:
            action = instr.get("action", "")
            if action == "replace_line":
                ln = instr.get("line_number", None)
                if ln is not None and 0 < ln <= len(lines):
                    lines[ln-1] = instr.get("new_text", "")
            elif action == "insert_after":
                ln = instr.get("line_number", None)
                if ln is not None and 0 < ln <= len(lines):
                    insertion = instr.get("new_text", "")
                    if ln == len(lines):
                        lines.append(insertion)
                    else:
                        lines.insert(ln, insertion)
            elif action == "remove_line":
                ln = instr.get("line_number", None)
                if ln is not None and 0 < ln <= len(lines):
                    lines.pop(ln-1)
            elif action == "search_replace":
                pattern = instr.get("pattern", "")
                repl = instr.get("replacement", "")
                count = instr.get("count", 0)
                line_str = '\n'.join(lines)
                if count > 0:
                    line_str = line_str.replace(pattern, repl, count)
                else:
                    line_str = line_str.replace(pattern, repl)
                lines = line_str.split('\n')

        updated = '\n'.join(lines)
        self.repo.write_file(filename, updated)
        return updated

repo = CodeRepository()
editor = CodeEditor(repo)

def get_code_search_tools(repo: CodeRepository) -> List[BaseTool]:
    async def search_code(ctx: Context, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        try:
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Search Agent] Searching for '{query}'"))
            results = repo.search_files(query=query, max_results=max_results)
            user_state = await ctx.get("user_state") or {}
            user_state["search_results"] = results
            user_state["search_history"].append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "count": len(results)
            })
            await ctx.set("user_state", user_state)
            return results
        except Exception as e:
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Search Agent] Error: {str(e)}"))
            return []
    return [FunctionToolWithContext.from_defaults(async_fn=search_code, name="search_code")]


def get_code_editor_tools(editor: CodeEditor) -> List[BaseTool]:
    async def create_file(ctx: Context, filename: str, content: str) -> Dict:
        try:
            editor.create_file(filename, content)
            user_state = await ctx.get("user_state")
            if not user_state:
                user_state = {}
            user_state.setdefault("created_files", []).append({
                "filename": filename,
                "content": content,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            await ctx.set("user_state", user_state)
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Editor Agent] Created file: {filename}"))
            return {"filename": filename, "status": "created"}
        except Exception as e:
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Editor Agent] Create error: {str(e)}"))
            raise

    async def update_file(ctx: Context, filename: str, new_content: str, mode: str = "replace") -> Dict:
        try:
            updated = editor.update_file(filename, new_content, mode)
            user_state = await ctx.get("user_state")
            files = user_state.get("created_files", [])
            file_entry = next((f for f in files if f["filename"] == filename), None)
            if not file_entry:
                file_entry = {
                    "filename": filename,
                    "content": updated,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                files.append(file_entry)
            else:
                file_entry["content"] = updated
                file_entry["updated_at"] = datetime.now().isoformat()

            user_state["created_files"] = files
            await ctx.set("user_state", user_state)
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Editor Agent] Updated file: {filename}"))
            return {"filename": filename, "status": "updated"}
        except Exception as e:
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Editor Agent] Update error: {str(e)}"))
            raise

    async def edit_code_lines(ctx: Context, filename: str, instructions: List[Dict[str, Any]]) -> Dict:
        try:
            updated = editor.edit_lines(filename, instructions)
            user_state = await ctx.get("user_state")
            files = user_state.get("created_files", [])
            file_entry = next((f for f in files if f["filename"] == filename), None)
            if not file_entry:
                file_entry = {
                    "filename": filename,
                    "content": updated,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                files.append(file_entry)
            else:
                file_entry["content"] = updated
                file_entry["updated_at"] = datetime.now().isoformat()

            user_state["created_files"] = files
            await ctx.set("user_state", user_state)
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Editor Agent] Edited lines in file: {filename}"))
            return {"filename": filename, "status": "lines_edited"}
        except Exception as e:
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Editor Agent] Line edit error: {str(e)}"))
            raise

    # Shell command execution tool (no human approval needed)
    async def execute_shell_command(ctx: Context, command: str) -> Dict:
        try:
            # Run command inside the repo base_dir to prevent arbitrary system-wide execution.
            result = subprocess.run(command, shell=True, cwd=repo.base_dir, capture_output=True, text=True)
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Editor Agent] Executed shell command: {command}"))
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            ctx.write_event_to_stream(ProgressEvent(msg=f"[Code Editor Agent] Shell execution error: {str(e)}"))
            raise

    return [
        FunctionToolWithContext.from_defaults(async_fn=create_file),
        FunctionToolWithContext.from_defaults(async_fn=update_file),
        FunctionToolWithContext.from_defaults(async_fn=edit_code_lines),
        FunctionToolWithContext.from_defaults(async_fn=execute_shell_command, name="execute_shell_command")
    ]


def get_analysis_tools() -> List[BaseTool]:
    async def analyze_codebase(ctx: Context) -> Dict:
        try:
            user_state = await ctx.get("user_state")
            results = user_state.get("search_results", [])
            if not results:
                return {"error": "No results to analyze"}
            llm = await ctx.get("llm")
            analysis_prompt = """
            Analyze the following code snippets for their structure, style, and potential improvements:
            """
            for i, r in enumerate(results):
                analysis_prompt += f"\nFile {i+1}: {r.get('filename', '')}\n{r.get('full_content', '')[:1000]}"

            analysis_prompt += "\nProvide a summary of improvements that can be made."

            response = await llm.achat(
                messages=[ChatMessage(role="user", content=analysis_prompt)]
            )
            analysis = {"analysis": response.message.content, "timestamp": datetime.now().isoformat()}
            user_state["analysis_notes"] = analysis
            await ctx.set("user_state", user_state)
            return analysis
        except Exception as e:
            return {"error": str(e)}

    return [FunctionToolWithContext.from_defaults(async_fn=analyze_codebase)]


def get_completion_tools() -> List[BaseTool]:
    async def assess_completion(ctx: Context) -> Dict:
        try:
            user_state = await ctx.get("user_state", {})
            files = user_state.get("created_files", [])
            if not files:
                return {"is_complete": False, "reason": "No files created."}

            doc_content = ""
            for f in files:
                doc_content += f"\n---\nFile: {f['filename']}\n{f['content'][:1000]}...\n"

            evaluation_prompt = f"""
            Evaluate the completeness and quality of these code files.
            Consider if the requested features are implemented, code is well-structured, and meets the user requirements.
            Return JSON:
            {{
                "is_complete": boolean,
                "recommendations": [ "string" ],
                "meets_standards": boolean,
                "rationale": "string"
            }}
            Code:
            {doc_content}
            """

            llm = await ctx.get("llm")
            response = await llm.achat(
                messages=[ChatMessage(role="user", content=evaluation_prompt)]
            )

            try:
                resp_text = response.message.content.strip()
                start = resp_text.find('{')
                end = resp_text.rfind('}') + 1
                if start >= 0 and end > start:
                    evaluation = json.loads(resp_text[start:end])
                else:
                    raise json.JSONDecodeError("No JSON found", resp_text, 0)
            except Exception:
                evaluation = {"is_complete": False, "recommendations": ["Failed to parse response"], "meets_standards": False, "rationale": "Parsing error"}

            user_state["document_evaluation"] = evaluation
            await ctx.set("user_state", user_state)
            return evaluation

        except Exception as e:
            return {"is_complete": False, "error": str(e)}

    return [FunctionToolWithContext.from_defaults(async_fn=assess_completion)]


def get_code_agent_configs() -> list[AgentConfig]:
    # Get prompts from config
    agent_prompts = config.get("agent_prompts", {})
    
    return [
        AgentConfig(
            name="Code Search Agent",
            description="Searches files in the codebase by a given query",
            system_prompt=agent_prompts.get("search_agent", ""),
            tools=get_code_search_tools(repo),
        ),
        AgentConfig(
            name="Code Editor Agent",
            description="Creates and updates code files and can execute shell commands within the codebase directory",
            system_prompt=agent_prompts.get("editor_agent", ""),
            tools=get_code_editor_tools(editor),
            tools_requiring_human_confirmation=["execute_shell_command"]
        ),
        AgentConfig(
            name="Analysis Agent",
            description="Analyzes code for improvements",
            system_prompt=agent_prompts.get("analysis_agent", ""),
            tools=get_analysis_tools(),
        ),
        AgentConfig(
            name="Completion Assessor Agent",
            description="Assesses if the code meets requirements",
            system_prompt=agent_prompts.get("completion_agent", ""),
            tools=get_completion_tools(),
        ),
    ]

async def main():
    llm = OpenAI(model=llm_config["model"], temperature=llm_config["temperature"])
    memory = ChatMemoryBuffer.from_defaults(llm=llm)
    initial_state = get_coder_state()
    agent_configs = get_code_agent_configs()
    workflow = ConciergeAgent(timeout=None)

    handler = workflow.run(
        user_msg="Hello!",
        agent_configs=agent_configs,
        llm=llm,
        chat_history=[],
        initial_state=initial_state,
    )

    while True:
        async for event in handler.stream_events():
            if isinstance(event, ToolRequestEvent):
                print(f"{SYSTEM_PREFIX_COLOR}SYSTEM{ARROW}{GOLD}Tool call needed:{Style.RESET_ALL}")
                print(event.tool_name, event.tool_kwargs)
                approved = input("Approve? (y/n): ")
                if "y" in approved.lower():
                    handler.ctx.send_event(
                        ToolApprovedEvent(
                            tool_id=event.tool_id,
                            tool_name=event.tool_name,
                            tool_kwargs=event.tool_kwargs,
                            approved=True,
                        )
                    )
                else:
                    reason = input("Reason for denial: ")
                    handler.ctx.send_event(
                        ToolApprovedEvent(
                            tool_name=event.tool_name,
                            tool_id=event.tool_id,
                            tool_kwargs=event.tool_kwargs,
                            approved=False,
                            response=reason,
                        )
                    )
            elif isinstance(event, ProgressEvent):
                print(f"{SYSTEM_PREFIX_COLOR}SYSTEM{ARROW}{GOLD}{event.msg}{Style.RESET_ALL}")

        result = await handler
        print(f"{AGENT_PREFIX}[AGENT]{ARROW}{MESSAGE_COLOR}{result['response']}{Style.RESET_ALL}")

        for i, msg in enumerate(result["chat_history"]):
            if i >= len(memory.get()):
                memory.put(msg)

        user_msg = input(f"USER{ARROW}")
        if user_msg.strip().lower() in ["exit", "quit", "bye"]:
            break

        handler = workflow.run(
            ctx=handler.ctx,
            user_msg=user_msg,
            agent_configs=agent_configs,
            llm=llm,
            chat_history=memory.get(),
            initial_state=initial_state,
        )


if __name__ == "__main__":
    asyncio.run(main())
