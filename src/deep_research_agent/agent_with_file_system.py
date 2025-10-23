import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from deep_research_agent.utils import format_messages
from prompts import LS_DESCRIPTION, READ_FILE_DESCRIPTION, WRITE_FILE_DESCRIPTION
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain_core.tools import InjectedToolCallId, tool
from langchain.agents import AgentState
from typing_extensions import List, Annotated, NotRequired, TypedDict, Literal 
from prompts import EMULATED_WEB_SEARCH_RESULTS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.
    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """
    content:str
    status: Literal["pending","ongoing","completed"]

def file_reducer(left, right):
    """Merge two file dictionaries, with right side taking precedence.

    Used as a reducer function for the files field in agent state,
    allowing incremental updates to the virtual file system.

    Args:
        left: Left side dictionary (existing files)
        right: Right side dictionary (new/updated files)

    Returns:
        Merged dictionary with right values overriding left values
    """
    if left is None:
        return right
    if right is None:
        return left
    return {**left,**right}

class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system.

    Inherits from LangGraph's AgentState and adds:
    - todos: List of Todo items for task planning and progress tracking
    - files: Virtual file system stored as dict mapping filenames to content
    """
    todos: List[Todo]
    files: Annotated[NotRequired[dict[str,str]], file_reducer]


@tool(description=LS_DESCRIPTION)
def ls_tool(state:Annotated[DeepAgentState, InjectedState])-> List[str]:
    """Lists all the files in the file system"""
    return list(state.get('files',{}).keys())

@tool(description=READ_FILE_DESCRIPTION)
def read_file_tool(
    file_path:str,
    state: Annotated[DeepAgentState, InjectedState],
    offset:int=0,
    limit:int=2000
):
    """Read the content of the file system with an optional offset and limit
    
    Args:
        file_path: Path to the file to read
        state: Agent state containing virtual filesystem (injected in tool node)
        offset: Line number to start reading from (default: 0)
        limit: Maximum number of lines to read (default: 2000)

    Returns:
        Formatted file content with line numbers, or error message if file not found
    """
    files = state.get('files',{})
    if file_path not in files:
        return f"Error file path {file_path} not found in the file system"
    content = files[file_path]
    if not content:
        return "System Information : File exists but does not have any content"
    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx+limit, len(lines))
    if start_idx > len(lines):
        return f"Line offset {offset} exceeds the current file length {len(lines)}"
    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]
        result_lines.append(f"{i+1:6d}\t{line_content}")
    
    return '\n'.join(result_lines)



@tool(description=WRITE_FILE_DESCRIPTION)
def write_file_tool(
    file_path:str,
    content:str,
    state: Annotated[DeepAgentState,InjectedState],
    tool_call_id : Annotated[str,InjectedToolCallId]
):
    """Writes content to a file in the virtual file system
    
    Args:
        file_path: Path where the file should be created/updated
        content: Content to write to the file
        state: Agent state containing virtual filesystem (injected in tool node)
        tool_call_id: Tool call identifier for message response (injected in tool node)

    Returns:
        Command to update agent state with new file content
    """
    files = state.get('files',{})
    files[file_path] = content
    return Command(
        update={
            "files":files,
            "messages":[
                ToolMessage(tool_call_id=tool_call_id, content=f"Updated file_path {file_path}")
            ]
        }
    )

FILE_USAGE_INSTRUCTIONS = """You have access to a virtual file system to help you retain and save context.                                                                                                                 
## Workflow Process                                                                                            
1. **Orient**: Use ls() to see existing files before starting work                                             
2. **Save**: Use write_file() to store the user's request so that we can keep it for later                     
3. **Read**: Once you are satisfied with the collected sources, read the saved file and use it to ensure that you directly answer the user's question."""

# Add mock research instructions
SIMPLE_RESEARCH_INSTRUCTIONS = """IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question."""

# Full prompt
INSTRUCTIONS = (
    FILE_USAGE_INSTRUCTIONS + "\n\n" + "=" * 80 + "\n\n" + SIMPLE_RESEARCH_INSTRUCTIONS
)


# Mock search tool
@tool(parse_docstring=True)
def web_search(
    query: str,
):
    """Search the web for information on a specific topic.

    This tool performs web searches and returns relevant results
    for the given query. Use this when you need to gather information from
    the internet about any topic.

    Args:
        query: The search query string. Be specific and clear about what
               information you're looking for.

    Returns:
        Search results from search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """
    return EMULATED_WEB_SEARCH_RESULTS

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0) 
tools = [ls_tool, read_file_tool, write_file_tool,web_search]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=INSTRUCTIONS,
    state_schema=DeepAgentState
)

# from IPython.display import display, Image
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Give me an overview of Model Context Protocol (MCP).",
            }
        ],
        "files": {},
    } # type: ignore
)

format_messages(result["messages"])

