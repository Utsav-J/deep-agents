from typing_extensions import Annotated, List, Literal, NotRequired, TypedDict
from langchain.agents import AgentState
from deep_research_agent.prompts import WRITE_TODOS_DESCRIPTION, TODO_USAGE_INSTRUCTIONS, EMULATED_WEB_SEARCH_RESULTS
from langchain.tools import tool
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from utils import format_messages
from dotenv import load_dotenv
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


@tool(description=WRITE_TODOS_DESCRIPTION, parse_docstring=True)
def write_todos(
    todos: List[Todo],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        tool_call_id: Tool call identifier for message response

    Returns:
        Command to update agent state with new TODO list
    """
    return Command(
        update={
            "todos":todos,
            "messages":[ToolMessage(f"Updated todos to : {todos}", tool_call_id=tool_call_id)]
        }
    )

@tool(parse_docstring=True)
def read_todos(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.

    Args:
        state: Injected agent state containing the current TODO list
        tool_call_id: Injected tool call identifier for message tracking

    Returns:
        Formatted string representation of the current TODO list
    """
    todos = state.get('todos',[])
    if not todos:
        return "No Todo's currently found"
    result = "Current todo list:\n"
    for i , item in enumerate(todos):
        state_emoji = {"pending":"⏳", "ongoing":"🔄", "completed":"✅"}
        emoji = state_emoji.get(item.get("status"),"❓")
        result += f"{i}. {emoji} {item['content']} ({item['status']})\n"
    
    return result.strip()

@tool(parse_docstring=True)
def web_search_tool(query:str):
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

tools=[write_todos, read_todos, web_search_tool]
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature=0.1)
SIMPLE_RESEARCH_INSTRUCTIONS = "IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question."
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=f"{TODO_USAGE_INSTRUCTIONS}\n\n{'='*15}\n{SIMPLE_RESEARCH_INSTRUCTIONS}",
    state_schema=DeepAgentState
)

result = agent.invoke(
    {
        "todos":[],
        "messages":[
            {
                "role":"user",
                "content":"Give me a short summary of model context protocol (mcp)"
            }
        ]
    } # type: ignore
)
format_messages(messages=result['messages'])