from typing import List
from typing_extensions import Literal, TypedDict, Annotated, NotRequired
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from prompts import TASK_DESCRIPTION_PREFIX
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent,AgentState
from prompts import SUBAGENT_USAGE_INSTRUCTIONS, EMULATED_WEB_SEARCH_RESULTS
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

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
    files: Annotated[NotRequired[dict[str,str]], file_reducer] # type: ignore


class SubAgent(TypedDict):
    name:str
    description:str
    prompt:str
    tools:NotRequired[list[str]]

def _create_task_tool(tools, subagents:list[SubAgent], model, state_schema):
    """Create a task delegation tool that enables context isolation through sub-agents.

    This function implements the core pattern for spawning specialized sub-agents with
    isolated contexts, preventing context clash and confusion in complex multi-step tasks.

    Args:
        tools: List of available tools that can be assigned to sub-agents
        subagents: List of specialized sub-agent configurations
        model: The language model to use for all agents
        state_schema: The state schema (typically DeepAgentState)

    Returns:
        A 'task' tool that can delegate work to specialized sub-agents
    """
    agents = {}
    tools_dict = {}
    for _tool in tools:
        if not isinstance(_tool, BaseTool):
            _tool = tool(_tool)
        tools_dict[_tool.name] = _tool
    
    #create specialized sub agents based on configurations
    for _agent in subagents:
        if "tools" in _agent:
            _tools = [tools_dict[t] for t in _agent['tools']]
        else:
            _tools = tools #default to every single tool
        agents[_agent['name']] = create_agent(
            model=model,
            tools=_tools,
            state_schema=state_schema,
        )
    
    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]


    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string))
    def task(
        description:str,
        subagent_type:str,
        state:Annotated[DeepAgentState, InjectedState],
        tool_call_id:Annotated[str,InjectedToolCallId]
    ):
        """Delegate a task to a specialized sub-agent with isolated context.

        This creates a fresh context for the sub-agent containing only the task description,
        preventing context pollution from the parent agent's conversation history.
        """
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
        subagent = agents[subagent_type]
        state['messages'] = [{"role":"user", "content":description}] #type:ignore
        result = subagent.invoke(state)
        return Command(
            update={
                "files": result.get("files", {}),  # Merge any file changes
                "messages": [
                    # Sub-agent result becomes a ToolMessage in parent context
                    ToolMessage(
                        result["messages"][-1].content, tool_call_id=tool_call_id
                    )
                ],
            }
        )
    return task

################## CREATING THE ACTUAL DEEP/MULTI AGENT SYSTEM #######################
max_concurrent_research_units = 3
max_researcher_iterations = 3

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
        Search results from the search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """
    return EMULATED_WEB_SEARCH_RESULTS

SIMPLE_RESEARCH_INSTRUCTIONS = "You are a researcher. Research the topic provided to you. IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the provided topic."

research_sub_agent:SubAgent = {
    "name":"research-agent",
    "description":"Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "prompt":SIMPLE_RESEARCH_INSTRUCTIONS,
    "tools":['web_search']
}

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
sub_agent_tools = [web_search]
task_tool = _create_task_tool(
    sub_agent_tools, [research_sub_agent], model, DeepAgentState
)
delegation_tools = [task_tool]
agent = create_agent(
    model=model,
    tools=delegation_tools,
    system_prompt=SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
        date=datetime.now().strftime("%H:%M:%S, %Y"),
    ),
    state_schema=DeepAgentState,
)

from utils import format_messages

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Give me an overview of Model Context Protocol (MCP).",
            }
        ],
    }
)

format_messages(result["messages"])
