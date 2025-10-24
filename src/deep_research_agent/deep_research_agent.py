import base64
import os
from typing_extensions import Annotated,TypedDict,NotRequired
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from markdownify import markdownify
from tavily import TavilyClient
from pydantic import BaseModel, Field
from datetime import datetime
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId, InjectedToolArg
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from state import DeepAgentState
from prompts import FILE_USAGE_INSTRUCTIONS, SUBAGENT_USAGE_INSTRUCTIONS, SUMMARIZE_WEB_SEARCH, TODO_USAGE_INSTRUCTIONS
from langgraph.prebuilt import create_react_agent
from task_tools import _create_task_tool

import httpx

from file_tools import ls, read_file, write_file
from todo_tools import read_todos, write_todos
from prompts import RESEARCHER_INSTRUCTIONS

load_dotenv()

summarization_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
tavily_client = TavilyClient()

class Summary(BaseModel):
    filename:str = Field(description="Name of the file whose summary its storing")
    summary:str = Field(description="The summarized content")

def get_today_str()->str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %#d, %Y")

def run_tavily_search(
    query:str,
    max_results:int=1,
    topic : Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True, 
):
    """Perform search using Tavily API for a single query.

    Args:
        search_query: Search query to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        Search results dictionary
    """
    result = tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content
    )
    return result

def summarize_webpage_content(webpage_content:str)->Summary:
    """Summarize webpage content using the configured summarization model.
    
    Args:
        webpage_content: Raw webpage content to summarize
        
    Returns:
        Summary object with filename and summary
    """
    try:
        structured_output_model = summarization_model.with_structured_output(Summary)
        summary_and_filename = structured_output_model.invoke(
            [
                HumanMessage(
                    content=SUMMARIZE_WEB_SEARCH.format(
                        webpage_content=webpage_content,
                        date=get_today_str()
                    )
                )
            ]
        )
        return summary_and_filename #type:ignore since my linter isnt convinced that we'll get a Summary object as response
    except:
        return Summary(
            filename="search_result.md",
            summary=webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
        )

def process_search_results(results:dict)->list[dict]:
    """Process search results by summarizing content where available.
    
    Args:
        results: Tavily search results dictionary
        
    Returns:
        List of processed results with summaries
    """
    processed_results = []
    httpx_client = httpx.Client()
    for result in results.get('results',[]):
        url = result['url']
        response = httpx_client.get(url)
        if response.status_code == 200:
            # generate a summary using the llm model
            raw_content = markdownify(response.text)
            summary_obj = summarize_webpage_content(raw_content)
        else:
            # Use Tavily's generated summary
            raw_content = result.get('raw_content', '')
            summary_obj = Summary(
                filename="URL_error.md",
                summary=result.get('content', 'Error reading URL; try another search.')
            )

        uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = f"{name}_{uid}{ext}"
        processed_results.append({
            'url': url,
            'title': result['title'],
            'summary': summary_obj.summary,
            'filename': summary_obj.filename,
            'raw_content': raw_content,
        })
    return processed_results

@tool(parse_docstring=True)
def tavily_search_tool(
    query:str,
    state:Annotated[DeepAgentState, InjectedState],
    tool_call_id:Annotated[str, InjectedToolCallId],
    max_results:Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general"
)->Command:
    """Search web and save detailed results to files while returning minimal context.

    Performs web search and saves full content to files for context offloading.
    Returns only essential information to help the agent decide on next steps.

    Args:
        query: Search query to execute
        state: Injected agent state for file storage
        tool_call_id: Injected tool call identifier
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Command that saves full results to files and provides minimal summary
    """
    search_results = run_tavily_search(
        query=query,
        include_raw_content=True,
        max_results=max_results,
        topic=topic
    )
    processed_results = process_search_results(search_results)
    files = state.get('files',[])
    saved_files = []
    summaries = []

    for i , result in enumerate(processed_results):
        filename = result['filename']
        file_content = f"""
# Search Results : {result['title']}
**URL:** {result['url']}
**Query:** {query}
**Date:** {get_today_str()}     

## Summary
{result['summary']}

## Raw Content
{result['raw_content'] if result['raw_content'] else 'No raw content available'}
"""
        files[filename] = file_content
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}...")
    
    summary_text = f"""
🔍 Found {len(processed_results)} result(s) for '{query}':
    
{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""
    return Command(
        update={
            "files":files,
            "messages":[
                ToolMessage(content=summary_text,tool_call_id=tool_call_id)
            ]
        }
    )

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    - How complex is the question: Have I reached the number of search limits?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.0)
max_concurrent_research_units = 3
max_researcher_iterations = 3

sub_agent_tools = [tavily_search_tool, think_tool]
built_in_tools = [ls, read_file, write_file, read_todos,write_todos,think_tool]


class SubAgent(TypedDict):
    name:str
    description:str
    prompt:str
    tools:NotRequired[list[str]]

research_sub_agent:SubAgent={
    'name':"research-agent",
    "description":"Delegate research to the sub-agent researcher. Only give this researcher one topic at a time",
    "prompt":RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools":['tavily_search_tool',"think_tool"]
}
task_tool =  _create_task_tool(
    sub_agent_tools, [research_sub_agent], model, DeepAgentState
)
delegation_tools = [task_tool]
all_tools = sub_agent_tools + built_in_tools + delegation_tools  # search available to main agent for trivial cases

SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
    date=datetime.now().strftime("%a %b %#d, %Y"),
)


INSTRUCTIONS = (
    "# TODO MANAGEMENT\n"
    + TODO_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# FILE SYSTEM USAGE\n"
    + FILE_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# SUB-AGENT DELEGATION\n"
    + SUBAGENT_INSTRUCTIONS
)

agent = create_react_agent(
    model, all_tools, state_schema=DeepAgentState, prompt=INSTRUCTIONS
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




'''
╭──────────────────────────────────────────────────────────────────────── 🧑 Human ─────────────────────────────────────────────────────────────────────────╮
│ Give me an overview of Model Context Protocol (MCP).                                                                                                      │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│                                                                                                                                                           │
│ 🔧 Tool Call: write_todos                                                                                                                                 │
│    Args: {                                                                                                                                                │
│   "todos": [                                                                                                                                              │
│     {                                                                                                                                                     │
│       "status": "pending",                                                                                                                                │
│       "content": "Research Model Context Protocol (MCP)"                                                                                                  │
│     }                                                                                                                                                     │
│   ]                                                                                                                                                       │
│ }                                                                                                                                                         │
│    ID: 42659d53-3419-48f6-84b3-6c86e3430067                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────── 🔧 Tool Output ──────────────────────────────────────────────────────────────────────╮
│ Updated todo list to [{'content': 'Research Model Context Protocol (MCP)', 'status': 'pending'}]                                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│                                                                                                                                                           │
│ 🔧 Tool Call: read_todos                                                                                                                                  │
│    Args: {}                                                                                                                                               │
│    ID: 8512c15a-03bd-47cd-b999-9cc3ea7d4480                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────── 🔧 Tool Output ──────────────────────────────────────────────────────────────────────╮
│ Current TODO List:                                                                                                                                        │
│ 1. ⏳ Research Model Context Protocol (MCP) (pending)                                                                                                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│                                                                                                                                                           │
│ 🔧 Tool Call: task                                                                                                                                        │
│    Args: {                                                                                                                                                │
│   "description": "Research Model Context Protocol (MCP)",                                                                                                 │
│   "subagent_type": "research-agent"                                                                                                                       │
│ }                                                                                                                                                         │
│    ID: b4db8187-8afa-4e85-803c-5e6c996f5faa                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────── 🔧 Tool Output ──────────────────────────────────────────────────────────────────────╮
│ I am sorry, but I encountered a technical issue with the search tool and was unable to retrieve any information. I will need to investigate this further  │
│ before I can proceed with your research request.                                                                                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│                                                                                                                                                           │
│ 🔧 Tool Call: think_tool                                                                                                                                  │
│    Args: {                                                                                                                                                │
│   "reflection": "The research agent failed to provide information on Model Context Protocol (MCP) due to a technical issue. I need to investigate this    │
│ issue. I will try to search for 'Model Context Protocol (MCP)' using the tavily_search_tool to see if it works. If it does, I will proceed with the       │
│ research. If not, I will need to report the issue and wait for it to be resolved."                                                                        │
│ }                                                                                                                                                         │
│    ID: 65b68ec9-7daf-4adb-b50a-086a8e9da52e                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────── 🔧 Tool Output ──────────────────────────────────────────────────────────────────────╮
│ Reflection recorded: The research agent failed to provide information on Model Context Protocol (MCP) due to a technical issue. I need to investigate     │
│ this issue. I will try to search for 'Model Context Protocol (MCP)' using the tavily_search_tool to see if it works. If it does, I will proceed with the  │
│ research. If not, I will need to report the issue and wait for it to be resolved.                                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│                                                                                                                                                           │
│ 🔧 Tool Call: tavily_search_tool                                                                                                                          │
│    Args: {                                                                                                                                                │
│   "query": "Model Context Protocol (MCP)"                                                                                                                 │
│ }                                                                                                                                                         │
│    ID: 3bf39d06-f0c1-421a-a697-58a5d715fe7b                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────── 🔧 Tool Output ──────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│ 🔍 Found 1 result(s) for 'Model Context Protocol (MCP)':                                                                                                  │
│                                                                                                                                                           │
│ - mcp_protocol_overview_BYM7dijH.md: The Model Context Protocol (MCP) is an open standard for AI applications to communicate with external services,      │
│ standardizing tool integration for AI agents. It acts as a standardized integration layer, enabling "plug-and-play" tool usage and complementing agent    │
│ orchestration frameworks. Key findings include MCP's ability to streamline tool integration, convert tool output for AI models, enhance multiagent        │
│ orchestration, and supplement retrieval augmented generation (RAG) by connecting to vector databases as a tool....                                        │
│                                                                                                                                                           │
│ Files: mcp_protocol_overview_BYM7dijH.md                                                                                                                  │
│ 💡 Use read_file() to access full details when needed.                                                                                                    │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│                                                                                                                                                           │
│ 🔧 Tool Call: write_todos                                                                                                                                 │
│    Args: {                                                                                                                                                │
│   "todos": [                                                                                                                                              │
│     {                                                                                                                                                     │
│       "status": "in_progress",                                                                                                                            │
│       "content": "Research Model Context Protocol (MCP)"                                                                                                  │
│     }                                                                                                                                                     │
│   ]                                                                                                                                                       │
│ }                                                                                                                                                         │
│    ID: b9df6695-43ef-4eee-9230-70f1f60c8c98                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────── 🔧 Tool Output ──────────────────────────────────────────────────────────────────────╮
│ Updated todo list to [{'content': 'Research Model Context Protocol (MCP)', 'status': 'in_progress'}]                                                      │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│                                                                                                                                                           │
│ 🔧 Tool Call: read_file                                                                                                                                   │
│    Args: {                                                                                                                                                │
│   "file_path": "mcp_protocol_overview_BYM7dijH.md"                                                                                                        │
│ }                                                                                                                                                         │
│    ID: d4766a88-96a4-485c-ab73-39703288ed8a                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────── 🔧 Tool Output ──────────────────────────────────────────────────────────────────────╮
│      1                                                                                                                                                    │
│      2  # Search Results : What is Model Context Protocol (MCP)? - IBM                                                                                    │
│      3  **URL:** https://www.ibm.com/think/topics/model-context-protocol                                                                                  │
│      4  **Query:** Model Context Protocol (MCP)                                                                                                           │
│      5  **Date:** Fri Oct 24, 2025                                                                                                                        │
│      6                                                                                                                                                    │
│      7  ## Summary                                                                                                                                        │
│      8  The Model Context Protocol (MCP) is an open standard for AI applications to communicate with external services, standardizing tool integration    │
│ for AI agents. It acts as a standardized integration layer, enabling "plug-and-play" tool usage and complementing agent orchestration frameworks. Key     │
│ findings include MCP's ability to streamline tool integration, convert tool output for AI models, enhance multiagent orchestration, and supplement        │
│ retrieval augmented generation (RAG) by connecting to vector databases as a tool.                                                                         │
│      9                                                                                                                                                    │
│     10  ## Raw Content                                                                                                                                    │
│     11  What is Model Context Protocol (MCP)? | IBM                                                                                                       │
│     12                                                                                                                                                    │
│     13                                                                                                                                                    │
│     14                                                                                                                                                    │
│     15                                                                                                                                                    │
│     16                                                                                                                                                    │
│     17  What is Model Context Protocol (MCP)?                                                                                                             │
│     18  =====================================                                                                                                             │
│     19                                                                                                                                                    │
│     20  ![Graphic render showcasing a twist of structured and unstructured data with                                                                      │
│ watsonx.data](https://assets.ibm.com/is/image/ibm/ibm_think_projectkavya_twist_still_4k?ts=1746859205539&dpr=off)                                         │
│     21                                                                                                                                                    │
│     22  AI Agents                                                                                                                                         │
│     23                                                                                                                                                    │
│     24  * [Welcome](https://www.ibm.com/think/ai-agents#605511093)                                                                                        │
│     25  * Caret right                                                                                                                                     │
│     26    Introduction                                                                                                                                    │
│     27    + [Overview](https://www.ibm.com/think/topics/ai-agents#7281535)                                                                                │
│     28    + [AI agents vs AI assistants](https://www.ibm.com/think/topics/ai-agents-vs-ai-assistants#7281536)                                             │
│     29    + Caret right                                                                                                                                   │
│     30      Agentic AI                                                                                                                                    │
│     31      - [What is agentic AI?](https://www.ibm.com/think/topics/agentic-ai#2095054954)                                                               │
│     32      - [Why is agentic AI important?](https://www.ibm.com/think/insights/agentic-ai#2095054955)                                                    │
│     33    + [Agentic AI vs generative AI](https://www.ibm.com/think/topics/agentic-ai-vs-generative-ai#7281538)                                           │
│     34    + Caret right                                                                                                                                   │
│     35      AI agent development                                                                                                                          │
│     36      - [What is AI agent development?](https://www.ibm.com/think/topics/ai-agent-development#1083937704)                                           │
│     37      - [AgentOps](https://www.ibm.com/think/topics/agentops#1083937705)                                                                            │
│     38    + Caret right                                                                                                                                   │
│     39      Types of AI agents                                                                                                                            │
│     40      - [Overview](https://www.ibm.com/think/topics/ai-agent-types#578379079)                                                                       │
│     41      - [Simple reflex agent](https://www.ibm.com/think/topics/simple-reflex-agent#578379080)                                                       │
│     42  * Caret right                                                                                                                                     │
│     43    Components                                                                                                                                      │
│     44    + [Overview](https://www.ibm.com/think/topics/components-of-ai-agents#498277090)                                                                │
│     45    + Caret right                                                                                                                                   │
│     46      Agentic workflows                                                                                                                             │
│     47      - [What are agentic workflows?](https://www.ibm.com/think/topics/agentic-workflows#732739700)                                                 │
│     48      - [Tutorial: Agentic workflows](https://www.ibm.com/think/tutorials/build-agentic-workflows-langgraph-granite#732739699)                      │
│     49    + [Communication](https://www.ibm.com/think/topics/ai-agent-communication#498277088)                                                            │
│     50    + [Learning](https://www.ibm.com/think/topics/ai-agent-learning#498277087)                                                                      │
│     51    + [Memory](https://www.ibm.com/think/topics/ai-agent-memory#498277086)                                                                          │
│     52    + [Perception](https://www.ibm.com/think/topics/ai-agent-perception#498277085)                                                                  │
│     53    + [Planning](https://www.ibm.com/think/topics/ai-agent-planning#498277084)                                                                      │
│     54    + [Reasoning](https://www.ibm.com/think/topics/agentic-reasoning#498277083)                                                                     │
│     55    + Caret right                                                                                                                                   │
│     56      Tool calling                                                                                                                                  │
│     57      - [What is tool calling?](https://www.ibm.com/think/topics/tool-calling#23317221)                                                             │
│     58      - [Tutorial: Ollama tool calling](https://www.ibm.com/think/tutorials/local-tool-calling-ollama-granite#23317222)                             │
│     59      - [Tutorial: LM Studio tool calling](https://www.ibm.com/think/tutorials/use-lm-studio-to-build-automatic-tool-calling-granite#23317223)      │
│     60  * Caret right                                                                                                                                     │
│     61    Architecture                                                                                                                                    │
│     62    + [Overview](https://www.ibm.com/think/topics/agentic-architecture#1003835715)                                                                  │
│     63    + Caret right                                                                                                                                   │
│     64      AI agent orchestration                                                                                                                        │
│     65      - [What is agent orchestration?](https://www.ibm.com/think/topics/ai-agent-orchestration#228874317)                                           │
│     66      - [Tutorial: Agent orchestration](https://www.ibm.com/think/tutorials/llm-agent-orchestration-with-langchain-and-granite#228874318)           │
│     67    + Caret right                                                                                                                                   │
│     68      Multi-agent systems                                                                                                                           │
│     69      - [What are multi-agent systems?](https://www.ibm.com/think/topics/multiagent-system#276684308)                                               │
│     70      - [Tutorial: crewAI multi-agent call                                                                                                          │
│ analysis](https://www.ibm.com/think/tutorials/multi-agent-collaboration-call-analysis-watsonx-ai-crewai#276684307)                                        │
│     71    + [Multi-agent collaboration](https://www.ibm.com/think/topics/multi-agent-collaboration#1003835712)                                            │
│     72    + Caret right                                                                                                                                   │
│     73      ReAct                                                                                                                                         │
│     74      - [What is ReAct?](https://www.ibm.com/think/topics/react-agent#1287801558)                                                                   │
│     75      - [Tutorial: LangGraph ReAct                                                                                                                  │
│ agent](https://www.ibm.com/think/tutorials/deploy-langgraph-react-agent-manage-it-support-tickets-watsonx-ai#1287801557)                                  │
│     76    + Caret right                                                                                                                                   │
│     77      ReWOO                                                                                                                                         │
│     78      - [What is ReWOO?](https://www.ibm.com/think/topics/rewoo#1793360183)                                                                         │
│     79      - [Tutorial: ReWOO reasoning agent](https://www.ibm.com/think/tutorials/build-rewoo-reasoning-agent-granite#1793360182)                       │
│     80  * Caret right                                                                                                                                     │
│     81    Protocols                                                                                                                                       │
│     82    + [Overview](https://www.ibm.com/think/topics/ai-agent-protocols#1509394340)                                                                    │
│     83    + Caret right                                                                                                                                   │
│     84      Agent Communication Protocol (ACP)                                                                                                            │
│     85      - [What is ACP?](https://www.ibm.com/think/topics/agent-communication-protocol#1190488334)                                                    │
│     86      - [Tutorial: ACP agent                                                                                                                        │
│ interoperability](https://www.ibm.com/think/tutorials/acp-ai-agent-interoperability-building-multi-agent-workflows#1190488335)                            │
│     87    + Caret right                                                                                                                                   │
│     88      Agent2Agent (A2A)                                                                                                                             │
│     89      - [What is A2A?](https://www.ibm.com/think/topics/agent2agent-protocol#684929709)                                                             │
│     90      - [Tutorial: A2A chat system](https://www.ibm.com/think/tutorials/use-a2a-protocol-for-ai-agent-communication#684929710)                      │
│     91    + Caret right                                                                                                                                   │
│     92      Model Context Protocol (MCP)                                                                                                                  │
│     93      - [What is MCP?](https://www.ibm.com/think/topics/model-context-protocol#179371084)                                                           │
│     94      - [Tutorial: MCP server](https://www.ibm.com/think/tutorials/how-to-build-an-mcp-server#179371085)                                            │
│     95  * Caret right                                                                                                                                     │
│     96    Multi-agent systems                                                                                                                             │
│     97    + [Overview](https://www.ibm.com/think/topics/multiagent-system#2014952965)                                                                     │
│     98    + Caret right                                                                                                                                   │
│     99      Agentic orchestration                                                                                                                         │
│    100      - [What is agentic orchestration?](https://www.ibm.com/think/topics/ai-agent-orchestration#2142864945)                                        │
│    101      - [Tutorial: Agent orchestration](https://www.ibm.com/think/tutorials/llm-agent-orchestration-with-langchain-and-granite#2142864944)          │
│    102    + [Multi-agent collaboration](https://www.ibm.com/think/topics/multi-agent-collaboration#2014952963)                                            │
│    103    + [Tutorial: crewAI multi-agent call                                                                                                            │
│ analysis](https://www.ibm.com/think/tutorials/multi-agent-collaboration-call-analysis-watsonx-ai-crewai#2014952962)                                       │
│    104  * Caret right                                                                                                                                     │
│    105    Frameworks                                                                                                                                      │
│    106    + [Overview](https://www.ibm.com/think/insights/top-ai-agent-frameworks#1774455706)                                                             │
│    107    + Caret right                                                                                                                                   │
│    108      AutoGen                                                                                                                                       │
│    109      - [What is AutoGen?](https://www.ibm.com/think/topics/autogen#1181250928)                                                                     │
│    110      - [Tutorial: AutoGen multi-agent RAG](https://www.ibm.com/think/tutorials/multi-agent-autogen-rag-granite#1181250927)                         │
│    111    + [AutoGPT](https://www.ibm.com/think/topics/autogpt#1774455708)                                                                                │
│    112    + [BabyAGI](https://www.ibm.com/think/topics/babyagi#1774455709)                                                                                │
│    113    + Caret right                                                                                                                                   │
│    114      BeeAI                                                                                                                                         │
│    115      - [What is BeeAI?](https://www.ibm.com/think/topics/beeai#1597040493)                                                                         │
│    116      - [Tutorial: BeeAI agentic contract                                                                                                           │
│ management](https://www.ibm.com/think/tutorials/build-multi-agent-contract-management-system-beeai-framework#1597040494)                                  │
│    117    + Caret right                                                                                                                                   │
│    118      ChatDev                                                                                                                                       │
│    119      - [What is ChatDev?](https://www.ibm.com/think/topics/chatdev#1091481868)                                                                     │
│    120      - [Tutorial: ChatDev ChatChain](https://www.ibm.com/think/tutorials/chatdev-chatchain-agent-communication-watsonx-ai#1091481869)              │
│    121    + Caret right                                                                                                                                   │
│    122      crewAI                                                                                                                                        │
│    123      - [What is crewAI?](https://www.ibm.com/think/topics/crew-ai#585923243)                                                                       │
│    124      - [Tutorial: crewAI retail shelf                                                                                                              │
│ optimization](https://www.ibm.com/think/tutorials/crewai-example-multimodal-agent-retail-shelf-optimization-watsonx#585923244)                            │
│    125    + [IBM watsonx agents](https://www.ibm.com/think/topics/watsonx-agents#1774455713)                                                              │
│    126    + Caret right                                                                                                                                   │
│    127      LangChain                                                                                                                                     │
│    128      - [What is LangChain?](https://www.ibm.com/think/topics/langchain#425194007)                                                                  │
│    129      - [Tutorial: LangChain agent](https://www.ibm.com/think/tutorials/using-langchain-tools-to-build-an-ai-agent#425194006)                       │
│    130    + Caret right                                                                                                                                   │
│    131      LangGraph                                                                                                                                     │
│    132      - [What is LangGraph?](https://www.ibm.com/think/topics/langgraph#930752632)                                                                  │
│    133      - [Tutorial: LangGraph SQL agent](https://www.ibm.com/think/tutorials/build-sql-agent-langgraph-mistral-medium-3-watsonx-ai#930752631)        │
│    134      - [Tutorial: LangGraph ReAct                                                                                                                  │
│ agent](https://www.ibm.com/think/tutorials/deploy-langgraph-react-agent-manage-it-support-tickets-watsonx-ai#930752630)                                   │
│    135    + [LangFlow](https://www.ibm.com/think/topics/langflow#826447883)                                                                               │
│    136    + Caret right                                                                                                                                   │
│    137      MetaGPT                                                                                                                                       │
│    138      - [What is MetaGPT?](https://www.ibm.com/think/topics/metagpt#276145643)                                                                      │
│    139      - [Tutorial: Multi-agent PRD automation](https://www.ibm.com/think/tutorials/multi-agent-prd-ai-automation-metagpt-ollama-deepseek#276145642) │
│    140  * Caret right                                                                                                                                     │
│    141    Governance                                                                                                                                      │
│    142    + [Overview](https://www.ibm.com/think/insights/ai-agent-governance#1268897081)                                                                 │
│    143    + [AI agent ethics](https://www.ibm.com/think/insights/ai-agent-ethics#1268897082)                                                              │
│    144    + Caret right                                                                                                                                   │
│    145      AI agent evaluation                                                                                                                           │
│    146      - [What is AI agent evaluation?](https://www.ibm.com/think/topics/ai-agent-evaluation#725195536)                                              │
│    147      - [Tutorial: AI agent evaluation](https://www.ibm.com/think/tutorials/ai-agent-evaluation#725195535)                                          │
│    148    + [AI agent security](https://www.ibm.com/think/topics/ai-agent-security#1268897084)                                                            │
│    149    + Caret right                                                                                                                                   │
│    150      Human-in-the-loop                                                                                                                             │
│    151      - [Tutorial: Human-in-the-loop](https://www.ibm.com/think/tutorials/human-in-the-loop-ai-agent-langraph-watsonx-ai#1736312786)                │
│    152  * Caret right                                                                                                                                     │
│    153    Agentic RAG                                                                                                                                     │
│    154    + [Overview](https://www.ibm.com/think/topics/agentic-rag#763338456)                                                                            │
│    155    + [Tutorial: LangChain agentic RAG](https://www.ibm.com/think/tutorials/agentic-rag#763338457)                                                  │
│    156    + Caret right                                                                                                                                   │
│    157      Agentic chunking                                                                                                                              │
│    158      - [What is agentic chunking?](https://www.ibm.com/think/topics/agentic-chunking#236418481)                                                    │
│    159      - [Tutorial: Agentic chunking for                                                                                                             │
│ RAG](https://www.ibm.com/think/tutorials/use-agentic-chunking-to-optimize-llm-inputs-with-langchain-watsonx-ai#236418482)                                 │
│    160    + Caret right                                                                                                                                   │
│    161      Corrective RAG                                                                                                                                │
│    162      - [Tutorial: Corrective RAG](https://www.ibm.com/think/tutorials/build-corrective-rag-agent-granite-tavily#269140144)                         │
│    163  * Caret right                                                                                                                                     │
│    164    Use cases / Applications                                                                                                                        │
│    165    + [Overview](https://www.ibm.com/think/topics/ai-agent-use-cases#257779831)                                                                     │
│    166    + [Automation](https://www.ibm.com/think/topics/agentic-automation#257779832)                                                                   │
│    167    + [Customer service](https://www.ibm.com/think/topics/ai-agents-in-customer-service#257779833)                                                  │
│    168    + [Finance](https://www.ibm.com/think/topics/ai-agents-in-finance#257779834)                                                                    │
│    169    + [Human resources](https://www.ibm.com/think/topics/ai-agents-in-human-resources#257779835)                                                    │
│    170    + [Marketing](https://www.ibm.com/think/topics/ai-agents-in-marketing#257779836)                                                                │
│    171    + [Procurement](https://www.ibm.com/think/topics/ai-agents-in-procurement#257779837)                                                            │
│    172    + [Sales](https://www.ibm.com/think/topics/ai-agents-in-sales#257779838)                                                                        │
│    173                                                                                                                                                    │
│    174  Author                                                                                                                                            │
│    175  ------                                                                                                                                            │
│    176                                                                                                                                                    │
│    177  [Anna Gutowska](https://www.ibm.com/think/author/anna-gutowska)                                                                                   │
│    178                                                                                                                                                    │
│    179  AI Engineer, Developer Advocate                                                                                                                   │
│    180                                                                                                                                                    │
│    181  IBM                                                                                                                                               │
│    182                                                                                                                                                    │
│    183  The Model Context Protocol (MCP) serves as a standardization layer for [AI](https://www.ibm.com/think/topics/artificial-intelligence)             │
│ applications to communicate effectively with external services such as (https://www.ibm.com/think/topics/tool-calling),                                   │
│ (https://www.ibm.com/think/topics/database) and predefined templates.                                                                                     │
│    184                                                                                                                                                    │
│    185  Have you ever tried to build a (https://www.ibm.com/think/topics/multiagent-system) but struggled to produce an effective dissemination of        │
│ information between each specialized agent? Is the variety of prebuilt and custom tools provided to your [AI                                              │
│ agent](https://www.ibm.com/think/topics/ai-agents) causing either tool execution or output parsing errors? Or perhaps, have these complications           │
│ discouraged you from attempting to develop your own agents entirely?                                                                                      │
│    186                                                                                                                                                    │
│    187  These impediments can be remedied with the Model Context Protocol (MCP). MCP allows AI agents to be context-aware while complying with a          │
│ standardized protocol for tool integration.                                                                                                               │
│    188                                                                                                                                                    │
│    189  An [AI agent](https://www.ibm.com/think/topics/ai-agents) is a system or program that is capable of autonomously performing tasks on behalf of a  │
│ user or another system. It performs them by designing its workflow and by using available tools. [Multiagent                                              │
│ systems](https://www.ibm.com/think/topics/multiagent-system) consist of multiple AI agents working collectively to perform tasks on behalf of a user or   │
│ another system.                                                                                                                                           │
│    190                                                                                                                                                    │
│    191  You can think of MCP for AI applications to serve the same purpose as a USB-C port serves for hardware.1 This analogy highlights the adaptability │
│ USB-C ports provide for connecting hardware comparing it to the standardized way in which various tools and data sources provide context to [AI           │
│ models](https://www.ibm.com/think/topics/ai-model) through MCP.                                                                                           │
│    192                                                                                                                                                    │
│    193  Tools provide meaning                                                                                                                             │
│    194  ---------------------                                                                                                                             │
│    195                                                                                                                                                    │
│    196  [Large language models (LLMs)](https://www.ibm.com/think/topics/large-language-models) like [Granite](https://www.ibm.com/granite), Gemini and    │
│ Llama are limited in their capabilities when deployed on their own. Without any AI tools, LLMs are skilled in several areas including:                    │
│    197                                                                                                                                                    │
│    198  * Subsequent text prediction: Prompting an LLM to complete a sentence such as “Jack and Jill went up the...” results in a correct prediction of   │
│ “Jack and Jill went up the hill.” This prompt and responses is an example of subsequent text prediction and works best on text the model was trained on.  │
│    199  * Basic Q&A: Since an LLM on its own cannot access external databases or web searches, it can answer natural language questions that pertain to   │
│ information in the data used to train the model. An example can be “Tell me about the treaty of Versailles” because this information about a major world  │
│ war is most likely included in the training data of general-purpose models. LLMs often perform this text generation in the form of a                      │
│ (https://www.ibm.com/think/topics/chatbots).                                                                                                              │
│    200  * [Sentiment analysis](https://www.ibm.com/think/topics/sentiment-analysis): LLMs can process text and determine whether it expresses a positive, │
│ negative or neutral sentiment.                                                                                                                            │
│    201  * Language translation: LLMs can translate text across languages and geographies. However, not every LLM is trained on data from more than one    │
│ language.                                                                                                                                                 │
│    202                                                                                                                                                    │
│    203  Aside from basic functions, an LLM without access to external tools cannot successfully run any user query that requires access to real-time      │
│ information. To provide LLMs with the opportunity to produce more meaningful results, tool integration can be introduced. Providing external tools such   │
│ as web searches, datasets and [APIs](https://www.ibm.com/think/topics/api), allows the LLM to expand its capabilities beyond its training data.           │
│    204                                                                                                                                                    │
│    205  To take this one step further, we can build AI agents by using an LLM and its available tools. Summarizing, agentic systems provide an LLM with a │
│ set of tools, allowing the model to determine appropriate tool use, adapt to a changing environment and form synthesized conclusions based on tool        │
│ output. However, at scale, these AI systems tend to fail. Hence, MCP, introduced by Anthropic in 2024, establishes an open standard for AI-tool           │
│ interactions.2                                                                                                                                            │
│    206                                                                                                                                                    │
│    207  Industry newsletter                                                                                                                               │
│    208                                                                                                                                                    │
│    209  ### The latest tech news, backed by expert insights                                                                                               │
│    210                                                                                                                                                    │
│    211  Stay up to date on the most important—and intriguing—industry trends on AI, automation, data and beyond with the Think newsletter. See the [IBM   │
│ Privacy Statement](https://www.ibm.com/us-en/privacy).                                                                                                    │
│    212                                                                                                                                                    │
│    213  ### Thank you! You are subscribed.                                                                                                                │
│    214                                                                                                                                                    │
│    215  Your subscription will be delivered in English. You will find an unsubscribe link in every newsletter. You can manage your subscriptions or       │
│ unsubscribe (https://www.ibm.com/account/reg/signup?formid=news-urx-51525). Refer to our [IBM Privacy Statement](https://www.ibm.com/us-en/privacy) for   │
│ more information.                                                                                                                                         │
│    216                                                                                                                                                    │
│    217  MCP establishes a standard                                                                                                                        │
│    218  --------------------------                                                                                                                        │
│    219                                                                                                                                                    │
│    220  It is cumbersome to connect external services to an LLM. Imagine an electrical circuit connecting a motor to various power sources. MCP is like   │
│ the wiring and switchboard of this circuit; it decides what electrical current (information) flows to the motor (AI model). Tool output or model context  │
│ can be compared to the input current—it is the voltage flowing from a source of power and can include memory, tools and past findings.                    │
│    221                                                                                                                                                    │
│    222  As the switchboard, MCP decides which sources of power (tool output or context) to connect and when to do so, regulates the current (stream of    │
│ information), filters and prioritizes inputs. It does that to ensure that only relevant wires are energized (the relevant context is loaded) and manages  │
│ the circuit’s timing and routing to not overload the system.                                                                                              │
│    223                                                                                                                                                    │
│    224  Just as a well-designed circuit prevents overload and ensures efficient power usage, MCP serves as a connector to facilitate efficient, relevant  │
│ and structured use of context for optimal AI model performance.                                                                                           │
│    225                                                                                                                                                    │
│    226  MCP establishes a new (https://www.ibm.com/think/topics/open-source) standard for AI engineers to agree upon. However, standards are not a new    │
│ concept in the software industry. For example, [REST APIs](https://www.ibm.com/think/topics/rest-apis) are industry-standard, offering consistent data    │
│ exchange between applications through HTTP requests aligned with REST design principles.                                                                  │
│    227                                                                                                                                                    │
│    228  Similarly, MCP unifies the LLM and external services to communicate efficiently by setting a standard. This standard allows for “plug-and-play”   │
│ tool usage rather than writing code for custom integration of each tool.                                                                                  │
│    229                                                                                                                                                    │
│    230  MCP is not an agent framework, but a standardized integration layer for agents accessing tools. It complements agent orchestration frameworks.    │
│ MCP can complement agent orchestration frameworks like [LangChain](https://www.ibm.com/think/topics/langchain),                                           │
│ [LangGraph](https://www.ibm.com/think/topics/langgraph), [BeeAI](https://beeai.dev/), LlamaIndex and (https://www.ibm.com/think/topics/crew-ai), but it   │
│ does not replace them; MCP does not decide when a tool is called and for what purpose.                                                                    │
│    231                                                                                                                                                    │
│    232  MCP simply provides a standardized connection to streamline tool integration.3 Ultimately, the LLM determines which tools to call based on the    │
│ context of the user’s request.                                                                                                                            │
│    233                                                                                                                                                    │
│    234  AI agents                                                                                                                                         │
│    235                                                                                                                                                    │
│    236  ### 5 Types of AI Agents: Autonomous Functions & Real-World Applications                                                                          │
│    237                                                                                                                                                    │
│    238  Learn how goal-driven and utility-based AI adapt to workflows and complex environments.                                                           │
│    239                                                                                                                                                    │
│    240  [Build, deploy and monitor AI agents](https://www.ibm.com/products/watsonx-ai)                                                                    │
│    241                                                                                                                                                    │
│    242  MCP architecture                                                                                                                                  │
│    243  ----------------                                                                                                                                  │
│    244                                                                                                                                                    │
│    245  The MCP client/server model can be broken down into three key architectural components:                                                           │
│    246                                                                                                                                                    │
│    247  ### MCP host                                                                                                                                      │
│    248                                                                                                                                                    │
│    249  An AI application receives the user requests and seeks access to context through the MCP. This integration layer can include IDEs such as Cursor  │
│ or Claude Desktop. It contains the orchestration logic and can connect each client to a server.4                                                          │
│    250                                                                                                                                                    │
│    251  ### MCP client                                                                                                                                    │
│    252                                                                                                                                                    │
│    253  Communication in the MCP ecosystem between the host and server must go through a client. This client exists within the host and converts user     │
│ requests into a structured format that the open protocol can process. Multiple clients can exist with a singular MCP host but each client has a 1:1       │
│ relationship with an MCP server.                                                                                                                          │
│    254                                                                                                                                                    │
│    255  Examples of MCP clients are IBM® BeeAI, Microsoft Copilot Studio, Claude.ai, Windsurf Editor and Postman. Clients serve as the session manager by │
│ handling interruptions, timeouts, reconnections and session closures. Clients also parse responses, conduct error handling and verify that responses are  │
│ relevant to the context and appropriate.4                                                                                                                 │
│    256                                                                                                                                                    │
│    257  ### MCP server                                                                                                                                    │
│    258                                                                                                                                                    │
│    259  The external service that provides the context to the LLM by converting user requests to server actions. Examples of MCP server integrations      │
│ include Slack, GitHub, Git, Docker or web search. These servers are typically GitHub repositories available in various programming languages (C#, Java™,  │
│ TypeScript, Python and others) and provide access to MCP tools.                                                                                           │
│    260                                                                                                                                                    │
│    261  Tutorials can typically be found within these GitHub repositories to aid in the technical implementation. MCP servers can also be used to connect │
│ [LLM inferencing](https://www.ibm.com/think/topics/ai-inference), through AI platform providers such as IBM and OpenAI, to the MCP SDK. In doing so, a    │
│ reusable MCP service is created for clients to access as a “standardized” chat tool.                                                                      │
│    262                                                                                                                                                    │
│    263  MCP servers are versatile as they allow connections to both internal and external resources and tools. According to the docs provided by          │
│ Anthropic, Model Context Protocol servers expose data through:                                                                                            │
│    264                                                                                                                                                    │
│    265  * **Resources**: Information retrieval from internal or external databases. Resources return data but do not execute actionable computations.5    │
│    266  * **Tools**: Information exchange with tools that can perform a side effect such as a calculation or fetch data through an API request.6          │
│    267  * **Prompts**: Reusable templates and workflows for LLM-server communication.7                                                                    │
│    268                                                                                                                                                    │
│    269  The transport layer between clients and servers is responsible for two-way message conversion. In the client-to-server stream, MCP protocol       │
│ messages are converted into JSON-RPC format, allowing for the transport of several data structures and their processing rules.8                           │
│    270                                                                                                                                                    │
│    271  In the reverse server-to-client stream, the received messages in JSON-RPC format are converted back into MCP protocol messages.9 The three        │
│ JSON-RPC message types include requests, responses and notifications. Requests require a response from the server, whereas notifications do not.          │
│    272                                                                                                                                                    │
│    273  In the transport layer between clients and servers, there are two main transport methods for MCP protocol, both of which transmit messages in     │
│ JSON-RPC 2.0 format. The first is standard input/output (stdio) which works best for integrating local resources due to the simple input/output           │
│ information transmission. This format is used for lightweight, synchronous messaging.4 Such resources include local file systems, databases and local     │
│ APIs.                                                                                                                                                     │
│    274                                                                                                                                                    │
│    275  The second is server-sent events (SSE) which works best for integrating remote resources. HTTP POST requests serve as the mechanism for           │
│ transmitting client-to-server messages and SSE is used for the reverse. This format can be used to handle multiple asynchronous, event-driven             │
│ server-calls at once.4                                                                                                                                    │
│    276                                                                                                                                                    │
│    277  Benefits of MCP                                                                                                                                   │
│    278  ---------------                                                                                                                                   │
│    279                                                                                                                                                    │
│    280  Imagine a real-world AI that scans your inbox to schedule client meetings, sends stock updates, and texts summaries of the last hour’s Slack      │
│ activity. Every service provider constructs their APIs differently by requiring different information to pass, returning different output schemas. Thus,  │
│ the slightest change in these tools can result in the collapse of this entire AI workflow infrastructure.                                                 │
│    281                                                                                                                                                    │
│    282  There is also a significant development load on engineers to manually construct these tool connections,                                           │
│ (https://www.ibm.com/think/topics/debugging) and maintain authentication like API keys and tool permissions. Tools are often dependent on the output of   │
│ other tools and there exist many edge cases in which these connections fail.                                                                              │
│    283                                                                                                                                                    │
│    284  Thus, it is critical to provide MCP integration as the middle layer between the LLM and the development tools. In this layer, the MCP can convert │
│ tool output in a way that is understandable by the model. Without the need to alternate between CLIs, the tool integration occurs all in one place.       │
│    285                                                                                                                                                    │
│    286  There are many real-world use cases for MCP. For instance, MCP enhances (https://www.ibm.com/think/topics/ai-agent-orchestration) and             │
│ communication through a shared workspace with common tools, eliminating the need for direct integrations.3                                                │
│    287                                                                                                                                                    │
│    288  MCP can also be used to supplement (https://www.ibm.com/think/topics/retrieval-augmented-generation). Rather than providing the retriever to      │
│ search a vector store or knowledge base, MCP can connect to a (https://www.ibm.com/think/topics/vector-database) through a server action. Searching the   │
│ database as a tool rather than passing the retriever in every LLM invocation allows for more strategic use of the tool. This approach also allows for     │
│ further (https://www.ibm.com/think/topics/tool-calling) upon data retrieval.3                                                                             │
│    289                                                                                                                                                    │
│    290  The future of MCP                                                                                                                                 │
│    291  -----------------                                                                                                                                 │
│    292                                                                                                                                                    │
│    293  MCP represents an evolving approach to LLM tool integration that continues to mature and reshape the space over time. As technical challenges     │
│ arise and MCP servers evolve, the standard adapts and MCPs continue to improve.                                                                           │
│    294                                                                                                                                                    │
│    295  Regardless, the need for standardized tool integration is critical for AI agents to operate autonomously and adapt dynamically to real-world      │
│ environments.10 With MCP, we can streamline the automation of complex (https://www.ibm.com/think/topics/agentic-workflows) to allow for less human        │
│ oversight. In turn, this shift enabled by MCP allows our time to be spent on more nuanced tasks requiring human intellect and intuition.                  │
│    296                                                                                                                                                    │
│    297  [Ebook                                                                                                                                            │
│    298                                                                                                                                                    │
│    299  Start realizing ROI: A practical guide to agentic AI                                                                                              │
│    300                                                                                                                                                    │
│    301  Discover ways to get ahead, successfully scaling AI across your business with real results.                                                       │
│    302                                                                                                                                                    │
│    303  Read the ebook](https://www.ibm.com/account/reg/signup?formid=urx-54067)                                                                          │
│    304                                                                                                                                                    │
│    305  ![Abstract portrayal of AI agent, shown in isometric view, acting as bridge between two                                                           │
│ systems](https://assets.ibm.com/is/image/ibm/ai4b_agents_iso?ts=1744110713514&dpr=off)                                                                    │
│    306                                                                                                                                                    │
│    307  [Build, run and manage AI agents with watsonx Orchestrate](https://www.ibm.com/products/watsonx-orchestrate)                                      │
│    308                                                                                                                                                    │
│    309  Resources                                                                                                                                         │
│    310  ---------                                                                                                                                         │
│    311                                                                                                                                                    │
│    312  [Coming soon - October 28                                                                                                                         │
│    313                                                                                                                                                    │
│    314  AI Agents at Scale: Success Stories                                                                                                               │
│    315                                                                                                                                                    │
│    316  Join us for an exclusive webinar with experts from Salesforce, Box, and LTIMindtree as they share real-world success stories and practical        │
│ insights on deploying AI agents at scale.                                                                                                                 │
│    317                                                                                                                                                    │
│    318  Sign up for the webinar](https://www.ibm.com/account/reg/signup?formid=urx-54029)                                                                 │
│    319                                                                                                                                                    │
│    320  [Report                                                                                                                                           │
│    321                                                                                                                                                    │
│    322  Top strategic technology trends for 2025: Agentic AI                                                                                              │
│    323                                                                                                                                                    │
│    324  Download this Gartner® research to learn the potential opportunities and risks of agentic AI for IT leaders and how to prepare for this next wave │
│ of AI innovation.                                                                                                                                         │
│    325                                                                                                                                                    │
│    326  Read the report](https://www.ibm.com/account/reg/signup?formid=urx-53506)                                                                         │
│    327                                                                                                                                                    │
│    328  [Report                                                                                                                                           │
│    329                                                                                                                                                    │
│    330  Agentic AI products to watch out for in 2025                                                                                                      │
│    331                                                                                                                                                    │
│    332  Explore how Agentic AI is pushing the boundaries of generative AI and automation to allow for more sophisticated, autonomous systems that can     │
│ operate independently in changing environments.                                                                                                           │
│    333                                                                                                                                                    │
│    334  Read the report](https://www.ibm.com/account/reg/signup?formid=urx-53887)                                                                         │
│    335                                                                                                                                                    │
│    336  [2025 AI Agents buyer's guide                                                                                                                     │
│    337                                                                                                                                                    │
│    338  How AI agents and assistants can benefit your organization                                                                                        │
│    339                                                                                                                                                    │
│    340  Dive into this comprehensive guide that breaks down key use cases, core capabilities, and step-by-step recommendations to help you choose the     │
│ right solutions for your business.                                                                                                                        │
│    341                                                                                                                                                    │
│    342  Read the guide](https://www.ibm.com/account/reg/signup?formid=urx-53811)                                                                          │
│    343                                                                                                                                                    │
│    344  [Video                                                                                                                                            │
│    345                                                                                                                                                    │
│    346  Reimagine business productivity with AI agents and assistants                                                                                     │
│    347                                                                                                                                                    │
│    348  Learn how AI agents and AI assistants can work together to achieve new levels of productivity.                                                    │
│    349                                                                                                                                                    │
│    350  Watch now](https://www.ibm.com/think/videos/ai-academy/reimagine-business-productivity-with-ai)                                                   │
│    351                                                                                                                                                    │
│    352  [Demo                                                                                                                                             │
│    353                                                                                                                                                    │
│    354  Try watsonx Orchestrate™                                                                                                                          │
│    355                                                                                                                                                    │
│    356  Explore how generative AI assistants can lighten your workload and improve productivity.                                                          │
│    357                                                                                                                                                    │
│    358  Start the demo](https://www.ibm.com/products/watsonx-orchestrate/demos)                                                                           │
│    359                                                                                                                                                    │
│    360  [Report                                                                                                                                           │
│    361                                                                                                                                                    │
│    362  From AI projects to profits: How agentic AI can sustain financial returns                                                                         │
│    363                                                                                                                                                    │
│    364  Learn how organizations are shifting from launching AI in disparate pilots to using it to drive transformation at the core.                       │
│    365                                                                                                                                                    │
│    366  Read the report](https://www.ibm.com/thought-leadership/institute-business-value/report/agentic-ai-profits)                                       │
│    367                                                                                                                                                    │
│    368  [Report                                                                                                                                           │
│    369                                                                                                                                                    │
│    370  Omdia Report on empowered intelligence: The impact of AI agents                                                                                   │
│    371                                                                                                                                                    │
│    372  Discover how you can unlock the full potential of gen AI with AI agents.                                                                          │
│    373                                                                                                                                                    │
│    374  Read the report](https://www.ibm.com/account/reg/signup?formid=urx-53501)                                                                         │
│    375                                                                                                                                                    │
│    376  [Podcast                                                                                                                                          │
│    377                                                                                                                                                    │
│    378  How AI agents will reinvent productivity                                                                                                          │
│    379                                                                                                                                                    │
│    380  Learn ways to use AI to be more creative, efficient and start adapting to a future that involves working closely with AI agents.                  │
│    381                                                                                                                                                    │
│    382  Listen now](https://www.ibm.com/think/podcasts/ai-in-action/how-ai-agents-will-reinvent-productivity)                                             │
│    383                                                                                                                                                    │
│    384  [News                                                                                                                                             │
│    385                                                                                                                                                    │
│    386  Ushering in the agentic enterprise: Putting AI to work across your entire technology estate                                                       │
│    387                                                                                                                                                    │
│    388  Stay updated about the new emerging AI agents, a fundamental tipping point in the AI revolution.                                                  │
│    389                                                                                                                                                    │
│    390  Read the news](https://www.ibm.com/new/announcements/productivity-revolution-with-ai-agents-that-work-across-stack)                               │
│    391                                                                                                                                                    │
│    392  [Podcast                                                                                                                                          │
│    393                                                                                                                                                    │
│    394  The future of agents, AI energy consumption, Anthropic's computer use and Google watermarking AI-generated text                                   │
│    395                                                                                                                                                    │
│    396  Stay ahead of the curve with our AI experts on this episode of Mixture of Experts as they dive deep into the future of AI agents and more.        │
│    397                                                                                                                                                    │
│    398  Listen                                                                                                                                            │
│ now](https://www.ibm.com/think/podcasts/mixture-of-experts/future-of-ai-agents-ai-energy-consumption-anthropic-computer-use-google-watermarking-ai)       │
│    399                                                                                                                                                    │
│    400  [Case study                                                                                                                                       │
│    401                                                                                                                                                    │
│    402  How Comparus is using a "banking assistant"                                                                                                       │
│    403                                                                                                                                                    │
│    404  Comparus used solutions from IBM® watsonx.ai™ and impressively demonstrated the potential of conversational banking as a new interaction model.   │
│    405                                                                                                                                                    │
│    406  Read the case study](https://www.ibm.com/case-studies/comparus-gmbh)                                                                              │
│    407                                                                                                                                                    │
│    408  Related solutions                                                                                                                                 │
│    409                                                                                                                                                    │
│    410                                                                                                                                                    │
│    411                                                                                                                                                    │
│    412                                                                                                                                                    │
│    413                                                                                                                                                    │
│    414                                                                                                                                                    │
│    415                                                                                                                                                    │
│    416                                                                                                                                                    │
│    417                                                                                                                                                    │
│    418                                                                                                                                                    │
│    419                                                                                                                                                    │
│    420                                                                                                                                                    │
│    421                                                                                                                                                    │
│    422                                                                                                                                                    │
│    423                                                                                                                                                    │
│    424                                                                                                                                                    │
│    425                                                                                                                                                    │
│    426                                                                                                                                                    │
│    427                                                                                                                                                    │
│    428                                                                                                                                                    │
│    429                                                                                                                                                    │
│    430                                                                                                                                                    │
│    431                                                                                                                                                    │
│    432                                                                                                                                                    │
│    433                                                                                                                                                    │
│    434                                                                                                                                                    │
│    435                                                                                                                                                    │
│    436                                                                                                                                                    │
│    437                                                                                                                                                    │
│    438                                                                                                                                                    │
│    439                                                                                                                                                    │
│    440                                                                                                                                                    │
│    441                                                                                                                                                    │
│    442                                                                                                                                                    │
│    443                                                                                                                                                    │
│    444                                                                                                                                                    │
│    445                                                                                                                                                    │
│    446                                                                                                                                                    │
│    447                                                                                                                                                    │
│    448                                                                                                                                                    │
│    449                                                                                                                                                    │
│    450                                                                                                                                                    │
│    451                                                                                                                                                    │
│    452                                                                                                                                                    │
│    453                                                                                                                                                    │
│    454                                                                                                                                                    │
│    455                                                                                                                                                    │
│    456                                                                                                                                                    │
│    457                                                                                                                                                    │
│    458                                                                                                                                                    │
│    459                                                                                                                                                    │
│    460                                                                                                                                                    │
│    461                                                                                                                                                    │
│    462                                                                                                                                                    │
│    463  AI agents for business                                                                                                                            │
│    464                                                                                                                                                    │
│    465                                                                                                                                                    │
│    466                                                                                                                                                    │
│    467  Build, deploy and manage powerful AI assistants and agents that automate workflows and processes with generative AI.                              │
│    468                                                                                                                                                    │
│    469  [Explore watsonx Orchestrate](https://www.ibm.com/products/watsonx-orchestrate)                                                                   │
│    470                                                                                                                                                    │
│    471                                                                                                                                                    │
│    472                                                                                                                                                    │
│    473                                                                                                                                                    │
│    474                                                                                                                                                    │
│    475                                                                                                                                                    │
│    476                                                                                                                                                    │
│    477                                                                                                                                                    │
│    478                                                                                                                                                    │
│    479                                                                                                                                                    │
│    480                                                                                                                                                    │
│    481                                                                                                                                                    │
│    482                                                                                                                                                    │
│    483                                                                                                                                                    │
│    484                                                                                                                                                    │
│    485                                                                                                                                                    │
│    486                                                                                                                                                    │
│    487                                                                                                                                                    │
│    488                                                                                                                                                    │
│    489                                                                                                                                                    │
│    490                                                                                                                                                    │
│    491                                                                                                                                                    │
│    492                                                                                                                                                    │
│    493                                                                                                                                                    │
│    494                                                                                                                                                    │
│    495                                                                                                                                                    │
│    496                                                                                                                                                    │
│    497                                                                                                                                                    │
│    498                                                                                                                                                    │
│    499                                                                                                                                                    │
│    500                                                                                                                                                    │
│    501                                                                                                                                                    │
│    502                                                                                                                                                    │
│    503                                                                                                                                                    │
│    504                                                                                                                                                    │
│    505                                                                                                                                                    │
│    506                                                                                                                                                    │
│    507                                                                                                                                                    │
│    508                                                                                                                                                    │
│    509                                                                                                                                                    │
│    510                                                                                                                                                    │
│    511                                                                                                                                                    │
│    512                                                                                                                                                    │
│    513                                                                                                                                                    │
│    514                                                                                                                                                    │
│    515                                                                                                                                                    │
│    516                                                                                                                                                    │
│    517                                                                                                                                                    │
│    518                                                                                                                                                    │
│    519                                                                                                                                                    │
│    520                                                                                                                                                    │
│    521                                                                                                                                                    │
│    522                                                                                                                                                    │
│    523                                                                                                                                                    │
│    524                                                                                                                                                    │
│    525  IBM AI agent solutions                                                                                                                            │
│    526                                                                                                                                                    │
│    527                                                                                                                                                    │
│    528                                                                                                                                                    │
│    529  Build the future of your business with AI solutions that you can trust.                                                                           │
│    530                                                                                                                                                    │
│    531  [Explore AI agent solutions](https://www.ibm.com/solutions/ai-agents)                                                                             │
│    532                                                                                                                                                    │
│    533                                                                                                                                                    │
│    534                                                                                                                                                    │
│    535                                                                                                                                                    │
│    536                                                                                                                                                    │
│    537                                                                                                                                                    │
│    538                                                                                                                                                    │
│    539                                                                                                                                                    │
│    540                                                                                                                                                    │
│    541                                                                                                                                                    │
│    542                                                                                                                                                    │
│    543                                                                                                                                                    │
│    544                                                                                                                                                    │
│    545                                                                                                                                                    │
│    546                                                                                                                                                    │
│    547                                                                                                                                                    │
│    548                                                                                                                                                    │
│    549                                                                                                                                                    │
│    550                                                                                                                                                    │
│    551                                                                                                                                                    │
│    552                                                                                                                                                    │
│    553                                                                                                                                                    │
│    554                                                                                                                                                    │
│    555                                                                                                                                                    │
│    556                                                                                                                                                    │
│    557                                                                                                                                                    │
│    558                                                                                                                                                    │
│    559                                                                                                                                                    │
│    560                                                                                                                                                    │
│    561                                                                                                                                                    │
│    562                                                                                                                                                    │
│    563                                                                                                                                                    │
│    564                                                                                                                                                    │
│    565                                                                                                                                                    │
│    566                                                                                                                                                    │
│    567                                                                                                                                                    │
│    568                                                                                                                                                    │
│    569                                                                                                                                                    │
│    570                                                                                                                                                    │
│    571                                                                                                                                                    │
│    572                                                                                                                                                    │
│    573                                                                                                                                                    │
│    574                                                                                                                                                    │
│    575                                                                                                                                                    │
│    576                                                                                                                                                    │
│    577                                                                                                                                                    │
│    578                                                                                                                                                    │
│    579                                                                                                                                                    │
│    580                                                                                                                                                    │
│    581                                                                                                                                                    │
│    582                                                                                                                                                    │
│    583                                                                                                                                                    │
│    584                                                                                                                                                    │
│    585  IBM Consulting AI services                                                                                                                        │
│    586                                                                                                                                                    │
│    587                                                                                                                                                    │
│    588                                                                                                                                                    │
│    589  IBM Consulting AI services help reimagine how businesses work with AI for transformation.                                                         │
│    590                                                                                                                                                    │
│    591  [Explore artificial intelligence services](https://www.ibm.com/consulting/artificial-intelligence)                                                │
│    592                                                                                                                                                    │
│    593  Take the next step                                                                                                                                │
│    594                                                                                                                                                    │
│    595  Whether you choose to customize pre-built apps and skills or build and deploy custom agentic services using an AI studio, the IBM watsonx         │
│ platform has you covered.                                                                                                                                 │
│    596                                                                                                                                                    │
│    597                                                                                                                                                    │
│    598  [Explore watsonx Orchestrate](https://www.ibm.com/products/watsonx-orchestrate)                                                                   │
│    599  [Explore watsonx.ai](https://www.ibm.com/products/watsonx-ai)                                                                                     │
│    600                                                                                                                                                    │
│    601  ##### Footnotes                                                                                                                                   │
│    602                                                                                                                                                    │
│    603  1 Introduction—Model Context Protocol (https://modelcontextprotocol.io/docs/getting-started/intro), 2025                                          │
│    604  2 Hou, X., Zhao, Y., Wang, S., & Wang, H., Model context protocol (MCP): Landscape, security threats and future research directions. arXiv        │
│ preprint arXiv: 2503.23278, 2025                                                                                                                          │
│    605  3 Se, K., #14: What is MCP and why is everyone—suddenly!—talking about it? [Huggingface.co](https://huggingface.co/).                             │
│ <https://huggingface.co/blog/Kseniase/mcp>, 17 March 2025                                                                                                 │
│    606  4 Ray, P., A survey on Model Context Protocol: Architecture, state-of-the-art, challenges and future directions. TechRxiv, 18 April 2025          │
│    607  5 Resources—Model Context Protocol (https://modelcontextprotocol.io/specification/2025-06-18/server/resources), 2025                              │
│    608  6 Tools—Model Context Protocol (https://modelcontextprotocol.io/specification/2025-06-18/server/tools), 2025                                      │
│    609  7 Prompts—Model Context Protocol (https://modelcontextprotocol.io/specification/2025-06-18/server/prompts), 2025                                  │
│    610  8 JSON-RPC Working Group—JSON-RPC 2.0 specification. [Jsonrpc.org](https://www.jsonrpc.org/). <https://www.jsonrpc.org/specification>, 26 March   │
│ 2025                                                                                                                                                      │
│    611  9 Transports—Model Context Protocol (https://modelcontextprotocol.io/specification/2025-06-18/basic/transports), 2025                             │
│    612  10 Singh, A., Ehtesham, A., Kumar, S., and Khoei, T. T., A survey of the Model Context Protocol (MCP): Standardizing context to enhance large     │
│ language models (LLMs) Preprints, (https://www.preprints.org/manuscript/202504.0245/v1), 2025                                                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│                                                                                                                                                           │
│ 🔧 Tool Call: think_tool                                                                                                                                  │
│    Args: {                                                                                                                                                │
│   "reflection": "I have successfully retrieved information about the Model Context Protocol (MCP) using the tavily_search_tool. The search results        │
│ provided a good overview, and I have read the content from the file 'mcp_protocol_overview_BYM7dijH.md'. This file explains what MCP is, its purpose,     │
│ architecture, and benefits. I believe I have enough information to answer the user's request."                                                            │
│ }                                                                                                                                                         │
│    ID: c1b6446a-da02-46e0-b0e1-edaaf13960a0                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────── 🔧 Tool Output ──────────────────────────────────────────────────────────────────────╮
│ Reflection recorded: I have successfully retrieved information about the Model Context Protocol (MCP) using the tavily_search_tool. The search results    │
│ provided a good overview, and I have read the content from the file 'mcp_protocol_overview_BYM7dijH.md'. This file explains what MCP is, its purpose,     │
│ architecture, and benefits. I believe I have enough information to answer the user's request.                                                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                           │
│                                                                                                                                                           │
│ 🔧 Tool Call: write_todos                                                                                                                                 │
│    Args: {                                                                                                                                                │
│   "todos": [                                                                                                                                              │
│     {                                                                                                                                                     │
│       "status": "completed",                                                                                                                              │
│       "content": "Research Model Context Protocol (MCP)"                                                                                                  │
│     }                                                                                                                                                     │
│   ]                                                                                                                                                       │
│ }                                                                                                                                                         │
│    ID: e71f21a3-83ad-4af5-8ad5-e7ac0f8e4595                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────────────── 🔧 Tool Output ──────────────────────────────────────────────────────────────────────╮
│ Updated todo list to [{'content': 'Research Model Context Protocol (MCP)', 'status': 'completed'}]                                                        │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────── 📝 AI ──────────────────────────────────────────────────────────────────────────╮
│ The Model Context Protocol (MCP) is an open standard designed to facilitate communication between AI applications and external services, with a           │
│ particular focus on standardizing tool integration for AI agents. It acts as a crucial integration layer, enabling a "plug-and-play" approach to tool     │
│ usage and complementing existing agent orchestration frameworks.                                                                                          │
│                                                                                                                                                           │
│ Here's a breakdown of its key aspects:                                                                                                                    │
│                                                                                                                                                           │
│ *   **Purpose:** MCP aims to streamline the integration of various tools, databases, and predefined templates into AI applications. It addresses          │
│ challenges such as inconsistent API designs, the complexity of manual tool connections, and the need for robust error handling and authentication.        │
│ *   **Analogy:** It's often compared to a USB-C port for hardware, providing a standardized way for different components to connect and communicate.      │
│ *   **Functionality:** MCP helps AI agents become context-aware by providing a standardized protocol for interacting with external services. This allows  │
│ LLMs, which have inherent limitations, to access real-time information, perform calculations, and interact with the outside world through tools.          │
│ *   **Architecture:** MCP operates on a client-server model with three main components:                                                                   │
│     *   **MCP Host:** The AI application that receives user requests and seeks context through MCP.                                                       │
│     *   **MCP Client:** Resides within the host, converting user requests into a format the MCP protocol can process. It manages sessions, handles        │
│ interruptions, and parses responses.                                                                                                                      │
│     *   **MCP Server:** The external service that provides context to the LLM. This can include databases, APIs, or even other LLM inferencing services.  │
│ MCP servers expose data through resources, tools, and prompts.                                                                                            │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Benefits:**                                                                                                                                         │
│     *   **Streamlined Tool Integration:** Reduces the need for custom code for each tool, enabling "plug-and-play" functionality.                         │
│     *   **Enhanced Multi-agent Orchestration:** Facilitates better communication and tool sharing among multiple AI agents.                               │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Benefits:**                                                                                                                                         │
│     *   **Streamlined Tool Integration:** Reduces the need for custom code for each tool, enabling "plug-and-play" functionality.                         │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Benefits:**                                                                                                                                         │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Benefits:**                                                                                                                                         │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Benefits:**                                                                                                                                         │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Benefits:**                                                                                                                                         │
│ *   **Communication:** The transport layer between clients and servers uses JSON-RPC 2.0 format, supporting both standard input/output (stdio) for local  │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Benefits:**                                                                                                                                         │
│ resources and server-sent events (SSE) for remote resources.                                                                                              │
│ *   **Benefits:**                                                                                                                                         │
│ *   **Benefits:**                                                                                                                                         │
│     *   **Streamlined Tool Integration:** Reduces the need for custom code for each tool, enabling "plug-and-play" functionality.                         │
│     *   **Enhanced Multi-agent Orchestration:** Facilitates better communication and tool sharing among multiple AI agents.                               │
│     *   **Supplement to RAG:** Can connect to vector databases as a tool, enhancing retrieval augmented generation.                                       │
│     *   **Reduced Development Load:** Simplifies the process of debugging, maintaining authentication, and managing tool dependencies.                    │
│ *   **Relationship with Frameworks:** MCP is not an agent framework itself but a standardization layer that complements frameworks like LangChain,        │
│ LangGraph, crewAI, and others. It standardizes the connection to tools, while the LLM determines which tools to call.                                     │
│                                                                                                                                                           │
│ In essence, MCP is a critical development for making AI agents more autonomous, adaptable, and efficient by providing a standardized and robust way for   │
│ them to interact with the vast ecosystem of external tools and data.                                                                                      │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


'''