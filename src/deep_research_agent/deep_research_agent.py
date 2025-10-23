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
from langchain.agents import create_agent
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
# Search Results : {result['tile']}
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

agent = create_agent(
    model, all_tools, system_prompt=INSTRUCTIONS, state_schema=DeepAgentState
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




