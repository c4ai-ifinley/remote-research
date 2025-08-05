import arxiv
import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from utils import atomic_write_json
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import SamplingMessage, TextContent
from mcp.shared.context import RequestContext
from mcp.types import CreateMessageRequestParams, CreateMessageResult

PAPER_DIR = Path("papers")

# Initialize FastMCP server
mcp = FastMCP("research")


@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)

    Returns:
        List of paper IDs found in the search
    """

    # Use arxiv to find the papers
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)

    # Create directory for this topic
    path = PAPER_DIR / topic.lower().replace(" ", "_")
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / "papers_info.json"

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date()),
        }
        papers_info[paper.get_short_id()] = paper_info

    # Save updated papers_info to json file
    atomic_write_json(papers_info, file_path)

    print(f"Results are saved in: {file_path}")

    return paper_ids


@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """

    for item_path in PAPER_DIR.iterdir():
        if item_path.is_dir():
            file_path = item_path / "papers_info.json"
            if file_path.is_file():
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There's no saved information related to paper {paper_id}."


@mcp.tool()
async def collect_recent_papers(
    topic: str = None, max_results: int = 5, ctx: Context = None
) -> str:
    """
    Collect papers with user-driven topic selection.
    If no topic is provided, will ask the user to specify one or allow cancellation.
    All papers found will be collected regardless of publication date.

    Args:
        topic: Research topic (optional - will prompt user if not provided)
        max_results: Maximum number of results to retrieve (default: 5)
    """

    # Handle missing topic through user sampling
    if not topic:
        topic = await _determine_research_topic(ctx)
        if not topic:
            return "mcp_research_server: Paper collection cancelled - no topic was provided."

    client = arxiv.Client()
    search = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    papers = client.results(search)

    collection_log = []
    papers_collected = []

    collection_log.append(f"Searching for papers on: '{topic}'")
    collection_log.append(f"Collecting all {max_results} most relevant papers")

    # Collect all papers found in the search
    for paper in papers:
        paper_year = paper.published.year
        paper_id = paper.get_short_id()

        papers_collected.append(
            {
                "id": paper_id,
                "title": paper.title,
                "year": paper_year,
            }
        )
        collection_log.append(f"Collected: {paper.title} ({paper_year})")

    # Save collected papers
    if papers_collected:
        path = PAPER_DIR / topic.lower().replace(" ", "_")
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "papers_info.json"

        try:
            with open(file_path, "r") as json_file:
                papers_info = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            papers_info = {}

        # Add full paper information for collected papers
        for paper_data in papers_collected:
            # Find the full paper info from the search results
            for paper in papers:
                if paper.get_short_id() == paper_data["id"]:
                    papers_info[paper_data["id"]] = {
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "summary": paper.summary,
                        "pdf_url": paper.pdf_url,
                        "published": str(paper.published.date()),
                        "year": paper_data["year"],
                        "collection_date": str(datetime.now().date()),
                    }
                    break

        atomic_write_json(papers_info, file_path)
        collection_log.append(f"Saved {len(papers_collected)} papers to {file_path}")

    # Generate report
    report = f"""# Papers Collection Report

**Topic**: {topic}
**Papers Found**: {len(list(papers))}
**Papers Collected**: {len(papers_collected)}

## Collection Process
{chr(10).join(collection_log)}

## Papers by Year
{_format_papers_by_year(papers_collected)}
"""

    return report


@mcp.tool()
async def collect_recent_papers_with_sampling(
    max_results: int = 5, ctx: Context = None
) -> str:
    """
    Demo tool that showcases MCP sampling functionality.
    Always asks the user for a research topic through sampling interface.
    """

    if not ctx:
        raise Exception("No context available for sampling")

    # Use sampling to get topic from user - following the official pattern
    prompt = """mcp_research_server: I need to collect recent research papers, but no specific topic was provided.

mcp_research_server: Please specify which research topic you'd like me to search for papers on. Some popular options include:
    • Machine Learning
    • Climate Change  
    • Quantum Computing
    • Artificial Intelligence
    • Biotechnology
    • Renewable Energy
    • Cybersecurity
    • Space Exploration

mcp_research_server: Please specify a research topic, or type "cancel" if you'd like to stop the process."""

    try:
        result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt),
                )
            ],
            max_tokens=100,
        )

        if result.content.type == "text":
            topic = result.content.text.strip()

            # Handle cancellation with proper MCP error
            if not topic or topic.lower() in ["cancel", "skip"]:
                raise Exception(
                    "USER_CANCELLED: User chose to cancel the sampling demo. No papers will be collected."
                )

            print(f"mcp_research_server: Using topic '{topic}' provided via sampling")
        else:
            raise Exception("Invalid response from sampling")

    except Exception as e:
        # Re-raise any exception (including our cancellation) to MCP client
        raise e

    # Now collect papers with the sampled topic
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    papers = client.results(search)

    papers_collected = []
    collection_log = [f"SAMPLING SUCCESS: Topic '{topic}' selected by user"]

    for paper in papers:
        paper_year = paper.published.year
        paper_id = paper.get_short_id()

        papers_collected.append(
            {
                "id": paper_id,
                "title": paper.title,
                "year": paper_year,
            }
        )
        collection_log.append(f"Collected: {paper.title} ({paper_year})")

    # Save papers
    if papers_collected:
        path = PAPER_DIR / f"sampling_demo_{topic.lower().replace(' ', '_')}"
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "papers_info.json"

        papers_info = {}
        for paper_data in papers_collected:
            for paper in papers:
                if paper.get_short_id() == paper_data["id"]:
                    papers_info[paper_data["id"]] = {
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "summary": paper.summary,
                        "pdf_url": paper.pdf_url,
                        "published": str(paper.published.date()),
                        "year": paper_data["year"],
                        "collection_date": str(datetime.now().date()),
                        "sampling_demo": True,
                    }
                    break

        atomic_write_json(papers_info, file_path)
        collection_log.append(f"Saved to {file_path}")

    report = f"""# MCP Sampling Demo - SUCCESS!

**DEBUG INFO**: Using topic '{topic}' provided via sampling
**Topic Selected via Sampling**: {topic}
**Papers Collected**: {len(papers_collected)}

## Sampling Workflow Completed
✓ Server requested user input through MCP sampling
✓ User provided topic: "{topic}"  
✓ Server collected {len(papers_collected)} papers

## Collection Log
{chr(10).join(collection_log)}

## Papers by Year
{_format_papers_by_year(papers_collected)}

**Demo Complete**: MCP sampling successfully enabled human-AI collaboration!
"""

    return report


async def _determine_research_topic(ctx: Context) -> str:
    """
    Determine research topic through user sampling.
    Asks the user for a topic or allows them to cancel.
    """
    if not ctx:
        # No context available - cannot proceed without user input
        print("mcp_research_server: No context available for user interaction")
        return None

    try:
        # Ask user for topic preference
        user_topic = await _ask_user_for_topic(ctx)
        if user_topic and user_topic.lower() not in ["skip", "cancel"]:
            print(f"mcp_research_server: Using topic '{user_topic}' provided by user")
            return user_topic
        else:
            # User chose to cancel
            print("mcp_research_server: User cancelled topic selection")
            return None

    except Exception as e:
        print(f"mcp_research_server: Error requesting topic from user: {e}")
        return None


async def _ask_user_for_topic(ctx: Context) -> str:
    """Ask user to specify a research topic via sampling with clear server identification"""

    topic_prompt = """mcp_research_server: I need to collect recent research papers, but no specific topic was provided.

mcp_research_server: Please specify which research topic you'd like me to search for papers on. Some popular options include:
    • Machine Learning
    • Quantum Computing
    • Biotechnology
    • Renewable Energy
    • Cybersecurity

mcp_research_server: Please specify a research topic, or type "cancel" if you'd like to stop the process."""

    try:
        result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=topic_prompt),
                )
            ],
            max_tokens=100,
        )

        if result.content.type == "text":
            response = result.content.text.strip()
            return response if response else None

    except Exception as e:
        print(f"mcp_research_server: Error in topic sampling: {e}")
        return None


def _format_papers_by_year(papers_collected: List[Dict]) -> str:
    """Format papers by publication year for the report"""
    if not papers_collected:
        return "No papers collected."

    # Group papers by year
    by_year = {}
    for paper in papers_collected:
        year = paper["year"]
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(paper["title"])

    # Format the output
    formatted = ""
    for year in sorted(by_year.keys(), reverse=True):
        papers_in_year = by_year[year]
        formatted += f"**{year}**: {len(papers_in_year)} paper{'s' if len(papers_in_year) != 1 else ''}\n"
        for title in papers_in_year:
            formatted += f"  - {title}\n"
        formatted += "\n"

    return formatted


@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """
    List all available topic folders in the papers directory.

    This resource provides a simple list of all available topic folders.
    """
    folders = []

    # Get all topic directories
    if PAPER_DIR.exists():
        for topic_path in PAPER_DIR.iterdir():
            if topic_path.is_dir():
                papers_file = topic_path / "papers_info.json"
                if papers_file.exists():
                    folders.append(topic_path.name)

    # Create a simple markdown list
    content = "# Available Topics\n\n"
    if folders:
        for folder in folders:
            content += f"- {folder}\n"
        content += f"\nUse @{folder} to access papers in that topic.\n"
    else:
        content += "No topics found.\n"

    return content


@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """
    Get detailed information about papers on a specific topic.

    Args:
        topic: The research topic to retrieve papers for
    """
    topic_dir = topic.lower().replace(" ", "_")
    papers_file = PAPER_DIR / topic_dir / "papers_info.json"

    if not papers_file.exists():
        return f"# No papers found for topic: {topic}\n\nTry searching for papers on this topic first."

    try:
        with open(papers_file, "r") as f:
            papers_data = json.load(f)

        # Create markdown content with paper details
        content = f"# Papers on {topic.replace('_', ' ').title()}\n\n"
        content += f"Total papers: {len(papers_data)}\n\n"

        for paper_id, paper_info in papers_data.items():
            content += f"## {paper_info['title']}\n"
            content += f"- **Paper ID**: {paper_id}\n"
            content += f"- **Authors**: {', '.join(paper_info['authors'])}\n"
            content += f"- **Published**: {paper_info['published']}\n"
            content += (
                f"- **PDF URL**: [{paper_info['pdf_url']}]({paper_info['pdf_url']})\n\n"
            )
            content += f"### Summary\n{paper_info['summary'][:500]}...\n\n"
            content += "---\n\n"

        return content
    except json.JSONDecodeError:
        return f"# Error reading papers data for {topic}\n\nThe papers data file is corrupted."


@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """Generate a prompt for Claude to find and discuss academic papers on a specific topic."""
    return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool. 

Follow these instructions:
1. First, search for papers using search_papers(topic='{topic}', max_results={num_papers})
2. For each paper found, extract and organize the following information:
   - Paper title
   - Authors
   - Publication date
   - Brief summary of the key findings
   - Main contributions or innovations
   - Methodologies used
   - Relevance to the topic '{topic}'

3. Provide a comprehensive summary that includes:
   - Overview of the current state of research in '{topic}'
   - Common themes and trends across the papers
   - Key research gaps or areas for future investigation
   - Most impactful or influential papers in this area

4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.

Please present both detailed information about each paper and a high-level synthesis of the research landscape in {topic}."""


@mcp.prompt()
def demo_sampling_prompt(num_papers: int = 5) -> str:
    """Demo prompt that showcases the MCP sampling functionality."""
    return f"""Demonstrate MCP server sampling by collecting {num_papers} research papers using the sampling interface.

Use the collect_recent_papers_with_sampling tool to trigger the sampling workflow:

1. The tool will detect that no topic was provided
2. The MCP server will use sampling to request a topic from you
3. You can specify any research topic you're interested in
4. The server will then collect {num_papers} relevant papers

Execute: collect_recent_papers_with_sampling(max_results={num_papers})

IMPORTANT: If the user cancels the sampling request (by typing "cancel"), do NOT try alternative approaches. Simply acknowledge that the sampling demo was cancelled and explain what would have happened if a topic was provided.

This showcases how MCP servers can request human guidance when they need additional information to complete their tasks."""


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
    print(os.environ.get("DLAI_LOCAL_URL").format(port=6277)[:-1])
