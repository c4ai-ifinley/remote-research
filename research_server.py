import arxiv
import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from utils import atomic_write_json
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import SamplingMessage, TextContent

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
    topic: str, max_results: int = 5, min_year: int = 2024, ctx: Context = None
) -> str:
    """
    Collect papers but ask human guidance for older papers.
    Demonstrates MCP sampling for autonomous decision-making.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    papers = client.results(search)

    collection_log = []
    papers_collected = []
    human_decisions = []

    collection_log.append(f"Searching for papers on: '{topic}'")
    collection_log.append(f"Auto-including papers from {min_year} or later")

    for paper in papers:
        paper_year = paper.published.year
        paper_id = paper.get_short_id()

        if paper_year >= min_year:
            papers_collected.append(
                {
                    "id": paper_id,
                    "title": paper.title,
                    "year": paper_year,
                    "decision": "auto_included",
                }
            )
            collection_log.append(f"Auto-included: {paper.title} ({paper_year})")
        else:
            collection_log.append(f"Found older paper: {paper.title} ({paper_year})")
            should_include = await _ask_human_about_older_paper(
                paper, min_year, topic, ctx
            )

            human_decisions.append(
                {
                    "paper_id": paper_id,
                    "paper_title": paper.title,
                    "paper_year": paper_year,
                    "decision": should_include,
                }
            )

            if should_include:
                papers_collected.append(
                    {
                        "id": paper_id,
                        "title": paper.title,
                        "year": paper_year,
                        "decision": "human_approved",
                    }
                )
                collection_log.append(f"Human approved: {paper.title} ({paper_year})")
            else:
                collection_log.append(f"Human skipped: {paper.title} ({paper_year})")

    if papers_collected:
        path = PAPER_DIR / topic.lower().replace(" ", "_")
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "papers_info.json"

        try:
            with open(file_path, "r") as json_file:
                papers_info = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            papers_info = {}

        for paper_data in papers_collected:
            papers_info[paper_data["id"]] = {
                "title": paper_data["title"],
                "year": paper_data["year"],
                "decision_method": paper_data["decision"],
                "collection_date": str(datetime.now().date()),
            }

        atomic_write_json(papers_info, file_path)
        collection_log.append(f"Saved {len(papers_collected)} papers to {file_path}")

    recent_count = len(
        [p for p in papers_collected if p["decision"] == "auto_included"]
    )
    human_approved_count = len(
        [p for p in papers_collected if p["decision"] == "human_approved"]
    )
    human_rejected_count = len([d for d in human_decisions if not d["decision"]])

    report = f"""# Recent Papers Collection Report

**Topic**: {topic}
**Date Threshold**: {min_year} or later
**Papers Found**: {max_results}
**Papers Collected**: {len(papers_collected)}

## Collection Summary
- Auto-included (recent): {recent_count} papers
- Human-approved (older): {human_approved_count} papers  
- Human-rejected (older): {human_rejected_count} papers

## Collection Process
{chr(10).join(collection_log)}

## Human Decisions
{_format_human_decisions(human_decisions)}
"""

    return report


async def _ask_human_about_older_paper(
    paper, min_year: int, topic: str, ctx: Context
) -> bool:
    """Ask human via MCP sampling whether to include an older paper."""
    if not ctx:
        print("No context available for sampling - defaulting to skip")
        return False

    paper_year = paper.published.year
    years_old = min_year - paper_year

    guidance_prompt = f"""I'm collecting recent papers on "{topic}" and found an older paper.

Paper: {paper.title}
Published: {paper_year} ({years_old} years before my {min_year} threshold)
Authors: {', '.join([author.name for author in paper.authors[:3]])}

Should I include this older paper? Respond with "yes" or "no": """

    try:
        result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=guidance_prompt),
                )
            ],
            max_tokens=100,
        )

        if result.content.type == "text":
            response_text = result.content.text.lower().strip()
            return response_text.startswith("yes")
        return False
    except Exception as e:
        print(f"Error requesting human guidance: {e}")
        return False


def _format_human_decisions(decisions: List[Dict]) -> str:
    """Format human decisions for the report"""
    if not decisions:
        return "No human decisions required."

    formatted = ""
    for decision in decisions:
        status = "INCLUDED" if decision["decision"] else "SKIPPED"
        formatted += (
            f"- {decision['paper_title']} ({decision['paper_year']}) - {status}\n"
        )

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


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
    print(os.environ.get("DLAI_LOCAL_URL").format(port=6277)[:-1])
