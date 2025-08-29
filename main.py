import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from crewai.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from serpapi import GoogleSearch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import asyncio
from typing import Optional

# Load API keys
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for request
class ResearchRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 200
    
    model_config = ConfigDict(protected_namespaces=())

# Initialize LLM
def get_llm(max_tokens=200):
    return ChatGroq(
        temperature=0,
        model="groq/llama-3.3-70b-versatile",
        max_tokens=max_tokens,
        api_key=os.getenv("GROQ_API_KEY")
    )

# Initialize Search Tool
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

# Define Google Trends tool using SerpAPI
@tool("google_trends_tool")
def google_trends_tool(keyword: str) -> str:
    """Fetch Google Trends data for a keyword using SerpAPI."""
    try:
        params = {
            "engine": "google_trends",
            "q": keyword,
            "data_type": "TIMESERIES",
            "api_key": os.getenv("SERPAPI_KEY")
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "interest_over_time" not in results:
            return f"No trend data found for '{keyword}'"

        # Extract last 5 data points
        timeseries = results["interest_over_time"]["timeline_data"][-5:]
        trend_points = {point["date"]: point["values"][0]["extracted_value"] for point in timeseries}

        return f"Google Trends for '{keyword}' (last 5 points): {trend_points}"

    except Exception as e:
        return f"Error fetching Google Trends: {e}"

@app.get("/")
async def root():
    """Simple root endpoint"""
    return {
        "message": "AI Research API is running",
        "usage": "Send POST request to /research with {'query': 'your research topic'}",
        "example_queries": [
            "latest AI trends in 2025",
            "renewable energy advancements",
            "quantum computing progress",
            "cybersecurity trends 2024",
            "space exploration updates",
            "healthcare technology innovations"
        ]
    }

@app.post("/research")
async def research_ai_trends(request: ResearchRequest):
    """
    Research AI trends based on user query using web search and Google Trends.
    Returns the detailed LLM response directly.
    """
    try:
        # Initialize LLM with custom token limit
        llm = get_llm(request.max_tokens)

        # Define the Research Agent
        research_agent = Agent(
            role="AI Research Analyst",
            goal=f"Research and summarize: {request.query} using web search and trend data.",
            backstory="An expert researcher specializing in technology trends analysis using search tools and Google Trends.",
            allow_delegation=False,
            verbose=True,
            llm=llm,
            tools=[search_tool, google_trends_tool]
        )

        # Create research task with user query
        research_task = Task(
            description=(
                f"Comprehensively research and analyze: {request.query} "
                "Include current trends, recent developments, and future outlook. "
                "Use web search for latest information and include Google Trends insights for relevant keywords."
            ),
            expected_output="Detailed summary including current trends, recent developments, and future outlook with supporting data.",
            agent=research_agent
        )

        # Setup Crew
        crew = Crew(
            agents=[research_agent],
            tasks=[research_task],
            verbose=True
        )

        # Run the research asynchronously
        result = await asyncio.to_thread(crew.kickoff)

        return {
            "status": "success",
            "query": request.query,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
