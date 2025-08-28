import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from crewai.tools import tool 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load API keys
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    model="groq/llama-3.3-70b-versatile",
    max_tokens=200,
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Search Tool (Serper for general web search)
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

# Define Google Trends tool using SerpAPI
@tool("google_trends_tool")
def google_trends_tool(keyword: str) -> str:
    """Fetch Google Trends data for a keyword using SerpAPI."""
    try:
        params = {
            "engine": "google_trends",
            "q": keyword,
            "data_type": "TIMESERIES",  # Options: TIMESERIES, RELATED_QUERIES, RELATED_TOPICS
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

# Define the Agent
research_agent = Agent(
    role="News Reporter",
    goal="Provide the latest current news using web search and SerpAPI tools.",
    backstory="A skilled reporter who stays up to date with the latest global happenings.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[search_tool, google_trends_tool]  
)

# Task setup
research_task = Task(
    description=(
        "Fetch and summarize the latest world news headlines today."
        
    ),
    expected_output="A clear summary of the top current news.",
    agent=research_agent
)

# Crew execution
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True
)

# Run the Crew
result = crew.kickoff()

print("\n--- Research Results ---\n")
print(result)
