import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool
)

search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Load API keys from .env
load_dotenv()

# Initialize ChatGroq LLM with OpenAI provider example
llm = ChatGroq(
    temperature=0,
    model="huggingface/mistralai/Mistral-7B-Instruct-v0.3",
    api_key=os.getenv("SERPER_API_KEY")
)


# Define Research Agent
research_agent = Agent(
    role="Researcher",
    goal="Find the latest information on AI advancements.",
    backstory="An expert researcher who knows how to search and summarize information.",
    tools=[search_tool, web_rag_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define a Task
research_task = Task(
    description="Research and provide a summary of the latest AI trends in 2025.",
    expected_output="A clear summary with key points of AI developments in 2025.",
    agent=research_agent,
    tools=[search_tool, web_rag_tool],
)

# Create a Crew
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    tools=[search_tool, web_rag_tool],
    verbose=True
)

# Run the crew
result = crew.kickoff()

print("\n--- Research Results ---\n")
print(result)
