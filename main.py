import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool   
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# Initialize ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    model="groq/llama-3.3-70b-versatile",
    max_tokens=200,
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Search Tool (Serper requires an API key)
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

# Define Research Agent with search capability
research_agent = Agent(
    role="Researcher",
    goal="Find the latest information on AI advancements.",
    backstory="An expert researcher who knows how to search and summarize information.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[search_tool]   
)

# Define a Task
research_task = Task(
    description="Research and provide a summary of the latest AI trends in 2025.",
    expected_output="A clear summary with key points of AI developments in 2025.",
    agent=research_agent
)

# Create a Crew
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True
)

# Run the crew
result = crew.kickoff()

print("\n--- Research Results ---\n")
print(result)
