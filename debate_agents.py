from crewai import Agent
import os
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM

load_dotenv()

llm = ChatLiteLLM(
    model ="gemini/gemini-2.5-pro",
    temperature = 0.5,
    verbose = True,
    api_key = os.getenv("GOOGLE_API_KEY")
)



hr_agent = Agent(
    role ="HR Specialist",
    goal ="Argue from an HR and employee welfare perspective.",
    backstory ="An empathetic HR manager who values work-life balance.",
    llm = llm
)
dev_agent = Agent(
    role="Software Developer",
    goal="Argue from a developer’s perspective focusing on productivity and focus.",
    backstory="A pragmatic coder who values deep work and minimal distractions.",
    llm=llm
)
manager_agent = Agent(
    role="Project Manager",
    goal="Argue from a management perspective focusing on deadlines and collaboration.",
    backstory="An efficiency-driven manager who emphasizes teamwork and delivery timelines.",
    llm=llm
)
moderator = Agent(
    role="Moderator",
    goal="Summarize the debate and extract the consensus or main points of disagreement.",
    backstory="A neutral observer ensuring fair summarization.",
    llm=llm
)


