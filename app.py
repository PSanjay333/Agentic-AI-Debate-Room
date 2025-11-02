import streamlit as st
from crewai import Crew, Agent, Task
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





st.set_page_config(page_title="Agentic Debate Room", layout="centered")

st.title("Agentic Debate Room")
st.write("Enter a topic and watch specialized AI agents debate from multiple perspectives!")

topic = st.text_input("Enter debate topic:", "Should remote work be mandatory for developers?")
# Defining Separate debate tasks
hr_task = Task(
    description=f"Debate on the topic: '{topic}'. Provide reasoning from an HR perspective.",
    agent=hr_agent,
    expected_output="A paragraph explaining the HR viewpoint on the topic."
)
dev_task = Task(
    description=f"Debate on the topic: '{topic}'. Provide reasoning from a developer perspective.",
    agent=dev_agent,
    expected_output="A paragraph explaining the developer's viewpoint on the topic."
)
manager_task = Task(
    description=f"Debate on the topic: '{topic}'. Provide reasoning from a management perspective.",
    agent=manager_agent,
    expected_output="A paragraph explaining the management viewpoint on the topic."
)
# Moderate task to summarize
summarize_task = Task(
    description=(
        f"Summarize the debate on '{topic}' between HR, Developer, and Manager. "
        "Extract 3 key insights and provide 1 concluding statement."
    ),
    agent=moderator,
    context=[hr_task, dev_task, manager_task],
    expected_output="A short summary with 3 short insights and 1 conclusion."
)


if st.button("Start Debate"):
    with st.spinner("Debate in progress..."):
        # HR Debate
        with st.spinner("HR Agent debating..."):
            hr_crew = Crew(agents=[hr_agent], tasks=[hr_task], verbose=True)
            hr_result = hr_crew.kickoff()
            st.subheader("HR Perspective")
            st.write(hr_result.raw)

        # Developer Debate
        with st.spinner("Developer Agent debating..."):
            dev_crew = Crew(agents=[dev_agent], tasks=[dev_task], verbose=True)
            dev_result = dev_crew.kickoff()
            st.subheader("Developer Perspective")
            st.write(dev_result.raw)

        # Manager Debate
        with st.spinner("Manager Agent debating..."):
            manager_crew = Crew(agents=[manager_agent], tasks=[manager_task], verbose=True)
            manager_result = manager_crew.kickoff()
            st.subheader("Manager Perspective")
            st.write(manager_result.raw)

        # Summary
        st.success("Debate complete!")
        st.markdown("### Debate Summary")
        with st.spinner("Summarizing discussion..."):
            summary_crew = Crew(agents=[moderator], tasks=[summarize_task], verbose=True)
            summary_result = summary_crew.kickoff()
            st.write(summary_result.raw)

