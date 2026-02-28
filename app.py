import streamlit as st
from typing import Literal
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

# ============ Load Environment Variables ============
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ============ Initialize LLM ============
llm = init_chat_model("groq:llama-3.1-8b-instant")

# ============ Define State ============
class SuperVisorState(MessagesState):
    """State for the multi-agent system"""
    next_agent: str = ""
    research_data: str = ""
    analysis: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""

# ============ Supervisor Agent ============
def create_supervisor_agent():
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of agents:
        1. Researcher – gathers information
        2. Analyst – analyzes data
        3. Writer – creates a final report

        Based on current state, decide who should work next.
        Respond only with the agent name: 'researcher', 'analyst', 'writer', or 'done'.

        Current state:
        - Has research: {has_research}
        - Has analysis: {has_analysis}
        - Has report: {has_report}
        """),
        ("human", "{task}")
    ])
    return supervisor_prompt | llm

def supervisor_agent(state: SuperVisorState) -> dict:
    messages = state['messages']
    task = messages[-1].content if messages else "no task"
    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))

    chain = create_supervisor_agent()
    decision = chain.invoke({
        "task": task,
        "has_research": has_research,
        "has_analysis": has_analysis,
        "has_report": has_report
    })

    decision_text = decision.content.strip().lower()

    if "done" in decision_text or has_report:
        next_agent = "end"
        msg = "✅ Supervisor: All tasks complete!"
    elif "researcher" in decision_text or not has_research:
        next_agent = "researcher"
        msg = "🧑‍🔬 Assigning task to Researcher..."
    elif "analyst" in decision_text or (has_research and not has_analysis):
        next_agent = "analyst"
        msg = "📊 Assigning task to Analyst..."
    elif "writer" in decision_text or (has_analysis and not has_report):
        next_agent = "writer"
        msg = "📝 Assigning task to Writer..."
    else:
        next_agent = "end"
        msg = "✅ Task completed."

    return {
        "messages": [AIMessage(content=msg)],
        "next_agent": next_agent,
        "current_task": task
    }

# ============ Researcher Agent ============
def researcher_agent(state: SuperVisorState) -> dict:
    task = state.get("current_task", "research topic")
    prompt = f"""As a research specialist, provide detailed information about: {task}
Include:
1. Key facts and background
2. Current developments
3. Relevant statistics
4. Case studies
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    research_data = response.content

    msg = f"🧑‍🔬 Researcher:\n{research_data[:1000]}"
    return {
        "messages": [AIMessage(content=msg)],
        "research_data": research_data,
        "next_agent": "supervisor"
    }

# ============ Analyst Agent ============
def analyst_agent(state: SuperVisorState) -> dict:
    research_data = state.get("research_data", "")
    task = state.get("current_task", "")
    prompt = f"""As an analyst, analyze this research data on '{task}'.
Provide:
1. Key insights
2. Trends & patterns
3. Risks and opportunities
4. Actionable recommendations
Data:
{research_data[:1500]}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content

    msg = f"📊 Analyst:\n{analysis[:1000]}"
    return {
        "messages": [AIMessage(content=msg)],
        "analysis": analysis,
        "next_agent": "supervisor"
    }

# ============ Writer Agent ============
def writer_agent(state: SuperVisorState) -> dict:
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")
    task = state.get("current_task", "")
    prompt = f"""As a professional writer, create an executive report on '{task}'.
Use the following:
Research Findings: {research_data[:1200]}
Analysis: {analysis[:1200]}

Include:
1. Executive Summary
2. Key Findings
3. Analysis & Insights
4. Recommendations
5. Conclusion
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content

    final_report = f"""
    🧾 Final Report
    {'='*50}
    Topic: {task}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    {'='*50}
    {report}
    {'='*50}
    """

    msg = "📝 Writer: Report completed!"
    return {
        "messages": [AIMessage(content=msg)],
        "final_report": final_report,
        "next_agent": "supervisor",
        "task_complete": True
    }

# ============ Router Function ============
def router(state: SuperVisorState) -> Literal["supervisor", "researcher", "analyst", "writer", "END"]:
    next_agent = state.get("next_agent", "supervisor")
    if next_agent == "end" or state.get("task_complete", False):
        return END
    return next_agent

# ============ Build LangGraph Workflow ============
workflow = StateGraph(SuperVisorState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)
workflow.set_entry_point("supervisor")

for node in ["supervisor", "researcher", "analyst", "writer"]:
    workflow.add_conditional_edges(node, router, {
        "supervisor": "supervisor",
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        END: END
    })

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# ============ Streamlit UI ============
st.set_page_config(page_title="Multi-Agent Workflow", layout="wide", page_icon="🤖")

st.title("🤖 Multi-Agent AI Workflow (Groq + LangGraph)")
st.write("Enter a topic or task and watch the agents collaborate step-by-step.")

task = st.text_input("🧠 Enter a task (e.g., 'AI impact on education')", "")

if st.button("Run Workflow") and task.strip():
    state = SuperVisorState(messages=[HumanMessage(content=task)])
    st.info("🚀 Starting multi-agent workflow...")
    report_placeholder = st.empty()

    full_messages = []
    for event in graph.stream(state):
        for value in event.values():
            if "messages" in value:
                for msg in value["messages"]:
                    full_messages.append(msg.content)
                    st.markdown(f"**{msg.content}**")
            if "final_report" in value and value["final_report"]:
                report_placeholder.markdown(f"### 🧾 Final Report\n\n```\n{value['final_report']}\n```")

    st.success("✅ Workflow completed!")

    # Save report for download
    final_report_text = value.get("final_report", "")
    if final_report_text:
        st.set_page_config(page_title="Multi-Agent Workflow", layout="wide", page_icon="🤖")

st.title("🤖 Multi-Agent AI Workflow (Groq + LangGraph)")
st.write("Enter a topic or task and watch the agents collaborate step-by-step.")

task = st.text_input("🧠 Enter a task (e.g., 'AI impact on education')", "")

if st.button("Run Workflow") and task.strip():
    state = SuperVisorState(messages=[HumanMessage(content=task)])
    st.info("🚀 Starting multi-agent workflow...")
    report_placeholder = st.empty()

    full_messages = []
    for event in graph.stream(state):
        for value in event.values():
            if "messages" in value:
                for msg in value["messages"]:
                    full_messages.append(msg.content)
                    st.markdown(f"**{msg.content}**")
            if "final_report" in value and value["final_report"]:
                report_placeholder.markdown(f"### 🧾 Final Report\n\n```\n{value['final_report']}\n```")

    st.success("✅ Workflow completed!")

    # Save report for download
    final_report_text = value.get("final_report", "")
    if final_report_text:
        st.set_page_config(page_title="Multi-Agent Workflow", layout="wide", page_icon="🤖")

st.title("🤖 Multi-Agent AI Workflow (Groq + LangGraph)")
st.write("Enter a topic or task and watch the agents collaborate step-by-step.")

task = st.text_input("🧠 Enter a task (e.g., 'AI impact on education')", "")

if st.button("Run Workflow") and task.strip():
    state = SuperVisorState(messages=[HumanMessage(content=task)])
    st.info("🚀 Starting multi-agent workflow...")
    report_placeholder = st.empty()

    full_messages = []
    for event in graph.stream(state):
        for value in event.values():
            if "messages" in value:
                for msg in value["messages"]:
                    full_messages.append(msg.content)
                    st.markdown(f"**{msg.content}**")
            if "final_report" in value and value["final_report"]:
                report_placeholder.markdown(f"### 🧾 Final Report\n\n```\n{value['final_report']}\n```")

    st.success("✅ Workflow completed!")

    # Save report for download
    final_report_text = value.get("final_report", "")
    if final_report_text:
        st.download_button(
            label="📥 Download Final Report",
            data=final_report_text,
            file_name="multi_agent_report.txt",
            mime="text/plain"
        )
