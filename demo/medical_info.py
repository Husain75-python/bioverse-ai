import os, io, requests
from datetime import datetime
import streamlit as st
from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM (Groq)
llm = init_chat_model("groq:llama-3.1-8b-instant")

# Define state
class MedState(MessagesState):
    next_agent: str = ""
    research_data: str = ""
    analysis: str = ""
    final_report: str = ""
    current_task: str = ""
    task_complete: bool = False

# Supervisor agent
def supervisor_agent(state: MedState):
    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))
    if not has_research:
        next_agent = "researcher"
        msg = "🔬 Assigning task to Researcher..."
    elif not has_analysis:
        next_agent = "analyst"
        msg = "📊 Assigning task to Analyst..."
    elif not has_report:
        next_agent = "writer"
        msg = "✍️ Assigning task to Writer..."
    else:
        next_agent = "end"
        msg = "✅ All tasks complete!"
    return {"messages": [AIMessage(content=msg)], "next_agent": next_agent}

# Researcher agent (Semantic Scholar)
import requests

def fetch_pubmed(topic: str, max_results=5):
    """Fetch recent papers from PubMed."""
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": topic,
            "retmax": max_results,
            "retmode": "json",
            "sort": "pub+date"
        }
        res = requests.get(url, params=params, timeout=15).json()
        ids = res.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No PubMed papers found."
        # Fetch summaries
        summaries = []
        for pid in ids:
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            summary_params = {"db": "pubmed", "id": pid, "retmode": "json"}
            sdata = requests.get(summary_url, params=summary_params, timeout=10).json()
            doc = sdata.get("result", {}).get(pid, {})
            title = doc.get("title", "No title")
            authors = ", ".join([a["name"] for a in doc.get("authors", []) if "name" in a])
            year = doc.get("pubdate", "Unknown").split(" ")[0]
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            summaries.append(f"{title} ({year})\nAuthors: {authors}\n{link}")
        return "\n\n".join(summaries)
    except Exception as e:
        return f"Error fetching PubMed: {e}"

def fetch_semantic_scholar(topic: str, max_results=5):
    """Fetch recent papers from Semantic Scholar."""
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": topic, "limit": max_results, "fields": "title,abstract,year,url"}
        res = requests.get(url, params=params, timeout=15).json()
        papers = res.get("data", [])
        if not papers:
            return "No Semantic Scholar papers found."
        return "\n\n".join([
            f"{p['title']} ({p['year']})\n{p.get('abstract', 'No abstract available')}\n{p['url']}"
            for p in papers
        ])
    except Exception as e:
        return f"Error fetching Semantic Scholar: {e}"

def fetch_openalex(topic: str, max_results=5):
    """Fetch recent papers from OpenAlex API."""
    try:
        url = "https://api.openalex.org/works"
        params = {"search": topic, "per-page": max_results}
        res = requests.get(url, params=params, timeout=15).json()
        results = res.get("results", [])
        if not results:
            return "No OpenAlex papers found."
        return "\n\n".join([
            f"{r['display_name']} ({r.get('publication_year','')})\n{r['id']}"
            for r in results
        ])
    except Exception as e:
        return f"Error fetching OpenAlex: {e}"

def researcher_agent(state: dict) -> dict:
    """Multi-source research aggregator."""
    topic = state.get("current_task", "")
    st.info(f"🔍 Searching for medical papers on '{topic}'...")

    pubmed_data = fetch_pubmed(topic)
    sem_data = fetch_semantic_scholar(topic)
    openalex_data = fetch_openalex(topic)

    research_text = f"""
=== PUBMED RESULTS ===
{pubmed_data}

=== SEMANTIC SCHOLAR RESULTS ===
{sem_data}

=== OPENALEX RESULTS ===
{openalex_data}
"""

    msg = (
        f"🧠 Researcher Agent:\n"
        f"Completed literature aggregation for **{topic}**.\n"
        f"Compiled verified medical research papers from PubMed, Semantic Scholar, and OpenAlex."  ) 
    return {
        "messages": [AIMessage(content=msg)],
        "research_data": research_text,
        "next_agent": "supervisor"
    }

# Analyst agent
def analyst_agent(state: MedState):
    topic = state.get("current_task")
    research_data = state.get("research_data", "")
    prompt = f"""You are a biomedical research analyst.
Analyze the following research on '{topic}'.
Provide:
1. Major findings and evidence strength
2. Key trends across studies
3. Gaps and limitations
4. Future research directions
Data:
{research_data[:2000]}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content
    msg = f"📊 Analyst:\n{analysis[:800]}"
    return {"messages": [AIMessage(content=msg)], "analysis": analysis, "next_agent": "supervisor"}

# Writer agent
def writer_agent(state: MedState):
    topic = state.get("current_task")
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")
    prompt = f"""Write a professional medical literature review on '{topic}'.
Sections:
1. Executive Summary
2. Key Findings
3. Trends & Gaps
4. Recommendations for clinicians/researchers
5. References
Base your report on:
Research Data: {research_data[:1500]}
Analysis: {analysis[:1500]}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content
    final_report = f"""
🧾 MEDICAL LITERATURE REVIEW
==================================================
Topic: {topic}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
==================================================
{report}
==================================================
"""
    return {"messages": [AIMessage(content="✍️ Writer: Final report completed!")],
            "final_report": final_report, "task_complete": True, "next_agent": "supervisor"}

# Router
def router(state: MedState) -> Literal["supervisor", "researcher", "analyst", "writer", "END"]:
    if state.get("task_complete", False):
        return END
    return state.get("next_agent", "supervisor")

# Build LangGraph workflow
workflow = StateGraph(MedState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)
workflow.set_entry_point("supervisor")

for node in ["supervisor", "researcher", "analyst", "writer"]:
    workflow.add_conditional_edges(node, router, {"supervisor": "supervisor", "researcher": "researcher",
                                                  "analyst": "analyst", "writer": "writer", END: END})

graph = workflow.compile(checkpointer=MemorySaver())

# PDF generator
def generate_pdf(topic, messages, final_report):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = [Paragraph(f"<b>Medical Literature Review</b>", styles['Title']),
               Spacer(1, 12),
               Paragraph(f"<b>Topic:</b> {topic}", styles['Heading2']),
               Spacer(1, 12)]
    for m in messages:
        content.append(Paragraph(m.replace("\n", "<br/>"), styles['BodyText']))
        content.append(Spacer(1, 8))
    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Final Report:</b>", styles['Heading2']))
    content.append(Paragraph(final_report.replace("\n", "<br/>"), styles['BodyText']))
    doc.build(content)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.set_page_config(page_title="Medical Literature Aggregator", layout="wide", page_icon="🧬")
st.title("🧬 AI Medical Literature Aggregator")
st.write("Enter a medical topic, and this system will fetch, analyze, and summarize recent research automatically.")

topic = st.text_input("Enter medical topic (e.g., 'Parkinson's Disease treatment')")

if st.button("Run Aggregator") and topic.strip():
    state = MedState(messages=[HumanMessage(content=topic)], current_task=topic)
    st.info("🚀 Starting AI Literature Review Workflow...")
    full_messages = []
    config = {"configurable": {"thread_id": f"thread_{datetime.now().timestamp()}"}}
    for event in graph.stream(state, config=config):
        for value in event.values():
            if "messages" in value:
                for msg in value["messages"]:
                    full_messages.append(msg.content)
                    st.markdown(f"**{msg.content}**")
            if "final_report" in value and value["final_report"]:
                st.markdown(f"### 🧾 Final Report\n\n```\n{value['final_report']}\n```")
                pdf_buffer = generate_pdf(topic, full_messages, value["final_report"])
                st.download_button(
                    label="📥 Download Medical Report as PDF",
                    data=pdf_buffer,
                    file_name=f"{topic.replace(' ', '_')}_Medical_Report.pdf",
                    mime="application/pdf"
                )
    st.success("✅ Completed! Medical report ready.")
