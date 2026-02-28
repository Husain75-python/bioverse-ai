import os
import re
import io
from datetime import datetime, timedelta
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv

# LangGraph / LangChain
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# SerpAPI
from serpapi import GoogleSearch

# ============ Configuration & Environment ============
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SERPAPI_API_KEY:
    raise RuntimeError("Please set SERPAPI_API_KEY in your environment (.env) before running the app")

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not set - LLM calls may fail if you rely on Groq. Set it in .env if you want to use Groq.")

llm = init_chat_model("groq:llama-3.1-8b-instant")

# ============ State Definition ============
class RFPState(MessagesState):
    next_agent: str = "supervisor"
    task_complete: bool = False
    current_task: str = ""
    discovered_rfps: list = []
    selected_rfp: dict = {}
    analysis: str = ""
    technical: str = ""
    pricing: str = ""
    final_report: str = ""

# ============ Helper utilities ============
DATE_PATTERNS = [
    r"(\d{4}-\d{2}-\d{2})",
    r"(\d{1,2} \w+ \d{4})",
    r"(\w+ \d{1,2}, \d{4})",
]

def parse_date_from_text(text: str) -> Optional[datetime]:
    if not text:
        return None
    for pat in DATE_PATTERNS:
        m = re.search(pat, text)
        if m:
            try:
                return datetime.fromisoformat(m.group(1))
            except Exception:
                try:
                    return datetime.strptime(m.group(1), "%d %B %Y")
                except Exception:
                    try:
                        return datetime.strptime(m.group(1), "%B %d, %Y")
                    except Exception:
                        continue
    return None

def days_until(d: datetime) -> int:
    return (d - datetime.now()).days

# ============ Supervisor Agent ============
def create_supervisor_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a supervisor that orchestrates RFP discovery and response agents. Reply only with one of: researcher, analyzer, technical, pricing, writer, done."),
        ("human", "Current state: has_selected_rfp={has_selected_rfp}, has_analysis={has_analysis}, has_technical={has_technical}, has_pricing={has_pricing}, has_report={has_report}. Task: {task}")
    ])
    return prompt | llm

def supervisor_agent(state: RFPState) -> dict:
    has_selected = bool(state.get("selected_rfp"))
    has_analysis = bool(state.get("analysis"))
    has_technical = bool(state.get("technical"))
    has_pricing = bool(state.get("pricing"))
    has_report = bool(state.get("final_report"))

    chain = create_supervisor_prompt()
    decision = chain.invoke({
        "has_selected_rfp": has_selected,
        "has_analysis": has_analysis,
        "has_technical": has_technical,
        "has_pricing": has_pricing,
        "has_report": has_report,
        "task": state.get("current_task", "RFP response")
    })
    decision_text = decision.content.strip().lower()

    if "done" in decision_text or has_report:
        next_agent = "end"
        msg = "✅ Supervisor: Workflow complete."
    elif "researcher" in decision_text or not has_selected:
        next_agent = "researcher"
        msg = "🔎 Supervisor: Launching Researcher to find RFPs..."
    elif "analyzer" in decision_text and has_selected and not has_analysis:
        next_agent = "analyzer"
        msg = "📑 Supervisor: Sending selected RFP to Analyzer..."
    elif "technical" in decision_text and has_analysis and not has_technical:
        next_agent = "technical"
        msg = "⚙️ Supervisor: Assigning Technical Agent..."
    elif "pricing" in decision_text and has_technical and not has_pricing:
        next_agent = "pricing"
        msg = "💰 Supervisor: Assigning Pricing Agent..."
    elif "writer" in decision_text and has_pricing and not has_report:
        next_agent = "writer"
        msg = "📝 Supervisor: Assigning Writer to compile final proposal..."
    else:
        if not has_selected:
            next_agent = "researcher"
            msg = "🔎 Supervisor: Need to find an RFP first."
        elif not has_analysis:
            next_agent = "analyzer"
            msg = "📑 Supervisor: Analyze the RFP."
        elif not has_technical:
            next_agent = "technical"
            msg = "⚙️ Supervisor: Create technical response."
        elif not has_pricing:
            next_agent = "pricing"
            msg = "💰 Supervisor: Create pricing response."
        else:
            next_agent = "writer"
            msg = "📝 Supervisor: Compile the final report."

    return {"messages": [AIMessage(content=msg)], "next_agent": next_agent}

# ============ Researcher (SerpAPI) ============
def researcher_agent(state: RFPState) -> dict:
    query = state.get("current_task", "IT, AI and data analytics RFPs India")
    st.info("Researcher: querying SerpAPI for RFPs...")

    try:
        client = GoogleSearch({
            "q": query,
            "hl": "en",
            "gl": "in",
            "num": 10,
            "api_key": SERPAPI_API_KEY
        })
        data = client.get_dict()
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Researcher failed to query SerpAPI: {e}")], "next_agent": "supervisor"}

    results = data.get("organic_results", [])
    if not results:
        return {"messages": [AIMessage(content="🔍 Researcher: No results found from SerpAPI.")], "next_agent": "supervisor"}

    candidates = []
    for c in results:
        title = c.get("title", "Untitled")
        url = c.get("link")
        snippet = c.get("snippet", "")
        date = parse_date_from_text(snippet)
        candidates.append({"title": title, "url": url, "snippet": snippet, "raw": snippet, "date": date})

    within_90 = [s for s in candidates if s["date"] and 0 <= days_until(s["date"]) <= 90]
    selected = within_90[0] if within_90 else (candidates[0] if candidates else None)

    if not selected:
        return {"messages": [AIMessage(content="🔍 Researcher: No valid RFPs found within 90 days.")], "next_agent": "supervisor"}

    state["discovered_rfps"] = candidates
    state["selected_rfp"] = selected

    msg = f"🔎 Researcher: Selected RFP — {selected.get('title')} | URL: {selected.get('url')} | Date: {selected.get('date') or 'Unknown'}"
    return {"messages": [AIMessage(content=msg)], "next_agent": "analyzer"}

# ============ Analyzer Agent ============
def analyzer_agent(state: RFPState) -> dict:
    rfp = state.get("selected_rfp", {})
    if not rfp:
        return {"messages": [AIMessage(content="❌ Analyzer: No RFP provided")], "next_agent": "supervisor"}

    prompt = f"Extract scope, deliverables, eligibility criteria, and key dates from the following RFP text:\nTitle: {rfp.get('title')}\nSnippet: {rfp.get('snippet')}\nContent: {rfp.get('raw')[:4000]}"
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content

    state["analysis"] = analysis
    return {"messages": [AIMessage(content=f"📑 Analyzer: Extracted key information (truncated)\n{analysis[:800]}")], "next_agent": "technical"}

# ============ Technical Agent ============
def technical_agent(state: RFPState) -> dict:
    analysis = state.get("analysis", "")
    prompt = f"Based on the extracted RFP analysis below, propose a suitable technical solution with architecture, timeline, team requirements and compliance considerations:\n{analysis[:4000]}"
    response = llm.invoke([HumanMessage(content=prompt)])
    technical = response.content

    state["technical"] = technical
    return {"messages": [AIMessage(content=f"⚙️ Technical: Drafted technical approach (truncated)\n{technical[:800]}")], "next_agent": "pricing"}

# ============ Pricing Agent ============
def pricing_agent(state: RFPState) -> dict:
    technical = state.get("technical", "")
    prompt = f"Create a pricing proposal (cost breakdown, assumptions, payment milestones) for the following technical approach:\n{technical[:4000]}\nInclude conservative estimates and optional cost-saving variants."
    response = llm.invoke([HumanMessage(content=prompt)])
    pricing = response.content

    state["pricing"] = pricing
    return {"messages": [AIMessage(content=f"💰 Pricing: Prepared pricing (truncated)\n{pricing[:800]}")], "next_agent": "writer"}

# ============ Writer Agent ============
def writer_agent(state: RFPState) -> dict:
    selected = state.get("selected_rfp", {})
    analysis = state.get("analysis", "")
    technical = state.get("technical", "")
    pricing = state.get("pricing", "")

    prompt = f"Compose a professional RFP response document. Include an executive summary, key solution points, timeline, pricing table, assumptions, and next steps.\nRFP Title: {selected.get('title')}\nRFP URL: {selected.get('url')}\nAnalysis:\n{analysis[:3000]}\nTechnical:\n{technical[:3000]}\nPricing:\n{pricing[:2000]}"
    response = llm.invoke([HumanMessage(content=prompt)])
    final_report = response.content

    state["final_report"] = final_report
    state["task_complete"] = True
    return {"messages": [AIMessage(content="📝 Writer: Final proposal compiled.")], "next_agent": "supervisor"}

# ============ Router ============
def router(state: RFPState) -> str:
    next_agent = state.get("next_agent", "supervisor")
    if next_agent == "end" or state.get("task_complete"):
        return END
    return next_agent

# ============ Build Graph ============
workflow = StateGraph(RFPState)
for node, func in {
    "supervisor": supervisor_agent,
    "researcher": researcher_agent,
    "analyzer": analyzer_agent,
    "technical": technical_agent,
    "pricing": pricing_agent,
    "writer": writer_agent,
}.items():
    workflow.add_node(node, func)
workflow.set_entry_point("supervisor")

for node in ["supervisor", "researcher", "analyzer", "technical", "pricing", "writer"]:
    workflow.add_conditional_edges(node, router, {
        "supervisor": "supervisor",
        "researcher": "researcher",
        "analyzer": "analyzer",
        "technical": "technical",
        "pricing": "pricing",
        "writer": "writer",
        END: END,
    })

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# ============ PDF generator ============
def generate_pdf(topic: str, rfp: dict, messages: List[str], final_report: str) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph(f"<b>Proposal for: {topic}</b>", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"<b>Selected RFP:</b> {rfp.get('title')}", styles["Heading2"]))
    content.append(Paragraph(f"URL: {rfp.get('url')}", styles["Normal"]))
    content.append(Paragraph(f"Date: {rfp.get('date')}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Agent Messages / Trace</b>", styles["Heading2"]))
    for m in messages:
        content.append(Paragraph(m.replace("\n", "<br/>"), styles["BodyText"]))
        content.append(Spacer(1, 6))

    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Final Proposal</b>", styles["Heading2"]))
    content.append(Paragraph(final_report.replace("\n", "<br/>"), styles["BodyText"]))
    doc.build(content)
    buf.seek(0)
    return buf

# ============ Streamlit UI ============
st.set_page_config(page_title="RFP Discovery & Response", layout="wide")
st.title("🏢 RFP Discovery & Response — Multi-Agent System")

with st.sidebar:
    st.header("Settings")
    query = st.text_input("Search query for RFPs", "RFPs IT services AI data analytics India")
    days_window = st.number_input("Submission within (days)", min_value=7, max_value=365, value=90)

if st.button("Run RFP Discovery & Response"):
    state = RFPState(messages=[HumanMessage(content=query)])
    state["current_task"] = query
    st.info(f"🚀 Starting workflow — Researcher will query SerpAPI and choose an RFP (deadline within {days_window} days)...")

    full_messages = []
    config = {"configurable": {"thread_id": f"rfp_thread_{datetime.now().timestamp()}"}}

    for event in graph.stream(state, config=config):
        for value in event.values():
            if "messages" in value:
                for msg in value["messages"]:
                    full_messages.append(msg.content)
                    st.markdown(f"**{msg.content}**")
            if "final_report" in value and value["final_report"]:
                st.markdown("### 🧾 Final Proposal\n\n```")
                st.code(value["final_report"][:12000])

    st.success("✅ Workflow completed")

    final_report_text = state.get("final_report", "")
    if final_report_text:
        pdf_buf = generate_pdf(query, state.get("selected_rfp", {}), full_messages, final_report_text)
        st.download_button("📥 Download Proposal PDF", data=pdf_buf, file_name=f"proposal_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf")
