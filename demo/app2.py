import os
import re
import io
from datetime import datetime, timedelta
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv

# LangGraph / LangChain / Tavily
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# Tavily integration
from langchain_community.tools.tavily_search import TavilySearchResults


# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ============ Configuration & Environment ============
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TAVILY_API_KEY:
    raise RuntimeError("Please set TAVILY_API_KEY in your environment (.env) before running the app")

# ============ Initialize LLM (Groq) ============
if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not set - LLM calls may fail if you rely on Groq. Set it in .env if you want to use Groq.")

llm = init_chat_model("groq:llama-3.1-8b-instant")

# ============ State Definition ============
class RFPState(MessagesState):
    next_agent: str = "supervisor"
    task_complete: bool = False
    current_task: str = ""
    # discovery outputs
    discovered_rfps: list = []
    selected_rfp: dict = {}
    # downstream outputs
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

# ============ Agents ============
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
        # fallback progression
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

# ============ Researcher (Tavily) ============

def researcher_agent(state: RFPState) -> dict:
    # build a query for Tavily - can be customized via UI
    query = state.get("current_task", "IT, AI and data analytics RFPs India")
    tavily = TavilySearchResults()

    st.info("Researcher: querying Tavily for RFPs...")
    try:
        results = tavily.run(query)
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Researcher failed to query Tavily: {e}" )], "next_agent": "supervisor"}

    # tavily.run may return complex json. Try to normalize into list of dicts
    candidates = []
    if isinstance(results, list):
        candidates = results
    elif isinstance(results, dict) and "results" in results:
        candidates = results["results"]
    else:
        # try to wrap whatever we got
        candidates = [results]

    # Each candidate ideally has: title, url, published or date, snippet, raw_content
    shortlisted = []
    for c in candidates:
        title = c.get("title") or c.get("heading") or c.get("name") or str(c.get("query", "Unnamed RFP"))
        url = c.get("url") or c.get("link") or c.get("source")
        snippet = c.get("snippet") or c.get("summary") or ""
        raw = c.get("raw_content") or c.get("content") or c.get("answer") or ""

        # try common date fields
        date = None
        for k in ["published", "date", "posted_at", "timestamp", "published_at", "submission_date"]:
            if k in c and c[k]:
                try:
                    date = datetime.fromisoformat(c[k])
                    break
                except Exception:
                    parsed = parse_date_from_text(str(c[k]))
                    if parsed:
                        date = parsed
                        break
        if not date:
            parsed = parse_date_from_text(snippet + "\n" + raw)
            if parsed:
                date = parsed

        candidates_info = {
            "title": title,
            "url": url,
            "snippet": snippet,
            "raw": raw,
            "date": date,
        }

        # Accept candidate if date exists and within 90 days OR if no date, keep for fallback
        shortlisted.append(candidates_info)

    # Filter by deadline (submission date within next 90 days)
    within_90 = []
    for s in shortlisted:
        if s["date"]:
            if 0 <= days_until(s["date"]) <= 90:
                within_90.append(s)

    selected = None
    if within_90:
        # choose the soonest deadline (smallest days_until)
        selected = sorted(within_90, key=lambda x: days_until(x["date"]))[0]
    else:
        # fallback: pick most recent by date if available
        with_dates = [s for s in shortlisted if s["date"]]
        if with_dates:
            selected = sorted(with_dates, key=lambda x: x["date"], reverse=True)[0]
        elif shortlisted:
            selected = shortlisted[0]

    if not selected:
        return {"messages": [AIMessage(content="🔍 Researcher: No RFPs found." )], "next_agent": "supervisor"}

    state["discovered_rfps"] = shortlisted
    state["selected_rfp"] = selected

    msg = f"🔎 Researcher: Selected RFP - {selected.get('title')} | URL: {selected.get('url')} | Date: {selected.get('date') or 'Unknown'}"
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
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyzer", analyzer_agent)
workflow.add_node("technical", technical_agent)
workflow.add_node("pricing", pricing_agent)
workflow.add_node("writer", writer_agent)
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
    query = st.text_input("Search query for RFPs (Tavily)", "RFPs IT services AI data analytics India")
    max_results = st.number_input("Max Tavily results", min_value=1, max_value=20, value=8)
    days_window = st.number_input("Submission within (days)", min_value=7, max_value=365, value=90)

if st.button("Run RFP Discovery & Response"):
    state = RFPState(messages=[HumanMessage(content=query)])
    state["current_task"] = query
    st.info("🚀 Starting workflow — Researcher will query Tavily and choose an RFP (deadline within {} days)...".format(days_window))

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

# ============ End of file ============

# Requirements (for your reference):
# pip install streamlit langgraph langchain-core langchain-tavily tavily-python langchain-community reportlab python-dotenv
