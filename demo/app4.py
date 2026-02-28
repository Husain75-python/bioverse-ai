import os
import re
import io
from datetime import datetime, timedelta
from typing import List

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
    raise RuntimeError("Please set SERPAPI_API_KEY in your environment (.env)")

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not set - LLM calls may fail if you rely on Groq.")

llm = init_chat_model("groq:llama-3.1-8b-instant")

# ============ State Definition ============
class SmartMedState(MessagesState):
    next_agent: str = "supervisor"
    task_complete: bool = False
    current_task: str = ""
    discovered_articles: list = []
    selected_articles: list = []
    analysis: str = ""
    evaluation: str = ""
    ranked_report: str = ""
    final_report: str = ""

# ============ Helper Utilities ============
def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

# ============ Supervisor Agent ============
def create_supervisor_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a supervisor orchestrating a Smart Medical Research Assistant workflow. Reply only with one of: researcher, analyzer, evaluator, ranker, writer, done."),
        ("human", "Current state: has_selected={has_selected}, has_analysis={has_analysis}, has_evaluation={has_evaluation}, has_ranked={has_ranked}, has_report={has_report}. Task: {task}")
    ])
    return prompt | llm

def supervisor_agent(state: SmartMedState) -> dict:
    has_selected = bool(state.get("selected_articles"))
    has_analysis = bool(state.get("analysis"))
    has_evaluation = bool(state.get("evaluation"))
    has_ranked = bool(state.get("ranked_report"))
    has_report = bool(state.get("final_report"))

    chain = create_supervisor_prompt()
    decision = chain.invoke({
        "has_selected": has_selected,
        "has_analysis": has_analysis,
        "has_evaluation": has_evaluation,
        "has_ranked": has_ranked,
        "has_report": has_report,
        "task": state.get("current_task", "Smart Medical Literature Aggregation")
    })
    decision_text = decision.content.strip().lower()

    if "done" in decision_text or has_report:
        next_agent = "end"
        msg = "✅ Supervisor: Workflow complete."
    elif "researcher" in decision_text or not has_selected:
        next_agent = "researcher"
        msg = "🔎 Supervisor: Launching Smart Medical Literature Researcher..."
    elif "analyzer" in decision_text and has_selected and not has_analysis:
        next_agent = "analyzer"
        msg = "📑 Supervisor: Sending selected articles to Analyzer..."
    elif "evaluator" in decision_text and has_analysis and not has_evaluation:
        next_agent = "evaluator"
        msg = "⚖️ Supervisor: Assigning Evaluator Agent..."
    elif "ranker" in decision_text and has_evaluation and not has_ranked:
        next_agent = "ranker"
        msg = "📊 Supervisor: Ranking articles by novelty and impact..."
    elif "writer" in decision_text and has_ranked and not has_report:
        next_agent = "writer"
        msg = "📝 Supervisor: Compiling final Smart Medical Research Report..."
    else:
        next_agent = "researcher"
        msg = "🔎 Supervisor: Proceeding to the next appropriate agent..."

    return {"messages": [AIMessage(content=msg)], "next_agent": next_agent}

# ============ Researcher Agent ============
def researcher_agent(state: SmartMedState) -> dict:
    query = state.get("current_task", "latest medical preprints and research")
    st.info("Researcher: querying PubMed, bioRxiv, medRxiv via SerpAPI...")

    # Fetch last 7 days articles automatically
    last_week = datetime.now() - timedelta(days=7)
    query += f" after:{last_week.strftime('%Y-%m-%d')}"

    try:
        client = GoogleSearch({
            "q": query + " site:pubmed.ncbi.nlm.nih.gov OR site:biorxiv.org OR site:medrxiv.org",
            "hl": "en",
            "gl": "us",
            "num": 15,
            "api_key": SERPAPI_API_KEY
        })
        data = client.get_dict()
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Researcher failed to query SerpAPI: {e}")], "next_agent": "supervisor"}

    results = data.get("organic_results", [])
    if not results:
        return {"messages": [AIMessage(content="🔍 Researcher: No recent medical articles found.")], "next_agent": "supervisor"}

    articles = []
    for c in results:
        title = c.get("title", "Untitled")
        url = c.get("link")
        snippet = clean_text(c.get("snippet", ""))
        articles.append({"title": title, "url": url, "snippet": snippet})

    selected = articles[:10]  # Top 10 recent articles
    state["discovered_articles"] = articles
    state["selected_articles"] = selected

    msg = "🔎 Researcher: Selected latest medical articles for analysis."
    return {"messages": [AIMessage(content=msg)], "next_agent": "analyzer"}

# ============ Analyzer Agent ============
def analyzer_agent(state: SmartMedState) -> dict:
    articles = state.get("selected_articles", [])
    if not articles:
        return {"messages": [AIMessage(content="❌ Analyzer: No articles provided")], "next_agent": "supervisor"}

    text_to_analyze = "\n".join([f"{a['title']}: {a['snippet']}" for a in articles])
    prompt = f"Summarize key findings, methods, and sources from the following medical articles:\n{text_to_analyze}"
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content

    state["analysis"] = analysis
    return {"messages": [AIMessage(content=f"📑 Analyzer: Summary (truncated)\n{analysis[:1000]}")], "next_agent": "evaluator"}

# ============ Evaluator Agent ============
def evaluator_agent(state: SmartMedState) -> dict:
    analysis = state.get("analysis", "")
    prompt = f"Evaluate relevance, novelty, and clinical impact of the following medical literature:\n{analysis[:4000]}"
    response = llm.invoke([HumanMessage(content=prompt)])
    evaluation = response.content

    state["evaluation"] = evaluation
    return {"messages": [AIMessage(content=f"⚖️ Evaluator: Evaluation complete (truncated)\n{evaluation[:1000]}")], "next_agent": "ranker"}

# ============ Ranker Agent ============
def ranker_agent(state: SmartMedState) -> dict:
    articles = state.get("selected_articles", [])
    evaluation = state.get("evaluation", "")
    prompt = f"Rank these articles by novelty, relevance, and impact:\nArticles:\n{', '.join([a['title'] for a in articles])}\n\nEvaluation:\n{evaluation[:3000]}"
    response = llm.invoke([HumanMessage(content=prompt)])
    ranked_report = response.content

    state["ranked_report"] = ranked_report
    return {"messages": [AIMessage(content="📊 Ranker: Articles ranked by impact and novelty.")], "next_agent": "writer"}

# ============ Writer Agent ============
def writer_agent(state: SmartMedState) -> dict:
    articles = state.get("selected_articles", [])
    analysis = state.get("analysis", "")
    evaluation = state.get("evaluation", "")
    ranked_report = state.get("ranked_report", "")

    prompt = f"Compose a daily smart medical research digest including summaries, analysis, evaluation, and ranked articles.\nArticles:\n{', '.join([a['title'] for a in articles])}\n\nAnalysis:\n{analysis[:3000]}\nEvaluation:\n{evaluation[:3000]}\nRanked:\n{ranked_report[:3000]}"
    response = llm.invoke([HumanMessage(content=prompt)])
    final_report = response.content

    state["final_report"] = final_report
    state["task_complete"] = True
    return {"messages": [AIMessage(content="📝 Writer: Smart Medical Research Report compiled.")], "next_agent": "supervisor"}

# ============ Router ============
def router(state: SmartMedState) -> str:
    next_agent = state.get("next_agent", "supervisor")
    if next_agent == "end" or state.get("task_complete"):
        return END
    return next_agent

# ============ Build Graph ============
workflow = StateGraph(SmartMedState)
for node, func in {
    "supervisor": supervisor_agent,
    "researcher": researcher_agent,
    "analyzer": analyzer_agent,
    "evaluator": evaluator_agent,
    "ranker": ranker_agent,
    "writer": writer_agent,
}.items():
    workflow.add_node(node, func)
workflow.set_entry_point("supervisor")

for node in ["supervisor","researcher","analyzer","evaluator","ranker","writer"]:
    workflow.add_conditional_edges(node, router, {
        "supervisor":"supervisor",
        "researcher":"researcher",
        "analyzer":"analyzer",
        "evaluator":"evaluator",
        "ranker":"ranker",
        "writer":"writer",
        END: END
    })

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# ============ PDF Generator ============
def generate_pdf(topic: str, articles: List[dict], messages: List[str], final_report: str) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph(f"<b>Smart Medical Research Digest for: {topic}</b>", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Selected Articles:</b>", styles["Heading2"]))
    for a in articles:
        content.append(Paragraph(f"{a['title']} - {a['url']}", styles["Normal"]))
        content.append(Paragraph(a['snippet'], styles["BodyText"]))
        content.append(Spacer(1, 6))

    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Agent Messages / Trace</b>", styles["Heading2"]))
    for m in messages:
        content.append(Paragraph(m.replace("\n","<br/>"), styles["BodyText"]))
        content.append(Spacer(1,6))

    content.append(Spacer(1,12))
    content.append(Paragraph("<b>Final Smart Medical Research Report</b>", styles["Heading2"]))
    content.append(Paragraph(final_report.replace("\n","<br/>"), styles["BodyText"]))
    doc.build(content)
    buf.seek(0)
    return buf

# ============ Streamlit UI ============
st.set_page_config(page_title="Smart Medical Research Assistant", layout="wide")
st.title("🩺 Smart Medical Research Assistant — Multi-Agent System")

with st.sidebar:
    st.header("Settings")
    query = st.text_input("Topic or Keywords", "COVID-19 latest research")
    days = st.number_input("Fetch articles from last (days)", 7, 1, 30, 7)

if st.button("Run Smart Medical Literature Aggregation"):
    state = SmartMedState(messages=[HumanMessage(content=query)])
    state["current_task"] = query
    st.info("🚀 Starting workflow — Researcher will fetch, summarize, evaluate, rank, and generate report...")

    full_messages = []
    config = {"configurable": {"thread_id": f"smart_med_thread_{datetime.now().timestamp()}"}}

    for event in graph.stream(state, config=config):
        for value in event.values():
            if "messages" in value:
                for msg in value["messages"]:
                    full_messages.append(msg.content)
                    st.markdown(f"**{msg.content}**")
            if "final_report" in value and value["final_report"]:
                st.markdown("### 🧾 Final Smart Medical Research Report\n\n```")
                st.code(value["final_report"][:12000])

    st.success("✅ Workflow completed")

    final_report_text = state.get("final_report", "")
    if final_report_text:
        pdf_buf = generate_pdf(query, state.get("selected_articles", []), full_messages, final_report_text)
        st.download_button("📥 Download PDF Report", data=pdf_buf, file_name=f"smart_med_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf")
