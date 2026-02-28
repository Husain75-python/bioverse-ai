import os, io, requests
from datetime import datetime
import streamlit as st
from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ============================================================
# Load environment variables
# ============================================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM (Groq)
llm = init_chat_model("groq:llama-3.1-8b-instant")

# ============================================================
# Define State
# ============================================================
class MedState(MessagesState):
    next_agent: str = ""
    research_data: str = ""
    drug_discovery: str = ""
    analysis: str = ""
    final_report: str = ""
    current_task: str = ""
    task_complete: bool = False

# ============================================================
# Helper: Fetch from APIs
# ============================================================
def fetch_pubmed(topic: str, max_results=5):
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": topic, "retmax": max_results, "retmode": "json", "sort": "pub+date"}
        res = requests.get(url, params=params, timeout=15).json()
        ids = res.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No PubMed papers found."
        summaries = []
        for pid in ids:
            sdata = requests.get(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": pid, "retmode": "json"},
                timeout=10
            ).json()
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
    try:
        url = "https://api.openalex.org/works"
        params = {"search": topic, "per-page": max_results}
        res = requests.get(url, params=params, timeout=15).json()
        results = res.get("results", [])
        if not results:
            return "No OpenAlex papers found."
        return "\n\n".join([f"{r['display_name']} ({r.get('publication_year','')})\n{r['id']}" for r in results])
    except Exception as e:
        return f"Error fetching OpenAlex: {e}"

def fetch_clinical_trials(topic: str, max_results=5):
    """Fetch ongoing clinical trials from 3 sources: ClinicalTrials.gov, WHO ICTRP, NCI."""
    trials = []

    # 1. ClinicalTrials.gov
    try:
        url = "https://clinicaltrials.gov/api/query/study_fields"
        params = {
            "expr": topic,
            "fields": "BriefTitle,Phase,OverallStatus,LocationCity,LocationCountry,StudyURL",
            "min_rnk": 1,
            "max_rnk": max_results,
            "fmt": "json"
        }
        res = requests.get(url, params=params, timeout=15).json()
        studies = res.get("StudyFieldsResponse", {}).get("StudyFields", [])
        if studies:
            ctgov_trials = [
                f"{s.get('BriefTitle',[''])[0]} | {s.get('Phase',[''])[0]} | {s.get('OverallStatus',[''])[0]} | {s.get('LocationCity',[''])[0]}, {s.get('LocationCountry',[''])[0]}\n{s.get('StudyURL',[''])[0]}"
                for s in studies
            ]
            trials.append("=== ClinicalTrials.gov ===\n" + "\n".join(ctgov_trials))
        else:
            trials.append("=== ClinicalTrials.gov ===\nNo trials found.")
    except Exception as e:
        trials.append(f"=== ClinicalTrials.gov ===\nError: {e}")

    # 2. WHO ICTRP (if API available)
    try:
        ictrp_url = f"https://trialsearch.who.int/TrialSearchAPI/search?query={topic}&page=1&pageSize={max_results}"
        res = requests.get(ictrp_url, timeout=15).json()
        records = res.get("records", [])
        if records:
            ictrp_trials = [
                f"{r.get('scientificTitle','')} | Status: {r.get('recruitmentStatus','')} | URL: {r.get('url','')}"
                for r in records
            ]
            trials.append("=== WHO ICTRP ===\n" + "\n".join(ictrp_trials))
        else:
            trials.append("=== WHO ICTRP ===\nNo trials found.")
    except Exception as e:
        trials.append(f"=== WHO ICTRP ===\nError: {e}")

    # 3. NCI Cancer Trials
    try:
        nci_url = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"
        params = {"disease": topic, "size": max_results}
        res = requests.get(nci_url, params=params, timeout=15).json()
        trials_data = res.get("trials", [])
        if trials_data:
            nci_trials = [
                f"{t.get('title','')} | Status: {t.get('status','')} | Phase: {t.get('phase','')} | URL: {t.get('url','')}"
                for t in trials_data
            ]
            trials.append("=== NCI Cancer Trials ===\n" + "\n".join(nci_trials))
        else:
            trials.append("=== NCI Cancer Trials ===\nNo trials found.")
    except Exception as e:
        trials.append(f"=== NCI Cancer Trials ===\nError: {e}")

    return "\n\n".join(trials)

# ============================================================
# Agents
# ============================================================

# Supervisor
def supervisor_agent(state: MedState):
    has_research = bool(state.get("research_data", ""))
    has_drug_discovery = bool(state.get("drug_discovery", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))

    if not has_research:
        next_agent = "researcher"
        msg = "🔬 Assigning task to Researcher..."
    elif not has_drug_discovery:
        next_agent = "drug_discovery"
        msg = "💊 Assigning task to Drug Discovery Agent..."
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

# Researcher
def researcher_agent(state: dict) -> dict:
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
    msg = f"🧠 Researcher Agent:\nCompiled verified medical papers for **{topic}**."
    return {"messages": [AIMessage(content=msg)],
            "research_data": research_text,
            "next_agent": "supervisor"}

# Drug Discovery Agent
def drug_discovery_agent(state: MedState):
    topic = state.get("current_task", "")
    st.info(f"💊 Searching for new medications and ongoing trials for '{topic}'...")

    search_query = f"{topic} new drug OR therapeutic OR treatment discovery"
    pubmed_drugs = fetch_pubmed(search_query)
    semantic_drugs = fetch_semantic_scholar(search_query)
    openalex_drugs = fetch_openalex(search_query)
    clinical_trials = fetch_clinical_trials(topic)

    combined_drug_data = f"""
=== PUBMED DRUG DISCOVERY ===
{pubmed_drugs}

=== SEMANTIC SCHOLAR DRUG DISCOVERY ===
{semantic_drugs}

=== OPENALEX DRUG DISCOVERY ===
{openalex_drugs}

=== ONGOING CLINICAL TRIALS ===
{clinical_trials}
"""
    prompt = f"""
You are a pharmacology research expert.
Summarize:
1. New or experimental drugs related to '{topic}'
2. Mechanisms of action
3. Clinical trial outcomes & ongoing studies
4. Future drug development directions
5. Key pharmaceutical companies or research groups

Data:
{combined_drug_data[:3000]}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    drug_summary = response.content
    msg = f"💊 Drug Discovery Agent:\n{drug_summary[:800]}"
    return {"messages": [AIMessage(content=msg)],
            "drug_discovery": drug_summary,
            "next_agent": "supervisor"}

# Analyst
def analyst_agent(state: MedState):
    topic = state.get("current_task")
    research_data = state.get("research_data", "")
    drug_data = state.get("drug_discovery", "")
    prompt = f"""You are a biomedical research analyst.
Analyze the combined research and drug data on '{topic}'.
Provide:
1. Major findings and evidence strength
2. Key trends across studies
3. Gaps and limitations
4. Future research directions
Data:
{research_data[:1500]}
Drug Discovery:
{drug_data[:1500]}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content
    msg = f"📊 Analyst:\n{analysis[:800]}"
    return {"messages": [AIMessage(content=msg)],
            "analysis": analysis,
            "next_agent": "supervisor"}

# Writer
def writer_agent(state: MedState):
    topic = state.get("current_task")
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")
    drug_data = state.get("drug_discovery", "")
    prompt = f"""Write a comprehensive medical literature review on '{topic}'.
Sections:
1. Executive Summary
2. Key Findings
3. New Drug Discoveries & Ongoing Trials
4. Trends & Gaps
5. Recommendations for clinicians/researchers
6. References
Base your report on:
Research Data: {research_data[:1000]}
Drug Data: {drug_data[:1000]}
Analysis: {analysis[:1000]}
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
    return {
        "messages": [AIMessage(content="✍️ Writer: Final report completed!")],
        "final_report": final_report,
        "task_complete": True,
        "next_agent": "supervisor"
    }

# ============================================================
# Router + Workflow Graph
# ============================================================
def router(state: MedState) -> Literal["supervisor", "researcher", "drug_discovery", "analyst", "writer", "END"]:
    if state.get("task_complete", False):
        return END
    return state.get("next_agent", "supervisor")

workflow = StateGraph(MedState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("drug_discovery", drug_discovery_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)
workflow.set_entry_point("supervisor")

for node in ["supervisor", "researcher", "drug_discovery", "analyst", "writer"]:
    workflow.add_conditional_edges(
        node, router,
        {
            "supervisor": "supervisor",
            "researcher": "researcher",
            "drug_discovery": "drug_discovery",
            "analyst": "analyst",
            "writer": "writer",
            END: END
        }
    )

graph = workflow.compile(checkpointer=MemorySaver())

# ============================================================
# PDF Generator
# ============================================================
def generate_pdf(topic, messages, final_report):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = [
        Paragraph(f"<b>Medical Research Review</b>", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"<b>Topic:</b> {topic}", styles['Heading2']),
        Spacer(1, 12)
    ]
    for m in messages:
        content.append(Paragraph(m.replace("\n", "<br/>"), styles['BodyText']))
        content.append(Spacer(1, 8))
    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Final Report:</b>", styles['Heading2']))
    content.append(Paragraph(final_report.replace("\n", "<br/>"), styles['BodyText']))
    doc.build(content)
    buffer.seek(0)
    return buffer

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Medical Literature Aggregator", layout="wide", page_icon="🧬")
st.title("🧬 BioVerse AI")
st.write("Enter a medical topic, and this system will fetch, analyze, discover new drugs, check ongoing trials, and summarize research automatically.")

topic = st.text_input("Enter medical topic (e.g., 'Pneumonia Disease treatment')")

if st.button("Start Research") and topic.strip():
    state = MedState(messages=[HumanMessage(content=topic)], current_task=topic)
    st.info("🚀 Starting AI Literature Review...")
    full_messages = []
    config = {"configurable": {"thread_id": f"thread_{datetime.now().timestamp()}"}}

    for event in graph.stream(state, config=config):
        for value in event:
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
