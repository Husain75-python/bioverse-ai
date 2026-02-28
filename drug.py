# bioverse_ai_streamlit.py
import os
import io
import time
import json
import hashlib
import tempfile
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from gtts import gTTS

# LLM imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

# Optional MySQL
try:
    import mysql.connector 
except Exception:
    mysql = None

# Optional networkx / plotting
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAVE_NX = True
except Exception:
    HAVE_NX = False

# -------------------- Load config --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# MySQL config
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))

# -------------------- Initialize LLM --------------------
try:
    llm = init_chat_model("groq:llama-3.1-8b-instant")
except Exception as e:
    llm = None
    st.warning(f"LLM init issue: {e}")

# -------------------- Utilities --------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def mysql_connect():
    """Return a mysql connector or None if not configured."""
    if not (MYSQL_HOST and MYSQL_USER and MYSQL_PASSWORD and MYSQL_DB and mysql):
        return None
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            autocommit=True
        )
        return conn
    except Exception as e:
        st.error(f"MySQL connection failed: {e}")
        return None

# create minimal schema if MySQL available
def ensure_schema():
    conn = mysql_connect()
    if not conn:
        return
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) UNIQUE,
        name VARCHAR(255),
        password_hash VARCHAR(255),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        topic TEXT,
        blood_group VARCHAR(10),
        `condition` TEXT,
        result_summary LONGTEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    ) ENGINE=InnoDB;
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        topic TEXT,
        pdf LONGBLOB,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    ) ENGINE=InnoDB;
    """)
    cur.close()
    conn.close()

# -------------------- Safe LLM invocation --------------------
def safe_invoke_llm(messages: List[HumanMessage], fallback="(LLM error)") -> str:
    if llm is None:
        return fallback
    try:
        resp = llm.invoke(messages)
        return getattr(resp, "content", str(resp))
    except Exception as e:
        return f"{fallback}: {e}"

# -------------------- Request helper --------------------
def requests_get_with_retry(url, params=None, headers=None, timeout=12, retries=2, backoff=1.0):
    last = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** i))
    raise last

# -------------------- MySQL CRUD helpers --------------------
def register_user(email: str, name: str, password: str) -> bool:
    conn = mysql_connect()
    if not conn:
        st.warning("MySQL not configured. Registration disabled.")
        return False
    cur = conn.cursor()
    try:
        password_hash = hash_password(password)
        cur.execute("INSERT INTO users (email, name, password_hash) VALUES (%s, %s, %s)", (email, name, password_hash))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Registration failed: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def login_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    conn = mysql_connect()
    if not conn:
        st.warning("MySQL not configured; login disabled for persistent users. Continue as guest.")
        return None
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        if user and user["password_hash"] == hash_password(password):
            return {"id": user["id"], "email": user["email"], "name": user["name"]}
        return None
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def save_query_to_mysql(user_id: Optional[int], topic: str, blood_group: str, condition: str, result_summary: str):
    conn = mysql_connect()
    if not conn:
        return False
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO queries (user_id, topic, blood_group, condition, result_summary) VALUES (%s,%s,%s,%s,%s)",
            (user_id, topic, blood_group, condition, result_summary)
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Save query failed: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def save_report_to_mysql(user_id: Optional[int], topic: str, pdf_bytes: bytes):
    conn = mysql_connect()
    if not conn:
        return False
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO reports (user_id, topic, pdf) VALUES (%s,%s,%s)",
            (user_id, topic, pdf_bytes)
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Save report failed: {e}")
        return False
    finally:
        cur.close()
        conn.close()
