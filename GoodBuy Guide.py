import streamlit as st
import os
import base64
import io
import json
from datetime import datetime
from PIL import Image
from streamlit.web.server.websocket_headers import _get_websocket_headers

# >>> ADDED (REAL DATA ONLY)
from pytrends.request import TrendReq
import pandas as pd

# --- LIBRARY IMPORTS ---
try:
    from google import genai
    from google.genai.types import GenerateContentConfig, Tool, GoogleSearch
except ImportError:
    pass
try:
    import openai
except ImportError:
    pass
try:
    import anthropic
except ImportError:
    pass
try:
    from duckduckgo_search import DDGS
except ImportError:
    pass

# --- DATABASE IMPORTS (GOOGLE SHEETS) ---
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    pass

# ===========================
# 1. PERSISTENT USAGE TRACKING
# ===========================
FREE_USAGE_LIMIT = 5
SHEET_NAME = "user_quotas"

def get_remote_ip():
    try:
        headers = _get_websocket_headers()
        if headers and "X-Forwarded-For" in headers:
            return headers["X-Forwarded-For"].split(",")[0]
        return "LOCALHOST_DEV_MACHINE"
    except Exception:
        return "UNKNOWN_CLIENT"

def get_google_sheet_client():
    try:
        if "gcp_service_account" in st.secrets:
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(
                dict(st.secrets["gcp_service_account"]), scope
            )
            return gspread.authorize(creds)
        return None
    except Exception:
        return None

def init_db():
    client = get_google_sheet_client()
    if client:
        try:
            sheet = client.open(SHEET_NAME).sheet1
            if not sheet.get_all_values():
                sheet.append_row(["ip_address", "count", "last_access"])
        except Exception:
            pass

def get_usage_count(ip):
    client = get_google_sheet_client()
    if not client:
        return 0
    try:
        sheet = client.open(SHEET_NAME).sheet1
        cell = sheet.find(ip)
        if cell:
            return int(sheet.cell(cell.row, 2).value)
        return 0
    except Exception:
        return 0

def increment_usage(ip):
    client = get_google_sheet_client()
    if not client:
        return 1
    try:
        sheet = client.open(SHEET_NAME).sheet1
        now = datetime.now().isoformat()
        cell = sheet.find(ip)
        if cell:
            count = int(sheet.cell(cell.row, 2).value) + 1
            sheet.update_cell(cell.row, 2, count)
            sheet.update_cell(cell.row, 3, now)
            return count
        sheet.append_row([ip, 1, now])
        return 1
    except Exception:
        return 1

init_db()

# ===========================
# 2. CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Product IQ: Agentic Shopper",
    page_icon="🛍️",
    layout="wide"
)

if "research_data" not in st.session_state:
    st.session_state.research_data = None
if "general_report" not in st.session_state:
    st.session_state.general_report = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "product_name" not in st.session_state:
    st.session_state.product_name = ""

# ===========================
# 3. SIDEBAR
# ===========================
with st.sidebar:
    st.header("⚙️ Engine Settings")
    provider = st.radio(
        "Select AI Provider:",
        ("Google Gemini", "OpenAI (ChatGPT)", "Anthropic (Claude)"),
        index=0
    )

    api_key = None
    model_id = None

    if provider == "Google Gemini":
        api_key = st.secrets.get("GEMINI_API_KEY") or st.text_input(
            "Gemini API Key", type="password"
