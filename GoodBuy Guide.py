import streamlit as st
import os
import base64
import io
import requests
from datetime import datetime
from PIL import Image
from streamlit.web.server.websocket_headers import _get_websocket_headers

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

# ===========================
# 1. PERSISTENT USAGE TRACKING (SUPABASE)
# ===========================

SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY")
FREE_USAGE_LIMIT = 5

def get_remote_ip():
    try:
        headers = _get_websocket_headers()
        if headers and "X-Forwarded-For" in headers:
            return headers["X-Forwarded-For"].split(",")[0]
        return "LOCALHOST_DEV_MACHINE"
    except Exception:
        return "UNKNOWN_CLIENT"

def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

def get_usage_count(ip):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/usage?ip_address=eq.{ip}&select=count",
        headers=supabase_headers()
    )
    data = r.json()
    return data[0]["count"] if data else 0

def increment_usage(ip):
    current = get_usage_count(ip)
    payload = {
        "ip_address": ip,
        "count": current + 1,
        "last_access": datetime.utcnow().isoformat()
    }
    requests.post(
        f"{SUPABASE_URL}/rest/v1/usage",
        headers=supabase_headers(),
        json=payload
    )
    return payload["count"]

# ===========================
# 2. CONFIGURATION
# ===========================
st.set_page_config(page_title="Product IQ: Agentic Shopper", page_icon="🛍️", layout="wide")

if "research_data" not in st.session_state: st.session_state.research_data = None
if "general_report" not in st.session_state: st.session_state.general_report = None
if "messages" not in st.session_state: st.session_state.messages = []
if "product_name" not in st.session_state: st.session_state.product_name = ""

# ===========================
# 3. SIDEBAR & SETTINGS
# ===========================
with st.sidebar:
    st.header("⚙️ Engine Settings")
    provider = st.radio("Select AI Provider:", ("Google Gemini", "OpenAI (ChatGPT)", "Anthropic (Claude)"))

    api_key = None
    model_id = None
    using_free_key = False

    # ---------- GEMINI ----------
    if provider == "Google Gemini":
        key_source = st.radio("API Key Source:", ("Use Free Default Key", "Enter My Own Key"))

        if key_source == "Use Free Default Key":
            using_free_key = True
            api_key = st.secrets.get("GEMINI_API_KEY")

            user_ip = get_remote_ip()
            used = get_usage_count(user_ip)
            st.progress(min(used / FREE_USAGE_LIMIT, 1.0), text=f"{used}/{FREE_USAGE_LIMIT} Free Uses")

            model_choice = st.selectbox(
                "Select Gemini Model:",
                ("2.5 Flash", "2.5 Pro")
            )

            model_id = "gemini-2.5-flash" if "Flash" in model_choice else "gemini-2.5-pro"

        else:
            api_key = st.text_input("Enter Gemini API Key", type="password")
            model_choice = st.selectbox(
                "Select Gemini Model:",
                ("2.5 Flash", "2.5 Pro", "3.0 Pro")
            )
            model_id = (
                "gemini-2.5-flash" if "Flash" in model_choice else
                "gemini-2.5-pro" if "2.5" in model_choice else
                "gemini-3-pro-preview"
            )

    # ---------- OPENAI ----------
    elif provider == "OpenAI (ChatGPT)":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_id = st.selectbox("Select Model:", ("gpt-5.2", "gpt-5.1", "gpt-5-mini"))

    # ---------- ANTHROPIC (FIXED LABELS) ----------
    elif provider == "Anthropic (Claude)":
        api_key = st.text_input("Enter Anthropic API Key", type="password")

        anthropic_models = {
            "Sonnet 4.5": "claude-sonnet-4-5-20250929",
            "Haiku 4.5": "claude-haiku-4-5-20251001",
            "Opus 4.5": "claude-opus-4-5-20251101",
        }

        choice = st.selectbox("Select Model:", anthropic_models.keys())
        model_id = anthropic_models[choice]

    client = None
    if api_key:
        if provider == "Google Gemini":
            client = genai.Client(api_key=api_key)
        elif provider == "OpenAI (ChatGPT)":
            client = openai.OpenAI(api_key=api_key)
        elif provider == "Anthropic (Claude)":
            client = anthropic.Anthropic(api_key=api_key)

# ===========================
# 4. UTILS (UNCHANGED)
# ===========================
def search_web_duckduckgo(query, max_results=5):
    try:
        results = DDGS().text(query, max_results=max_results)
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search failed: {e}"

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# ===========================
# 5. LLM CALLER (UNCHANGED)
# ===========================
def call_llm(system_instruction, user_prompt, use_search=False, search_query=None, image_data=None):
    if not client:
        return "Error: Client not initialized."

    if provider == "Google Gemini":
        tools = [Tool(google_search=GoogleSearch())] if use_search else None
        config = GenerateContentConfig(
            tools=tools,
            system_instruction=system_instruction,
            temperature=0.1
        )
        contents = [user_prompt]
        if image_data:
            contents.append(image_data)
        return client.models.generate_content(model=model_id, contents=contents, config=config).text

    final_prompt = user_prompt
    if use_search and search_query:
        web_data = search_web_duckduckgo(search_query)
        final_prompt = f"WEB CONTEXT:\n{web_data}\n\nUSER QUERY:\n{user_prompt}"

    if provider == "OpenAI (ChatGPT)":
        return client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": final_prompt}
            ]
        ).choices[0].message.content

    if provider == "Anthropic (Claude)":
        return client.messages.create(
            model=model_id,
            system=system_instruction,
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=4000
        ).content[0].text

# ===========================
# 6. USAGE INCREMENT (PERSISTENT)
# ===========================
def handle_free_usage():
    if using_free_key:
        ip = get_remote_ip()
        if get_usage_count(ip) >= FREE_USAGE_LIMIT:
            st.error("🚫 Free quota exhausted.")
            st.stop()
        increment_usage(ip)
