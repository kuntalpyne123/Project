import streamlit as st
import os
import base64
import io
import json
import time
import pandas as pd
from datetime import datetime
from PIL import Image
from pytrends.request import TrendReq
from streamlit.web.server.websocket_headers import _get_websocket_headers

# ===========================
# LIBRARY IMPORTS
# ===========================
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

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    pass

# ===========================
# CONFIG
# ===========================
FREE_USAGE_LIMIT = 5
SHEET_NAME = "user_quotas"

st.set_page_config(
    page_title="Product IQ: Agentic Shopper",
    page_icon="🛍️",
    layout="wide"
)

# ===========================
# GOOGLE TRENDS (REAL DATA)
# ===========================
def fetch_google_trends(keywords, timeframe="today 12-m", geo=""):
    """
    REAL Google Trends data.
    No synthetic values.
    """
    try:
        pytrends = TrendReq(hl="en-US", tz=330)
        pytrends.build_payload(
            kw_list=keywords,
            timeframe=timeframe,
            geo=geo
        )
        df = pytrends.interest_over_time()
        if df.empty:
            return None
        return df.drop(columns=["isPartial"], errors="ignore")
    except Exception:
        return None

def extract_products_for_trends(product_name, research_data, limit=4):
    products = [product_name]

    for line in research_data.splitlines():
        if "Rival" in line or "vs" in line:
            parts = line.split(":")
            if len(parts) > 1:
                p = parts[1].strip()
                if p and p not in products:
                    products.append(p)
        if len(products) >= limit:
            break

    return products

# ===========================
# SESSION STATE
# ===========================
for key in ["research_data", "general_report", "product_name", "messages"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ===========================
# SIDEBAR
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
        api_key = st.secrets.get("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
        model_id = st.selectbox(
            "Model",
            ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview")
        )

    elif provider == "OpenAI (ChatGPT)":
        api_key = st.text_input("OpenAI API Key", type="password")
        model_id = st.selectbox("Model", ("gpt-5", "gpt-5-mini"))

    elif provider == "Anthropic (Claude)":
        api_key = st.text_input("Anthropic API Key", type="password")
        model_id = st.selectbox(
            "Model",
            ("claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101")
        )

    client = None
    if api_key:
        if provider == "Google Gemini":
            client = genai.Client(api_key=api_key)
        elif provider == "OpenAI (ChatGPT)":
            client = openai.OpenAI(api_key=api_key)
        elif provider == "Anthropic (Claude)":
            client = anthropic.Anthropic(api_key=api_key)

# ===========================
# UTILS
# ===========================
def search_web_duckduckgo(query):
    try:
        return "\n".join(
            f"- {r['title']}: {r['body']}"
            for r in DDGS().text(query, max_results=5)
        )
    except Exception:
        return ""

def call_llm(system, prompt, use_search=False, search_query=None, image=None):
    if not client:
        return "LLM not configured."

    final_prompt = prompt

    if use_search and search_query:
        web = search_web_duckduckgo(search_query)
        final_prompt = f"WEB DATA:\n{web}\n\n{prompt}"

    if provider == "Google Gemini":
        config = GenerateContentConfig(system_instruction=system)
        return client.models.generate_content(
            model=model_id,
            contents=[final_prompt],
            config=config
        ).text

    if provider == "OpenAI (ChatGPT)":
        return client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": final_prompt}
            ]
        ).choices[0].message.content

    if provider == "Anthropic (Claude)":
        return client.messages.create(
            model=model_id,
            system=system,
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=4000
        ).content[0].text

# ===========================
# UI
# ===========================
st.title("🛍️ The GoodBuy Guide")

product = st.text_input("Enter product name")

if st.button("🚀 Analyze") and product:
    with st.spinner("Researching..."):
        research = call_llm(
            "Senior Market Analyst",
            f"Find issues, competitors, pricing and sentiment for {product}",
            use_search=True,
            search_query=f"{product} reviews competitors pricing issues"
        )
        report = call_llm(
            "Tech Reviews Editor",
            f"Write a buying guide for {product} using:\n{research}"
        )

        st.session_state.product_name = product
        st.session_state.research_data = research
        st.session_state.general_report = report

# ===========================
# RESULTS
# ===========================
if st.session_state.general_report:
    st.subheader(f"📘 Analysis for {st.session_state.product_name}")
    st.markdown(st.session_state.general_report)

    # ===========================
    # REAL MARKET TREND CHART
    # ===========================
    st.subheader("📈 Market Demand Trend (Google Search Interest)")
    st.caption("Based on real Google Trends data. Relative interest, not sales volume.")

    products = extract_products_for_trends(
        st.session_state.product_name,
        st.session_state.research_data
    )

    with st.spinner("Fetching real-time Google Trends data..."):
        trend_df = fetch_google_trends(products)

    if trend_df is not None:
        st.line_chart(trend_df)
    else:
        st.warning("Google Trends data temporarily unavailable.")

    st.caption("ℹ️ Source: Google Trends (real-time public data)")

    with st.expander("🔍 Raw Research Data"):
        st.text_area("Research", st.session_state.research_data, height=200)
