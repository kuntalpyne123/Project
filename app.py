import streamlit as st
import os
import base64
import io
import json
import re
import pandas as pd
import altair as alt  # NEW: For better charts
from datetime import datetime
from PIL import Image

# Handle specialized Streamlit import safely
try:
    from streamlit.web.server.websocket_headers import _get_websocket_headers
except ImportError:
    pass

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

# --- DATABASE IMPORTS ---
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    pass

# ===========================
# 1. PERSISTENT USAGE TRACKING (CLOUD BASED)
# ===========================
FREE_USAGE_LIMIT = 5
SHEET_NAME = "user_quotas"

def get_remote_ip():
    try:
        headers = _get_websocket_headers()
        if headers and "X-Forwarded-For" in headers: return headers["X-Forwarded-For"].split(",")[0]
        return "LOCALHOST_DEV_MACHINE"
    except Exception: return "UNKNOWN_CLIENT"

def get_google_sheet_client():
    """Authenticates with Google Sheets using Streamlit Secrets."""
    try:
        if "gcp_service_account" in st.secrets:
            # Create a credential object from the secrets dictionary
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
            return gspread.authorize(creds)
        else:
            return None
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None

def init_db():
    """Checks if the Google Sheet exists, creates headers if empty."""
    client = get_google_sheet_client()
    if client:
        try:
            try:
                sheet = client.open(SHEET_NAME).sheet1
            except gspread.SpreadsheetNotFound:
                st.warning(f"Cloud DB Error: Please create a Google Sheet named '{SHEET_NAME}' and share it with your service account email.")
                return
            
            # Check if headers exist
            if not sheet.get_all_values():
                sheet.append_row(["ip_address", "count", "last_access"])
        except Exception as e:
            pass # Fail silently in UI, log in console

def get_usage_count(ip):
    client = get_google_sheet_client()
    if not client:
        return 0 # Fallback if DB not configured
    
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
        return 1 # Fallback
        
    try:
        sheet = client.open(SHEET_NAME).sheet1
        current_time = datetime.now().isoformat()
        
        try:
            cell = sheet.find(ip)
        except:
            cell = None

        if cell:
            # Update existing user
            current_count = int(sheet.cell(cell.row, 2).value)
            new_count = current_count + 1
            sheet.update_cell(cell.row, 2, new_count)
            sheet.update_cell(cell.row, 3, current_time)
            return new_count
        else:
            # Add new user
            sheet.append_row([ip, 1, current_time])
            return 1
    except Exception as e:
        st.error(f"DB Write Error: {e}")
        return 1

# Initialize Cloud DB
init_db()

# ===========================
# 2. CONFIGURATION
# ===========================
st.set_page_config(page_title="Product IQ: Agentic Shopper", page_icon="🛍️", layout="wide")
st.markdown("""
<style>
    .report-box { border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

if "research_data" not in st.session_state: st.session_state.research_data = None
if "general_report" not in st.session_state: st.session_state.general_report = None
if "chart_data" not in st.session_state: st.session_state.chart_data = None # Renamed to handle both charts
if "messages" not in st.session_state: st.session_state.messages = []
if "product_name" not in st.session_state: st.session_state.product_name = ""

# ===========================
# 3. SIDEBAR & SETTINGS
# ===========================
with st.sidebar:
    st.header("⚙️ Engine Settings")
    provider = st.radio("Select AI Provider:", ("Google Gemini", "OpenAI (ChatGPT)", "Anthropic (Claude)"), index=0)

    api_key = None
    model_id = None
    using_free_key = False 
    
    # --- GOOGLE GEMINI SETTINGS ---
    if provider == "Google Gemini":
        st.info("⚡ Native Search Grounding (Most Accurate)")
        key_source = st.radio("API Key Source:", ("Use Free Default Key", "Custom Key to access GEMINI PRO models"))
        
        if key_source == "Use Free Default Key":
            using_free_key = True 
            user_ip = get_remote_ip()
            current_usage = get_usage_count(user_ip)
            st.caption(f"🔒 ID: ...{user_ip[-4:] if len(user_ip)>4 else user_ip}") 
            usage_left = FREE_USAGE_LIMIT - current_usage
            st.progress(min(current_usage / FREE_USAGE_LIMIT, 1.0), text=f"Quota: {current_usage}/{FREE_USAGE_LIMIT} used")
            if usage_left <= 0: st.error("🚫 Quota Exceeded.")  
            
            # --- FIX FOR RENDER DEPLOYMENT ---
            # 1. Try Environment Variable (Render)
            env_key = os.environ.get("GEMINI_API_KEY")
            if env_key:
                api_key = env_key
            else:
                # 2. Try Local Secrets (Fallback)
                try:
                    if "GEMINI_API_KEY" in st.secrets: api_key = st.secrets["GEMINI_API_KEY"]
                except: st.error("Secrets not available locally.")
        else:
            api_key = st.text_input("Enter Gemini API Key", type="password")
        
        if using_free_key:
            gemini_options = ("2.5 Flash", "3 Flash Preview")
        else:
            gemini_options = ("2.5 Flash", "2.5 Pro", "3 Flash (Latest)", "3 Pro (Most Powerful)")
            
        model_choice = st.selectbox("Select Gemini Model:", gemini_options)
        
        if "2.5 Flash" in model_choice: model_id = "gemini-2.5-flash"
        elif "2.5 Pro" in model_choice: model_id = "gemini-2.5-pro"
        elif "3 Flash" in model_choice: model_id = "gemini-2.5-flash" # Fallback mapping
        else: model_id = "gemini-2.0-pro-exp-02-05" # Example ID

    # --- OPENAI SETTINGS ---
    elif provider == "OpenAI (ChatGPT)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_id = st.selectbox("Select Model:", ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"))

    # --- ANTHROPIC SETTINGS ---
    elif provider == "Anthropic (Claude)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter Anthropic API Key", type="password")
        
        anthropic_models = {
            "Sonnet 3.5": "claude-3-5-sonnet-20240620",
            "Haiku 3": "claude-3-haiku-20240307",
            "Opus 3": "claude-3-opus-20240229"
        }
        
        selected_display_name = st.selectbox("Select Model:", list(anthropic_models.keys()))
        model_id = anthropic_models[selected_display_name]

    client = None
    if api_key:
        if provider == "Google Gemini": client = genai.Client(api_key=api_key)
        elif provider == "OpenAI (ChatGPT)": client = openai.OpenAI(api_key=api_key)
        elif provider == "Anthropic (Claude)": client = anthropic.Anthropic(api_key=api_key)

# ===========================
# 4. UTILS
# ===========================
def search_web_duckduckgo(query, max_results=5):
    try:
        results = DDGS().text(query, max_results=max_results)
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e: return f"Search failed: {str(e)}"

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def call_llm(system_instruction, user_prompt, use_search=False, search_query=None, image_data=None):
    if not client: return "Error: Client not initialized."
    
    # --- GEMINI ---
    if provider == "Google Gemini":
        tools = [Tool(google_search=GoogleSearch())] if use_search else None
        # KEEPING TEMP AT 0.1 for Reliability
        config = GenerateContentConfig(tools=tools, system_instruction=system_instruction, temperature=0.1)
        contents = [user_prompt]
        if image_data: contents.append(image_data)
        try: return client.models.generate_content(model=model_id, contents=contents, config=config).text
        except Exception as e: return f"Gemini Error: {e}"

    # --- OTHERS (WEB BRIDGE) ---
    final_prompt = user_prompt
    if use_search and search_query:
        with st.spinner(f"🕵️ Searching web for {provider}..."):
            web_data = search_web_duckduckgo(search_query)
            final_prompt = f"WEB CONTEXT:\n{web_data}\n\nUSER QUERY:\n{user_prompt}"

    # --- OPENAI ---
    if provider == "OpenAI (ChatGPT)":
        content_payload = [{"type": "text", "text": final_prompt}]
        if image_data:
            buffered = io.BytesIO()
            image_data.save(buffered, format="JPEG")
            img_str = encode_image(buffered.getvalue())
            content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})
        messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": content_payload}]
        return client.chat.completions.create(model=model_id, messages=messages).choices[0].message.content

    # --- ANTHROPIC ---
    elif provider == "Anthropic (Claude)":
        content_payload = [{"type": "text", "text": final_prompt}]
        if image_data:
            buffered = io.BytesIO()
            image_data.save(buffered, format="JPEG")
            img_str = encode_image(buffered.getvalue())
            content_payload.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_str}})
        return client.messages.create(model=model_id, system=system_instruction, messages=[{"role": "user", "content": content_payload}], max_tokens=4000).content[0].text

# ===========================
# 5. INTELLIGENT AGENTS
# ===========================

# --- VISION AGENT ---
def identify_product_from_image(image):
    instruction = "You are a Tech Hardware Forensic Expert."
    prompt = """
    Identify this product.
    INTERNAL DEBATE:
    1. Does it look like a budget phone or a flagship? Check materials.
    2. IF UNSURE: Admit ambiguity.
    OUTPUT RULES:
    1. If ambiguous (e.g. S24 vs S25), output STRICTLY: "Product A vs Product B".
    2. Otherwise, return the exact Product Name.
    """
    return call_llm(instruction, prompt, image_data=image)

# --- RESEARCHER AGENT ---
RESEARCHER_INSTRUCTION = """
ROLE: Senior Market Intelligence Analyst.
GOAL: Compile a detailed, raw evidence dossier on {product_name}.
RULE: DO NOT SUMMARIZE. I need specific numbers, quotes, and technical details.

MANDATORY INTELLIGENCE GATHERING:
1.  **Pricing Forensics:**
    - What is the official MSRP?
    - What is the current "Street Price"?

2.  **Broader Market Context (CRITICAL):**
    - Do NOT just look at Samsung and Apple.
    - Identify the top 5 players in this specific category (e.g., Pixel, Xiaomi, OnePlus if phones; Sony, Bose if audio).
    - Find specific SALES VOLUMES, SHIPMENT NUMBERS, or MARKET SHARE percentages for 2024/2025.

3.  **Competitive Landscape:**
    - Identify 3 specific rivals.
    - Find the "Number of Reviews" on major platforms for all of them.

4.  **Technical Deep Dive & Feature Scoring:**
    - Look for reviews discussing: Camera, Battery, Performance, and Value.
    - Note the general sentiment (e.g., "Camera is better than iPhone but video is worse").

OUTPUT: A dense, detailed, unformatted text file with all these facts.
"""

def run_research(product_name):
    instruction = RESEARCHER_INSTRUCTION.format(product_name=product_name)
    prompt = f"""
    CONDUCT A DEEP-DIVE INVESTIGATION ON: {product_name}
    
    1. FIND precise pricing data.
    2. [IMPORTANT] FIND GLOBAL SALES DATA or MARKET SHARE DATA for {product_name} and at least 4-5 competitors (e.g. Xiaomi, Oppo, Vivo, Google, etc).
    3. If sales data is missing, find "Total Review Counts" on Amazon/BestBuy as a popularity proxy.
    4. Collect sentiment data on: Camera, Battery, Performance, Value.
    
    Provide the RAW DATA now.
    """
    # Updated query to be broader
    search_query = f"{product_name} sales volume market share 2024 2025 vs competitors shipments statistics top 5 brands market analysis"
    return call_llm(instruction, prompt, use_search=True, search_query=search_query)

# --- DATA ANALYST AGENT (MODIFIED FOR 2 CHARTS) ---
def analyze_data_for_charts(product_name, research_data):
    """Parses text to extract numbers for TWO distinct charts."""
    instruction = "You are a Data Analyst. Convert unstructured text into strict JSON."
    prompt = f"""
    Analyze the following research data for {product_name}:
    {research_data}

    TASK: Create JSON data for two separate charts.

    --- CHART 1: MARKET PRESENCE (Vertical Bar Chart) ---
    Rules for 'market_stats':
    1.  **PRIORITY 1:** SALES DATA (e.g. "10 Million Units Sold", "$5B Revenue").
    2.  **PRIORITY 2:** MARKET SHARE % (e.g. "22% Market Share").
    3.  **STRICT RULE:** Do NOT use "Shipments" unless it is the only data available. Do NOT use "Review Counts".
    4.  **Breadth:** Include {product_name} AND at least 3-4 other major competitors found in the text.
    5.  **Label:** "metric_name" should describe what you found (e.g., "Est. Annual Sales (Units)").

    --- CHART 2: FEATURE SCORECARD (Horizontal Grouped Bar Chart) ---
    Rules for 'feature_scores':
    1.  Based on the sentiment in the text, assign a score from 1 (Poor) to 5 (Excellent).
    2.  Categories: "Camera", "Battery", "Performance", "Value", "Durability".
    3.  Include {product_name} and the top 2-3 competitors.

    OUTPUT FORMAT (STRICT JSON):
    {{
        "market_stats": {{
            "metric_name": "Global Market Share 2024 (%)",
            "data": {{
                "{product_name}": 20,
                "Competitor A": 18,
                "Competitor B": 12,
                "Competitor C": 8
            }}
        }},
        "feature_scores": [
            {{"product": "{product_name}", "category": "Camera", "score": 5}},
            {{"product": "{product_name}", "category": "Battery", "score": 4}},
            {{"product": "Competitor A", "category": "Camera", "score": 4}},
            {{"product": "Competitor A", "category": "Battery", "score": 5}}
            // ... repeat for others
        ]
    }}
    """
    response = call_llm(instruction, prompt, use_search=False)
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return None

# --- EDITOR AGENT ---
EDITOR_INSTRUCTION = """
ROLE: Lead Reviews Editor at a Top Tech Publication.
TONE: Authoritative, Professional, Nuanced.
GOAL: Write the definitive Buying Guide for {product_name}.

INPUT DATA: Use the Researcher's raw notes.

STRICT WRITING RULES:
1.  **No "Form Filling":** Write a flowing article with H2 headers.
2.  **Table Formatting:** The Comparison Matrix must be a properly formatted Markdown table.
3.  **Ambiguity Logic:** IF (and ONLY IF) there is ambiguity (e.g., "S24 vs S25"), include a specific section clarifying it.
4.  **Detail Level:** Be specific (e.g., "14 hours battery", not "Good battery").

--- REPORT SKELETON ---

# The Definitive Review: {product_name}

### 🏆 The Executive Verdict
(3-4 sentence distinct paragraph. Is it a buy? Who is it for?)

**Quick Ratings:**
* **Reliability:** (X/5)
* **Value:** (X/5)
* **Future Proofing:** (X/5)

---

### 📉 Market Analysis & Pricing
(Discuss the current street price vs MSRP. Is it a good time to buy?)

---

### ⚔️ The Competition Matrix (MANDATORY)

| Product | Price | The "Win" (Pro) | The "Loss" (Con) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **{product_name}** | $X | ... | ... | ... |
| **(Rival 1)** | $Y | ... | ... | ... |
| **(Rival 2)** | $Z | ... | ... | ... |
| **(Rival 3)** | $A | ... | ... | ... |

*(Add a paragraph below the table analyzing these choices).*

---

### 🕵️ The "Hidden" Truths (Long-Term Ownership)
(Detail wear-and-tear, Reddit complaints, maintenance costs.)

---

### 📝 Final Buying Advice
* **✅ Buy it if:** (Scenario A), (Scenario B).
* **❌ Skip it if:** (Scenario C), (Scenario D).
"""

def generate_report(product_name, research_data):
    instruction = EDITOR_INSTRUCTION.format(product_name=product_name)
    prompt = f"Research Data:\n{research_data}\n\nGenerate Guide with the Comparison Table."
    return call_llm(instruction, prompt)

# --- PERSONALIZER AGENT ---
PERSONALIZER_INSTRUCTION = """
ROLE: Sales Engineer.
GOAL: Re-evaluate {product_name} for the USER'S PROFILE.

INSTRUCTIONS:
1. **Be Detailed:** Write a consultation letter (200-300 words).
2. **Structure:**
   - "The Fit Check"
   - "Day-in-the-Life Simulation"
   - "The Hard Truth"
   - "Final Verdict"

OUTPUT: A detailed, empathetic, and logic-driven letter.
"""

def generate_personal_rec(product_name, research_data, user_profile):
    instruction = PERSONALIZER_INSTRUCTION.format(product_name=product_name)
    prompt = f"Research Data: {research_data}\nUser Profile: {user_profile}\nGenerate a detailed personalized recommendation letter."
    return call_llm(instruction, prompt)

# ===========================
# 6. APP INTERFACE
# ===========================
st.title("🛍️ The GoodBuy Guide")
st.caption(f"Powered by **{provider}**")

# --- PHASE 1: INPUT ---
with st.container(border=True):
    col_input, col_attach, col_run = st.columns([0.85, 0.05, 0.1], vertical_alignment="bottom")
    
    with col_input:
        text_input = st.text_input(
            "Product Name", 
            placeholder="Type product name or upload an image...", 
            label_visibility="collapsed"
        )
        
    with col_attach:
        with st.popover("📷", help="Upload Image"):
            st.markdown("### 📤 Upload Product Image")
            image_input = st.file_uploader(
                "Upload", 
                type=["jpg", "png", "jpeg"], 
                label_visibility="collapsed"
            )
            
    with col_run:
        start_btn = st.button("⏩", type="primary", use_container_width=True)

if image_input:
    with st.expander("✅ Image Attached (Click to view)", expanded=False):
        st.image(image_input, width=150)

# --- MAIN LOGIC ---
if start_btn:
    if not api_key:
        st.error("🔑 API Key missing.")
        st.stop()
        
    if using_free_key:
        user_ip = get_remote_ip()
        count = get_usage_count(user_ip)
        if count >= FREE_USAGE_LIMIT:
            st.error("🚫 Free Quota Exceeded. Please enter your own API Key.")
            st.stop()
    
    # Reset
    st.session_state.general_report = None
    st.session_state.chart_data = None
    st.session_state.messages = []
    
    status = st.status("🕵️ Agentic Workflow Started...", expanded=True)
    
    try:
        # Identification
        if image_input:
            status.write("📸 **Vision Agent:** Analyzing materials & design...")
            image = Image.open(image_input)
            st.session_state.image_obj = image
            identified_name = identify_product_from_image(image).strip()
            
            if " vs " in identified_name.lower():
                status.write(f"🤔 **Ambiguity Detected:** Resembles {identified_name}. Switching to Comparative Mode.")
            else:
                status.write(f"👁️ **Identified:** {identified_name}")
                
        elif text_input:
            identified_name = text_input
        else:
            status.error("No input provided.")
            st.stop()
            
        st.session_state.product_name = identified_name
        
        # Research
        status.write(f"🌍 **Researcher:** Scouring web for '{identified_name}' global data...")
        data = run_research(identified_name)
        st.session_state.research_data = data
        
        # [MODIFIED: Extract Dual-Chart Data]
        status.write("📈 **Data Analyst:** Analyzing market share & feature scores...")
        chart_json = analyze_data_for_charts(identified_name, data)
        st.session_state.chart_data = chart_json
        
        # Report
        status.write("📔✒️ **Editor:** Compiling Product Report...")
        report = generate_report(identified_name, data)
        st.session_state.general_report = report
        
        if using_free_key: 
            increment_usage(get_remote_ip())
        
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"System Error: {e}")

# --- RESULT DISPLAY ---
if st.session_state.general_report:
    st.divider()
    
    # Header & Correction Widget
    c1, c2 = st.columns([3, 1])
    with c1: st.subheader(f"Analysis for: {st.session_state.product_name}")
    with c2:
        with st.popover("Wrong Product?"):
            new_name = st.text_input("Correct Name:")
            if st.button("🔄 Retry Analysis"):
                st.session_state.product_name = new_name
                with st.spinner(f"Correction: Analyzing {new_name}..."):
                    data = run_research(new_name)
                    st.session_state.research_data = data
                    # Update Chart
                    chart_json = analyze_data_for_charts(new_name, data)
                    st.session_state.chart_data = chart_json
                    # Update Report
                    report = generate_report(new_name, data)
                    st.session_state.general_report = report
                st.rerun()

    if "image_obj" in st.session_state:
        st.image(st.session_state.image_obj, width=150)

    # --- NEW VISUALIZATION SECTION ---
    if st.session_state.chart_data:
        
        # 1. MARKET CHART
        m_stats = st.session_state.chart_data.get("market_stats", {})
        if m_stats and m_stats.get("data"):
            st.markdown(f"### 📊 {m_stats.get('metric_name', 'Market Comparison')}")
            
            # Prepare Data
            chart_dict = m_stats["data"]
            df_market = pd.DataFrame(list(chart_dict.items()), columns=["Brand", "Value"])
            
            # Altair Vertical Bar Chart
            base = alt.Chart(df_market).encode(
                x=alt.X('Brand', sort='-y', axis=alt.Axis(title="Brand/Product", labelAngle=-45)),
                y=alt.Y('Value', axis=alt.Axis(title=m_stats.get('metric_name', 'Value'))),
                tooltip=['Brand', 'Value']
            )
            bar = base.mark_bar().encode(
                color=alt.Color('Brand', legend=None) # Different colors for each bar
            )
            # Add text labels on top of bars
            text = base.mark_text(dy=-10, color='black').encode(text='Value')
            
            st.altair_chart((bar + text).properties(height=300), use_container_width=True)
            st.caption("Data Source: Extracted from public search records (Shipments, Sales, or Review Counts).")
        else:
            st.info("ℹ️ No specific sales figures or review counts found publicly for comparison.")

        st.divider()

        # 2. FEATURE SCORE CHART
        f_scores = st.session_state.chart_data.get("feature_scores", [])
        if f_scores:
            st.markdown("### 🏆 Feature Face-Off (1-5 Score)")
            df_scores = pd.DataFrame(f_scores)
            
            # Altair Horizontal Grouped Bar Chart
            chart_features = alt.Chart(df_scores).mark_bar().encode(
                y=alt.Y('category', title=None),
                x=alt.X('score', title='Score (1-5)', scale=alt.Scale(domain=[0, 5])),
                color=alt.Color('product', title='Product', legend=alt.Legend(orient='top')),
                yOffset='product', # This creates the grouping effect
                tooltip=['product', 'category', 'score']
            ).properties(
                height=250 # Adjust height based on categories
            )
            
            st.altair_chart(chart_features, use_container_width=True)
            st.caption("Scores derived from sentiment analysis of technical reviews.")

    # Report Display
    st.markdown(st.session_state.general_report)
    
    with st.expander("🔍 Raw Intelligence"):
        st.text_area("Data", st.session_state.research_data, height=150)
    
    st.divider()
    
    # --- PERSONALIZATION SECTION ---
    st.markdown("## 👤 The Private Investigator")
    with st.container(border=True):
        st.markdown("#### Let's see if this product is the RIGHT CHoice for YOU")
        st.caption("Tell us a bit about yourself")
        
        user_profile = st.text_area("Profile", placeholder="e.g. 'I commute 2 hours a day, love bass-heavy music, but have a strict budget of $200...'")

        if st.button("✨ Generate My Personal Verdict"):
            if not user_profile:
                st.warning("Please tell us a little about yourself.....")
            else:
                with st.spinner("🤖 Analyzing User Type and Requirements..."):
                    rec = generate_personal_rec(st.session_state.product_name, st.session_state.research_data, user_profile)
                    st.markdown("### 💌 Your Personal Verdict")
                    st.markdown(rec)

    st.divider()
    
    # --- CHAT SECTION ---
    st.markdown("### 💬 Know more about this product or other options....")
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = call_llm(f"Expert on {st.session_state.product_name}. Context: {st.session_state.research_data}", prompt)
                st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
