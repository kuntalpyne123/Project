import streamlit as st
import os
import sqlite3
import base64
import io
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
# 1. PERSISTENT USAGE TRACKING
# ===========================
DB_FILE = "user_quotas.db"
FREE_USAGE_LIMIT = 5

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS usage (ip_address TEXT PRIMARY KEY, count INTEGER, last_access TEXT)''')
    conn.commit()
    conn.close()

def get_remote_ip():
    try:
        headers = _get_websocket_headers()
        if headers and "X-Forwarded-For" in headers: return headers["X-Forwarded-For"].split(",")[0]
        return "LOCALHOST_DEV_MACHINE"
    except Exception: return "UNKNOWN_CLIENT"

def get_usage_count(ip):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT count FROM usage WHERE ip_address=?", (ip,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0

def increment_usage(ip):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT count FROM usage WHERE ip_address=?", (ip,))
    result = c.fetchone()
    current_time = datetime.now().isoformat()
    if result:
        new_count = result[0] + 1
        c.execute("UPDATE usage SET count=?, last_access=? WHERE ip_address=?", (new_count, current_time, ip))
    else:
        new_count = 1
        c.execute("INSERT INTO usage (ip_address, count, last_access) VALUES (?, ?, ?)", (ip, 1, current_time))
    conn.commit()
    conn.close()
    return new_count

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
    
    if provider == "Google Gemini":
        st.info("⚡ Native Search Grounding (Most Accurate)")
        key_source = st.radio("API Key Source:", ("Use Free Default Key", "Enter My Own Key"))
        if key_source == "Use Free Default Key":
            using_free_key = True 
            user_ip = get_remote_ip()
            current_usage = get_usage_count(user_ip)
            st.caption(f"🔒 ID: {user_ip}") 
            usage_left = FREE_USAGE_LIMIT - current_usage
            st.progress(min(current_usage / FREE_USAGE_LIMIT, 1.0), text=f"Quota: {current_usage}/{FREE_USAGE_LIMIT} used")
            if usage_left <= 0: st.error("🚫 Quota Exceeded.")  
            try:
                if "GEMINI_API_KEY" in st.secrets: api_key = st.secrets["GEMINI_API_KEY"]
            except: st.error("Secrets not available locally.")
        else:
            api_key = st.text_input("Enter Gemini API Key", type="password")
        
        model_choice = st.selectbox("Select Gemini Model:", ("2.5 Flash (Fast)", "2.5 Pro (Stable)", "3.0 Pro (Latest)"))
        if "Flash" in model_choice: model_id = "gemini-2.5-flash"
        elif "2.5" in model_choice: model_id = "gemini-2.5-pro"
        else: model_id = "gemini-3-pro-preview"

    elif provider == "OpenAI (ChatGPT)":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_id = st.selectbox("Select Model:", ("gpt-4o", "gpt-4o-mini"))

    elif provider == "Anthropic (Claude)":
        api_key = st.text_input("Enter Anthropic API Key", type="password")
        model_id = st.selectbox("Select Model:", ("claude-3-5-sonnet-20241022", "claude-3-opus-20240229"))

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
        config = GenerateContentConfig(tools=tools, system_instruction=system_instruction, temperature=0.3)
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

# --- VISION AGENT WITH SELF-CORRECTION ---
def identify_product_from_image(image):
    instruction = "You are a Tech Hardware Forensic Expert."
    
    prompt = """
    Identify this product.
    
    INTERNAL DEBATE (Self-Correction):
    1. Does it look like a budget phone (A-series) or a flagship (S-series/Edge)?
    2. LOOK CLOSER: Check the frame material (Plastic vs Metal). Check the bezel thickness.
    3. IF UNSURE: It is better to admit ambiguity than to guess the cheap model.
    
    OUTPUT RULES:
    1. If the product resembles multiple versions (e.g. S24 vs S25, or A55 vs S24), output STRICTLY: "Product A vs Product B".
    2. Otherwise, return the exact Product Name.
    3. No filler words.
    """
    return call_llm(instruction, prompt, image_data=image)

# --- RESEARCHER AGENT ---
RESEARCHER_INSTRUCTION = """
ROLE: Product Intelligence Engine.
GOAL: Investigate {product_name}. 

AMBIGUITY PROTOCOL:
If {product_name} contains "vs" or "or":
1. PERFORM COMPARATIVE ANALYSIS.
2. Investigate BOTH products mentioned.
3. Highlight key differences (Price, Specs, Issues).

MANDATORY:
1. Sales/Market Status.
2. Reliability & "Hidden Gotchas" (Repairs/Fees).
3. Fake Review Check.
4. Competitors (if not already comparing).
OUTPUT: Raw, detailed notes.
"""

def run_research(product_name):
    instruction = RESEARCHER_INSTRUCTION.format(product_name=product_name)
    prompt = f"Investigate {product_name}. Check reliability, market status, and rivals."
    search_query = f"{product_name} reviews problems reliability vs competitors 2025"
    return call_llm(instruction, prompt, use_search=True, search_query=search_query)

# --- EDITOR AGENT ---
EDITOR_INSTRUCTION = """
ROLE: Transparent Shopping Consultant.
GOAL: Master Report for {product_name}.

STRUCTURE:
1. **Visuals:** .
2. **Ambiguity Handling:** If input is "A vs B", start with a "Visual Identification Crisis" table comparing them.
3. **Verdict:** "The Main Pick", "Budget Alt", "Performance Upgrade".
4. **Reliability:** "Life after 1 year".
5. **Transparency:** State conflicting data.

OUTPUT: Markdown.
"""

def generate_report(product_name, research_data):
    instruction = EDITOR_INSTRUCTION.format(product_name=product_name)
    prompt = f"Research Data:\n{research_data}\n\nGenerate Guide."
    return call_llm(instruction, prompt)

# ===========================
# 6. APP INTERFACE
# ===========================
st.title("🛍️ Product IQ: Agentic Shopper")
st.caption(f"Powered by **{provider} ({model_id})**")

# --- INPUT SECTION ---
with st.container(border=True):
    col1, col2 = st.columns([1, 1])
    with col1: text_input = st.text_input("Type Product Name", placeholder="e.g. Dyson Airwrap")
    with col2: image_input = st.file_uploader("Or Upload Image", type=["jpg", "png", "jpeg"])
    
    # ONE CLICK ACTION
    start_btn = st.button("🚀 Analyze Product", type="primary")

# --- MAIN LOGIC ---
if start_btn:
    if not api_key:
        st.error("🔑 API Key missing.")
        st.stop()
    
    # 1. Reset & Start
    st.session_state.general_report = None
    st.session_state.messages = []
    
    status = st.status("🕵️ Agentic Workflow Started...", expanded=True)
    
    try:
        # 2. Identification (Text or Image)
        if image_input:
            status.write("📸 **Vision Agent:** Analyzing materials & design language...")
            image = Image.open(image_input)
            st.session_state.image_obj = image
            
            # AI Guess
            identified_name = identify_product_from_image(image).strip()
            
            # Logic: If ambiguity ("vs"), the system is designed to handle it.
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

        # 3. Research (Automated)
        status.write(f"🌍 **Research Agent:** Scouring web for '{identified_name}'...")
        data = run_research(identified_name)
        st.session_state.research_data = data
        
        # 4. Report (Automated)
        status.write("📊 **Editorial Agent:** Compiling Master Guide...")
        report = generate_report(identified_name, data)
        st.session_state.general_report = report
        
        # 5. Usage
        if using_free_key: increment_usage(get_remote_ip())
        
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"System Error: {e}")

# --- RESULT DISPLAY & CORRECTION UI ---
if st.session_state.general_report:
    st.divider()
    
    # --- HEADER WITH CORRECTION WIDGET ---
    # This is the "Escape Hatch" for hallucinations without breaking automation
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"Analysis for: {st.session_state.product_name}")
    with c2:
        # If the AI got it wrong, the user can fix it HERE and RERUN
        with st.popover("Wrong Product?"):
            new_name = st.text_input("Correct Name:")
            if st.button("🔄 Retry Analysis"):
                st.session_state.product_name = new_name
                # Trigger rerun logic manually by clearing report and rerunning research
                with st.spinner(f"Correction: Analyzing {new_name}..."):
                    data = run_research(new_name)
                    st.session_state.research_data = data
                    report = generate_report(new_name, data)
                    st.session_state.general_report = report
                st.rerun()

    if "image_obj" in st.session_state:
        st.image(st.session_state.image_obj, width=150)

    # REPORT
    st.markdown(st.session_state.general_report)
    
    with st.expander("🔍 Raw Intelligence"):
        st.text_area("Data", st.session_state.research_data, height=150)
    
    st.divider()
    
    # PERSONALIZATION & CHAT (Standard)
    st.markdown("### 💬 Ask about this product")
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = call_llm(f"Expert on {st.session_state.product_name}. Context: {st.session_state.research_data}", prompt)
                st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
