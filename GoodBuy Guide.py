import streamlit as st
import os
import sqlite3
from datetime import datetime
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
# 1. PERSISTENT USAGE TRACKING (SQLite)
# ===========================

DB_FILE = "user_quotas.db"
FREE_USAGE_LIMIT = 5

def init_db():
    """Initialize the SQLite database to track usage by IP."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage (
            ip_address TEXT PRIMARY KEY,
            count INTEGER,
            last_access TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_remote_ip():
    """
    Robust IP detection that works on Localhost and Deployed Cloud.
    """
    try:
        headers = _get_websocket_headers()  
       # 1. If running on Streamlit Cloud or behind a proxy
        if headers and "X-Forwarded-For" in headers:
            return headers["X-Forwarded-For"].split(",")[0]
        # 2. If running on Localhost (No X-Forwarded-For header exists)
        # We return a static string so the DB knows it's the SAME local machine.
        return "LOCALHOST_DEV_MACHINE" 
    except Exception:
        # Fallback for safety
        return "UNKNOWN_CLIENT"

def get_usage_count(ip):
    """Fetch current usage count for an IP."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT count FROM usage WHERE ip_address=?", (ip,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0

def increment_usage(ip):
    """Increment the usage count for an IP."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Check if exists
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

# Initialize DB on app load
init_db()

# ===========================
# 2. CONFIGURATION & SETUP
# ===========================

st.set_page_config(page_title="Product IQ: Agentic Shopper", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.85em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .badge-trust { background-color: #28a745; color: white; }
    .badge-warning { background-color: #ffc107; color: black; }
    .badge-danger { background-color: #dc3545; color: white; }
    .badge-info { background-color: #17a2b8; color: white; }
    .report-box { border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "research_data" not in st.session_state: st.session_state.research_data = None
if "general_report" not in st.session_state: st.session_state.general_report = None
if "messages" not in st.session_state: st.session_state.messages = []
if "product_name" not in st.session_state: st.session_state.product_name = ""

# ===========================
# 3. SIDEBAR CONFIGURATION
# ===========================

with st.sidebar:
    st.header("⚙️ Engine Settings")

    # --- A. PROVIDER SELECTION ---
    provider = st.radio(
        "Select AI Provider:",
        ("Google Gemini", "OpenAI (ChatGPT)", "Anthropic (Claude)"),
        index=0
    )

    api_key = None
    model_id = None
    using_free_key = False # Flag to track if we need to enforce limits
    
    # --- B. KEY MANAGEMENT ---
    
    # 1. GOOGLE GEMINI CONFIG
    if provider == "Google Gemini":
        st.info("⚡ Native Search Grounding (Most Accurate)")
        
        key_source = st.radio(
            "API Key Source:", 
            ("Use Free Default Key", "Enter My Own Key"),
            help="Default key is limited to 5 requests per IP address."
        )

        if key_source == "Use Free Default Key":
            using_free_key = True 
            
            # --- GET IP & SHOW USAGE ---
            user_ip = get_remote_ip()
            current_usage = get_usage_count(user_ip)
            
            # DEBUGGER: Shows you what the App sees. 
            # If this says "LOCALHOST_DEV_MACHINE", it is working correctly.
            st.caption(f"🔒 ID: {user_ip}") 
            
            usage_left = FREE_USAGE_LIMIT - current_usage
            
            # Progress bar calculation
            st.progress(min(current_usage / FREE_USAGE_LIMIT, 1.0), 
                        text=f"Quota: {current_usage}/{FREE_USAGE_LIMIT} used")
            if usage_left <= 0:
                st.error("🚫 Quota Exceeded. Please enter your own API Key.")  
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
                else:
                    st.error("🚨 Default key not found in secrets!")
            except StreamlitAPIException:
                st.error("Secrets not available locally.")
        else:
            api_key = st.text_input("Enter Gemini API Key", type="password")
        
        # Model Selection
        model_choice = st.selectbox(
            "Select Gemini Model:",
            ("2.5 Flash (Fast)", "2.5 Pro (Stable)", "3.0 Pro (Latest)")
        )
        if "Flash" in model_choice: model_id = "gemini-2.5-flash"
        elif "2.5" in model_choice: model_id = "gemini-2.5-pro"
        else: model_id = "gemini-3-pro-preview"

    # 2. OPENAI CONFIG
    elif provider == "OpenAI (ChatGPT)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_id = st.selectbox("Select Model:", ("gpt-4o", "gpt-4o-mini"))

    # 3. ANTHROPIC CONFIG
    elif provider == "Anthropic (Claude)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter Anthropic API Key", type="password")
        model_id = st.selectbox("Select Model:", ("claude-3-5-sonnet-20241022", "claude-3-opus-20240229"))

    # --- C. INITIALIZATION ---
    client = None
    if api_key:
        if provider == "Google Gemini":
            try: client = genai.Client(api_key=api_key)
            except Exception as e: st.error(f"Gemini Error: {e}")
        elif provider == "OpenAI (ChatGPT)":
            try: client = openai.OpenAI(api_key=api_key)
            except Exception as e: st.error(f"OpenAI Error: {e}")
        elif provider == "Anthropic (Claude)":
            try: client = anthropic.Anthropic(api_key=api_key)
            except Exception as e: st.error(f"Anthropic Error: {e}")

# ===========================
# 4. WEB SEARCH BRIDGE
# ===========================

def search_web_duckduckgo(query, max_results=5):
    try:
        results = DDGS().text(query, max_results=max_results)
        return "\n".join([f"- {r['title']}: {r['body']} (Source: {r['href']})" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

# ===========================
# 5. UNIFIED LLM WRAPPER
# ===========================

def call_llm(system_instruction, user_prompt, use_search=False, search_query=None):
    if not client:
        return "Error: Client not initialized. Check API Key."

    # --- GOOGLE GEMINI HANDLER ---
    if provider == "Google Gemini":
        tools = [Tool(google_search=GoogleSearch())] if use_search else None
        config = GenerateContentConfig(tools=tools, system_instruction=system_instruction, temperature=0.3)
        try:
            return client.models.generate_content(model=model_id, contents=user_prompt, config=config).text
        except Exception as e: return f"Gemini Error: {e}"

    # --- SEARCH INJECTION FOR OTHERS ---
    final_prompt = user_prompt
    if use_search and search_query:
        with st.spinner(f"🕵️ Bridging to live web via DuckDuckGo for {provider}..."):
            web_data = search_web_duckduckgo(search_query)
            final_prompt = f"CONTEXT FROM LIVE WEB SEARCH:\n{web_data}\n\nUSER QUERY:\n{user_prompt}"

    # --- OPENAI HANDLER ---
    if provider == "OpenAI (ChatGPT)":
        try:
            messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": final_prompt}]
            response = client.chat.completions.create(model=model_id, messages=messages, temperature=0.3)
            return response.choices[0].message.content
        except Exception as e: return f"OpenAI Error: {e}"

    # --- ANTHROPIC HANDLER ---
    elif provider == "Anthropic (Claude)":
        try:
            response = client.messages.create(model=model_id, system=system_instruction, messages=[{"role": "user", "content": final_prompt}], max_tokens=4000, temperature=0.3)
            return response.content[0].text
        except Exception as e: return f"Claude Error: {e}"

# ===========================
# 6. AGENT PERSONAS
# ===========================

RESEARCHER_INSTRUCTION = """
ROLE: You are the \"Product Intelligence Engine\". You do not just search; you investigate.
GOAL: Gather deep, conflict-aware data for {product_name} AND its top 3 competitors.

MANDATORY DATA POINTS TO FETCH:
1.  **Market Status:** Is it Limited Edition? Discontinued? What are the sales trends/popularity in major regions (US, EU, Asia)?
2.  **The \"Hidden Gotchas\":** Find maintenance costs, subscription fees, accessory requirements, and common repair issues after 6 months.
3.  **Fake Review Detection:** Scan for patterns—disparity between \"professional\" and \"user\" reviews, or floods of 5-star vague reviews.
4.  **Competitor Intelligence:** Find 2-3 direct rivals. Compare Price vs. Performance.
5.  **Technical Specs:** The hard numbers (dimensions, battery life, materials).
6.  **Price Intelligence:** Current street price, MSRP, and discount history.

OUTPUT:
Raw, detailed, unsummarized notes. Cite every claim.
"""

EDITOR_INSTRUCTION = """
ROLE: You are the \"Transparent Shopping Consultant\". You hate industry jargon and sponsored bias.
GOAL: Create a robust, easy-to-read Master Report for {product_name}.

INPUT: Use the provided Raw Research Data.

STRUCTURE & RULES:
1.  **Visuals:** Start with specific image tags (e.g. ).
2.  **Trust Badges:** At the top, assign badges based on data:
    - \"✅ High Reliability\" (if few defects found)
    - \"⚠️ Fake Review Risk\" (if suspicious patterns found)
    - \"🏆 Best Value\" (if beats competitors)
3.  **The \"0-Second Summary\":** A 3-bullet TL;DR.
4.  **Decision Stress Eliminator:**
    - **\"The Main Pick\":** {product_name} (Why?)
    - **\"The Budget Alternative\":** (Name a cheaper rival found in data)
    - **\"The Performance Upgrade\":** (Name a better rival found in data)
5.  **Long-Term Ownership:** A dedicated section on \"Life with this product after 1 year\" (Maintenance, wear & tear).
6.  **Transparency Box:** Explicitly state: \"We found X conflicting data points...\" or \"This recommendation is based on...\"
7.  **Tone:** Empowering, clear, direct. No fluff.

OUTPUT FORMAT: Clean Markdown.
"""

PERSONALIZER_INSTRUCTION = """
ROLE: You are a hyper-personalized Sales Engineer.
GOAL: Re-evaluate {product_name} specifically for the USER'S PROFILE.

INPUT:
1. General Research Data (Context)
2. User's \"About Me\" (Profile)

TASK:
1.  **Match/Mismatch:** Does this product fit their specific lifestyle? (e.g., \"You said you hate noise, but this unit is 60dB...\")
2.  **The Verdict:** \"YES, BUY IT\" or \"NO, AVOID IT\". Be decisive.
3.  **Usage Simulation:** \"Here is how this fits into your daily routine...\"
4.  **Do Not Buy List:** If it fails their constraints, create a \"Why this is on your Blacklist\".

OUTPUT: A short, punchy personal letter to the user.
"""

# ===========================
# 7. APP LOGIC
# ===========================

def run_research(product_name):
    instruction = RESEARCHER_INSTRUCTION.format(product_name=product_name)
    prompt = f"""
    Investigate {product_name}.
    - Compare with 3 rivals.
    - Find long-term reliability issues.
    - Check for fake review patterns.
    - Get sales trends and availability status.
    """
    search_query = f"{product_name} reviews vs competitors reliability issues 2025"
    return call_llm(instruction, prompt, use_search=True, search_query=search_query)

def generate_report(product_name, research_data):
    instruction = EDITOR_INSTRUCTION.format(product_name=product_name)
    prompt = f"Research Data:\n{research_data}\n\nGenerate the Master Buying Guide."
    return call_llm(instruction, prompt)

def generate_personal_rec(product_name, research_data, user_profile):
    instruction = PERSONALIZER_INSTRUCTION.format(product_name=product_name)
    prompt = f"""
    Research Data: {research_data}
    User Profile: {user_profile}
    Generate a personalized recommendation letter.
    """
    return call_llm(instruction, prompt)

# ===========================
# 8. APP INTERFACE
# ===========================

st.title("🛍️ Product IQ: Agentic Shopper")
st.markdown("### We Do the Homework. You Get the Best.")
st.caption(f"Powered by **{provider} ({model_id})**")

# --- PHASE 1: INPUT ---
with st.form("research_form"):
    product_input = st.text_input("What are you thinking of buying?", placeholder="e.g. Sony WH-1000XM5, Dyson Airwrap")
    submitted = st.form_submit_button("🔎 Show Me the Truth")

if submitted and product_input:
    # --- RATE LIMIT CHECK (DATABASE BASED) ---
    if using_free_key:
        user_ip = get_remote_ip()
        current_usage = get_usage_count(user_ip)
        
        if current_usage >= FREE_USAGE_LIMIT:
            st.error(f"🛑 Free Usage Limit Reached for IP {user_ip} ({FREE_USAGE_LIMIT}/{FREE_USAGE_LIMIT}).")
            st.warning("To continue, please select 'Enter My Own Key' in the sidebar.")
            st.stop() # Halt execution
    
    # --- CHECK MISSING KEY ---
    if not api_key:
        st.error("🔑 API Key missing. Please configure settings in the sidebar.")
        st.stop()

    st.session_state.product_name = product_input
    st.session_state.messages = [] 
    st.session_state.general_report = None 
    
    status = st.status(f"🕵️ Initiating Deep Dive via {provider}...", expanded=True)
    
    try:
        # 1. Research
        status.write(f"🌍 **The Deep Hunter:** Scouring global markets, rivals, and forums...")
        research_data = run_research(product_input)
        st.session_state.research_data = research_data
        
        # 2. Report
        status.write("📊 **The Analyst:** Balancing performance, quality, and value…")
        report_text = generate_report(product_input, research_data)
        st.session_state.general_report = report_text
        
        # --- INCREMENT USAGE COUNTER (DATABASE) ---
        if using_free_key:
            new_count = increment_usage(user_ip)
            st.toast(f"Free Quota Used: {new_count}/{FREE_USAGE_LIMIT}")
        
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
        
    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"System Error: {e}")

# --- PHASE 2: DISPLAY REPORT ---
if st.session_state.general_report:
    st.divider()
    
    # 2.1 The Report
    st.markdown(st.session_state.general_report)
    
    # 2.2 Source Transparency
    with st.expander("🔍 View Raw Intelligence & Citations (Transparency Layer)"):
        if provider == "Google Gemini": st.info("✅ Verified with Google Search")
        else: st.info("✅ Verified with DuckDuckGo Search")
        st.text_area("Raw Data", st.session_state.research_data, height=200)

    st.divider()

    # --- PHASE 3: PERSONALIZATION ENGINE ---
    st.markdown("## 👤 The Private Investigator")
    st.markdown("Want to know if this product is **RIGHT for YOU**.")
    
    with st.container(border=True):
        st.markdown("#### Tell us about yourself")
        user_profile = st.text_area(
            "Type in", 
            placeholder="e.g. 'I am a student on a budget...'",
        )

        if st.button("✨ Generate My Personal Verdict"):
            if not user_profile:
                st.warning("Please tell us a little about yourself first.")
            else:
                with st.spinner("🤖 Simulating your ownership experience..."):
                    personal_rec = generate_personal_rec(
                        st.session_state.product_name,
                        st.session_state.research_data,
                        user_profile
                    )
                    st.markdown("### 💌 Your Personal Verdict")
                    st.markdown(personal_rec)

    st.divider()

    # --- PHASE 4: AGENTIC CHAT ---
    st.markdown("## 💬 Ask Me Anything")
    st.caption(f"Want to know more about the **{st.session_state.product_name}** ?")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about maintenance, rivals, or specific specs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = call_llm(
                    f"You are an expert on {st.session_state.product_name}. Answer based on this research: {st.session_state.research_data}", 
                    prompt
                )
                st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
