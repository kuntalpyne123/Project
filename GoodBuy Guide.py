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
# 1. PERSISTENT USAGE TRACKING (SQLite)
# ===========================

DB_FILE = "user_quotas.db"
FREE_USAGE_LIMIT = 5

def init_db():
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
    try:
        headers = _get_websocket_headers()  
        if headers and "X-Forwarded-For" in headers:
            return headers["X-Forwarded-For"].split(",")[0]
        return "LOCALHOST_DEV_MACHINE" 
    except Exception:
        return "UNKNOWN_CLIENT"

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
if "confirmed_product" not in st.session_state: st.session_state.confirmed_product = False
if "identified_name" not in st.session_state: st.session_state.identified_name = None
if "is_ambiguous" not in st.session_state: st.session_state.is_ambiguous = False

# ===========================
# 3. SIDEBAR CONFIGURATION
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
            if usage_left <= 0: st.error("🚫 Quota Exceeded. Please enter your own API Key.")  
            try:
                if "GEMINI_API_KEY" in st.secrets: api_key = st.secrets["GEMINI_API_KEY"]
                else: st.error("🚨 Default key not found in secrets!")
            except Exception: st.error("Secrets not available locally.")
        else:
            api_key = st.text_input("Enter Gemini API Key", type="password")
        
        model_choice = st.selectbox("Select Gemini Model:", ("2.5 Flash (Fast)", "2.5 Pro (Stable)", "3.0 Pro (Latest)"))
        if "Flash" in model_choice: model_id = "gemini-2.5-flash"
        elif "2.5" in model_choice: model_id = "gemini-2.5-pro"
        else: model_id = "gemini-3-pro-preview"

    elif provider == "OpenAI (ChatGPT)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_id = st.selectbox("Select Model:", ("gpt-4o", "gpt-4o-mini"))

    elif provider == "Anthropic (Claude)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter Anthropic API Key", type="password")
        model_id = st.selectbox("Select Model:", ("claude-3-5-sonnet-20241022", "claude-3-opus-20240229"))

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

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def call_llm(system_instruction, user_prompt, use_search=False, search_query=None, image_data=None):
    if not client: return "Error: Client not initialized. Check API Key."

    # --- GOOGLE GEMINI HANDLER ---
    if provider == "Google Gemini":
        tools = [Tool(google_search=GoogleSearch())] if use_search else None
        config = GenerateContentConfig(tools=tools, system_instruction=system_instruction, temperature=0.3)
        contents = [user_prompt]
        if image_data: contents.append(image_data)
        try:
            return client.models.generate_content(model=model_id, contents=contents, config=config).text
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
            content_payload = [{"type": "text", "text": final_prompt}]
            if image_data:
                buffered = io.BytesIO()
                image_data.save(buffered, format="JPEG")
                img_str = encode_image(buffered.getvalue())
                content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})
            messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": content_payload}]
            response = client.chat.completions.create(model=model_id, messages=messages, temperature=0.3)
            return response.choices[0].message.content
        except Exception as e: return f"OpenAI Error: {e}"

    # --- ANTHROPIC HANDLER ---
    elif provider == "Anthropic (Claude)":
        try:
            content_payload = [{"type": "text", "text": final_prompt}]
            if image_data:
                buffered = io.BytesIO()
                image_data.save(buffered, format="JPEG")
                img_str = encode_image(buffered.getvalue())
                content_payload.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_str}})
            response = client.messages.create(model=model_id, system=system_instruction, messages=[{"role": "user", "content": content_payload}], max_tokens=4000, temperature=0.3)
            return response.content[0].text
        except Exception as e: return f"Claude Error: {e}"

# ===========================
# 6. AGENT PERSONAS & LOGIC
# ===========================

# --- UPDATED: GENERALIZED FORENSIC VISION AGENT ---
def identify_product_from_image(image, user_hint=None):
    hint_instruction = f"USER HINT: The user believes this is '{user_hint}'. Verify if the visual evidence supports this." if user_hint else ""
    
    instruction = "You are a Universal Product Recognition Expert. You identify everything from Tech to Fashion, Home Goods, and Cars."
    
    prompt = f"""
    Analyze this product image deeply. {hint_instruction}
    
    STEP 1: VISUAL FORENSICS
    - Material Analysis: (Plastic vs Metal, Leather vs Synthetic, etc.)
    - Design Identifiers: (Logos, Button placement, Bezels, Stitching patterns)
    - Distinguishing Features: What makes this SPECIFIC model unique?
    
    STEP 2: AMBIGUITY CHECK (CRITICAL)
    - If the product looks exactly like two different versions (e.g., iPhone 13 vs 14, or two similar handbags), you MUST admit confusion.
    - Format ambiguous results as: "Product A vs Product B" (e.g., "Samsung S24 Ultra vs S25 Ultra").
    
    STEP 3: OUTPUT
    - Return ONLY the Product Name (or the "A vs B" string).
    - No filler words like "This is".
    """
    return call_llm(instruction, prompt, image_data=image)

RESEARCHER_INSTRUCTION = """
ROLE: You are the "Product Intelligence Engine". 
GOAL: Investigate {product_name}. 

SPECIAL INSTRUCTION FOR AMBIGUOUS INPUTS ("A vs B"):
If the product name implies a comparison (e.g. "S24 vs S25"):
1. Treat this as a COMPARATIVE RESEARCH task.
2. Investigate BOTH products.
3. Explicitly find the differences in specs, price, and issues.
4. Still find 2 OTHER external competitors if possible.

MANDATORY DATA POINTS:
1.  **Market Status:** Sales trends, Availability.
2.  **The "Hidden Gotchas":** Maintenance, Fees, Repairs.
3.  **Fake Review Detection:** Disparity between pro vs user reviews.
4.  **Competitor Intelligence:** Price vs Performance.
5.  **Technical Specs:** Hard numbers.

OUTPUT: Raw, detailed, unsummarized notes. Cite every claim.
"""

EDITOR_INSTRUCTION = """
ROLE: You are the "Transparent Shopping Consultant".
GOAL: Create a Master Report for {product_name}.

STRUCTURE & RULES:
1.  **Visuals:** Start with .
2.  **Ambiguity Handling (CRITICAL):** - If the input was "Product A vs Product B", start the report with a **"Visual Identification Crisis"** section.
    - Explain: "The image provided resembles both A and B. Here is the comparative analysis:"
    - Create a comparison table between A and B immediately.
3.  **Trust Badges:** Assign badges (Reliability, Value, etc.).
4.  **The "0-Second Summary":** TL;DR.
5.  **Decision Stress Eliminator:**
    - "The Main Pick": {product_name}
    - "The Budget Alternative"
    - "The Performance Upgrade"
6.  **Transparency Box:** Explicitly state conflicting data points.

OUTPUT FORMAT: Clean Markdown.
"""

PERSONALIZER_INSTRUCTION = """
ROLE: You are a hyper-personalized Sales Engineer.
GOAL: Re-evaluate {product_name} specifically for the USER'S PROFILE.
OUTPUT: A short, punchy personal letter to the user.
"""

def run_research(product_name):
    instruction = RESEARCHER_INSTRUCTION.format(product_name=product_name)
    prompt = f"Investigate {product_name}. Compare, find issues, check reviews."
    search_query = f"{product_name} reviews vs competitors reliability issues 2025"
    return call_llm(instruction, prompt, use_search=True, search_query=search_query)

def generate_report(product_name, research_data):
    instruction = EDITOR_INSTRUCTION.format(product_name=product_name)
    prompt = f"Research Data:\n{research_data}\n\nGenerate the Master Buying Guide."
    return call_llm(instruction, prompt)

def generate_personal_rec(product_name, research_data, user_profile):
    instruction = PERSONALIZER_INSTRUCTION.format(product_name=product_name)
    prompt = f"Research Data: {research_data}\nUser Profile: {user_profile}\nGenerate a personalized recommendation letter."
    return call_llm(instruction, prompt)

# ===========================
# 7. APP INTERFACE
# ===========================

st.title("🛍️ Product IQ: Agentic Shopper")
st.markdown("### We Do the Homework. You Get the Best.")
st.caption(f"Powered by **{provider} ({model_id})**")

# --- PHASE 1: INPUT ---
with st.container(border=True):
    st.markdown("#### 🔎 Start Your Search")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        text_input = st.text_input("Type Product Name", placeholder="e.g. Dyson Airwrap")
    with col2:
        image_input = st.file_uploader("Or Upload an Image", type=["jpg", "jpeg", "png"])

    analyze_btn = st.button("🚀 Analyze Product", type="primary")

# --- LOGIC CONTROL FLOW ---
if analyze_btn:
    st.session_state.confirmed_product = False
    st.session_state.identified_name = None
    st.session_state.is_ambiguous = False
    st.session_state.messages = []
    st.session_state.general_report = None
    
    # 1. Image Logic 
    if image_input:
        if not api_key:
            st.error("🔑 API Key missing. Please configure settings.")
            st.stop()
            
        with st.spinner("📸 Visual Cortex: Scanning design, materials, and potential ambiguities..."):
            image = Image.open(image_input)
            st.session_state.image_obj = image 
            
            # CALL THE VISION AGENT
            identified = identify_product_from_image(image)
            st.session_state.identified_name = identified.strip()
            
            # CHECK FOR AMBIGUITY ("vs" or "or" in the name)
            if " vs " in identified.lower() or " or " in identified.lower():
                st.session_state.is_ambiguous = True

    # 2. Text Logic
    elif text_input:
        st.session_state.identified_name = text_input
        st.session_state.confirmed_product = True 
    
    else:
        st.warning("Please provide a name or image.")

# --- PHASE 2: CONFIRMATION & AMBIGUITY HANDLING ---
if st.session_state.identified_name and not st.session_state.confirmed_product:
    st.divider()
    
    col_img, col_act = st.columns([1, 2])
    with col_img:
        if "image_obj" in st.session_state:
            st.image(st.session_state.image_obj, width=200, caption="Your Upload")
            
    with col_act:
        # --- SCENARIO A: AMBIGUOUS IDENTIFICATION (User Req Step 1 & 2) ---
        if st.session_state.is_ambiguous:
            st.warning(f"🤔 Visual Similarity Detected: The user has probably uploaded the image of **{st.session_state.identified_name}**.")
            st.info("💡 I will give the comparative analysis of all these products identified by me from the uploaded image.")
            
            if st.button("✅ Proceed with Comparative Analysis"):
                st.session_state.product_name = st.session_state.identified_name
                st.session_state.confirmed_product = True
                st.rerun()
                
        # --- SCENARIO B: SINGLE IDENTIFICATION ---
        else:
            st.success(f"🤖 AI Identified: **{st.session_state.identified_name}**")
            st.caption("Please confirm if this is correct.")
            
            c1, c2 = st.columns(2)
            if c1.button("✅ Yes, Correct"):
                st.session_state.product_name = st.session_state.identified_name
                st.session_state.confirmed_product = True
                st.rerun()
                
            if c2.button("❌ No, Let me Fix"):
                st.session_state.manual_correction_mode = True

    if st.session_state.get("manual_correction_mode"):
        correct_name = st.text_input("Type the correct name:", value=st.session_state.identified_name)
        if st.button("Run Analysis with Correct Name"):
            st.session_state.product_name = correct_name
            st.session_state.confirmed_product = True
            st.session_state.manual_correction_mode = False
            st.rerun()

# --- PHASE 3: EXECUTE RESEARCH (NORMAL WORKFLOW) ---
if st.session_state.confirmed_product and not st.session_state.general_report:
    # Rate Limit Check
    if using_free_key:
        user_ip = get_remote_ip()
        current = get_usage_count(user_ip)
        if current >= FREE_USAGE_LIMIT:
            st.error("🛑 Free Usage Limit Reached.")
            st.stop()

    status = st.status(f"🕵️ Deep Diving into **{st.session_state.product_name}**...", expanded=True)
    try:
        # Research (Handles "A vs B" automatically due to updated Prompt)
        status.write("🌍 **The Deep Hunter:** Scouring global markets, comparisons & reliability logs...")
        data = run_research(st.session_state.product_name)
        st.session_state.research_data = data
        
        # Report (Will generate comparative table if ambiguous)
        status.write("📊 **The Analyst:** Compiling Master Guide...")
        report = generate_report(st.session_state.product_name, data)
        st.session_state.general_report = report
        
        # Increment Usage
        if using_free_key:
            increment_usage(user_ip)
            
        status.update(label="✅ Complete!", state="complete", expanded=False)
        st.rerun()
        
    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"System Error: {e}")

# --- PHASE 4: DISPLAY RESULTS ---
if st.session_state.general_report:
    st.divider()
    st.markdown(st.session_state.general_report)
    
    with st.expander("🔍 View Raw Intelligence (Transparency)"):
        st.text_area("Raw Data", st.session_state.research_data, height=200)
    
    st.divider()

    # --- PERSONALIZATION ---
    st.markdown("## 👤 The Private Investigator")
    with st.container(border=True):
        st.markdown("#### Tell us about yourself")
        user_profile = st.text_area("Profile", placeholder="e.g. 'I am a student on a budget...'")

        if st.button("✨ Generate My Personal Verdict"):
            if not user_profile:
                st.warning("Please tell us a little about yourself first.")
            else:
                with st.spinner("🤖 Simulating ownership..."):
                    rec = generate_personal_rec(st.session_state.product_name, st.session_state.research_data, user_profile)
                    st.markdown("### 💌 Your Personal Verdict")
                    st.markdown(rec)

    st.divider()

    # --- CHAT ---
    st.markdown("## 💬 Ask Me Anything")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about details..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = call_llm(f"Expert on {st.session_state.product_name}. Context: {st.session_state.research_data}", prompt)
                st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
