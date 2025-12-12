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
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_id = st.selectbox("Select Model:", ("gpt-4o", "gpt-4o-mini"))

    elif provider == "Anthropic (Claude)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
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
    - What is the current "Street Price" on Amazon/BestBuy/Resale sites?
    - Has it hit an all-time low recently? (Give the price).

2.  **The "Hidden" Reality (Negative Bias Search):**
    - Search Reddit/Forums for "failed after X months".
    - identifying recurring defects (e.g., "hinge crack," "battery drain," "stitching issues").
    - Are there subscription walls or expensive accessories required?

3.  **Competitive Landscape (Specifics Required):**
    - Identify exactly 3 rivals:
      A. Direct Rival (Same Price).
      B. Budget Killer (Cheaper).
      C. The "Dream" Upgrade (More expensive).
    - For EACH rival, find: Name, Price, Main Advantage over {product_name}, and Main Weakness.

4.  **Technical Deep Dive:**
    - Weight, Dimensions, Battery Life (Real-world vs Claimed), Materials used.

5.  **Social Sentiment:**
    - What do the 1-star reviews say?
    - What do the 5-star reviews say?

OUTPUT: A dense, detailed, unformatted text file with all these facts.
"""

def run_research(product_name):
    # We pass the DETAILED instruction to the System Role
    instruction = RESEARCHER_INSTRUCTION.format(product_name=product_name)
    
    # We give a specific Prompt to trigger the behavior
    prompt = f"""
    CONDUCT A DEEP-DIVE INVESTIGATION ON: {product_name}
    
    1. SCOUR the web for "Reddit {product_name} issues", "long term review", and "{product_name} vs competitors".
    2. FIND precise pricing data.
    3. FILL the Competitor Matrix with 3 specific rivals.
    
    Provide the RAW DATA now.
    """
    search_query = f"{product_name} detailed specs price history vs competitors reliability reddit issues"
    # Execute
    return call_llm(instruction, prompt, use_search=True, search_query=search_query)

# --- EDITOR AGENT ---
EDITOR_INSTRUCTION = """
ROLE: Lead Reviews Editor at a Top Tech Publication (e.g., The Verge, Wirecutter).
TONE: Authoritative, Professional, Nuanced, and Comprehensive.
GOAL: Write the definitive Buying Guide for {product_name}.

INPUT DATA: Use the Researcher's raw notes.

STRICT WRITING RULES:
1.  **No "Form Filling":** Do not output "Section 1", "Section 2". Write a flowing article with H2 headers.
2.  **Table Formatting:** The Comparison Matrix must be a properly formatted Markdown table.
3.  **Ambiguity Logic:** IF (and ONLY IF) the Researcher identified an ambiguity (e.g., "S24 vs S25"), include a specific section clarifying it. If not, SKIP IT.
4.  **Detail Level:** Do not say "Good battery." Say "The battery lasts approx. 14 hours, which is 2 hours less than the..."

--- REPORT SKELETON (Use this as a guide, not a checklist) ---

# The Definitive Review: {product_name}



### 🏆 The Executive Verdict
(Write a 3-4 sentence distinct paragraph. Is it a buy? Who is it for? Be decisive.)

**Quick Ratings:**
* **Reliability:** (e.g., 4/5 - "Solid build but prone to scratches")
* **Value:** (e.g., 5/5 - "Unbeatable for the price")
* **Future Proofing:** (e.g., 3/5 - "Successor rumored in Q4")

---

### 📉 Market Analysis & Pricing
(Discuss the current street price vs MSRP. Is it a good time to buy? Are there fake listings?)

---

### ⚔️ The Competition Matrix (MANDATORY)
*Here is how {product_name} stacks up against the market:*

| Product | Price | The "Win" (Pro) | The "Loss" (Con) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **{product_name}** | $X | ... | ... | ... |
| **(Budget Rival)** | $Y | ... | ... | ... |
| **(Direct Rival)** | $Z | ... | ... | ... |
| **(Premium Rival)** | $A | ... | ... | ... |

*(Add a paragraph below the table analyzing these choices deeply).*

---

### 🕵️ The "Hidden" Truths (Long-Term Ownership)
(Detail the wear-and-tear issues. Mention the Reddit/Forum complaints found by the researcher. Discuss maintenance costs.)

---

### 📝 Final Buying Advice
* **✅ Buy it if:** (Scenario A), (Scenario B).
* **❌ Skip it if:** (Scenario C), (Scenario D).

"""

def generate_report(product_name, research_data):
    instruction = EDITOR_INSTRUCTION.format(product_name=product_name)
    prompt = f"Research Data:\n{research_data}\n\nGenerate Guide with the Comparison Table."
    return call_llm(instruction, prompt)

# --- PERSONALIZER AGENT (UPDATED) ---
PERSONALIZER_INSTRUCTION = """
ROLE: You are a hyper-personalized Sales Engineer.
GOAL: Re-evaluate {product_name} specifically for the USER'S PROFILE.

INSTRUCTIONS FOR OUTPUT:
1. **Be Detailed & Thoughtful:** Do not be brief. Write a consultation letter (approx 200-300 words).
2. **Structure:**
   - **"The Fit Check":** Analyze how the product specs specifically match the user's mentioned habits/needs.
   - **"Day-in-the-Life Simulation":** Describe a specific scenario where this product will help or hinder them based on their profile.
   - **"The Hard Truth":** If the user is on a budget, warn them about hidden costs. If they are a power user, warn them about limitations.
   - **"Final Verdict":** A definitive "Buy" or "Skip" tailored to them.

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
st.caption(f"Powered by **{provider} ({model_id})**")

# --- INPUT SECTION ---
with st.container(border=True):
    col1, col2 = st.columns([1, 1])
    with col1: text_input = st.text_input("Type Product Name", placeholder="e.g. Dyson Airwrap")
    with col2: image_input = st.file_uploader("Or Upload Image", type=["jpg", "png", "jpeg"])
    
    start_btn = st.button("🚀 Analyze Product", type="primary")

# --- MAIN LOGIC ---
if start_btn:
    if not api_key:
        st.error("🔑 API Key missing.")
        st.stop()
    
    # Reset
    st.session_state.general_report = None
    st.session_state.messages = []
    
    status = st.status("🕵️ Product Analysis started...", expanded=True)
    
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
        status.write(f"🌍 **Researcher:** Scouring web for '{identified_name}' & alternatives...")
        data = run_research(identified_name)
        st.session_state.research_data = data
        
        # Report
        status.write("📊 **Editor:** Compiling Product Report...")
        report = generate_report(identified_name, data)
        st.session_state.general_report = report
        
        if using_free_key: increment_usage(get_remote_ip())
        
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
                    report = generate_report(new_name, data)
                    st.session_state.general_report = report
                st.rerun()

    if "image_obj" in st.session_state:
        st.image(st.session_state.image_obj, width=150)

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
