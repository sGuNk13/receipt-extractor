import streamlit as st
import sqlite3
import json
from datetime import datetime
import PyPDF2
import io
import os

# Page config
st.set_page_config(page_title="KKBS AI: PDF Data Extractor", page_icon="📄", layout="wide")

# Initialize database
def init_db():
    conn = sqlite3.connect('pdf_extractor.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  project_name TEXT NOT NULL,
                  responsible_person TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS budget_items
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  project_id INTEGER,
                  activity_name TEXT,
                  description TEXT,
                  amount REAL,
                  FOREIGN KEY (project_id) REFERENCES projects (id))''')
    conn.commit()
    return conn

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    total_pages = len(pdf_reader.pages)
    
    # Extract ALL pages with page markers
    pages_text = {}
    for page_num in range(total_pages):
        page = pdf_reader.pages[page_num]
        page_content = page.extract_text()
        pages_text[page_num] = page_content
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page_content
    
    return text, pages_text

# Find relevant pages for extraction
def find_relevant_pages(pages_text):
    relevant_pages = set()
    page_14_start = None
    page_15_start = None
    
    for page_num, content in pages_text.items():
        # Check if this page contains important sections
        if "1. ชื่อโครงการ" in content or "1.ชื่อโครงการ" in content:
            relevant_pages.add(page_num)
        
        if "2. ผู้รับผิดชอบ" in content or "2.ผู้รับผิดชอบ" in content:
            relevant_pages.add(page_num)
        
        # Find where section 14 starts
        if ("14. รายละเอียดงบประมาณ" in content or "14.รายละเอียดงบประมาณ" in content) and page_14_start is None:
            page_14_start = page_num
        
        # Find where section 15 starts
        if ("15." in content or "15 ." in content) and page_14_start is not None and page_15_start is None:
            # Check if this is actually section 15 (not just "15" as a number)
            if "15. ตัวชี้วัด" in content or "15.ตัวชี้วัด" in content or "15. " in content[:200]:
                page_15_start = page_num
    
    # Add all pages from section 14 to section 15 (or end of section 14)
    if page_14_start is not None:
        if page_15_start is not None:
            # Include from page 14 to page 15 (inclusive)
            for p in range(page_14_start, page_15_start + 1):
                relevant_pages.add(p)
        else:
            # If no section 15 found, include page 14 and next 2 pages as fallback
            relevant_pages.add(page_14_start)
            relevant_pages.add(page_14_start + 1)
            relevant_pages.add(page_14_start + 2)
    
    return sorted(list(relevant_pages))

# Build optimized text from relevant pages only
def build_optimized_text(pages_text, relevant_pages):
    optimized_text = ""
    for page_num in relevant_pages:
        if page_num in pages_text:
            optimized_text += f"\n--- Page {page_num + 1} ---\n"
            optimized_text += pages_text[page_num]
    return optimized_text

# Extract data using Groq LLM
def extract_with_groq(pdf_text, api_key):
    try:
        import requests
        
        prompt = f"""Extract the following information from this Thai academic PDF text and return ONLY a valid JSON object:

PDF Text (ALL PAGES):
{pdf_text[:12000]}

Instructions:
1. Find "1. ชื่อโครงการ" and extract the project name that comes after it
2. Find "2. ผู้รับผิดชอบ" and extract the person's name (before "หลักสูตร")
3. IMPORTANT: Find "14. รายละเอียดงบประมาณ" section (it may be on later pages) and extract ALL budget line items. Look for:
   - Activity names (like "Finance and Accounting Automation", "Accounting Systems and ERP")
   - Line items with format: "ค่า..." followed by calculations in parentheses and amounts
   - Extract the description, calculation, and amount for each item
   - Amounts are numbers with commas (like 36,000 or 12,000)

IMPORTANT: Extract ALL budget items you can find, even if there are 10+ items.

Return ONLY this JSON (no markdown, no explanation):
{{
  "project_name": "project name here",
  "responsible_person": "person name here",
  "budget_items": [
    {{"activity_name": "activity 1", "description": "ค่าตอบแทนวิทยากร (2 คน x 2 วัน x 6 ชั่วโมง x 1,500 บาท)", "amount": 36000}},
    {{"activity_name": "activity 1", "description": "ค่าที่พักวิทยากร (2 ห้อง x 2 คืน x 2,500 บาท)", "amount": 10000}}
  ]
}}"""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return None, f"API Error: {response.status_code} - {response.text}"
        
        result = response.json()
        response_text = result['choices'][0]['message']['content'].strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        data = json.loads(response_text.strip())
        return data, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# Save to database
def save_to_db(conn, data):
    c = conn.cursor()
    c.execute("INSERT INTO projects (project_name, responsible_person) VALUES (?, ?)",
              (data['project_name'], data['responsible_person']))
    project_id = c.lastrowid
    
    for item in data['budget_items']:
        c.execute("INSERT INTO budget_items (project_id, activity_name, description, amount) VALUES (?, ?, ?, ?)",
                  (project_id, item['activity_name'], item['description'], item['amount']))
    
    conn.commit()
    return project_id

# Main app
def main():
    st.title("📄 KKBS AI: PDF Data Extractor")
    st.markdown("Extract project information from Thai academic PDFs using AI")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Try to get API key from secrets first
        default_api_key = st.secrets.get("GROQ_API_KEY", "")
        
        if default_api_key:
            st.success("✅ API Key loaded from secrets")
            api_key = default_api_key
        else:
            api_key = st.text_input("Groq API Key", type="password", help="Get free API key from https://console.groq.com")
        
        # Business mode toggle (from secrets)
        enable_save = st.secrets.get("ENABLE_SAVE", "false").lower() == "true"
        
        st.markdown("---")
        st.markdown("### 📊 Extracted Fields")
        st.markdown("- ชื่อโครงการ (Project Name)")
        st.markdown("- ผู้รับผิดชอบ (Responsible Person)")
        st.markdown("- รายละเอียดงบประมาณ (Budget)")
        
        if enable_save and st.button("🗄️ View All Projects"):
            st.session_state.show_projects = True
    
    # Initialize database
    conn = init_db()
    
    # Show all projects
    if st.session_state.get('show_projects', False):
        st.header("📋 All Projects")
        projects = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        
        if projects:
            for proj in projects:
                with st.expander(f"🗂️ {proj[1]} (ID: {proj[0]})"):
                    st.write(f"**Responsible:** {proj[2]}")
                    st.write(f"**Created:** {proj[3]}")
                    
                    budget = conn.execute("SELECT * FROM budget_items WHERE project_id = ?", (proj[0],)).fetchall()
                    if budget:
                        st.write("**Budget Items:**")
                        total = 0
                        for item in budget:
                            st.write(f"- {item[2]}: {item[3]} - ฿{item[4]:,.2f}")
                            total += item[4]
                        st.write(f"**Total: ฿{total:,.2f}**")
        else:
            st.info("No projects yet. Upload a PDF to get started!")
        
        if st.button("⬅️ Back to Upload"):
            st.session_state.show_projects = False
            st.rerun()
        return
    
    # File upload
    st.header("1️⃣ Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        if not api_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar")
            st.info("Get a free API key from: https://console.groq.com")
            return
        
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Extract text
        with st.spinner("📖 Reading PDF..."):
            try:
                full_text, pages_text = extract_text_from_pdf(uploaded_file)
                
                # Find relevant pages
                relevant_pages = find_relevant_pages(pages_text)
                optimized_text = build_optimized_text(pages_text, relevant_pages)
                
                st.success(f"✅ PDF processed successfully ({len(pages_text)} pages)")
                
                # Store optimized text
                st.session_state.pdf_text = optimized_text
                
            except Exception as e:
                st.error(f"❌ Error reading PDF: {e}")
                return
        
        # Extract with Groq
        if st.button("🤖 Extract Data with AI", type="primary"):
            if 'pdf_text' not in st.session_state:
                st.error("Please upload a PDF first")
                return
                
            with st.spinner("🧠 AI is analyzing your PDF..."):
                data, error = extract_with_groq(st.session_state.pdf_text, api_key)
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 Tips: Make sure your API key is valid and has credits")
                return
            
            if data:
                st.session_state.extracted_data = data
                st.success("✅ Data extracted successfully!")
                st.rerun()
    
    # Show extracted data
    if 'extracted_data' in st.session_state:
        st.header("2️⃣ Review & Edit Extracted Data")
        data = st.session_state.extracted_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            data['project_name'] = st.text_input("Project Name (ชื่อโครงการ)", value=data.get('project_name', ''))
        
        with col2:
            data['responsible_person'] = st.text_input("Responsible Person (ผู้รับผิดชอบ)", value=data.get('responsible_person', ''))
        
        st.subheader("Budget Items (รายละเอียดงบประมาณ)")
        
        budget_items = data.get('budget_items', [])
        
        if budget_items:
            for idx, item in enumerate(budget_items):
                with st.expander(f"Item {idx + 1}: {item.get('activity_name', 'Unnamed')}", expanded=True):
                    col1, col2, col3 = st.columns([2, 3, 1])
                    with col1:
                        item['activity_name'] = st.text_input(f"Activity {idx+1}", value=item.get('activity_name', ''), key=f"act_{idx}")
                    with col2:
                        item['description'] = st.text_input(f"Description {idx+1}", value=item.get('description', ''), key=f"desc_{idx}")
                    with col3:
                        item['amount'] = st.number_input(f"Amount {idx+1}", value=float(item.get('amount', 0)), key=f"amt_{idx}")
            
            total = sum(item.get('amount', 0) for item in budget_items)
            st.metric("Total Budget", f"฿{total:,.2f}")
        else:
            st.warning("No budget items found")
        
        # Save button
        st.header("3️⃣ Save to Database")
        
        if not enable_save:
            st.warning("💼 **Demo Mode**: Save function is disabled. Contact us to unlock full features!")
            st.info("📧 Email: skamolthip.filos.ai@gmail.com | 📞 Phone: +66-64-142-6195")
        else:
            if st.button("💾 Save to Database", type="primary"):
                if not data.get('project_name'):
                    st.error("❌ Project name is required")
                elif not data.get('responsible_person'):
                    st.error("❌ Responsible person is required")
                elif not budget_items:
                    st.error("❌ At least one budget item is required")
                else:
                    try:
                        project_id = save_to_db(conn, data)
                        st.success(f"✅ Saved successfully! Project ID: {project_id}")
                        del st.session_state.extracted_data
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error saving: {e}")

if __name__ == "__main__":
    main()
