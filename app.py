import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import streamlit as st
from prompt import PROMPT_WORKAW
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from document_reader import get_kmutnb_summary


# โหลดค่า environment จากไฟล์ .env แล้วตั้งค่า API key
load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise RuntimeError("GENAI_API_KEY not set. Add it to .env or your environment variables.")

genai.configure(api_key=api_key)

# ----------------- CONFIG -----------------
generation_config = {
    "temperature": 0.0,
    "top_p": 0.90,
    "top_k": 40,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
    "candidate_count": 1,
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# ----------------- SYNONYMS -----------------
SYNONYMS = {
    "ค่าเฉลี่ย": ["mean", "average", "ค่ากลาง", "ค่าเฉลี่ยเลขคณิต"],
    "มัธยฐาน": ["median", "ค่ากลาง"],
    "ฐานนิยม": ["mode", "โหมด"],
    "ส่วนเบี่ยงเบนมาตรฐาน": ["standard deviation", "SD", "ความแปรปรวน"],
    "ความแปรปรวน": ["variance", "var"],
    "พิสัย": ["range", "ค่าพิสัย"],
    "ควอไทล์": ["quartile", "Q1", "Q2", "Q3"],
    "เปอร์เซ็นไทล์": ["percentile"],
    "การแจกแจงความถี่": ["frequency distribution", "ตารางแจกแจงความถี่"],
    "ฮิสโตแกรม": ["histogram", "กราฟแท่ง"],
    "กราฟ": ["chart", "graph", "แผนภูมิ"],
    "ความน่าจะเป็น": ["probability", "โอกาส"],
    "การสุ่มตัวอย่าง": ["sampling", "ตัวอย่าง"],
    "ประชากร": ["population"],
    "การทดสอบสมมติฐาน": ["hypothesis testing", "การทดสอบ"],
    "ค่า Z": ["Z-score", "คะแนนมาตรฐาน"],
    "ค่า T": ["T-score", "t-test"],
    "สหสัมพันธ์": ["correlation", "ความสัมพันธ์"],
    "การถดถอย": ["regression", "เส้นถดถอย"],
    "ข้อมูลเชิงปริมาณ": ["quantitative data", "ข้อมูลตัวเลข"],
    "ข้อมูลเชิงคุณภาพ": ["qualitative data", "ข้อมูลเชิงกลุ่ม"],
    "ตัวแปรต่อเนื่อง": ["continuous variable"],
    "ตัวแปรไม่ต่อเนื่อง": ["discrete variable"],
}

def expand_synonyms(text: str) -> list:
    """ขยายคำค้นจาก SYNONYMS"""
    expanded = set()
    lower = text.lower()
    for key, alts in SYNONYMS.items():
        if key.lower() in lower:
            for a in alts:
                expanded.add(a)
        for a in alts:
            if a.lower() in lower:
                expanded.add(key)
                expanded.update([x for x in alts if x != a])
    return sorted(expanded)

# ----------------- MODEL (Gemini 2.5) -----------------
@st.cache_resource(show_spinner=False)
def get_model():
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",  
        safety_settings=SAFETY_SETTINGS,
        generation_config=generation_config,
        system_instruction=PROMPT_WORKAW
    )

model = get_model()

# ----------------- PDF READER (อ่านทุกหน้า) -----------------
def read_full_pdf(pdf_path: str) -> str:
    """อ่าน PDF ทุกหน้าด้วย PyPDF2"""
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return text.strip()
    except ImportError:
        st.error("ต้องติดตั้ง PyPDF2: pip install PyPDF2")
        return ""
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# ----------------- FILE PATH MANAGEMENT -----------------
def find_dataset_file():
    """ค้นหาไฟล์ dataset จากหลายที่"""
    possible_paths = [
        "DataSetMath.pdf",
        "./DataSetMath.pdf",
        os.path.join(os.path.dirname(__file__), "DataSetMath.pdf"),
        os.path.join(os.getcwd(), "DataSetMath.pdf"),
        "/app/DataSetMath.pdf",
        "/mount/src/DataSetMath.pdf",
        os.path.join("data", "DataSetMath.pdf"),
        os.path.join("assets", "DataSetMath.pdf"),
        os.path.join("documents", "DataSetMath.pdf"),
        "dataset_Math.pdf",
        "DataSetMath.pdf",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def get_dataset_path():
    """หา path ของไฟล์ dataset พร้อมแสดง debug info"""
    found_path = find_dataset_file()
    if found_path:
        return found_path

    st.sidebar.write("🔍 **Debug Info:**")
    st.sidebar.write(f"Current working directory: `{os.getcwd()}`")
    try:
        script_dir = os.path.dirname(__file__)
    except NameError:
        script_dir = "(no __file__ in this env)"
    st.sidebar.write(f"Script directory: `{script_dir}`")

    try:
        files = os.listdir(os.getcwd())
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        st.sidebar.write(f"PDF files found: {pdf_files}")
        st.sidebar.write(f"All files in current dir: {files[:10]}...")
    except Exception as e:
        st.sidebar.write(f"Error listing files: {e}")

    return None

# ----------------- IO & CACHE -----------------
@st.cache_data(show_spinner=True)
def load_kmutnb_summary(path: str) -> str:
    """Load และ cache ข้อมูลจาก PDF (อ่านทุกหน้า)"""
    try:
        # ลองใช้ read_full_pdf ก่อน
        content = read_full_pdf(path)
        if content:
            return content
        # ถ้าไม่ได้ ใช้ document_reader เดิม
        return get_kmutnb_summary(path)
    except Exception as e:
        return f"Error loading PDF: {str(e)}"

# ----------------- UPLOAD FALLBACK -----------------
def handle_file_upload():
    """ให้ user อัปโหลดไฟล์เองถ้าหาไฟล์ไม่เจอ"""
    st.warning("⚠️ ไม่พบไฟล์ DataSetMath.pdf ในระบบ")
    st.info("💡 กรุณาอัปโหลดไฟล์ dataset ของคุณ")

    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ PDF Dataset",
        type=['pdf'],
        help="อัปโหลดไฟล์ DataSetMath.pdf หรือไฟล์ PDF ที่มีข้อมูลรายวิชาสถิติ KMUTNB"
    )

    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ อัปโหลดไฟล์ {uploaded_file.name} เรียบร้อยแล้ว")
        return temp_path

    return None

# ----------------- UI -----------------
def clear_history():
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดี! มีอะไรให้ช่วยเกี่ยวกับรายวิชาสถิติ KMUTNB"}
    ]
    st.session_state.pop("chat_session", None)
    st.rerun()

with st.sidebar:
    if st.button("Clear History"):
        clear_history()

    st.markdown("---")
    st.subheader("📁 File Status")

st.title("💬 วิชาสถิติ Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "model",
            "content": "สวัสดี! มีอะไรให้ช่วยเกี่ยวกับรายวิชาสถิติ KMUTNB",
        }
    ]

# ----------------- LOAD DATASET -----------------
file_path = get_dataset_path()

if file_path is None:
    file_path = handle_file_upload()

if file_path is None:
    st.error("❌ ไม่สามารถโหลดไฟล์ dataset ได้ กรุณาอัปโหลดไฟล์หรือตรวจสอบการติดตั้ง")
    st.stop()

with st.sidebar:
    st.success(f"✅ Using file: `{os.path.basename(file_path)}`")
    st.caption(f"Full path: `{file_path}`")

try:
    file_content = load_kmutnb_summary(file_path)
    if isinstance(file_content, str) and file_content.startswith("Error"):
        st.error(file_content)
        st.info("💡 ลองตรวจสอบไฟล์ PDF หรืออัปโหลดไฟล์ใหม่")
        st.stop()
    else:
        with st.sidebar:
            st.info(f"📄 Content loaded: {len(file_content)} characters")
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# ----------------- CREATE / REUSE CHAT SESSION -----------------
def ensure_chat_session():
    if "chat_session" not in st.session_state:
        base_history = [
            {
                "role": "model",
                "parts": [{"text": "พร้อมให้บริการข้อมูลจากเอกสารที่แนบไว้"}],
            },
            {
                "role": "user",
                "parts": [{
                    "text": (
                        "นี่คือข้อมูล รายวิชาสถิติ ทั้งหมด:\n\n"
                        + file_content
                    )
                }],
            },
        ]
        st.session_state["chat_session"] = model.start_chat(history=base_history)

ensure_chat_session()

# ----------------- RENDER HISTORY -----------------
def render_messages(limit_last:int = 20):
    for msg in st.session_state["messages"][-limit_last:]:
        st.chat_message(msg["role"]).write(msg["content"])

render_messages()

# ----------------- HANDLE INPUT -----------------
prompt = st.chat_input(placeholder="พิมพ์คำถามเกี่ยวกับ รายวิชาสถิติ ✨")

def trim_history(max_pairs:int = 8):
    """จำกัดความยาว history ใน UI"""
    msgs = st.session_state["messages"]
    if len(msgs) > (2 * max_pairs + 1):
        st.session_state["messages"] = msgs[-(2 * max_pairs + 1):]

def generate_response(user_text: str):
    st.session_state["messages"].append({"role": "user", "content": user_text})
    st.chat_message("user").write(user_text)

    # โหมดสั้น ๆ ขอบคุณ
    if user_text.lower().startswith("add") or user_text.lower().endswith("add"):
        reply = "ขอบคุณสำหรับคำแนะนำ"
        st.session_state["messages"].append({"role": "model", "content": reply})
        st.chat_message("model").write(reply)
        trim_history()
        return

    # เตรียม prompt แบบไม่เสริมคำ
    syns = expand_synonyms(user_text)
    syn_hint = f" (คำที่เกี่ยวข้อง: {', '.join(syns)})" if syns else ""
    
    final_prompt = f"""{user_text}{syn_hint}

กฎการตอบ:
- ตอบเฉพาะข้อมูลที่มีใน Dataset
- ห้ามแต่งคำตอบ ห้ามคาดเดา
- ตอบสั้น กระชับ ตรงประเด็น
- ไม่ต้องใช้คำว่า "ครับ" หรือ "ค่ะ"
- ถ้าไม่มีข้อมูล ให้ตอบว่า "ไม่พบข้อมูลนี้ใน Dataset ของรายวิชานี้ ขออภัย" """

    placeholder = st.chat_message("model")
    stream_box = placeholder.empty()
    collected = []

    try:
        for chunk in st.session_state["chat_session"].send_message(final_prompt, stream=True):
            piece = getattr(chunk, "text", None)
            if piece:
                collected.append(piece)
                stream_box.write("".join(collected))
        final_text = "".join(collected).strip() or "ไม่พบข้อมูล"
            
    except Exception as e:
        final_text = f"เกิดข้อผิดพลาด: {e}"

    st.session_state["messages"].append({"role": "model", "content": final_text})
    trim_history()

if prompt:
    generate_response(prompt)