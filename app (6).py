%%writefile app.py
import streamlit as st
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tempfile
import base64
import io
from datetime import datetime

# Image Processing
from PIL import Image

# ViT Transformers
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

# LangChain Core & Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Modern Agent
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ═════════════════════════════════════════════
# CUSTOM CSS
# ═════════════════════════════════════════════

def load_custom_css():
    st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%); }
    [data-testid="stSidebar"] * { color: white !important; }
    .stApp > header { background-color: transparent; }
    .stChatMessage {
        background-color: rgba(255,255,255,0.95);
        border-radius: 15px; padding: 15px; margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: white !important; font-weight: 700 !important; }
    .stTextInput > div > div > input { border-radius: 10px; border: 2px solid #667eea; }
    .stButton > button {
        border-radius: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 600; border: none;
        padding: 10px 24px; transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.2); }
    .stFileUploader { background-color: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #667eea; font-weight: 700; }
    .stAlert { border-radius: 10px; border-left: 5px solid #667eea; }
    .dataframe { border-radius: 10px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# CORE CONFIGURATION
# ═════════════════════════════════════════════

def get_llm(api_key: str, temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature
    )

@st.cache_resource(show_spinner="⚙️ Loading ViT model (first time only)...")
def load_vit_model():
    """Load and cache ViT model so it loads only once across sessions."""
    model_name = "google/vit-base-patch16-224"
    processor  = ViTImageProcessor.from_pretrained(model_name)
    model      = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 JPEG string for Gemini multimodal input."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_text_from_response(response):
    """Extract plain text from agent response dict."""
    if isinstance(response, dict) and "messages" in response:
        last = response["messages"][-1]
        content = last.content if hasattr(last, 'content') else last.get('content', '')
        if isinstance(content, list):
            return '\n'.join(
                item.get('text','') if isinstance(item, dict) else str(item)
                for item in content if isinstance(item, dict) and item.get('type')=='text' or isinstance(item, str)
            )
        return str(content)
    return str(response)

# ═════════════════════════════════════════════
# TOOL DEFINITIONS
# ═════════════════════════════════════════════

def create_eda_tools(df: pd.DataFrame):
    @tool
    def get_data_summary() -> str:
        """Returns comprehensive info about the dataset: columns, types, missing values, statistics."""
        summary = {
            "columns": df.columns.tolist(),
            "types":   df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "shape":   df.shape,
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {}
        }
        return json.dumps(summary, indent=2)

    @tool
    def generate_visualization(plot_type: str, column: str) -> str:
        """Creates histogram or boxplot. plot_type: 'histogram' or 'boxplot'. column: exact column name."""
        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type.lower() == "histogram":
            sns.histplot(df[column], kde=True, ax=ax, color='#667eea')
            ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        elif plot_type.lower() == "boxplot":
            sns.boxplot(x=df[column], ax=ax, color='#764ba2')
            ax.set_title(f'Boxplot of {column}', fontsize=14, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        return "Plot displayed successfully in the UI."

    return [get_data_summary, generate_visualization]

def create_code_generation_tools():
    @tool
    def generate_python_code(description: str) -> str:
        """Generates Python code based on description."""
        return f"Generating code for: {description}"
    @tool
    def explain_code(code_snippet: str) -> str:
        """Explains what a code snippet does."""
        return "Analyzing code snippet..."
    return [generate_python_code, explain_code]

def create_web_search_tools():
    @tool
    def search_information(query: str) -> str:
        """Searches for information on the web (simulated)."""
        return f"🔍 Searching for: {query}\n\nNote: Web search is simulated in this demo."
    return [search_information]

def create_sql_tools():
    @tool
    def generate_sql_query(description: str) -> str:
        """Generates SQL query from natural language."""
        return f"Generating SQL for: {description}"
    @tool
    def explain_sql_query(query: str) -> str:
        """Explains what an SQL query does."""
        return "Explaining SQL query..."
    return [generate_sql_query, explain_sql_query]

# ═════════════════════════════════════════════
# STREAMLIT UI SETUP
# ═════════════════════════════════════════════

st.set_page_config(
    page_title="🤖 Gemini AI Nexus",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)
load_custom_css()

# ═════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🤖 AI Nexus Control")
    st.markdown("---")

    api_keys = [
        'AIzaSyBxnhFqUTKTNk9_4ku3EHRjykuCHPSGXO4',
        'AIzaSyBzIhFqiVx-uGD3-q9HzglYTBkW6kIy5bo',
        'AIzaSyD4jfyDs-w-mcpw3h45cmXdQMttmb4UoME',
        'AIzaSyBegUDL9QDxBAmFw7qERw4Rf2GkDaNY3YI',
        'AIzaSyDU3B_BxKtqRqlqoZbtDtUYWiGB0BDsKNA',
        'AIzaSyDWZqrmM7-WTK-06ieW9ay6ND5gVG90IpM'
    ]
    if 'api_key' not in st.session_state:
        st.session_state.api_key = random.choice(api_keys)
    api_key = st.session_state.api_key
    st.success("✅ API Key Auto-Selected")

    st.markdown("---")
    st.markdown("### 🎯 Select Agent")
    chat_mode = st.selectbox(
        "Choose your AI assistant",
        [
            "💬 General Chat",
            "📊 Data Analyst",
            "📄 Document RAG",
            "💻 Code Generator",
            "🔍 Web Research",
            "🗄️ SQL Assistant",
            "🎨 Creative Writer",
            "🖼️ Image Analyzer"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        st.info("Higher temperature = more creative responses")

    st.markdown("---")
    agent_info = {
        "💬 General Chat":   "General purpose conversational AI assistant",
        "📊 Data Analyst":   "Analyze CSV files with visualizations",
        "📄 Document RAG":   "Question-answering from PDF documents",
        "💻 Code Generator": "Generate and explain code snippets",
        "🔍 Web Research":   "Research assistant with web search",
        "🗄️ SQL Assistant":  "Generate and explain SQL queries",
        "🎨 Creative Writer":"Creative content generation",
        "🖼️ Image Analyzer": "ViT classification + Gemini vision analysis"
    }
    st.markdown(f"**Current Agent:**\n{agent_info.get(chat_mode, '')}")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0
    st.metric("Messages", st.session_state.message_count)

    llm = get_llm(api_key, temperature)

# ═════════════════════════════════════════════
# MAIN HEADER
# ═════════════════════════════════════════════

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🤖 Gemini AI Nexus")
    st.markdown(f"*{chat_mode} - Powered by Google Gemini*")
with col2:
    st.markdown(f"### {datetime.now().strftime('%I:%M %p')}")
    st.markdown(f"{datetime.now().strftime('%B %d, %Y')}")

st.markdown("---")

# ═════════════════════════════════════════════
# AGENT IMPLEMENTATIONS
# ═════════════════════════════════════════════

# ── DATA ANALYST ──────────────────────────────
if chat_mode == "📊 Data Analyst":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>📊 Data Analyst Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Upload a CSV file to explore your data. Use <b>Visualize</b> to generate charts manually, and <b>Chat</b> to ask questions about your dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("📝 Rows",    df.shape[0])
        with col2: st.metric("📊 Columns", df.shape[1])
        with col3: st.metric("🔢 Numeric", len(df.select_dtypes(include='number').columns))
        with col4: st.metric("⚠️ Missing", df.isnull().sum().sum())
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["👀 Data Preview", "📈 Visualize", "💬 Chat with Data"])

        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
            with st.expander("📋 Column Info"):
                st.dataframe(pd.DataFrame({
                    "Column":  df.columns,
                    "Type":    df.dtypes.astype(str).values,
                    "Missing": df.isnull().sum().values,
                    "Unique":  df.nunique().values
                }), use_container_width=True)

        with tab2:
            st.markdown("#### 📈 Create Visualizations")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            all_cols     = df.columns.tolist()
            vcol1, vcol2, vcol3 = st.columns([2, 2, 1])
            with vcol1:
                plot_type = st.selectbox("Plot Type", ["Histogram","Boxplot","Scatter Plot","Bar Chart","Line Chart"])
            with vcol2:
                if plot_type == "Scatter Plot":
                    x_col = st.selectbox("X Column", numeric_cols, key="x_col")
                    y_col = st.selectbox("Y Column", numeric_cols, key="y_col")
                elif plot_type in ["Histogram","Boxplot","Line Chart"]:
                    selected_col = st.selectbox("Select Column", numeric_cols)
                else:
                    selected_col = st.selectbox("Select Column", all_cols)
            with vcol3:
                st.markdown("<br>", unsafe_allow_html=True)
                generate_btn = st.button("🎨 Generate", use_container_width=True)
            if generate_btn:
                try:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    if plot_type == "Histogram":
                        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color='#667eea')
                        ax.set_title(f'Distribution of {selected_col}', fontweight='bold')
                        ax.set_xlabel(selected_col); ax.set_ylabel("Frequency")
                    elif plot_type == "Boxplot":
                        sns.boxplot(x=df[selected_col].dropna(), ax=ax, color='#764ba2')
                        ax.set_title(f'Boxplot of {selected_col}', fontweight='bold')
                        ax.set_xlabel(selected_col)
                    elif plot_type == "Scatter Plot":
                        pdf = df[[x_col, y_col]].dropna()
                        ax.scatter(pdf[x_col], pdf[y_col], color='#667eea', alpha=0.6, s=50)
                        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                        ax.set_title(f'{x_col} vs {y_col}', fontweight='bold'); ax.grid(True, alpha=0.3)
                    elif plot_type == "Bar Chart":
                        vc = df[selected_col].value_counts().head(15)
                        ax.bar(vc.index.astype(str), vc.values, color='#667eea')
                        ax.set_title(f'Top values in {selected_col}', fontweight='bold')
                        ax.set_xlabel(selected_col); ax.set_ylabel("Count")
                        plt.xticks(rotation=45, ha='right')
                    elif plot_type == "Line Chart":
                        ax.plot(df[selected_col].dropna().values, color='#667eea', linewidth=1.5)
                        ax.set_title(f'Line Chart of {selected_col}', fontweight='bold')
                        ax.set_xlabel("Index"); ax.set_ylabel(selected_col); ax.grid(True, alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with tab3:
            msgs = StreamlitChatMessageHistory(key="data_chat_history")
            if len(msgs.messages) == 0:
                msgs.add_ai_message("👋 Dataset loaded! Ask me anything about your data.")
            tools = create_eda_tools(df)
            agent = create_agent(llm, tools)
            for msg in msgs.messages:
                with st.chat_message(msg.type): st.markdown(msg.content)
            if user_input := st.chat_input("Ask about your data..."):
                st.session_state.message_count += 1
                with st.chat_message("human"): st.markdown(user_input)
                with st.chat_message("ai"):
                    with st.spinner("🤔 Analyzing..."):
                        response = agent.invoke({
                            "messages": [
                                {"role": "system", "content": "You are an expert Data Scientist. Use get_data_summary tool. Always use exact column names."},
                                *[{"role": m.type, "content": m.content} for m in msgs.messages],
                                {"role": "user", "content": user_input}
                            ]
                        })
                        output = extract_text_from_response(response)
                        st.markdown(output)
                        msgs.add_user_message(user_input); msgs.add_ai_message(output)

# ── GENERAL CHAT ──────────────────────────────
elif chat_mode == "💬 General Chat":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>💬 General Chat Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>A general-purpose AI assistant powered by Gemini. Ask anything — trivia, explanations, advice, summaries, translations, and more.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Multi-turn conversation &nbsp;|&nbsp; ✅ Any topic &nbsp;|&nbsp; ✅ Fast responses</p>
    </div>
    """, unsafe_allow_html=True)
    msgs = StreamlitChatMessageHistory(key="gen_chat_history")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("👋 Hello! I'm your AI assistant. How can I help you today?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if prompt := st.chat_input("Type your message..."):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(prompt)
        with st.chat_message("ai"):
            with st.spinner("💭 Thinking..."):
                res = llm.invoke(prompt)
                st.markdown(res.content)
                msgs.add_user_message(prompt); msgs.add_ai_message(res.content)

# ── IMAGE ANALYZER ────────────────────────────
elif chat_mode == "🖼️ Image Analyzer":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>🖼️ Image Analyzer Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>
            Two-stage pipeline: <b>Stage 1</b> — ViT Vision Transformer classifies the image into 1000 ImageNet categories with confidence scores.
            <b>Stage 2</b> — Gemini Vision receives the <i>actual image</i> directly and generates a rich, detailed explanation.
            Then chat with Gemini about the image!
        </p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ ViT Classification &nbsp;|&nbsp; ✅ Top-5 Predictions &nbsp;|&nbsp; ✅ Gemini Vision (sees image) &nbsp;|&nbsp; ✅ Follow-up Chat</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ How does the pipeline work?"):
        st.markdown("""
        **Stage 1 — ViT (Vision Transformer)** `google/vit-base-patch16-224`:
        | Step | What happens |
        |------|-------------|
        | 🖼️ Patch Splitting | Image divided into 16×16 pixel patches |
        | 🔢 Linear Embedding | Each patch flattened and projected to a vector |
        | 📍 Positional Encoding | Position information added to each patch vector |
        | 🔄 Transformer Encoder | Multi-head self-attention across all patches |
        | 🏷️ Classification Head | [CLS] token outputs probabilities over 1000 classes |

        **Stage 2 — Gemini Vision (gemini-2.5-flash)**:
        - Receives the **actual image as base64** via multimodal input
        - Also receives ViT predictions as context
        - Sees the full image and provides natural language understanding
        - Supports **follow-up chat** — ask anything about the image!
        """)

    st.markdown("---")
    uploaded_img = st.file_uploader("📤 Upload an Image", type=["jpg","jpeg","png","webp","bmp"])

    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")

        # Image preview + metadata
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col_info:
            st.markdown("#### 📐 Image Details")
            st.markdown(f"- **Filename:** `{uploaded_img.name}`")
            st.markdown(f"- **Format:** `{uploaded_img.type}`")
            st.markdown(f"- **Dimensions:** `{img.size[0]} × {img.size[1]} px`")
            st.markdown(f"- **Color Mode:** `{img.mode}`")
            st.markdown(f"- **Total Pixels:** `{img.size[0]*img.size[1]:,}`")
            st.markdown(f"- **File Size:** `{uploaded_img.size/1024:.1f} KB`")

        st.markdown("---")

        # ── STAGE 1: ViT ─────────────────────────
        st.markdown("## 🔬 Stage 1 — ViT Classification")

        with st.spinner("🔍 Running ViT Vision Transformer..."):
            try:
                processor, vit_model = load_vit_model()
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    logits = vit_model(**inputs).logits
                probs       = torch.nn.functional.softmax(logits, dim=-1)[0]
                top5        = torch.topk(probs, 5)
                top5_idx    = top5.indices.tolist()
                top5_scores = top5.values.tolist()
                top5_labels = [vit_model.config.id2label[i] for i in top5_idx]

                st.markdown("### 🏷️ Top-5 Predictions")
                rank_emojis = ["🥇","🥈","🥉","4️⃣","5️⃣"]
                for i, (label, score) in enumerate(zip(top5_labels, top5_scores)):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.markdown(f"{rank_emojis[i]} **{label.replace('_',' ').title()}**")
                        st.markdown(f"`{score*100:.2f}%`")
                    with c2:
                        st.progress(float(score))

                # Confidence chart
                st.markdown("### 📊 Confidence Chart")
                fig, ax = plt.subplots(figsize=(10, 4))
                colors       = ['#667eea','#764ba2','#a855f7','#c084fc','#e9d5ff']
                short_labels = [l.replace('_',' ')[:28] for l in top5_labels]
                bars = ax.barh(short_labels[::-1], [s*100 for s in top5_scores[::-1]], color=colors[::-1])
                for bar, score in zip(bars, top5_scores[::-1]):
                    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                            f'{score*100:.2f}%', va='center', fontweight='bold')
                ax.set_xlabel("Confidence (%)"); ax.set_title("ViT Top-5 Confidence", fontweight='bold')
                ax.set_xlim(0, max(top5_scores)*100+15); ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                # Build predictions text for Gemini context
                predictions_text = "\n".join([
                    f"{i+1}. {label.replace('_',' ')} ({score*100:.2f}%)"
                    for i, (label, score) in enumerate(zip(top5_labels, top5_scores))
                ])

                # ── STAGE 2: Gemini Vision ────────────
                st.markdown("---")
                st.markdown("## 🤖 Stage 2 — Gemini Vision Analysis")

                # Convert image to base64 for Gemini multimodal
                img_b64 = image_to_base64(img)

                with st.spinner("💭 Gemini is analyzing the image directly..."):
                    gemini_msg = HumanMessage(content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        },
                        {
                            "type": "text",
                            "text": f"""You are analyzing this image directly as a vision AI.

A ViT Vision Transformer also classified it with these predictions:
{predictions_text}

Please provide a comprehensive analysis:
1. **What you see** — Describe the image in detail (objects, scene, colors, composition, background)
2. **ViT Validation** — Do the ViT predictions match what you see? Agree or disagree with reasoning
3. **Context & Setting** — What real-world scenario, location, or situation does this show?
4. **Key Visual Features** — Shapes, textures, patterns, lighting that helped with identification
5. **Interesting Insights** — Any notable details, unusual elements, or fun facts about the subject

Be detailed, clear, and engaging!"""
                        }
                    ])
                    gemini_resp = llm.invoke([gemini_msg])
                    st.markdown(gemini_resp.content)

                # ── Follow-up Chat ─────────────────────
                st.markdown("---")
                st.markdown("### 💬 Chat with Gemini About This Image")
                st.caption("Gemini can see the image directly — ask anything about it!")

                # Store image data in session state per uploaded file
                if st.session_state.get("current_img_name") != uploaded_img.name:
                    st.session_state.current_img_name  = uploaded_img.name
                    st.session_state.current_img_b64   = img_b64
                    st.session_state.vit_predictions   = predictions_text
                    st.session_state.img_chat_history  = []

                # Show previous chat
                for chat in st.session_state.img_chat_history:
                    with st.chat_message(chat["role"]):
                        st.markdown(chat["content"])

                if follow_up := st.chat_input("Ask about the image..."):
                    st.session_state.message_count += 1
                    st.session_state.img_chat_history.append({"role": "human", "content": follow_up})

                    with st.chat_message("human"):
                        st.markdown(follow_up)

                    with st.chat_message("ai"):
                        with st.spinner("🔍 Gemini is looking at the image..."):
                            # Always include the image + prior history as text
                            messages = []

                            # System context with image
                            messages.append(HumanMessage(content=[
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.current_img_b64}"}
                                },
                                {
                                    "type": "text",
                                    "text": f"This is the image we are discussing. ViT classified it as: {st.session_state.vit_predictions.splitlines()[0] if st.session_state.vit_predictions else 'unknown'}. Answer the user's question about it."
                                }
                            ]))

                            # Add conversation history (skip last user msg, added below)
                            for prev in st.session_state.img_chat_history[:-1]:
                                if prev["role"] == "human":
                                    messages.append(HumanMessage(content=prev["content"]))
                                else:
                                    messages.append(AIMessage(content=prev["content"]))

                            # Current question
                            messages.append(HumanMessage(content=follow_up))

                            reply = llm.invoke(messages)
                            st.markdown(reply.content)
                            st.session_state.img_chat_history.append({"role": "ai", "content": reply.content})

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Ensure `transformers`, `torch`, and `Pillow` are installed: `pip install transformers torch Pillow`")

# ── DOCUMENT RAG ──────────────────────────────
elif chat_mode == "📄 Document RAG":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>📄 Document RAG Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Upload any PDF and ask questions. The agent reads, chunks, and indexes your document using vector search, then answers based on the content.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ PDF Q&A &nbsp;|&nbsp; ✅ Semantic Search &nbsp;|&nbsp; ✅ Multi-page Documents</p>
    </div>
    """, unsafe_allow_html=True)
    msgs     = StreamlitChatMessageHistory(key="rag_chat_history")
    pdf_file = st.file_uploader("📎 Upload PDF Document", type="pdf")
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_file.getvalue()); tmp_path = tmp.name
        with st.spinner("📖 Processing document..."):
            docs    = PyPDFLoader(tmp_path).load()
            splits  = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
            vs      = FAISS.from_documents(splits, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
            retriever = vs.as_retriever(search_kwargs={"k": 3})
        col1, col2 = st.columns(2)
        with col1: st.metric("📄 Pages", len(docs))
        with col2: st.metric("🧩 Chunks", len(splits))
        st.markdown("---")
        if len(msgs.messages) == 0:
            msgs.add_ai_message("📚 Document loaded! Ask me anything about its content.")
        for msg in msgs.messages:
            with st.chat_message(msg.type): st.markdown(msg.content)
        if query := st.chat_input("Ask a question about the document..."):
            st.session_state.message_count += 1
            with st.chat_message("human"): st.markdown(query)
            msgs.add_user_message(query)
            with st.chat_message("ai"):
                with st.spinner("🔍 Searching..."):
                    ctx  = "\n\n".join([d.page_content for d in retriever.invoke(query)])
                    resp = llm.invoke(f"Use this context to answer. If not found, say so.\n\nContext:\n{ctx}\n\nQuestion: {query}\n\nAnswer:")
                    st.markdown(resp.content); msgs.add_ai_message(resp.content)
        if os.path.exists(tmp_path): os.remove(tmp_path)

# ── CODE GENERATOR ────────────────────────────
elif chat_mode == "💻 Code Generator":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>💻 Code Generator Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Describe what you need and get clean, well-documented code instantly. Python, JavaScript, SQL, Java, C++, and more.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Multi-language &nbsp;|&nbsp; ✅ Explanation &nbsp;|&nbsp; ✅ Debugging &nbsp;|&nbsp; ✅ Best Practices</p>
    </div>
    """, unsafe_allow_html=True)
    msgs  = StreamlitChatMessageHistory(key="code_chat_history")
    tools = create_code_generation_tools()
    agent = create_agent(llm, tools)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("👨‍💻 Hi! I can generate code, explain snippets, and debug. What would you like to code today?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if user_input := st.chat_input("Describe what code you need..."):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("⚡ Generating..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "system", "content": "You are an expert programmer. Generate clean, well-documented code. Use markdown code blocks."},
                        *[{"role": m.type, "content": m.content} for m in msgs.messages],
                        {"role": "user", "content": user_input}
                    ]
                })
                output = extract_text_from_response(response)
                st.markdown(output); msgs.add_user_message(user_input); msgs.add_ai_message(output)

# ── WEB RESEARCH ──────────────────────────────
elif chat_mode == "🔍 Web Research":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>🔍 Web Research Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>AI-powered research assistant. Summaries, comparisons, explanations, deep dives on any topic.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Topic Research &nbsp;|&nbsp; ✅ Summarization &nbsp;|&nbsp; ✅ Comparisons &nbsp;|&nbsp; ✅ Fact Finding</p>
    </div>
    """, unsafe_allow_html=True)
    msgs  = StreamlitChatMessageHistory(key="research_chat_history")
    tools = create_web_search_tools()
    agent = create_agent(llm, tools)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("🔬 Hello! I'm your research assistant. What would you like to research?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if user_input := st.chat_input("What would you like to research?"):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("🔍 Researching..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "system", "content": "You are a research assistant. Provide comprehensive, well-sourced answers."},
                        *[{"role": m.type, "content": m.content} for m in msgs.messages],
                        {"role": "user", "content": user_input}
                    ]
                })
                output = extract_text_from_response(response)
                st.markdown(output); msgs.add_user_message(user_input); msgs.add_ai_message(output)

# ── SQL ASSISTANT ─────────────────────────────
elif chat_mode == "🗄️ SQL Assistant":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>🗄️ SQL Assistant Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Convert plain English into SQL. SELECT, JOIN, GROUP BY, CTEs, subqueries. MySQL, PostgreSQL, SQLite, SQL Server.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Query Generation &nbsp;|&nbsp; ✅ Explanation &nbsp;|&nbsp; ✅ Optimization &nbsp;|&nbsp; ✅ Schema Design</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**💡 Try asking:**")
    ex1, ex2, ex3 = st.columns(3)
    with ex1: st.code("Get top 10 customers by revenue", language=None)
    with ex2: st.code("Join orders with users table",    language=None)
    with ex3: st.code("Monthly sales trend last year",   language=None)
    st.markdown("---")
    msgs  = StreamlitChatMessageHistory(key="sql_chat_history")
    tools = create_sql_tools()
    agent = create_agent(llm, tools)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("🗄️ Hi! I can generate SQL queries, explain them, and optimize performance. What do you need?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if user_input := st.chat_input("Describe your SQL query need..."):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("💾 Processing..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "system", "content": "You are an expert SQL assistant. Format all queries in SQL code blocks."},
                        *[{"role": m.type, "content": m.content} for m in msgs.messages],
                        {"role": "user", "content": user_input}
                    ]
                })
                output = extract_text_from_response(response)
                st.markdown(output); msgs.add_user_message(user_input); msgs.add_ai_message(output)

# ── CREATIVE WRITER ───────────────────────────
elif chat_mode == "🎨 Creative Writer":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>🎨 Creative Writer Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Stories, poems, blog posts, ad copy, lyrics, scripts. Higher creativity mode enabled.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Stories &nbsp;|&nbsp; ✅ Poems &nbsp;|&nbsp; ✅ Blog Posts &nbsp;|&nbsp; ✅ Brainstorming &nbsp;|&nbsp; ✅ Ad Copy</p>
    </div>
    """, unsafe_allow_html=True)
    msgs         = StreamlitChatMessageHistory(key="creative_chat_history")
    creative_llm = get_llm(api_key, temperature=0.7)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("✨ Hello! Stories, poems, articles, ad copy — what shall we create today?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if user_input := st.chat_input("What would you like to create?"):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("✍️ Creating..."):
                response = creative_llm.invoke(user_input)
                st.markdown(response.content)
                msgs.add_user_message(user_input); msgs.add_ai_message(response.content)

# ═════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:white;padding:20px;'>
    <p>🤖 Powered by Google Gemini AI | Built with Streamlit & LangChain</p>
    <p>Made with ❤️ for AI Enthusiasts</p>
</div>
""", unsafe_allow_html=True)
