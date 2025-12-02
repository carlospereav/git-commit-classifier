import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Git Commit Classifier",
    page_icon="🤖",
    layout="centered"
)

# Constants
MODEL_PATH = "./model"
COLOR_MAP = {
    "feat": "#2ea44f", "fix": "#cb2431", "docs": "#0366d6", "style": "#6f42c1",
    "refactor": "#d73a49", "test": "#f9c513", "chore": "#959da5", 
    "perf": "#6f42c1", "ci": "#24292e", "build": "#24292e"
}

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    """Loads the model and tokenizer only once."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main Container */
    .main {
        background-color: #f8f9fa;
    }
    /* Title */
    h1 {
        color: #1e1e1e;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    /* Subheader */
    .stMarkdown p {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
    }
    /* Magic Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        background-color: white;
        color: #333;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        border-color: #2196F3;
        color: #2196F3;
        background-color: #e3f2fd;
    }
    /* Result Card */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- UI Layout ---
st.title("🤖 Git Commit Classifier")
st.markdown("Enter a commit message to automatically detect its **Conventional Commit** type.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://git-scm.com/images/logos/downloads/Git-Icon-1788C.png", width=50)
    st.header("About")
    st.info(
        """
        **Supported Types:**
        - `feat`: New feature
        - `fix`: Bug fix
        - `docs`: Documentation
        - `style`: Formatting
        - `refactor`: Restructuring
        - `test`: Adding tests
        - `chore`: Maintenance
        - `perf`: Performance
        - `ci`: CI/CD
        - `build`: Build system
        """
    )
    st.markdown("---")
    st.caption("Powered by **DistilBERT** & **Streamlit**")

import pandas as pd
from utils.github_helper import fetch_recent_commits

# ... (Previous imports and config remain)

# Main Input
tab1, tab2, tab3 = st.tabs(["✍️ Single Message", "🐙 Analyze Repo", "🧠 How it Works"])

with tab1:
    st.subheader("⚡ Quick Test")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🐛 Fix Bug"):
        st.session_state.text_input = "prevent crash when user inputs empty string"
    if col2.button("✨ New Feat"):
        st.session_state.text_input = "add new dark mode toggle to settings"
    if col3.button("📚 Docs"):
        st.session_state.text_input = "update installation instructions in README"
    if col4.button("🚀 Perf"):
        st.session_state.text_input = "optimize image loading speed"

    # Text Area (connected to session state)
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

    user_input = st.text_area("Commit Message:", value=st.session_state.text_input, height=100, placeholder="Type your commit message here...")

    _, col_center, _ = st.columns([1, 2, 1])
    if col_center.button("🔍 Classify Message", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a message first.")
        elif model is None:
            st.error("Model not loaded.")
        else:
            # Inference
            with st.spinner("Analyzing semantics..."):
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    
                # Get top prediction
                confidence, predicted_class_id = torch.max(probs, dim=-1)
                predicted_label = model.config.id2label[predicted_class_id.item()]
                confidence_score = confidence.item()

                # Color mapping for badges (Hex codes)
                bg_color = COLOR_MAP.get(predicted_label, "#959da5")
                
                # Display Result in a nice card
                st.markdown(f"""
                <div class="result-card">
                    <div class="badge" style="background-color: {bg_color};">{predicted_label.upper()}</div>
                    <p style="color: #586069; margin: 0;">Confidence: <strong>{confidence_score:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)

                if confidence_score < 0.5:
                    st.warning("⚠️ Low confidence. The model is unsure about this prediction.")

                # --- Confidence Chart ---
                st.markdown("### 📊 Confidence Distribution")
                
                # Get top 3 predictions
                top_probs, top_indices = torch.topk(probs, 3)
                top_probs = top_probs.flatten().tolist()
                top_indices = top_indices.flatten().tolist()
                top_labels = [model.config.id2label[idx] for idx in top_indices]

                fig = go.Figure(go.Bar(
                    x=top_probs,
                    y=top_labels,
                    orientation='h',
                    text=[f"{p:.1%}" for p in top_probs],
                    textposition='auto',
                    marker=dict(color=[bg_color, '#e1e4e8', '#e1e4e8']) # Top match gets the color
                ))
                
                fig.update_layout(
                    xaxis_title="Probability",
                    yaxis=dict(autorange="reversed"),
                    height=200,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🐙 Analyze Repository")
    st.markdown("Fetch the latest commits from a public repository and classify them all at once.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        repo_url = st.text_input("GitHub Repository URL:", placeholder="https://github.com/huggingface/transformers")
    with col2:
        limit = st.number_input("Commits to fetch", min_value=5, max_value=50, value=10, step=5)
    
    if st.button("Fetch & Analyze", type="primary"):
        if not repo_url:
            st.warning("Please enter a URL.")
        else:
            with st.spinner(f"Fetching last {limit} commits from GitHub..."):
                result = fetch_recent_commits(repo_url, limit=limit)
                
            if "error" in result:
                st.error(result["error"])
            else:
                commits = result["data"]
                st.success(f"Fetched {len(commits)} commits. Analyzing...")
                
                results_data = []
                
                # Bulk Inference
                progress_bar = st.progress(0)
                for i, commit in enumerate(commits):
                    msg = commit['message'].split('\n')[0] # Only take first line
                    
                    inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = F.softmax(outputs.logits, dim=-1)
                        conf, pred_id = torch.max(probs, dim=-1)
                        
                    label = model.config.id2label[pred_id.item()]
                    
                    results_data.append({
                        "SHA": commit['sha'],
                        "Message": msg,
                        "Author": commit['author'],
                        "Type": label,
                        "Confidence": f"{conf.item():.1%}"
                    })
                    progress_bar.progress((i + 1) / len(commits))
                
                # Display DataFrame
                df = pd.DataFrame(results_data)
                st.dataframe(
                    df.style.applymap(lambda x: f"background-color: {'#d4edda' if 'feat' in x else '#f8d7da' if 'fix' in x else ''}", subset=['Type']),
                    use_container_width=True
                )
                
                # Stats
                st.markdown("### 📈 Commit Distribution")
                type_counts = df['Type'].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=type_counts.index, 
                    values=type_counts.values, 
                    hole=.4, # Donut chart
                    marker=dict(colors=[COLOR_MAP.get(label, 'grey') for label in type_counts.index])
                )])
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🧠 Under the Hood: The NLP Pipeline")
    
    st.markdown("""
    ### 1. The Architecture: DistilBERT
    At the core of this application lies **DistilBERT**, a distilled version of the BERT (Bidirectional Encoder Representations from Transformers) architecture. 
    
    While standard BERT models are powerful, they are often too heavy for real-time applications. We chose DistilBERT because it reduces the model size by 40% and increases inference speed by 60% while retaining over 97% of BERT's performance capabilities.
    
    The model consists of **6 Transformer layers** (compared to BERT-base's 12). Each layer uses **Self-Attention mechanisms** to understand the context of every word in relation to every other word in the sentence. This is crucial for commit messages, where the meaning of a word like "fix" depends heavily on whether it's used as a verb ("fix bug") or a noun ("hot fix").

    ### 2. The Semantic Challenge & Strategy
    A common pitfall in classifying commit messages is **overfitting to prefixes**. If a model simply learns that the word "feat" at the start of a sentence equals the `feat` label, it hasn't learned anything useful. It fails immediately when a developer writes "added a new login button" instead of "feat: added login button".
    
    To solve this, we implemented a **Robustness Strategy** during the data preparation phase:
    
    We trained the model specifically on **masked commit messages**. We algorithmically stripped the Conventional Commit prefixes (e.g., removing `feat:`, `fix:`) from the training data. This forced the model's attention mechanism to focus on the **semantic content** of the message—the verbs ("add", "remove", "optimize") and the technical objects ("database", "API", "css")—rather than relying on the easy shortcut of the prefix.
    
    ### 3. Fine-Tuning Process
    We utilized **Transfer Learning** to adapt the pre-trained DistilBERT model (which already understands English syntax and grammar) to our specific domain of software engineering logs.
    
    - **Tokenizer:** We used the WordPiece tokenizer to break down technical jargon into sub-word units (e.g., "refactoring" &rarr; "refactor" + "##ing"), ensuring the model handles out-of-vocabulary technical terms gracefully.
    - **Optimization:** The model was fine-tuned using the **AdamW optimizer** with a linear learning rate scheduler. We monitored the **Cross-Entropy Loss** to minimize the divergence between the predicted probability distribution and the actual labels.
    - **Classification Head:** On top of the transformer encoder, we added a linear classification layer that projects the 768-dimensional hidden states of the `[CLS]` token into our 10 target classes (`feat`, `fix`, `docs`, etc.).
    
    This approach resulted in a model that doesn't just "read" text, but "understands" the intent behind code changes.
    """)
