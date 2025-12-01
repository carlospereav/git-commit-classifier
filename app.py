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

# --- UI Layout ---
st.title("🤖 Git Commit Classifier")
st.markdown("Enter a commit message to automatically detect its **Conventional Commit** type.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        """
        This AI model classifies git commit messages into 10 categories:
        - **feat**: New feature
        - **fix**: Bug fix
        - **docs**: Documentation
        - **style**: Formatting
        - **refactor**: Code restructuring
        - **test**: Adding tests
        - **chore**: Maintenance
        - **perf**: Performance
        - **ci**: CI/CD
        - **build**: Build system
        """
    )
    st.markdown("---")
    st.caption("Powered by DistilBERT & Streamlit")

# Main Input
st.subheader("Test with Examples")
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

user_input = st.text_area("Commit Message:", value=st.session_state.text_input, height=100)

if st.button("Classify", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    elif model is None:
        st.error("Model not loaded.")
    else:
        # Inference
        with st.spinner("Analyzing..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                
            # Get top prediction
            confidence, predicted_class_id = torch.max(probs, dim=-1)
            predicted_label = model.config.id2label[predicted_class_id.item()]
            confidence_score = confidence.item()

            # Display Result
            st.markdown("### Result")
            
            # Color mapping for badges
            color_map = {
                "feat": "green", "fix": "red", "docs": "blue", "style": "cyan",
                "refactor": "orange", "test": "yellow", "chore": "grey", 
                "perf": "violet", "ci": "grey", "build": "grey"
            }
            color = color_map.get(predicted_label, "grey")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f":{color}[**{predicted_label.upper()}**]")
            with col2:
                if confidence_score < 0.5:
                    st.warning(f"Low confidence ({confidence_score:.1%}). Are you sure?")
                else:
                    st.success(f"Confidence: {confidence_score:.1%}")

            # --- Confidence Chart ---
            st.markdown("#### Confidence Distribution")
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probs, 3)
            top_probs = top_probs.flatten().tolist()
            top_indices = top_indices.flatten().tolist()
            top_labels = [model.config.id2label[idx] for idx in top_indices]

            fig = go.Figure(go.Bar(
                x=top_probs,
                y=top_labels,
                orientation='h',
                marker=dict(color=['#4CAF50', '#2196F3', '#FFC107']) # Custom colors
            ))
            
            fig.update_layout(
                xaxis_title="Probability",
                yaxis=dict(autorange="reversed"), # Top probability at the top
                height=250,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
