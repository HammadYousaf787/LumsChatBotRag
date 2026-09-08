"""
Streamlit Frontend for LUMS Academic Knowledge Assistant
With full LLM configuration controls and logging
"""

import streamlit as st
from rag_pipeline import LUMSRAGPipeline
from logging_config import get_logger
from datetime import datetime
from pathlib import Path
import os

# Initialize logger
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="LUMS Academic Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
    
if "pipeline_initialized" not in st.session_state:
    st.session_state.pipeline_initialized = False
    
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background-color: #1e3c72;
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        border-left: 4px solid #1e3c72;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        border-left: 4px solid #2ecc71;
    }
    .config-box {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1e3c72;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #2a5298;
        color: white;
    }
    .chat-message {
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #f0f2f6;
        text-align: right;
    }
    .assistant-message {
        background-color: #e8f4f8;
        text-align: left;
    }
    .log-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 0.5rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        max-height: 200px;
        overflow-y: auto;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 LUMS Academic Knowledge Assistant</h1>
    <p>Ask questions about LUMS policies, courses, and schedules</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Configuration
with st.sidebar:
    st.markdown("## ⚙️ LLM Configuration")
    st.markdown("*Adjust parameters and click 'Apply Configuration' to update*")
    
    # LLM Model Selection
    st.markdown("### 🤖 Model Selection")
    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.2-90b-text-preview",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    selected_model = st.selectbox(
        "Model",
        options=model_options,
        index=0,
        help="Select the Groq model to use"
    )
    
    st.markdown("### 🎛️ Generation Parameters")
    
    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower = more factual, Higher = more creative"
    )
    
    # Max Tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum length of response"
    )
    
    # Conversation Memory Toggle
    st.markdown("### 🧠 Conversation Memory")
    use_memory = st.toggle("Enable Conversation Memory", value=True)
    
    # Apply Configuration Button
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Apply Configuration", use_container_width=True, type="primary"):
            logger.info("Applying new configuration from UI")
            st.session_state.pipeline_initialized = False
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            logger.info("Clearing chat history")
            st.session_state.messages = []
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear_history()
            st.rerun()
    
    # Display current configuration
    st.markdown("---")
    st.markdown("### 📋 Current Config")
    st.markdown(f"""
    - **Model**: `{selected_model}`
    - **Temperature**: {temperature}
    - **Max Tokens**: {max_tokens}
    - **Memory**: {'✅ Enabled' if use_memory else '❌ Disabled'}
    """)
    
    # System Status
    st.markdown("---")
    st.markdown("### 🟢 System Status")
    
    if st.session_state.pipeline_initialized:
        st.success("✅ Pipeline Ready")
        if st.session_state.rag_pipeline:
            llm_info = st.session_state.rag_pipeline.get_llm_info()
            st.info(f"🤖 {llm_info}")
            
            # Show stats
            stats = st.session_state.rag_pipeline.get_stats()
            st.caption(f"📊 Queries: {stats['total_queries']}")
            st.caption(f"📝 Memory: {stats['memory_entries']} entries")
    else:
        st.warning("⏳ Pipeline not initialized")
        st.caption("Ask a question or apply config to start")
    
    # Log Viewer
    st.markdown("---")
    st.markdown("### 📋 Logs")
    
    if st.button("📊 View Recent Logs", use_container_width=True):
        try:
            log_dir = Path("logs")
            if log_dir.exists():
                log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # Show last 5 logs
                        log_text = "".join(lines[-5:])
                        st.code(log_text, language="json")
                else:
                    st.info("No logs for today")
            else:
                st.info("No logs found")
        except Exception as e:
            st.error(f"Could not read logs: {e}")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
            
            # Display timing info if available
            if message["role"] == "assistant" and "elapsed_time" in message:
                st.caption(f"⏱️ Response time: {message['elapsed_time']:.2f}s")

    # Chat input
    if prompt := st.chat_input("Ask a question about LUMS..."):
        logger.info(f"User question: {prompt[:100]}...")
        st.session_state.query_count += 1
        
        # Initialize pipeline if not already
        if not st.session_state.pipeline_initialized:
            with st.spinner("Initializing pipeline with new configuration..."):
                try:
                    logger.info("Initializing pipeline...")
                    pipeline = LUMSRAGPipeline(
                        chroma_dir="chroma_db",
                        use_conversation=use_memory
                    )
                    pipeline.apply_config(
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    st.session_state.rag_pipeline = pipeline
                    st.session_state.pipeline_initialized = True
                    logger.info("Pipeline initialized successfully")
                except Exception as e:
                    logger.error(f"Pipeline initialization failed: {e}")
                    st.error(f"Failed to initialize pipeline: {str(e)}")
                    st.stop()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching for answer..."):
                if not st.session_state.pipeline_initialized:
                    response = "⚠️ The RAG pipeline is not initialized. Please apply configuration."
                    sources = []
                    elapsed_time = 0
                else:
                    try:
                        result = st.session_state.rag_pipeline.query(prompt)
                        response = result["answer"]
                        sources = result.get("sources", [])
                        elapsed_time = result.get("elapsed_time", 0)
                        logger.info(f"Query completed in {elapsed_time:.2f}s")
                    except Exception as e:
                        logger.error(f"Query error: {e}")
                        response = f"❌ An error occurred: {str(e)}"
                        sources = []
                        elapsed_time = 0
            
            st.markdown(response)
            
            # Display sources
            if sources:
                with st.expander("📚 View Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")
            
            # Display timing
            if elapsed_time > 0:
                st.caption(f"⏱️ Response time: {elapsed_time:.2f}s")
            
            # Store assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources,
                "elapsed_time": elapsed_time
            })

with col2:
    # Quick stats
    st.markdown("### 📊 Session Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Queries", st.session_state.query_count)
    
    if st.session_state.pipeline_initialized and st.session_state.rag_pipeline:
        stats = st.session_state.rag_pipeline.get_stats()
        st.metric("Total Queries", stats['total_queries'])
        st.metric("Errors", stats['errors'])
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    example_queries = [
        "What is the minimum CGPA required for graduation?",
        "What is the grading breakdown for ACCT 100?",
        "When is ACCT 130 scheduled?",
        "What happens if I fail a course?",
        "What are the attendance requirements?"
    ]
    
    for query in example_queries:
        if st.button(f"🔍 {query[:40]}...", key=query):
            logger.info(f"Example query clicked: {query[:40]}...")
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()
    
    # Quick actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Reset Pipeline", use_container_width=True):
        logger.info("Resetting pipeline")
        st.session_state.pipeline_initialized = False
        st.session_state.rag_pipeline = None
        st.rerun()
    
    if st.button("📊 View Pipeline Stats", use_container_width=True):
        if st.session_state.pipeline_initialized and st.session_state.rag_pipeline:
            stats = st.session_state.rag_pipeline.get_stats()
            st.json(stats)
        else:
            st.info("Pipeline not initialized")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    LUMS Academic Knowledge Assistant | Powered by RAG | Data sourced from official LUMS documents
    <br>
    <span style="color: #999;">Questions? Feedback? Contact the development team</span>
</div>
""", unsafe_allow_html=True)