import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import QuestionClassifier
from retriever import DobbiRetriever
from generator import ResponseGenerator

st.set_page_config(
    page_title="Dobbi CS Assistant",
    page_icon="🧺",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    classifier = QuestionClassifier()
    retriever = DobbiRetriever(db_path="./chroma_db")
    generator = ResponseGenerator()
    return classifier, retriever, generator

classifier, retriever, generator = load_pipeline()

st.title("🧺 Dobbi CS Assistant")
st.markdown("*AI-powered draft responses for customer service*")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Customer Message")
    
    channel = st.radio(
        "Channel",
        ["📧 Email", "💬 WhatsApp", "📝 Manual"],
        horizontal=True
    )
    
    customer_message = st.text_area(
        "Paste customer message here",
        height=200,
        placeholder="Hoi, hoeveel kost het om een winterjas te reinigen?"
    )
    
    if st.button("🔍 Analyze & Generate Response", type="primary", use_container_width=True):
        if customer_message:
            with st.spinner("Analyzing..."):
                classification = classifier.classify(customer_message)
                st.session_state['classification'] = classification
                
                retrieved_docs = retriever.retrieve(customer_message, k=5)
                st.session_state['retrieved_docs'] = retrieved_docs
                
                result = generator.generate(
                    customer_message=customer_message,
                    category=classification['category'],
                    retrieved_docs=retrieved_docs
                )
                st.session_state['result'] = result
                st.session_state['analyzed'] = True
        else:
            st.warning("Please enter a customer message first.")

with col2:
    st.subheader("📤 Suggested Response")
    
    if st.session_state.get('analyzed'):
        cls = st.session_state['classification']
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Category", cls['category'])
        col_b.metric("Confidence", f"{cls['confidence']:.0%}")
        col_c.metric("Sentiment", cls['sentiment'])
        
        result = st.session_state['result']
        
        edited_response = st.text_area(
            "Draft response (edit if needed)",
            value=result['draft_response'],
            height=300
        )
        
        with st.expander("📚 Sources used"):
            for doc in st.session_state['retrieved_docs'][:3]:
                st.markdown(f"**{doc['metadata']['source']}** (distance: {doc['distance']:.3f})")
                st.caption(doc['content'][:200])
                st.divider()
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.rerun()
        with btn_col2:
            if st.button("🚩 Flag Issue", use_container_width=True):
                st.warning("Flagged for review")
        with btn_col3:
            if st.button("📋 Copy", type="primary", use_container_width=True):
                st.write("Response copied!")
                st.code(edited_response, language=None)
    else:
        st.info("👈 Paste a customer message and click 'Analyze' to get started.")

with st.sidebar:
    st.header("📊 Stats")
    st.caption("Coming soon: usage statistics")