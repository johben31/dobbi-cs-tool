import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import QuestionClassifier
from retriever import DobbiRetriever
from generator import ResponseGenerator

STATS_FILE = Path(__file__).parent.parent / "stats.json"


def format_category_name(category: str) -> str:
    """Convert category_code to Category Name."""
    return category.replace("_", " ").title()


def load_stats() -> list[dict]:
    """Load usage statistics from JSON file."""
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_stats(stats: list[dict]) -> None:
    """Save usage statistics to JSON file."""
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def record_analysis(category: str, confidence: float, message_length: int) -> None:
    """Record a message analysis to the stats file."""
    stats = load_stats()
    stats.append({
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "confidence": confidence,
        "message_length": message_length
    })
    save_stats(stats)


def compute_stats(stats: list[dict]) -> dict:
    """Compute aggregate statistics from raw data."""
    if not stats:
        return {
            "total": 0,
            "category_breakdown": {},
            "avg_confidence": 0,
            "today": 0,
            "this_week": 0
        }

    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday())

    category_counts = {}
    total_confidence = 0
    today_count = 0
    week_count = 0

    for entry in stats:
        category = entry["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
        total_confidence += entry["confidence"]

        entry_time = datetime.fromisoformat(entry["timestamp"])
        if entry_time >= today_start:
            today_count += 1
        if entry_time >= week_start:
            week_count += 1

    total = len(stats)
    category_breakdown = {
        cat: count / total for cat, count in category_counts.items()
    }

    return {
        "total": total,
        "category_breakdown": category_breakdown,
        "avg_confidence": total_confidence / total,
        "today": today_count,
        "this_week": week_count
    }


st.set_page_config(
    page_title="Dobbi CS Assistant",
    page_icon="🧺",
    layout="wide"
)

# Custom CSS with Dobbi brand colors
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem !important;
    }
    
    /* Primary button - Dobbi green */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #85B17E, #7CB19D, #75B1B3);
        border: none;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #75A16E, #6CA18D, #65A1A3);
        border: none;
        color: white;
    }
    
    /* Regular buttons */
    .stButton > button {
        border: 2px solid #85B17E;
        color: #85B17E;
    }
    .stButton > button:hover {
        border: 2px solid #75B1B3;
        color: #75B1B3;
    }
    
    /* Header styling */
    h1 {
        color: #85B17E !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Subheaders */
    h2, h3 {
        color: #7CB19D !important;
    }
    
    /* Metrics - smaller font */
    [data-testid="stMetricValue"] {
        color: #85B17E;
        font-size: 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #85B17E15, #75B1B315);
    }
    
    /* Info boxes */
    .stAlert {
        border-left-color: #7CB19D;
    }
    
    /* Text area focus */
    .stTextArea textarea:focus {
        border-color: #85B17E;
    }
    
    /* Radio buttons */
    .stRadio > div {
        color: #85B17E;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #7CB19D;
    }
    
    /* Divider */
    hr {
        border-color: #75B1B3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    import os
    from indexer import KnowledgeBaseIndexer
    
    # Auto-index if database doesn't exist
    if not os.path.exists("./chroma_db"):
        indexer = KnowledgeBaseIndexer()
        
        print("Checking knowledge base files...")
        print(f"faq_en.json exists: {os.path.exists('knowledge_base/faq_en.json')}")
        print(f"faq_nl.json exists: {os.path.exists('knowledge_base/faq_nl.json')}")
        print(f"terms_en.json exists: {os.path.exists('knowledge_base/terms_en.json')}")
        print(f"prices.csv exists: {os.path.exists('knowledge_base/prices.csv')}")
        
        if os.path.exists("knowledge_base/faq_en.json"):
            indexer.index_faq("knowledge_base/faq_en.json")
        if os.path.exists("knowledge_base/faq_nl.json"):
            indexer.index_faq("knowledge_base/faq_nl.json")
        if os.path.exists("knowledge_base/terms_en.json"):
            indexer.index_faq("knowledge_base/terms_en.json")
        if os.path.exists("knowledge_base/prices.csv"):
            indexer.index_prices("knowledge_base/prices.csv")
        
        print(f"Total items indexed: {indexer.collection.count()}")
    
    classifier = QuestionClassifier()
    retriever = DobbiRetriever(db_path="./chroma_db")
    generator = ResponseGenerator()
    return classifier, retriever, generator

classifier, retriever, generator = load_pipeline()

st.title("🧺 Dobbi CS Assistant")
st.caption("AI-powered draft responses for customer service")

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
        height=150,
        placeholder="Hoi, hoeveel kost het om een winterjas te reinigen?"
    )
    
    if st.button("🔍 Analyze & Generate Response", type="primary", use_container_width=True):
        if customer_message:
            with st.spinner("Analyzing..."):
                classification = classifier.classify(customer_message)
                st.session_state['classification'] = classification
                
                retrieved_docs = retriever.retrieve(customer_message, k=15)
                st.session_state['retrieved_docs'] = retrieved_docs
                
                result = generator.generate(
                    customer_message=customer_message,
                    category=classification['category'],
                    retrieved_docs=retrieved_docs
                )
                st.session_state['result'] = result
                st.session_state['analyzed'] = True

                record_analysis(
                    category=classification['category'],
                    confidence=classification['confidence'],
                    message_length=len(customer_message)
                )
        else:
            st.warning("Please enter a customer message first.")

with col2:
    st.subheader("📤 Suggested Response")
    
    if st.session_state.get('analyzed'):
        cls = st.session_state['classification']
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Category", format_category_name(cls['category']))
        col_b.metric("Confidence", f"{cls['confidence']:.0%}")
        col_c.metric("Sentiment", cls['sentiment'].title())
        
        result = st.session_state['result']
        
        edited_response = st.text_area(
            "Draft response (edit if needed)",
            value=result['draft_response'],
            height=200
        )
        
        # Main copy button - full width, prominent
        if st.button("📋 Copy Response", type="primary", use_container_width=True):
            st.code(edited_response, language=None)
            st.success("Click the copy icon above ☝️")

        # Regenerate and Bad Response buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.rerun()
        with btn_col2:
            if st.button("👎 Bad Response", use_container_width=True):
                st.warning("Flagged for review")

        # Internal order details panel (only for order queries)
        if result.get('order_details'):
            st.divider()
            st.subheader("📦 Internal Order Details")
            details = result['order_details']
            
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Carrier", details['carrier'])
            col_d2.metric("Shipment ID", details['shipment_id'] or "N/A")
            col_d3.metric("Packages", details['package_count'])
            
            if details['packages']:
                with st.expander("Package Details"):
                    for pkg in details['packages']:
                        status_icon = "✅" if pkg['status'] == "Order succesvol uitgevoerd" else "⏳"
                        st.text(f"{status_icon} {pkg['barcode']}: {pkg['status']}")
        
        # Sources expander at the bottom
        with st.expander("📚 Sources used"):
            for doc in st.session_state['retrieved_docs'][:3]:
                st.markdown(f"**{doc['metadata']['source']}** (distance: {doc['distance']:.3f})")
                st.caption(doc['content'][:200])
                st.divider()
    else:
        st.info("👈 Paste a customer message and click 'Analyze' to get started.")

with st.sidebar:
    st.header("📊 Stats")

    stats_data = load_stats()
    computed = compute_stats(stats_data)

    st.metric("Total Messages", computed["total"])

    col_today, col_week = st.columns(2)
    col_today.metric("Today", computed["today"])
    col_week.metric("This Week", computed["this_week"])

    if computed["total"] > 0:
        st.metric("Avg Confidence", f"{computed['avg_confidence']:.0%}")

        st.subheader("Category Breakdown")
        
        import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import QuestionClassifier
from retriever import DobbiRetriever
from generator import ResponseGenerator

STATS_FILE = Path(__file__).parent.parent / "stats.json"


def format_category_name(category: str) -> str:
    """Convert category_code to Category Name."""
    return category.replace("_", " ").title()


def load_stats() -> list[dict]:
    """Load usage statistics from JSON file."""
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_stats(stats: list[dict]) -> None:
    """Save usage statistics to JSON file."""
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def record_analysis(category: str, confidence: float, message_length: int) -> None:
    """Record a message analysis to the stats file."""
    stats = load_stats()
    stats.append({
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "confidence": confidence,
        "message_length": message_length
    })
    save_stats(stats)


def compute_stats(stats: list[dict]) -> dict:
    """Compute aggregate statistics from raw data."""
    if not stats:
        return {
            "total": 0,
            "category_breakdown": {},
            "avg_confidence": 0,
            "today": 0,
            "this_week": 0
        }

    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday())

    category_counts = {}
    total_confidence = 0
    today_count = 0
    week_count = 0

    for entry in stats:
        category = entry["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
        total_confidence += entry["confidence"]

        entry_time = datetime.fromisoformat(entry["timestamp"])
        if entry_time >= today_start:
            today_count += 1
        if entry_time >= week_start:
            week_count += 1

    total = len(stats)
    category_breakdown = {
        cat: count / total for cat, count in category_counts.items()
    }

    return {
        "total": total,
        "category_breakdown": category_breakdown,
        "avg_confidence": total_confidence / total,
        "today": today_count,
        "this_week": week_count
    }


# Custom CSS with Dobbi brand colors
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem !important;
    }
    
    /* Primary button - Dobbi green */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #85B17E, #7CB19D, #75B1B3);
        border: none;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #75A16E, #6CA18D, #65A1A3);
        border: none;
        color: white;
    }
    
    /* Regular buttons */
    .stButton > button {
        border: 2px solid #85B17E;
        color: #85B17E;
    }
    .stButton > button:hover {
        border: 2px solid #75B1B3;
        color: #75B1B3;
    }
    
    /* Header styling */
    h1 {
        color: #85B17E !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Subheaders */
    h2, h3 {
        color: #7CB19D !important;
    }
    
    /* Metrics - smaller font */
    [data-testid="stMetricValue"] {
        color: #85B17E;
        font-size: 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #85B17E15, #75B1B315);
    }
    
    /* Info boxes */
    .stAlert {
        border-left-color: #7CB19D;
    }
    
    /* Text area focus */
    .stTextArea textarea:focus {
        border-color: #85B17E;
    }
    
    /* Radio buttons */
    .stRadio > div {
        color: #85B17E;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #7CB19D;
    }
    
    /* Divider */
    hr {
        border-color: #75B1B3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    import os
    from indexer import KnowledgeBaseIndexer
    
    # Auto-index if database doesn't exist
    if not os.path.exists("./chroma_db"):
        indexer = KnowledgeBaseIndexer()
        
        print("Checking knowledge base files...")
        print(f"faq_en.json exists: {os.path.exists('knowledge_base/faq_en.json')}")
        print(f"faq_nl.json exists: {os.path.exists('knowledge_base/faq_nl.json')}")
        print(f"terms_en.json exists: {os.path.exists('knowledge_base/terms_en.json')}")
        print(f"prices.csv exists: {os.path.exists('knowledge_base/prices.csv')}")
        
        if os.path.exists("knowledge_base/faq_en.json"):
            indexer.index_faq("knowledge_base/faq_en.json")
        if os.path.exists("knowledge_base/faq_nl.json"):
            indexer.index_faq("knowledge_base/faq_nl.json")
        if os.path.exists("knowledge_base/terms_en.json"):
            indexer.index_faq("knowledge_base/terms_en.json")
        if os.path.exists("knowledge_base/prices.csv"):
            indexer.index_prices("knowledge_base/prices.csv")
        
        print(f"Total items indexed: {indexer.collection.count()}")
    
    classifier = QuestionClassifier()
    retriever = DobbiRetriever(db_path="./chroma_db")
    generator = ResponseGenerator()
    return classifier, retriever, generator

classifier, retriever, generator = load_pipeline()

st.title("🧺 Dobbi CS Assistant")
st.caption("AI-powered draft responses for customer service")

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
        height=150,
        placeholder="Hoi, hoeveel kost het om een winterjas te reinigen?"
    )
    
    if st.button("🔍 Analyze & Generate Response", type="primary", use_container_width=True):
        if customer_message:
            with st.spinner("Analyzing..."):
                classification = classifier.classify(customer_message)
                st.session_state['classification'] = classification
                
                retrieved_docs = retriever.retrieve(customer_message, k=15)
                st.session_state['retrieved_docs'] = retrieved_docs
                
                result = generator.generate(
                    customer_message=customer_message,
                    category=classification['category'],
                    retrieved_docs=retrieved_docs
                )
                st.session_state['result'] = result
                st.session_state['analyzed'] = True

                record_analysis(
                    category=classification['category'],
                    confidence=classification['confidence'],
                    message_length=len(customer_message)
                )
        else:
            st.warning("Please enter a customer message first.")

with col2:
    st.subheader("📤 Suggested Response")
    
    if st.session_state.get('analyzed'):
        cls = st.session_state['classification']
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Category", format_category_name(cls['category']))
        col_b.metric("Confidence", f"{cls['confidence']:.0%}")
        col_c.metric("Sentiment", cls['sentiment'].title())
        
        result = st.session_state['result']
        
        edited_response = st.text_area(
            "Draft response (edit if needed)",
            value=result['draft_response'],
            height=200
        )
        
        # Main copy button - full width, prominent
        if st.button("📋 Copy Response", type="primary", use_container_width=True):
            st.code(edited_response, language=None)
            st.success("Click the copy icon above ☝️")

        # Regenerate and Bad Response buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.rerun()
        with btn_col2:
            if st.button("👎 Bad Response", use_container_width=True):
                st.warning("Flagged for review")

        # Internal order details panel (only for order queries)
        if result.get('order_details'):
            st.divider()
            st.subheader("📦 Internal Order Details")
            details = result['order_details']
            
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Carrier", details['carrier'])
            col_d2.metric("Shipment ID", details['shipment_id'] or "N/A")
            col_d3.metric("Packages", details['package_count'])
            
            if details['packages']:
                with st.expander("Package Details"):
                    for pkg in details['packages']:
                        status_icon = "✅" if pkg['status'] == "Order succesvol uitgevoerd" else "⏳"
                        st.text(f"{status_icon} {pkg['barcode']}: {pkg['status']}")
        
        # Sources expander at the bottom
        with st.expander("📚 Sources used"):
            for doc in st.session_state['retrieved_docs'][:3]:
                st.markdown(f"**{doc['metadata']['source']}** (distance: {doc['distance']:.3f})")
                st.caption(doc['content'][:200])
                st.divider()
    else:
        st.info("👈 Paste a customer message and click 'Analyze' to get started.")

with st.sidebar:
    st.header("📊 Stats")

    stats_data = load_stats()
    computed = compute_stats(stats_data)

    st.metric("Total Messages", computed["total"])

    col_today, col_week = st.columns(2)
    col_today.metric("Today", computed["today"])
    col_week.metric("This Week", computed["this_week"])

    if computed["total"] > 0:
        st.metric("Avg Confidence", f"{computed['avg_confidence']:.0%}")

        st.subheader("Category Breakdown")
        
        # Bar chart
        import plotly.express as px
        
        categories = [format_category_name(cat) for cat in computed["category_breakdown"].keys()]
        values = [pct * 100 for pct in computed["category_breakdown"].values()]
        
        fig = px.bar(
            y=categories,
            x=values,
            orientation='h',
            color=categories,
            color_discrete_sequence=['#85B17E', '#E07A5F', '#F2CC8F', '#81B29A', '#3D405B']
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="",
            yaxis_title=""
        )
        
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={'displayModeBar': False}
        )
