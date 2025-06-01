import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agent.supplier_agent import SupplierAgent
from agent.utils import DataValidator, ReportFormatter
import io
from PIL import Image
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="Supplier Management Agent",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .page-nav {
        background: #e9ecef;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = None
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        # Page selection
        page = st.radio(
            "Select Page:",
            ["📊 Analysis", "💬 Chat with AI"],
            key="page_selector"
        )
        
        st.divider()
        
        # Show analysis status
        if st.session_state.analysis_results:
            st.success("✅ Analysis Data Available")
            scores = st.session_state.analysis_results.get("score_table", {})
            st.info(f"📈 {len(scores)} suppliers analyzed")
        else:
            st.info("ℹ️ No analysis data yet")
            st.caption("Run analysis first to enable AI chat")
            st.info("(Sample test data is avalible for use in the GITHUB repo under the DATA folder)")
        
        st.divider()
        
        # API Key for both pages
        st.subheader("🔑 API Configuration")
        default_key = ""
        try:
            default_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
        
        api_key = st.text_input(
            "OpenAI API Key",
            value=default_key,
            type="password",
            help="Required for AI features"
        )
        
        if default_key and api_key == default_key:
            st.success("✅ API key loaded")
    
    # Route to appropriate page
    if page == "📊 Analysis":
        show_analysis_page(api_key)
    elif page == "💬 Chat with AI":
        show_chat_page(api_key)

def show_analysis_page(api_key):
    """Main analysis page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Supplier Management Agent</h1>
        <p>AI-Powered Supplier Performance Analysis using LangGraph</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to proceed")
        st.stop()
    
    # File upload section
    st.header("📁 Upload Your Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Supplier Data")
        excel_file = st.file_uploader(
            "Upload Excel/CSV file",
            type=['xlsx', 'xls', 'csv'],
            help="Upload your supplier performance data"
        )
        
        if excel_file:
            st.success(f"✅ {excel_file.name} uploaded")
    
    with col2:
        st.subheader("📋 Audit Reports")
        audit_files = st.file_uploader(
            "Upload Word documents",
            type=['docx'],
            accept_multiple_files=True,
            help="Upload supplier audit reports"
        )
        
        if audit_files:
            st.success(f"✅ {len(audit_files)} audit files uploaded")
    
    # Processing options
    if excel_file and audit_files:
        st.header("🔧 Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            show_raw_data = st.checkbox("Show data preview", value=True)
        with col2:
            show_weights = st.checkbox("Show LLM weights", value=True)
        
        # Data preview
        if show_raw_data:
            with st.expander("📊 Data Preview", expanded=False):
                try:
                    if excel_file.name.endswith('.csv'):
                        df = pd.read_csv(excel_file)
                    else:
                        df = pd.read_excel(excel_file)
                    
                    validation_results = DataValidator.validate_supplier_data(df)
                    
                    if validation_results["errors"]:
                        for error in validation_results["errors"]:
                            st.error(f"❌ {error}")
                    
                    if validation_results["warnings"]:
                        for warning in validation_results["warnings"]:
                            st.warning(f"⚠️ {warning}")
                    
                    if validation_results["is_valid"]:
                        st.success("✅ Data validation passed")
                        
                        if validation_results.get("column_mapping"):
                            st.info("🔗 **Column Mapping Applied:**")
                            mapping_df = pd.DataFrame([
                                {"Required Column": k, "Found Column": v} 
                                for k, v in validation_results["column_mapping"].items()
                            ])
                            st.dataframe(mapping_df, use_container_width=True)
                    
                    st.dataframe(df.head(10), use_container_width=True)
                    excel_file.seek(0)
                    
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Analysis button
        st.header("🚀 Run Analysis")
        if st.button("🚀 Start Supplier Analysis", type="primary", use_container_width=True):
            run_analysis(api_key, excel_file, audit_files, show_weights)
    
    elif not excel_file or not audit_files:
        show_demo_section()
    
    # Show results if available
    if st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results, show_weights)

def show_chat_page(api_key):
    """Dedicated chat page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💬 AI Chat Assistant</h1>
        <p>Ask questions about your supplier performance analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use chat")
        st.stop()
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ No analysis data available. Please run analysis first from the Analysis page.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📊 Go to Analysis Page", type="primary"):
                st.session_state.page_selector = "📊 Analysis"
                st.rerun()
        
        st.stop()
    
    # Chat interface
    st.header("🗨️ Chat with Your Data")
    
    # Show analysis summary
    with st.expander("📊 Current Analysis Summary", expanded=False):
        results = st.session_state.analysis_results
        scores = results.get("score_table", {})
        
        if scores:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Suppliers", len(scores))
            with col2:
                avg_score = round(sum(scores.values()) / len(scores), 2)
                st.metric("Average Score", f"{avg_score}")
            with col3:
                best_supplier = max(scores, key=scores.get)
                st.metric("Top Performer", best_supplier)
    
    # Create data context for AI
    data_context = create_data_context(st.session_state.analysis_results)
    
    # Chat input (main chat interface)
    st.subheader("💭 Ask Your Question")
    user_question = st.text_input(
        "Type your question here:",
        placeholder="e.g., Which supplier should I focus on improving first?",
        key="chat_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Send Question", type="primary", disabled=not user_question):
            if user_question:
                process_chat_question(user_question, data_context, api_key)
    
    with col2:
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Suggested questions
    if not st.session_state.chat_history:
        st.subheader("💡 Suggested Questions")
        
        suggested_questions = [
            "Who is my best performing supplier and why?",
            "Which suppliers need immediate improvement?", 
            "What are the biggest risks in my supplier base?",
            "How can I improve overall supplier performance?",
            "What should be my top 3 priorities?",
            "Which suppliers have the most audit findings?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            col = cols[i % 2]
            with col:
                if st.button(f"💭 {question}", key=f"suggested_{i}", use_container_width=True):
                    process_chat_question(question, data_context, api_key)
    
    # Chat history display
    if st.session_state.chat_history:
        st.subheader("📝 Conversation History")
        
        # Display in reverse order (newest first)
        for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**🙋 Question {len(st.session_state.chat_history) - i}:**")
                st.info(question)
                st.markdown("**🤖 AI Response:**")
                st.success(answer)
                st.divider()
        
        # Export chat option
        if st.button("📥 Export Chat History"):
            chat_export = "\n\n".join([
                f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history
            ])
            st.download_button(
                "📥 Download Chat History",
                data=chat_export,
                file_name="supplier_chat_history.txt",
                mime="text/plain"
            )

def process_chat_question(question, data_context, api_key):
    """Process a chat question and add to history"""
    
    with st.spinner("🤖 AI is thinking..."):
        try:
            response = generate_chat_response(question, data_context, api_key)
            st.session_state.chat_history.append((question, response))
            st.rerun()
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history.append((question, error_msg))

def run_analysis(api_key, excel_file, audit_files, show_weights):
    """Run the supplier analysis using the agent"""
    
    with st.spinner("🤖 Initializing AI Agent..."):
        try:
            agent = SupplierAgent(api_key)
            st.session_state.current_agent = agent
        except Exception as e:
            st.error(f"❌ Error initializing agent: {str(e)}")
            return
    
    excel_file.seek(0)
    for audit_file in audit_files:
        audit_file.seek(0)
    
    with st.spinner("🔄 Processing supplier data..."):
        result = agent.process_supplier_data(excel_file, audit_files)
    
    if not result["success"]:
        st.error(f"❌ Analysis failed: {result['error']}")
        return
    
    # Store results in session state
    st.session_state.analysis_results = result["data"]
    st.success("✅ Analysis completed successfully! You can now use the Chat feature.")

def display_analysis_results(state, show_weights):
    """Display the analysis results"""
    
    st.header("📊 Analysis Results")
    
    scores = state["score_table"]
    if scores:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Suppliers", len(scores))
        with col2:
            avg_score = round(sum(scores.values()) / len(scores), 2)
            st.metric("Average Score", f"{avg_score}")
        with col3:
            best_supplier = max(scores, key=scores.get)
            st.metric("Top Performer", best_supplier)
        with col4:
            best_score = scores[best_supplier]
            st.metric("Best Score", f"{best_score}")
        
        # Performance chart
        st.subheader("📈 Performance Scores")
        fig = go.Figure(data=[
            go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color='lightblue',
                marker_line_color='navy',
                marker_line_width=2
            )
        ])
        
        fig.update_layout(
            title="Supplier Performance Scores",
            xaxis_title="Suppliers",
            yaxis_title="Performance Score",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show weights if requested
        if show_weights and state["weights"]:
            st.subheader("⚖️ Performance Weights")
            weights_df = pd.DataFrame(list(state["weights"].items()), 
                                     columns=['Metric', 'Weight'])
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.1%}")
            st.dataframe(weights_df, use_container_width=True)
        
        # Summary
        st.subheader("📋 AI-Generated Summary")
        if state["summary"]:
            st.markdown(state["summary"])
        
        # Detailed scores table
        st.subheader("📊 Detailed Scores")
        scores_df = ReportFormatter.format_score_table(scores)
        st.dataframe(scores_df, use_container_width=True)
        
        # Audit findings table
        if state["audit_findings"]:
            st.subheader("🔍 Audit Findings Summary")
            findings_df = ReportFormatter.format_findings_summary(state["audit_findings"])
            st.dataframe(findings_df, use_container_width=True)
        
        # Download options
        st.subheader("📥 Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Generate Word Report"):
                if st.session_state.current_agent:
                    with st.spinner("Generating report..."):
                        try:
                            doc_bytes = st.session_state.current_agent.save_to_doc(state)
                            if doc_bytes:
                                st.download_button(
                                    label="📥 Download Report",
                                    data=doc_bytes,
                                    file_name="Supplier_Performance_Report.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                                st.success("✅ Report generated!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with col2:
            csv_data = pd.DataFrame(list(scores.items()), 
                                   columns=['Supplier', 'Score']).to_csv(index=False)
            st.download_button(
                label="📊 Download Scores (CSV)",
                data=csv_data,
                file_name="supplier_scores.csv",
                mime="text/csv"
            )

def show_demo_section():
    """Show demo/example when no files are uploaded"""
    
    st.info("👆 Upload your supplier data and audit reports to get started!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Supplier Data (Excel/CSV):**")
        sample_data = pd.DataFrame({
            'Supplier': ['Supplier 1', 'Supplier 2', 'Supplier 3'],
            'OnTimeDeliveryRate': [95.5, 87.2, 92.1],
            'DefectRate': [2.1, 5.3, 3.2],
            'Location': ['San Jose, CA', 'Sacramento, CA', 'Austin, TX']
        })
        st.dataframe(sample_data, use_container_width=True)
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            "📥 Download Sample Data",
            csv,
            "sample_supplier_data.csv",
            "text/csv"
        )
    
    with col2:
        st.write("**Audit Reports (Word Documents):**")
        st.markdown("""
        - Upload Word documents (.docx format)
        - Should contain audit findings and observations
        - Agent will automatically extract finding counts
        - Multiple files can be uploaded at once
        """)

def create_data_context(state):
    """Create a comprehensive context about the supplier data for the AI"""
    
    scores = state.get("score_table", {})
    findings = state.get("audit_findings", {})
    weights = state.get("weights", {})
    summary = state.get("summary", "")
    supplier_data = state.get("structured_data", [])
    
    context = f"""
SUPPLIER PERFORMANCE ANALYSIS CONTEXT:

PERFORMANCE SCORES:
{', '.join([f"{supplier}: {score}" for supplier, score in scores.items()])}

AUDIT FINDINGS:
{', '.join([f"{supplier}: {count} findings" for supplier, count in findings.items()])}

PERFORMANCE WEIGHTS USED:
{', '.join([f"{metric}: {weight:.1%}" for metric, weight in weights.items()])}

DETAILED SUPPLIER DATA:
"""
    
    for supplier in supplier_data:
        context += f"\n{supplier.get('Supplier', 'Unknown')}: "
        context += f"On-Time Delivery: {supplier.get('OnTimeDeliveryRate', 'N/A')}%, "
        context += f"Defect Rate: {supplier.get('DefectRate', 'N/A')}%"
    
    context += f"\n\nAI GENERATED SUMMARY:\n{summary}"
    
    return context

def generate_chat_response(question, data_context, api_key):
    """Generate AI response to user question using supplier data context"""
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        
        prompt = f"""
You are a supplier management expert analyzing performance data. Answer the user's question based on the supplier data provided.

SUPPLIER DATA CONTEXT:
{data_context}

USER QUESTION: {question}

Guidelines for your response:
- Be specific and reference actual data from the context
- Provide actionable insights and recommendations
- Use supplier names and specific scores/metrics when relevant
- Keep responses concise but comprehensive
- If the question can't be answered from the data, say so clearly
- Focus on business value and practical next steps

Answer:
"""
        
        result = llm.invoke(prompt)
        return result.content
        
    except Exception as e:
        return f"I'm sorry, I couldn't process your question due to an error: {str(e)}"

if __name__ == "__main__":
    main()
