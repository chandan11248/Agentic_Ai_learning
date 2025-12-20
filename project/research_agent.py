import streamlit as st
import arxiv
import fitz  # PyMuPDF
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.chains.summarize import load_summarize_chain
# --- PAGE CONFIG ---
st.set_page_config(page_title="Deep Research Agent (Safe Mode)", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    
    st.divider()
    mode = st.radio("Mode", ["Quick Search (RAG)", "Deep Study (Map-Reduce)"])
    st.info("Quick Search: Finds specific facts (Cheap).\n\nDeep Study: Summarizes the first 10 pages (Expensive).")

# --- FUNCTIONS ---

def search_arxiv(query, max_results=3):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for r in client.results(search):
        results.append({
            "title": r.title,
            "summary": r.summary,
            "pdf_url": r.pdf_url,
            "published": r.published.strftime("%Y-%m-%d")
        })
    return results

def download_and_extract_text(pdf_url):
    import requests
    try:
        response = requests.get(pdf_url)
        pdf_path = "temp_paper.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        
        doc = fitz.open(pdf_path)
        text = ""
        # LIMIT TO 10 PAGES to save tokens (Free Tier Safety)
        for page in doc[:10]: 
            text += page.get_text()
            
        doc.close()
        os.remove(pdf_path)
        return text
    except Exception as e:
        return ""

# --- LLM SETUP ---
def get_llm():
    if not groq_api_key:
        return None
    return ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile", 
        temperature=0.3
    )

def run_map_reduce(text_content):
    """
    Summarizes text using Map-Reduce.
    SAFEGUARD: Truncates text to 30k chars to prevent rate limit crashes.
    """
    llm = get_llm()
    if not llm: return "Error: No API Key"
    
    # 1. Safety Truncation (Max ~30k chars is approx 8-10k tokens)
    if len(text_content) > 30000:
        text_content = text_content[:30000]
        st.toast("⚠️ Note: Paper truncated to first 30,000 characters to save daily tokens.")

    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=200)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text_content)]
    
    # 3. Run Chain with Error Handling
    try:
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        res = chain.invoke(docs)
        return res['output_text']
    except Exception as e:
        if "rate_limit" in str(e).lower():
            return "⚠️ Rate Limit Hit: You have used your daily free tokens. Please wait or switch to Quick Search."
        return f"Error: {str(e)}"

# --- MAIN UI ---

st.title("📚 Deep Research Agent")

if "papers" not in st.session_state:
    st.session_state.papers = []
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. SEARCH PHASE
query = st.text_input("Enter a research topic:")
if st.button("🔍 Find Papers") and query:
    if not groq_api_key:
        st.error("Please enter your Groq API Key.")
    else:
        with st.spinner("Searching ArXiv..."):
            results = search_arxiv(query)
            st.session_state.papers = results
            st.success(f"Found {len(results)} papers!")

# 2. SELECTION
if st.session_state.papers:
    st.subheader("Select a Paper")
    
    paper_titles = [p['title'] for p in st.session_state.papers]
    selected_title = st.selectbox("Choose a paper to analyze:", paper_titles)
    selected_paper = next(p for p in st.session_state.papers if p['title'] == selected_title)

    if st.button(f"🧠 Analyze: {mode}"):
        with st.status("Processing Paper...", expanded=True) as status:
            status.write("Downloading PDF...")
            text = download_and_extract_text(selected_paper['pdf_url'])
            st.session_state.full_text = text
            
            if mode == "Deep Study (Map-Reduce)":
                status.write("Running Map-Reduce Strategy...")
                summary = run_map_reduce(text)
                
                # Check if it was an error message
                if "Error" in summary or "Rate Limit" in summary:
                    st.error(summary)
                    status.update(label="Failed", state="error", expanded=True)
                else:
                    st.session_state.messages = [{"role": "assistant", "content": f"**Full Paper Analysis:**\n\n{summary}"}]
                    status.update(label="Complete!", state="complete", expanded=False)
                    st.rerun()
                
            else: # Quick Search Mode (RAG)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
                st.session_state.retriever = BM25Retriever.from_documents(docs)
                st.session_state.retriever.k = 5
                st.session_state.messages = [{"role": "assistant", "content": "I'm ready to answer specific questions about this paper."}]
                status.update(label="Ready for Q&A", state="complete", expanded=False)
                st.rerun()

# 3. CHAT PHASE
if st.session_state.messages:
    st.divider()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        if not groq_api_key:
            st.error("API Key missing.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm = get_llm()
                    try:
                        if mode == "Deep Study (Map-Reduce)":
                            messages = [
                                SystemMessage(content="You are a helpful research assistant. Answer based on the analysis provided."),
                                HumanMessage(content=user_input)
                            ]
                            # Use last analysis if available
                            if len(st.session_state.messages) > 0:
                                messages.insert(1, SystemMessage(content=f"Context: {st.session_state.messages[0]['content']}"))
                            
                            response = llm.invoke(messages)
                            st.write(response.content)
                            st.session_state.messages.append({"role": "assistant", "content": response.content})
                            
                        else: # Quick Search (RAG)
                            if st.session_state.retriever:
                                relevant_docs = st.session_state.retriever.invoke(user_input)
                                context = "\n\n".join([d.page_content for d in relevant_docs])
                                system_prompt = f"Answer using this context:\n{context}"
                                messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
                                response = llm.invoke(messages)
                                st.write(response.content)
                                st.session_state.messages.append({"role": "assistant", "content": response.content})
                            else:
                                st.error("Please analyze a paper first.")
                                
                    except Exception as e:
                        st.error(f"Error: {str(e)}")