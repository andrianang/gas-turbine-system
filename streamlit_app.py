import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dataclasses import dataclass
from typing import List, Dict
import plotly.express as px
import os
import gc
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading

st.set_page_config(
    page_title="Gas Turbine RAG System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class ModelSummary:
    model_name: str
    summary: str
    response_time: float

@dataclass
class SearchResult:
    document: str
    metadata: Dict
    score: float

if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cpu")  # Force CPU for free tier
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "t5-small"  # Most reliable for free tier
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'simple_search_docs' not in st.session_state:
    st.session_state.simple_search_docs = None
if 'ai_model_status' not in st.session_state:
    st.session_state.ai_model_status = None
if 'ai_model_name' not in st.session_state:
    st.session_state.ai_model_name = None

@st.cache_resource
def load_embedding_model():
    try:
        st.info("🔄 Loading sentence transformer embedding model...")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception as e:
        st.warning(f"⚠️ Sentence transformer failed: {str(e)}")
        try:
            st.info("🔄 Using default ChromaDB embedding function...")
            return embedding_functions.DefaultEmbeddingFunction()
        except Exception as e2:
            st.error(f"❌ All embedding functions failed: {str(e2)}")
            st.info("🔄 Creating basic text-based embedding...")
            return None

@st.cache_resource
def load_ai_model(model_id):
    try:        
        st.info(f"🤖 Loading {model_id}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        
        st.success(f"✅ Successfully loaded {model_id}")
        return tokenizer, model, device
        
    except Exception as e:
        st.warning(f"⚠️ Could not load AI model: {str(e)}")
        try:
            # Try an even simpler alternative
            model_id = "t5-small"
            st.info(f"🤖 Trying fallback model {model_id}...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            
            device = torch.device("cpu")
            model.to(device)
            model.eval()
            
            st.success(f"✅ Successfully loaded fallback {model_id}")
            return tokenizer, model, device
            
        except Exception as e2:
            st.error(f"❌ All AI models failed to load. Using search-only mode.")
            return None, None, None

@st.cache_data
def load_dataset():
    try:
        possible_files = [
            "./src/Dataset_MGT10_CGT05.xlsx",
            "Dataset_MGT10_CGT05.xlsx",
            "./Dataset_MGT10_CGT05.xlsx"
        ]
        
        data_file = None
        for file_name in possible_files:
            if os.path.exists(file_name):
                data_file = file_name
                break
        
        if data_file is None:
            st.error("❌ Dataset file not found. Available files:")
            for file in os.listdir("."):
                if file.endswith((".xlsx", ".csv")):
                    st.write(f"- {file}")
            return None
            
        sheet_names = ['CGT_MGT', 'Use Case', 0]
        df = None
        
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(data_file, sheet_name=sheet_name)
                st.success(f"✅ Loaded dataset from {data_file} (sheet: {sheet_name})")
                break
            except Exception as e:
                continue
        
        if df is None:
            try:
                xl_file = pd.ExcelFile(data_file)
                st.error(f"❌ Could not load data. Available sheets: {xl_file.sheet_names}")
            except:
                st.error(f"❌ Could not read Excel file: {data_file}")
            return None
        
        df = df.dropna(how='all')
        st.info(f"📊 Cleaned dataset: {len(df)} records")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

def format_row(row):
    key_fields = [
        ("Gas Turbine Type", "Turbine Type"),
        ("Component Type", "Component"),
        ("FailureMode", "Failure Mode"),
        ("failurecause1", "Failure Cause"),
        ("failureeffect", "Failure Effect"),
        ("TaskDesc(ToPreventorToDetect)", "Task Description"),
    ]
    
    doc_parts = []
    for col, label in key_fields:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            value = str(row[col]).strip()[:200]
            doc_parts.append(f"{label}: {value}")
    
    return "\n".join(doc_parts) if doc_parts else "No data available"

def process_documents_batch(df, start_idx, end_idx):
    documents = []
    metadatas = []
    ids = []
    
    for idx in range(start_idx, min(end_idx, len(df))):
        row = df.iloc[idx]
        
        doc_text = format_row(row)
        if len(doc_text.strip()) < 10:
            continue
            
        documents.append(doc_text)
        
        metadata = {}
        for key in ['Gas Turbine Type', 'Component Type', 'FailureMode', 'failureeffect']:
            if key in row and pd.notna(row[key]):
                metadata[key] = str(row[key])[:100]
        
        metadatas.append(metadata)
        ids.append(f"doc-{idx}")
    
    return documents, metadatas, ids

def initialize_system():
    if st.session_state.system_initialized:
        return True
    
    init_container = st.container()
    
    with init_container:
        st.info("🚀 Initializing Gas Turbine RAG System...")
        
        df = load_dataset()
        if df is None:
            return False
        
        st.session_state.df = df
        
        embedding_fn = load_embedding_model()
        if embedding_fn is None:
            st.warning("⚠️ Embedding function unavailable. Using basic text search mode.")
            # Store documents in a simple format for text-based search
            st.session_state.simple_search_docs = []
            for idx, row in df.iterrows():
                doc_text = format_row(row)
                if len(doc_text.strip()) > 10:
                    metadata = {}
                    for key in ['Gas Turbine Type', 'Component Type', 'FailureMode', 'failureeffect']:
                        if key in row and pd.notna(row[key]):
                            metadata[key] = str(row[key])[:100]
                    
                    st.session_state.simple_search_docs.append({
                        'document': doc_text,
                        'metadata': metadata,
                        'id': f"doc-{idx}"
                    })
            
            st.success(f"✅ Initialized simple text search with {len(st.session_state.simple_search_docs)} documents")
            st.session_state.system_initialized = True
            return True
        
        st.session_state.chroma_client = chromadb.EphemeralClient()
        
        try:
            st.session_state.collection = st.session_state.chroma_client.get_collection(name="gas_turbine_kb")
            
            collection_count = st.session_state.collection.count()
            if collection_count > 0:
                st.success(f"✅ Found existing knowledge base with {collection_count} documents - loading...")
                time.sleep(0.5)
                st.session_state.system_initialized = True
                return True
            else:
                st.info("📝 Collection exists but empty - will populate with documents...")
                
        except Exception:
            try:
                st.session_state.collection = st.session_state.chroma_client.create_collection(
                    name="gas_turbine_kb",
                    embedding_function=embedding_fn
                )
                st.info("📝 Created new collection - will populate with documents...")
            except Exception as e:
                st.error(f"❌ Error creating ChromaDB collection: {str(e)}")
                return False
        
        st.info("🔄 Processing and adding documents to knowledge base...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_size = 50
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            status_text.text(f"Processing batch {batch_idx + 1}/{total_batches}...")
            
            documents, metadatas, ids = process_documents_batch(df, start_idx, end_idx)
            
            if documents:
                try:
                    st.session_state.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                except Exception as e:
                    st.error(f"❌ Error adding batch {batch_idx + 1}: {str(e)}")
                    continue
            
            progress = (batch_idx + 1) / total_batches
            progress_bar.progress(progress)
            
            if batch_idx % 5 == 0:
                gc.collect()
                time.sleep(0.1)
        
        progress_bar.progress(1.0)
        status_text.text("✅ System initialized successfully!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.system_initialized = True
        return True

def simple_text_search(query: str, n_results: int = 3) -> List[SearchResult]:
    """Simple text-based search when embeddings are unavailable"""
    if 'simple_search_docs' not in st.session_state:
        return []
    
    query_lower = query.lower()
    query_words = query_lower.split()
    
    scored_docs = []
    for doc in st.session_state.simple_search_docs:
        doc_text_lower = doc['document'].lower()
        
        # Simple scoring based on keyword matches
        score = 0
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                count = doc_text_lower.count(word)
                score += count * len(word)  # Longer words get higher weight
        
        if score > 0:
            # Normalize score roughly
            normalized_score = min(score / 100, 1.0)
            scored_docs.append((doc, normalized_score))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Convert to SearchResult objects
    results = []
    for doc, score in scored_docs[:n_results]:
        results.append(SearchResult(
            document=doc['document'],
            metadata=doc['metadata'],
            score=score
        ))
    
    return results

def search_documents(query: str, n_results: int = 3) -> List[SearchResult]:
    if not query:
        return []
    
    # Check if we're using simple text search mode
    if hasattr(st.session_state, 'simple_search_docs'):
        return simple_text_search(query, n_results)
    
    # Use ChromaDB search
    if st.session_state.collection is None:
        return []
    
    try:
        results = st.session_state.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        search_results = []
        for doc, meta, score in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            search_results.append(SearchResult(
                document=doc,
                metadata=meta,
                score=1 - score
            ))
        
        return search_results
        
    except Exception as e:
        st.error(f"❌ Search error: {str(e)}")
        return []

def generate_summary(context: str, query: str) -> ModelSummary:
    start_time = time.time()
    
    tokenizer, model, device = load_ai_model()
    if tokenizer is None or model is None:
        # Provide a meaningful fallback analysis
        keywords = query.lower().split()
        context_lower = context.lower()
        
        relevant_info = []
        for keyword in keywords:
            if keyword in context_lower and len(keyword) > 3:
                sentences = context.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 10:
                        relevant_info.append(sentence.strip())
                        break
        
        if relevant_info:
            fallback_response = f"Based on the maintenance records, here are key findings related to '{query}':\n\n" + "\n\n".join(relevant_info[:3])
        else:
            fallback_response = f"I found relevant maintenance records for '{query}'. Please review the source documents below for detailed information about failure causes, effects, and maintenance procedures."
        
        return ModelSummary(
            model_name="Search-Only Mode",
            summary=fallback_response,
            response_time=time.time() - start_time
        )
    
    # Determine model name from tokenizer
    model_name = getattr(tokenizer, 'name_or_path', 'AI-Model')
    if 'flan-t5-small' in model_name.lower():
        display_name = "Flan-T5-Small"
    elif 'flan-t5-base' in model_name.lower():
        display_name = "Flan-T5-Base"
    elif 't5-small' in model_name.lower():
        display_name = "T5-Small"
    else:
        display_name = "AI-Model"
    
    prompt = f"""Task: Answer the following question about gas turbine maintenance using only the provided context.

Context:
{context[:1200]}

Question: {query}

Instructions: Provide a detailed answer that includes specific failure causes, maintenance procedures, and recommendations based on the context above."""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                min_new_tokens=20,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
                early_stopping=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.replace(prompt, "").strip()
        
        if not answer or len(answer) < 10:
            answer = f"Based on the maintenance data, I found relevant information about {query.lower()}. Please check the detailed search results below for specific maintenance procedures and failure analysis."
        
        return ModelSummary(
            model_name=display_name,
            summary=answer,
            response_time=time.time() - start_time
        )
        
    except Exception as e:
        return ModelSummary(
            model_name=display_name,
            summary=f"Generated analysis from search results: Please review the source documents below for detailed information about {query.lower()} including failure causes, effects, and maintenance procedures.",
            response_time=time.time() - start_time
        )

def display_search_results(search_results: List[SearchResult]):
    for i, result in enumerate(search_results):
        with st.expander(f"📄 Document {i+1} - Similarity: {result.score:.3f}", expanded=i==0):
            st.text_area(
                "Document Content:",
                result.document,
                height=200,
                key=f"doc_content_{i}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Component:**", result.metadata.get('Component Type', 'N/A'))
                st.write("**Turbine:**", result.metadata.get('Gas Turbine Type', 'N/A'))
            
            with col2:
                failure_mode = result.metadata.get('FailureMode', 'N/A')
                if len(str(failure_mode)) > 100:
                    failure_mode = str(failure_mode)[:100] + "..."
                st.write("**Failure Mode:**", failure_mode)
                
                failure_effect = result.metadata.get('failureeffect', 'N/A')
                if len(str(failure_effect)) > 100:
                    failure_effect = str(failure_effect)[:100] + "..."
                st.write("**Failure Effect:**", failure_effect)

def main():
    st.title("⚙️ Gas Turbine RAG System")
    st.markdown("*AI-powered maintenance analysis for gas turbines*")
    
    with st.expander("ℹ️ About This System", expanded=False):
        st.markdown("""
        This **Retrieval-Augmented Generation (RAG)** system provides intelligent gas turbine maintenance analysis optimized for **Hugging Face Free Tier**:
        
        - 🔍 **Semantic Search**: Finds relevant maintenance records using ChromaDB + sentence transformers
        - 🤖 **Lightweight AI Models**: Optimized models that work within free tier memory limits
        - 📊 **Comprehensive Insights**: Combines search results with AI reasoning for detailed answers
        
        **How it works:**
        1. **Select your AI model** (Flan-T5-Small recommended for free tier)
        2. Enter your question about gas turbine failures or maintenance
        3. System searches 3000+ maintenance records for relevant context
        4. AI model analyzes your question using found context for expert-level insights
        5. View both AI analysis and source documents for complete understanding
        
        **Free Tier Optimized Models:**
        - **T5-Small**: 🛡️ Most reliable - loads in 1-2 minutes, good quality
        - **Flan-T5-Small**: 🚀 Balanced - loads in 2-5 minutes, better quality  
        - **Flan-T5-Base**: ⚡ Higher quality - loads in 5-10 minutes, much better quality
        - **Flan-T5-Large**: 🎯 Best quality - loads in 10-15+ minutes, highest quality
        
        **Free Tier Tips for Large Models:**
        - Try during **off-peak hours** (late night/early morning UTC)
        - **Be patient** - large models can take 15+ minutes to download
        - **Don't refresh** the page during loading
        - If it fails, wait 10-15 minutes and try again
        - Consider **Flan-T5-Base** as a good middle ground
        """)
    
    if not initialize_system():
        st.stop()
    
    # Main model selection section
    st.header("🤖 Quick Model Selection")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        quick_options = {
            "t5-small": "🛡️ T5-Small (Most Reliable)",
            "google/flan-t5-small": "🚀 Flan-T5-Small (Good Quality)", 
            "google/flan-t5-base": "⚡ Flan-T5-Base (Better Quality)",
            "google/flan-t5-large": "🎯 Flan-T5-Large (Best Quality)"
        }
        
        current_selection = st.selectbox(
            "Choose your AI model:",
            options=list(quick_options.keys()),
            index=list(quick_options.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in quick_options else 0,
            format_func=lambda x: quick_options[x],
            help="Large models provide best quality but may timeout on free tier. Try during off-peak hours for better success.",
            key="quick_model_select"
        )
        
        if current_selection != st.session_state.selected_model:
            st.session_state.selected_model = current_selection
            st.rerun()
    
    with col2:
        if st.session_state.selected_model in st.session_state.model_cache:
            st.success("✅ Model Ready")
        else:
            if st.button("📥 Load Model", use_container_width=True):
                load_ai_model(st.session_state.selected_model)
                st.rerun()
    
    with col3:
        if st.button("🔧 More Options", use_container_width=True, help="See sidebar for additional model options"):
            st.info("👈 Check the sidebar for more model options and advanced settings!")
    
    # Free tier optimization warning
    st.warning("🌐 **Hugging Face Free Tier**: Large models available but may take 10-15+ minutes to load. Try during off-peak hours for best success. T5-Small loads fastest.")
    
    st.markdown("---")
    
    with st.sidebar:
        st.header("🤖 AI Model Selection")
        
        model_options = {
            "t5-small": "T5-Small (Ultra Reliable for Free Tier)",
            "google/flan-t5-small": "Flan-T5-Small (Good Quality, Reliable)",
            "google/flan-t5-base": "Flan-T5-Base (Better Quality, May Timeout)",
            "google/flan-t5-large": "Flan-T5-Large (Best Quality, High Risk)",
            "t5-base": "T5-Base (Alternative Medium Model)",
            "t5-large": "T5-Large (Large Alternative)",
            "google/t5-v1_1-small": "T5-v1.1-Small (Efficient)",
            "google/t5-v1_1-base": "T5-v1.1-Base (Improved Medium)",
            "google/t5-v1_1-large": "T5-v1.1-Large (Improved Large)"
        }
        
        model_categories = {
            "Most Reliable": ["t5-small"],
            "Recommended": ["google/flan-t5-small", "google/t5-v1_1-small"],
            "Medium Models": ["google/flan-t5-base", "t5-base", "google/t5-v1_1-base"],
            "Large Models (Risky on Free Tier)": ["google/flan-t5-large", "t5-large", "google/t5-v1_1-large"]
        }
        
        # Model selection method choice
        selection_method = st.radio(
            "Selection Method:",
            ["📋 Dropdown", "🎛️ Category View"],
            horizontal=True
        )
        
        if selection_method == "📋 Dropdown":
            selected = st.selectbox(
                "Choose AI Model:",
                options=list(model_options.keys()),
                index=0,
                format_func=lambda x: model_options[x],
                help="Choose model size vs reliability trade-off. Large models give best quality but may timeout on free tier. Try during off-peak hours for large models."
            )
        else:
            st.write("**Select by Category:**")
            selected = None
            
            for category, models in model_categories.items():
                with st.expander(f"{category} Models", expanded=(category == "Free Tier Optimized")):
                    for model in models:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if st.button(
                                model_options[model], 
                                key=f"btn_{model}",
                                use_container_width=True,
                                type="primary" if model == st.session_state.selected_model else "secondary"
                            ):
                                selected = model
                        with col2:
                            if model == st.session_state.selected_model:
                                st.write("✅")
        
        # Update selected model
        if selected and selected != st.session_state.selected_model:
            st.session_state.selected_model = selected
            st.rerun()
        
        # Current selection display
        st.markdown("---")
        st.write(f"**Current Model:** {model_options[st.session_state.selected_model]}")
        
        # Load model controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Load Model", help="Load or reload the selected model"):
                if st.session_state.selected_model in st.session_state.model_cache:
                    del st.session_state.model_cache[st.session_state.selected_model]
                load_ai_model(st.session_state.selected_model)
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache", help="Clear all cached models to free memory"):
                st.session_state.model_cache.clear()
                st.success("Cache cleared!")
                st.rerun()
        
        # Free tier optimization tip
        st.info("💡 **Free Tier Tips:**\n- T5-Small: Ultra reliable, 1-2 min load\n- Flan-T5-Small: Good quality, 2-5 min load\n- Flan-T5-Large: Best quality, 10-15+ min load\n- Try large models during off-peak hours (late night UTC)")
        
        # Model information
        with st.expander("ℹ️ Model Information", expanded=False):
            if "flan-t5" in st.session_state.selected_model:
                model_size = st.session_state.selected_model.split("-")[-1]
                st.info(f"**Flan-T5 {model_size.title()}**\n\nInstruction-tuned model optimized for question-answering tasks. Good for maintenance analysis.")
            elif "t5-v1_1" in st.session_state.selected_model:
                st.info("**T5 v1.1**\n\nImproved version of T5 with better performance and efficiency.")
            else:
                st.info("**T5 Model**\n\nText-to-text transfer transformer. General purpose language model.")
        
        st.header("📊 System Status")
        if st.session_state.df is not None:
            st.success(f"✅ {len(st.session_state.df)} records loaded")
            
            # Show AI model status
            if st.session_state.selected_model in st.session_state.model_cache:
                model_name = model_options[st.session_state.selected_model].split(" (")[0]
                st.success(f"🤖 {model_name} Ready")
            else:
                st.warning(f"⚠️ {model_options[st.session_state.selected_model].split(' (')[0]} Not Loaded")
            
            # Show search method
            if hasattr(st.session_state, 'simple_search_docs') and st.session_state.simple_search_docs:
                st.info("🔍 Simple Text Search")
            else:
                st.info("💾 ChromaDB Vector Search")
        
        st.header("⚙️ Search Settings")
        n_results = st.slider(
            "Number of results:",
            min_value=1,
            max_value=8,
            value=3,
            help="How many relevant documents to retrieve"
        )
        
        st.header("📈 Dataset Info")
        if st.session_state.df is not None:
            st.metric("Total Records", len(st.session_state.df))
            
            if 'Component Type' in st.session_state.df.columns:
                unique_components = st.session_state.df['Component Type'].nunique()
                st.metric("Unique Components", unique_components)
            
            if 'Gas Turbine Type' in st.session_state.df.columns:
                turbine_types = st.session_state.df['Gas Turbine Type'].value_counts()
                st.write("**Turbine Types:**")
                for turbine, count in turbine_types.head(5).items():
                    st.write(f"• {turbine}: {count}")
    
    st.header("🔍 Ask Your Question")
    
    sample_questions = [
        "What failures commonly occur in battery banks?",
        "How should I maintain circuit breakers?",
        "What causes coupling vibration problems?",
        "Tell me about terminal corrosion issues",
        "What are the effects of insufficient cooling?"
    ]
    
    st.write("**💡 Try these sample questions:**")
    cols = st.columns(len(sample_questions))
    for i, question in enumerate(sample_questions):
        with cols[i]:
            if st.button(f"📝 {question[:25]}...", key=f"sample_{i}"):
                st.session_state.current_query = question
    
    query = st.text_input(
        "Enter your question:",
        value=st.session_state.get('current_query', ''),
        placeholder="e.g., What causes battery failure in gas turbines?",
        key="main_query"
    )
    
    if st.button("🚀 Search & Analyze", type="primary"):
        if not query:
            st.warning("Please enter a question first!")
        else:
            with st.spinner("🔍 Searching knowledge base..."):
                search_results = search_documents(query, n_results)
                
                if not search_results:
                    st.warning("❌ No relevant documents found. Try rephrasing your question.")
                else:
                    st.subheader("🤖 AI Analysis of Your Question")
                    
                    with st.spinner("🧠 AI is analyzing your question using maintenance data..."):
                        context = "\n\n".join([result.document for result in search_results])
                        summary = generate_summary(context, query)
                        
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown("**🎯 Expert Analysis:**")
                            st.write(summary.summary)
                        
                        with col2:
                            st.metric("⏱️ Response Time", f"{summary.response_time:.2f}s")
                            st.metric("🤖 AI Model", summary.model_name)
                    
                    st.subheader(f"📚 Source Documents ({len(search_results)} found)")
                    display_search_results(search_results)
                    
                    if len(search_results) > 1:
                        st.subheader("📊 Search Results Analysis")
                        
                        scores = [result.score for result in search_results]
                        doc_labels = [f"Doc {i+1}" for i in range(len(scores))]
                        
                        fig = px.bar(
                            x=doc_labels,
                            y=scores,
                            title="Document Similarity Scores",
                            labels={'x': 'Document', 'y': 'Similarity Score'},
                            color=scores,
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 14px;'>
        🚀 Deployed on Hugging Face Spaces | Built with Streamlit, ChromaDB & Flan-T5
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()