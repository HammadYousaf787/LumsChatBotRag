"""
RAG Pipeline for LUMS Academic Knowledge Assistant
Using advanced logging configuration
"""

import os
import re
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Import advanced logging
from logging_config import get_logger, LoggerContext, log_function_call

# Initialize logger
logger = get_logger(__name__)

# Try to import config, fallback to hardcoded values if needed
try:
    import config
except (ImportError, ValueError, SyntaxError) as e:
    logger.warning(f"Config loading failed: {e}. Using fallback values.")
    
    # Fallback config
    class config:
        GROQ_API_KEY = "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"
        CHROMA_DIR = "chroma_db"
        DATA_DIR = "data"
        DEFAULT_MODEL = "llama-3.3-70b-versatile"
        DEFAULT_TEMPERATURE = 0.3
        DEFAULT_MAX_TOKENS = 500
        LOG_LEVEL = "INFO"

def validate_config():
    """Validate configuration settings."""
    if not config.GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY not found in config.py!")
        logger.error("Please set GROQ_API_KEY in config.py")
        raise ValueError("GROQ_API_KEY is required")

# Validate configuration
validate_config()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Import from langchain-classic
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# Custom Exceptions
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class APIError(PipelineError):
    """Exception for API-related errors."""
    pass

class RetrievalError(PipelineError):
    """Exception for retrieval errors."""
    pass

class ConfigurationError(PipelineError):
    """Exception for configuration errors."""
    pass

class LUMSRAGPipeline:
    """RAG pipeline for LUMS academic knowledge assistant with LangChain memory."""
    
    def __init__(self, chroma_dir: str = None, use_conversation: bool = True, retry_attempts: int = 3):
        """
        Initialize the RAG pipeline.
        
        Args:
            chroma_dir: Directory containing ChromaDB (defaults to config)
            use_conversation: Whether to enable conversation memory
            retry_attempts: Number of retry attempts for API calls
        """
        logger.info("🚀 Initializing LUMS RAG Pipeline...")
        
        # Get configuration from config.py
        self.chroma_dir = Path(chroma_dir or config.CHROMA_DIR)
        self.use_conversation = use_conversation
        self.retry_attempts = retry_attempts
        
        # Get default values from config
        self.default_model = config.DEFAULT_MODEL
        self.default_temperature = config.DEFAULT_TEMPERATURE
        self.default_max_tokens = config.DEFAULT_MAX_TOKENS
        
        # Get API key from config
        self.api_key = config.GROQ_API_KEY
        
        self.config = {}
        self.last_retrieval_info = {}
        self.error_log = []
        self.query_count = 0
        self.total_queries = 0
        
        # Log configuration
        logger.info(f"📋 Configuration:")
        logger.info(f"   Model: {self.default_model}")
        logger.info(f"   Temperature: {self.default_temperature}")
        logger.info(f"   Max Tokens: {self.default_max_tokens}")
        logger.info(f"   Chroma Dir: {self.chroma_dir}")
        logger.info(f"   Conversation: {'Enabled' if use_conversation else 'Disabled'}")
        
        # Validate chroma directory
        if not self.chroma_dir.exists():
            logger.warning(f"⚠️ Chroma directory not found: {self.chroma_dir}")
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        try:
            with LoggerContext(logger, "Pipeline Initialization"):
                # Initialize embeddings
                self.embeddings = self._initialize_embeddings()
                
                # Load vectorstore
                self.vectorstore = self._initialize_vectorstore()
                
                # Create retriever
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                
                # Initialize LLM
                self.llm = self._initialize_llm()
                
                # Initialize LangChain Memory
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
                
                # Create the conversational chain
                if self.use_conversation:
                    self.qa_chain = self._create_conversational_chain()
                else:
                    self.qa_chain = None
            
            logger.info("✅ Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            raise PipelineError(f"Failed to initialize pipeline: {str(e)}")
    
    def get_llm_info(self):
        """Get information about the current LLM."""
        if hasattr(self.llm, 'model_name'):
            return f"{self.llm.model_name}"
        elif hasattr(self.llm, '__class__'):
            return self.llm.__class__.__name__
        return "Unknown LLM"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        memory_entries = 0
        if self.memory:
            try:
                memory_vars = self.memory.load_memory_variables({})
                memory_entries = len(memory_vars.get('chat_history', []))
            except:
                pass
        
        return {
            "total_queries": self.total_queries,
            "session_queries": self.query_count,
            "errors": len(self.error_log),
            "conversation_enabled": self.use_conversation,
            "memory_entries": memory_entries
        }
    
    def get_config(self):
        """Get current configuration."""
        return self.config
    
    def get_error_log(self):
        """Get error log."""
        return self.error_log
    
    @log_function_call(logger)
    def _initialize_embeddings(self):
        """Initialize embeddings with error handling."""
        try:
            logger.info("🔄 Loading embeddings model...")
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embeddings model loaded successfully")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            raise PipelineError(f"Embeddings initialization failed: {str(e)}")
    
    @log_function_call(logger)
    def _initialize_vectorstore(self):
        """Initialize vectorstore with error handling."""
        try:
            logger.info(f"🔄 Loading vectorstore from: {self.chroma_dir}")
            vectorstore = Chroma(
                persist_directory=str(self.chroma_dir),
                embedding_function=self.embeddings,
                collection_name="lums_academic_knowledge"
            )
            
            # Check if vectorstore is empty
            count = vectorstore._collection.count()
            if count == 0:
                logger.warning("⚠️ Vectorstore is empty! Please run ingest.py first.")
                raise RetrievalError("No documents found in the database")
            
            logger.info(f"✅ Vectorstore loaded with {count} documents")
            return vectorstore
        except Exception as e:
            logger.error(f"❌ Failed to load vectorstore: {e}")
            raise RetrievalError(f"Vectorstore loading failed: {str(e)}")
    
    @log_function_call(logger)
    def _initialize_llm(self):
        """Initialize the language model with Groq with retry logic."""
        if not self.api_key:
            logger.error("❌ No API key available. Using mock LLM.")
            return MockLLM()
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                from langchain_groq import ChatGroq
                config = getattr(self, 'config', {})
                model = config.get('model', self.default_model)
                temperature = config.get('temperature', self.default_temperature)
                max_tokens = config.get('max_tokens', self.default_max_tokens)
                
                logger.info(f"🔄 Initializing Groq with model: {model} (attempt {attempt}/{self.retry_attempts})")
                llm = ChatGroq(
                    model=model,
                    temperature=temperature,
                    api_key=self.api_key,
                    max_tokens=max_tokens,
                    timeout=30
                )
                
                # Test the connection
                test_response = llm.invoke("Say hello")
                logger.info("✅ Groq connection successful")
                return llm
                
            except ImportError:
                logger.error("❌ langchain-groq not installed. Run: pip install langchain-groq")
                return MockLLM()
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt} failed: {e}")
                if attempt == self.retry_attempts:
                    logger.error(f"❌ All {self.retry_attempts} attempts failed")
                    return MockLLM()
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return MockLLM()
    
    @log_function_call(logger)
    def _create_conversational_chain(self):
        """Create conversational chain with error handling."""
        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                verbose=True,
                chain_type="stuff",
                combine_docs_chain_kwargs={
                    "prompt": self._create_prompt_template()
                }
            )
            return chain
        except Exception as e:
            logger.error(f"❌ Failed to create conversational chain: {e}")
            raise PipelineError(f"Conversational chain creation failed: {str(e)}")
    
    def _create_prompt_template(self):
        """Create the system prompt template with chat history."""
        template = """You are a LUMS Academic Assistant. Your purpose is to help students find information about LUMS policies, courses, and schedules.

Instructions:
1. Answer questions ONLY using the provided context from LUMS documents.
2. If the answer is not present in the provided context, respond: "I could not find this information in the provided LUMS documents."
3. Do not fabricate information or use outside knowledge.
4. Always provide sources/citations for your answers using the document metadata.
5. Be precise, concise, and helpful in your responses.

Context from Documents:
{context}

Chat History:
{chat_history}

Current Question:
{question}

Remember: Use the chat history to understand follow-up questions. Only use information from the context above.

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
    
    def apply_config(self, model=None, temperature=None, max_tokens=None):
        """Apply configuration and reinitialize the chain."""
        try:
            logger.info("🔄 Applying new configuration...")
            
            self.config = {
                'model': model or self.default_model,
                'temperature': temperature if temperature is not None else self.default_temperature,
                'max_tokens': max_tokens or self.default_max_tokens
            }
            
            logger.info(f"📋 New config: {self.config}")
            
            self.llm = self._initialize_llm()
            
            if self.use_conversation:
                self.qa_chain = self._create_conversational_chain()
            
            logger.info(f"✅ Configuration applied successfully")
            return self.config
        except Exception as e:
            logger.error(f"❌ Configuration update failed: {e}")
            raise ConfigurationError(f"Failed to apply configuration: {str(e)}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a user query and return response with sources."""
        start_time = time.time()
        self.total_queries += 1
        self.query_count += 1
        
        # Log query
        logger.info(f"📝 Query #{self.total_queries}: {question[:100]}...")
        
        # Validate input
        if not question or not question.strip():
            logger.warning("⚠️ Empty question received")
            return {
                "answer": "Please ask a valid question.",
                "sources": []
            }
        
        try:
            with LoggerContext(logger, f"Query #{self.total_queries}"):
                # If using conversation chain
                if self.use_conversation and self.qa_chain:
                    try:
                        response = self.qa_chain.invoke({"question": question})
                    except Exception as e:
                        logger.error(f"❌ Chain invocation failed: {e}")
                        # Try to recover by recreating the chain
                        logger.info("🔄 Attempting to recover by recreating chain...")
                        self.qa_chain = self._create_conversational_chain()
                        response = self.qa_chain.invoke({"question": question})
                    
                    answer = response.get("answer", "No answer generated")
                    
                    # Extract sources from the retrieved docs
                    sources = []
                    if "source_documents" in response:
                        for doc in response["source_documents"]:
                            source_info = self._format_source(doc)
                            if source_info not in sources:
                                sources.append(source_info)
                    
                    # Ensure sources are in the answer
                    if sources and "Sources:" not in answer:
                        answer += f"\n\n**Sources:**\n" + "\n".join(sources)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Query completed in {elapsed:.2f}s")
                    logger.info(f"   Sources: {len(sources)} documents")
                    
                    return {
                        "answer": answer,
                        "sources": sources,
                        "chat_history": self.memory.load_memory_variables({}),
                        "elapsed_time": elapsed
                    }
                
                else:
                    # Fallback to manual query without memory
                    docs = self.retriever.get_relevant_documents(question)
                    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
                    sources = []
                    for doc in docs:
                        source_info = self._format_source(doc)
                        if source_info not in sources:
                            sources.append(source_info)
                    
                    prompt = self._create_prompt_template().format(
                        context=context,
                        chat_history="",
                        question=question
                    )
                    
                    if hasattr(self.llm, 'invoke'):
                        response_obj = self.llm.invoke(prompt)
                        answer = response_obj.content
                        
                        # Ensure sources are in the answer
                        if sources and "Sources:" not in answer:
                            answer += f"\n\n**Sources:**\n" + "\n".join(sources)
                    else:
                        answer = f"Mock response to: {question}"
                    
                    elapsed = time.time() - start_time
                    
                    return {
                        "answer": answer,
                        "sources": sources,
                        "elapsed_time": elapsed
                    }
                
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            self.error_log.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "error": str(e)
            })
            return {
                "answer": "I apologize, but I encountered an error. Please try again later.",
                "sources": [],
                "error": str(e)
            }
    
    def _format_source(self, doc: Document) -> str:
        """Format source information from document metadata with better detail."""
        metadata = doc.metadata
        doc_type = metadata.get("document_type", "unknown")

        # Get the filename (full name with extension)
        file_name = metadata.get("file_name", metadata.get("source", "Unknown Document"))

        # Get page number
        page = metadata.get("page", "N/A")

        # Get course code if available
        course_code = metadata.get("course_code", "")

        # Clean up filename (remove extension for display)
        display_name = file_name.replace(".pdf", "").replace(".PDF", "")

        # Truncate long names if needed (keep first 50 chars)
        if len(display_name) > 60:
            display_name = display_name[:57] + "..."

        if doc_type == "handbook":
            return f"📖 {display_name} (Page {page})"
        elif doc_type == "course_outline":
            if course_code:
                return f"📚 {course_code} - {display_name} (Page {page})"
            else:
                return f"📚 {display_name} (Page {page})"
        elif doc_type == "schedule":
            day = metadata.get("day", "")
            time_info = metadata.get("time", "")
            if course_code and day:
                return f"📅 {course_code} - {day} (Page {page})"
            elif course_code:
                return f"📅 {course_code} Schedule (Page {page})"
            elif day and time_info:
                return f"📅 {display_name} - {day} {time_info} (Page {page})"
            else:
                return f"📅 {display_name} (Page {page})"
        else:
            return f"📄 {display_name} (Page {page})"
    
    def clear_history(self):
        """Clear conversation history using LangChain's memory."""
        if self.memory:
            self.memory.clear()
            logger.info("🗑️ Conversation history cleared")
    
    def get_history(self):
        """Get conversation history from LangChain's memory."""
        if self.memory:
            return self.memory.load_memory_variables({})
        return {}

class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self):
        self.model_name = "mock-llm"
    
    def invoke(self, prompt):
        class Response:
            def __init__(self, content):
                self.content = content
        return Response("🔧 Mock response. Set GROQ_API_KEY in config.py for real answers!")