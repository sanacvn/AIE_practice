
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter

# Load API key
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY tidak ditemukan di .env")

# 1. Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI Engineering tutor."
    ),
    (
        "human",
        """
        Explain the concept simply:

        {input}
        """
    )
])

# 2. Model
model = ChatOpenRouter(
    model="openrouter/free",
    temperature=0.3,
    api_key=API_KEY
)

# 3. Output Parser
parser = StrOutputParser()

# 4. LCEL Chain
chain = prompt | model | parser

# 5. Invoke
result = chain.invoke({
    "input": "RAG"
})

print(result)

# Answer

# # RAG: Retrieval-Augmented Generation
# ### *The "Open-Book Exam" for LLMs*

# ---

# ## 1. The Core Concept (TL;DR)
# **RAG is an architecture that connects an LLM to an external knowledge base (your data) so it can answer questions using specific, up-to-date, or private information it wasn't trained on.**

# > **Analogy:** 
# > *   **Standard LLM:** A student taking a **closed-book exam**. They rely solely on memory (training weights). They hallucinate if they forget details.
# > *   **RAG:** A student taking an **open-book exam**. They search the textbook (Vector DB), read the relevant pages (Retrieval), and *then* write the answer (Generation).

# ---

# ## 2. Why do we need it? (The Problem)
# LLMs have three fundamental limitations:

# | Limitation | Description |
# | :--- | :--- |
# | **Knowledge Cutoff** | Training data is frozen in time (e.g., "I don't know about the iPhone 16"). |
# | **No Private Data** | They don't know your company's HR policy, your codebase, or your PDF contracts. |
# | **Hallucination** | They confidently generate plausible-sounding lies when they don't know the answer. |

# **RAG solves all three** by grounding the model in *retrieved facts* rather than *parametric memory*.

# ---

# ## 3. The Architecture: The 3-Stage Pipeline

# ### Phase 0: Offline Ingestion (Done once / periodically)
# > **Goal:** Make your data searchable.

# 1.  **Load:** Read PDFs, Notion, Confluence, Code, SQL, APIs.
# 2.  **Split (Chunking):** Break large docs into small, semantically meaningful pieces (e.g., 512 tokens with overlap).
# 3.  **Embed:** Pass chunks through an **Embedding Model** (e.g., `text-embedding-3-large`, `bge-large`, `nomic-embed-text`) → **Vectors (arrays of floats)**.
# 4.  **Store & Index:** Save vectors + metadata (source, page #) in a **Vector Database** (Pinecone, Weaviate, Qdrant, Milvus, PGVector, Chroma).

# ---

# ### Phase 1: Retrieval (Runtime - "The Search")
# > **Goal:** Find the *top-k* most relevant chunks for the user query.

# 1.  **Embed Query:** Convert user question into a vector using the *same* embedding model.
# 2.  **Vector Search (ANN):** Approximate Nearest Neighbor search in the Vector DB (Cosine Similarity / Dot Product).
# 3.  **(Optional) Hybrid Search:** Combine **Semantic Search** (vectors) + **Keyword Search** (BM25/Sparse vectors) for exact matches (error codes, IDs).
# 4.  **(Advanced) Reranking:** Feed `Query + Top 20 Candidates` into a **Cross-Encoder Reranker** (e.g., `bge-reranker`, Cohere Rerank) to reorder by true relevance.

# ---

# ### Phase 2: Augmentation (Runtime - "The Prompt Engineering")
# > **Goal:** Build a context-rich prompt.

# ```text
# SYSTEM: You are a helpful assistant. Answer ONLY using the context below. 
#         If the answer isn't there, say "I don't know."

# CONTEXT:
# [Chunk 1: "Acme Corp refund policy: 30 days, no questions asked..."]
# [Chunk 2: "Exceptions: Custom furniture is final sale..."]
# [Chunk 3: "Contact support@acme.com for escalations..."]

# USER QUESTION: "Can I return a custom sofa?"
# ```

# ---

# ### Phase 3: Generation (Runtime - "The Answer")
# > **Goal:** Synthesize the answer.

# *   Send the augmented prompt to the **LLM** (GPT-4o, Claude 3.5, Llama 3, Mistral).
# *   **Stream** the response back to the user.
# *   **Citations:** Parse LLM output to link sentences back to source chunks (critical for trust).

# ---

# ## 4. Key Technical Components Deep Dive

# ### A. Chunking Strategies (Critical for Quality)
# | Strategy | Best For | Pros/Cons |
# | :--- | :--- | :--- |
# | **Fixed Size + Overlap** | General purpose | Simple; breaks semantic flow. |
# | **Recursive (LangChain default)** | General text | Tries separators `["\n\n", "\n", " ", ""]` hierarchically. |
# | **Semantic Chunking** | Complex docs | Uses embedding similarity to split where topic changes. Expensive. |
# | **Document Specific** | Code/Markdown/HTML | Splits by function/class (code) or headers (MD/HTML). **Highly recommended.** |

# ### B. Embedding Models
# *   **Open Source (Local):** `bge-m3` (multi-lingual + dense/sparse/colbert), `nomic-embed-text-v1.5`, `e5-mistral-7b`.
# *   **API:** OpenAI `text-embedding-3-large` (best general perf), Cohere `embed-english-v3.0`, Voyage AI (code/finance/legal specialized).
# *   **Matters:** Dimension size (storage cost), Max tokens (chunk size limit), Matryoshka support (truncatable dims).

# ### C. Vector Databases
# | Category | Examples | Use Case |
# | :--- | :--- | :--- |
# | **Managed Cloud** | Pinecone, Weaviate Cloud, Zilliz, Upstash | Zero ops, auto-scaling, enterprise features. |
# | **Self-Hosted / OSS** | Qdrant, Milvus, Weaviate, Chroma | Cost control, data privacy, custom hardware. |
# | **SQL Extensions** | **PGVector (Postgres)**, SQLite-VSS | **Best for simple stacks:** ACID, SQL joins, single DB. |

# ---

# ## 5. Advanced RAG Patterns (Beyond "Naive RAG")

# ### 1. Query Transformation (Better Retrieval)
# *   **Query Rewriting:** LLM rewrites vague user query into precise search query.
# *   **HyDE (Hypothetical Document Embeddings):** LLM generates a *fake answer*, embeds *that*, searches for similar real docs.
# *   **Step-back Prompting:** Ask high-level concept question first, retrieve broad context, then answer specific question.
# *   **Sub-query Decomposition:** Break complex query ("Compare Q1 vs Q2 revenue") into parallel sub-queries.

# ### 2. Advanced Retrieval
# *   **Parent Document Retriever:** Store small chunks (child) for search, but return larger parent chunks (whole section/page) to LLM for context.
# *   **Sentence Window Retrieval:** Embed single sentences, retrieve surrounding window (e.g., ±2 sentences).
# *   **Knowledge Graph RAG (GraphRAG):** Extract entities/relationships → Build Graph → Traverse graph for multi-hop reasoning ("Who is the CEO of the company that acquired X?").

# ### 3. Agentic RAG (The Current Frontier)
# The LLM becomes an **Agent** with tools:
# *   `search_vector_db(query)`
# *   `search_web(query)`
# *   `query_sql(sql)`
# *   `read_file(path)`
# *   **Loop:** Plan → Tool Call → Observe → Reflect → Re-plan → Answer.

# ---

# ## 6. Evaluation: How do you know it works?
# Don't just "vibe check." Measure:

# | Metric | What it measures | Tooling |
# | :--- | :--- | :--- |
# | **Context Relevance** | Are retrieved chunks actually relevant? | `Ragas`, `TruLens`, `DeepEval` |
# | **Faithfulness / Groundedness** | Does the answer *only* use the context? (Anti-hallucination) | `Ragas`, `TruLens` |
# | **Answer Relevance** | Does it answer the specific user question? | `Ragas` |
# | **Hit Rate / MRR** | Did the *correct* chunk make it into Top-K? | Custom eval scripts |

# **Golden Rule:** Build an **Eval Set** (50-100 Q/A pairs with ground truth context) *before* you tune chunk size, embedding model, or prompts.

# ---

# ## 7. Common Pitfalls & Fixes

# | Symptom | Likely Cause | Fix |
# | :--- | :--- | :--- |
# | "I don't know" on known data | Retrieval failed (bad chunks / bad embedding) | Check chunking; try Hybrid Search; try Reranker. |
# | Hallucinates despite context | Prompt instructions weak; Context too long/noisy | Strict system prompt; Rerank; Reduce Top-K; "Cite sources". |
# | Slow latency | Large chunks, heavy reranker, slow LLM | Smaller chunks; Async retrieval; Smaller LLM (Haiku/3.5 Sonnet/8B); Cache embeddings. |
# | High Cost | Sending too many tokens to LLM | Summarize chunks before sending; Parent Document Retrieval; Route simple queries to small model. |

# ---

# ## 8. Minimal "Hello World" Code (Python / LangChain / OpenAI / Chroma)

# ```python
# # pip install langchain-openai langchain-chroma langchain-text-splitters pypdf

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

# # 1. LOAD & SPLIT
# loader = PyPDFLoader("company_handbook.pdf")
# docs = loader.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = splitter.split_documents(docs)

# # 2. EMBED & STORE (Vector DB)
# vectorstore = Chroma.from_documents(
#     documents=chunks, 
#     embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
#     persist_directory="./chroma_db" # Persists to disk
# )
# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# # 3. PROMPT & LLM
# system_prompt = (
#     "You are an HR assistant. Use the retrieved context to answer. "
#     "If you don't know, say you don't know. Cite sources."
#     "\n\n{context}"
# )
# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}"),
# ])

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# # 4. CHAIN: Retrieval -> Augmentation -> Generation
# # create_stuff_documents_chain handles the "Augmentation" (stuffing docs into {context})
# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)

# # 5. RUN
# response = rag_chain.invoke({"input": "What is the parental leave policy?"})
# print(response["answer"])
# print("\nSources:", [doc.metadata.get("page") for doc in response["context"]])
# ```

# ---

# ## 9. Summary Checklist for Production
# - [ ] **Data Hygiene:** Clean PDFs (remove headers/footers), parse tables correctly.
# - [ ] **Metadata Filtering:** Filter by `doc_type`, `date`, `department`, `version` *before* vector search (Pre-filtering).
# - [ ] **Reranker:** Almost always worth the ~50-100ms latency for quality jump.
# - [ ] **Streaming:** Stream tokens from LLM to UI immediately (UX critical).
# - [ ] **Observability:** Log `Query -> Retrieved Chunks -> Prompt -> Response -> User Feedback (👍/👎)`.
# - [ ] **Guardrails:** PII detection, Prompt Injection detection, Competitor mention blocking.

# ---

# **Next Steps for You:**
# 1.  Build the "Hello World" above with a PDF you have lying around.
# 2.  Swap `Chroma` for `PGVector` (if you use Postgres) or `Qdrant`.
# 3.  Add a **Reranker** (Cohere Rerank API or local `bge-reranker-v2-m3` via `sentence-transformers`).
# 4.  Build an **Evaluation Set** (JSONL of Q/A) and run `Ragas` on it.
# LangChain berhasil!