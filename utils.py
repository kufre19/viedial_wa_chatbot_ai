import os
import sys

try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
import datetime
import openai
import chromadb
import traceback
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pypdf
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging
from sentence_transformers import CrossEncoder

load_dotenv()
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'app_cache'
# Load the cross-encoder model once at startup (avoids reloading on every request)
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME'),
    token=os.getenv("HF_TOKEN"),
    model_kwargs={"cache_dir": os.getenv('SENTENCE_TRANSFORMERS_HOME') }
    )


# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# Initialize OpenAI client with API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=openai_api_key)

# Create persistent storage directory if it doesn't exist
db_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(db_directory, exist_ok=True)

# Create logs directory if it doesn't exist
logs_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_directory, exist_ok=True)

# Initialize ChromaDB with persistent storage and OpenAI embedding function
chroma_client = chromadb.PersistentClient(path=db_directory)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-3-small"
)
diabetes_collection = chroma_client.get_or_create_collection(
    name="diabetes_docs", embedding_function=openai_ef
)


def log_interaction(user_question,formulated_query, history, context_chunks, prompt, response):
    """Log the interaction details to a file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_directory, f"interaction_{timestamp}.json")

    log_data = {
        "timestamp": timestamp,
        "user_question": user_question,
        "formulated_query": formulated_query,
        "history": history,
        "context_chunks": context_chunks,
        "prompt": prompt,
        "response": response,
    }

    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_file


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF, page by page."""
    pages = []
    try:
        # Open the PDF
        doc = pypdf.PdfReader(pdf_path)

        # Extract text from each page and store separately
        for i, page in enumerate(doc.pages):
            page_text = page.extract_text()
            if page_text.strip():
                pages.append(page_text)

        return pages
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise


def semantic_chunk(text, max_sentences_per_chunk=10, similarity_threshold=0.7):
    """Split text into semantic chunks using embedding similarity."""
    # Split into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    # Get embeddings for all sentences
    response = openai_client.embeddings.create(
        input=sentences,
        model="text-embedding-3-small"
    )
    embeddings = np.array([embedding.embedding for embedding in response.data])
    
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = [embeddings[0]]
    
    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0]
        
        if (len(current_chunk) < max_sentences_per_chunk and 
            sim > similarity_threshold):
            current_chunk.append(sentences[i])
            current_embedding.append(embeddings[i])
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
            current_embedding = [embeddings[i]]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks




def rerank_chunks(query, chunks, top_n=5):
    """
    Rerank a list of chunks against the query using a cross-encoder model.

    The cross-encoder scores each (query, chunk) pair together — unlike ChromaDB
    which compares separate embeddings — giving more accurate relevance scores.
    Returns the top_n most relevant chunks sorted by score descending.
    """
    # Build (query, chunk) pairs for the model to score
    pairs = [(query, chunk) for chunk in chunks]

    # Score all pairs — returns a float score per pair (higher = more relevant)
    scores = reranker_model.predict(pairs)

    # Sort chunks by score descending and return the top N
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]


def search_similar_chunks(query, initial_n=20, final_n=5):
    """
    Retrieve candidate chunks from ChromaDB then rerank them.

    We cast a wide net (initial_n=20) with ChromaDB's fast vector search,
    then use the cross-encoder to pick the most relevant final_n chunks.
    """
    results = diabetes_collection.query(query_texts=[query], n_results=initial_n)
    candidate_chunks = results["documents"][0]

    if not candidate_chunks:
        return []

    # Rerank candidates and return only the best ones
    return rerank_chunks(query, candidate_chunks, top_n=final_n)


def generate_response(query, context_chunks, history=[]):
    """Generate a response from OpenAI with the context pages."""
    system_prompt = """
    You are a helpful medical assistant specializing in diabetes, hypertension, or cardiovascular disease care and management and also during pregnancy.
    You are answering questions from patients who use the Viedial mobile and web application for the following topics only:
    1.Diabetes
    2.Hypertension
    3.Cardiovascular Disease
    4.Gestational Diabetes
    5.Hypertension and Pregnancy
  

    IF A QUESTION FROM A USER DOES NOT PERTAIN TO DIABETES PLEASE RESPOND THAT YOU DO NOT HAVE AN ANSWER TO THAT.
    IF THE PROVIDED INFORMATION DOES NOT CONTAIN SOMETHING RELEVANT TO THE USER QUESTION, ACKNOWLEDGE THAT AND RESPOND THAT YOU DO NOT HAVE
    AN ANSWER TO THAT!
    IF THE CONTEXT INFORMATION IS EMPTY, PLEASE RESPOND WITH:
    "I apologize, but I don't have enough reliable medical information on that, to properly answer your question about diabetes, hypertension, or cardiovascular disease.

    Here are some follow-up questions you may want to ask your healthcare provider:
    1. What are the most reliable sources for diabetes, hypertension, or cardiovascular disease information?
    2. Can you recommend any diabetes, hypertension, or cardiovascular disease education programs?
    3. What specific aspects of diabetes, hypertension, or cardiovascular disease would you like me to learn more about?"

    When responding:
    1. Be compassionate and conversational, but accurate and precise.
    2. Only use information from the provided context.
    3. If you don't have enough information to answer confidently, acknowledge that.
    4. Don't include citations or references to the source documents.
    5. Provide 2-3 suggested follow-up questions at the end of your response that the user might want to ask next.
    """

    context_str = ' '.join(context_chunks) if context_chunks else "No context available."

    user_prompt = f"""
    Here is some relevant information about diabetes, hypertension, or cardiovascular disease extracted from reliable medical documents:

    CONTEXT INFORMATION:
    {context_str}

    USER QUESTION:
    {query}
    """

    messages = [
        {"role": "system", "content": system_prompt.strip()},
    ] + history + [
        {"role": "user", "content": user_prompt.strip()},
    ]

    stream = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        stream=True

    )

    
    for response in stream:
        content = response.choices[0].delta.content
        
        if content:          # filters None
            yield content


    




def auto_generate_chat_title(question):
    """Auto generate a chat title based on the question."""
    prompt = f"""
    You are a helpful medical assistant specializing in diabetes, hypertension, or cardiovascular disease care and management.
    You are answering questions from patients for the following topics only:
    1.Daibetes
    2.Hypertension
    3.Cardiovascular Disease


    I'll provide you with a question from a user. generate a title for a chat based on the question and return only the title and keep it short.

    USER QUESTION:
    {question}

    RESPONSE (ONLY THE TITLE):
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful, accurate, and compassionate diabetes, hypertension, or cardiovascular disease care assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1000,
    )
    response_content = response.choices[0].message.content
    return {"success": True, "title": response_content}






def auto_generate_question_query(topic):
    """Auto generate a question/query to be used based on the topic."""
    prompt = f"""
    You are a helpful medical assistant specializing in diabetes, hypertension, or cardiovascular disease care and management.
    You help generate question/search queries that will be used by the agent system to query our database for more information for the following topics only:
    1.Daibetes
    2.Hypertension
    3.Cardiovascular Disease
    4.Pregnancy and Diabetes/Hypertension
    
    Your query should be able to invoke rich content when used to query the database, users should be able to get full information on what the topic is about and
    the information provided to them should be usesful.



    TOPIC:
    {topic}

    RESPONSE (ONLY THE SEARCH QUERY):
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful, accurate, and compassionate diabetes, hypertension, or cardiovascular disease care assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1000,
    )
    response_content = response.choices[0].message.content
    return {"success": True, "query": response_content}






def get_answer_for_question(question, history=[]):
    """Get answer for a user question using retrieval augmented generation."""

    try:
      

        # Retrieve from ChromaDB and rerank — returns best final_n chunks
        context_pages = search_similar_chunks(question)


        # Generate response using OpenAI with original question
       
        # log_interaction(question, question, history, context_pages, "prompt_log", response)
        
    
        for response in generate_response(question, context_pages, history):
            yield response

    except Exception as e:
        yield False


def create_embeddings_from_folder(folder_path):
    """Extract text from all PDFs in a folder, chunk semantically, and store embeddings in ChromaDB."""
    log_data = {"folder": folder_path, "status": "started", "steps": []}

    try:
        # Step 1: Create collection
        log_data["steps"].append({"step": "create_collection", "status": "started"})
        collection = chroma_client.get_or_create_collection(
            name="diabetes_docs", embedding_function=openai_ef
        )
        
        log_data["steps"][-1].update(
            {"status": "completed", "collection_name": "diabetes_docs"}
        )

        # Step 2: Clear existing data
        # log_data["steps"].append({"step": "clear_existing_data", "status": "started"})
        # try:
        #     existing_ids = collection.get()["ids"]
        #     if existing_ids:
        #         collection.delete(ids=existing_ids)
        #     log_data["steps"][-1].update(
        #         {
        #             "status": "completed",
        #             "ids_deleted": len(existing_ids) if existing_ids else 0,
        #         }
        #     )
        # except Exception as delete_error:
        #     log_data["steps"][-1].update(
        #         {
        #             "status": "failed",
        #             "error": str(delete_error),
        #             "note": "Failed to delete existing data, continuing with adding new chunks",
        #         }
        #     )
        #     print(f"Warning: Failed to delete existing data: {delete_error}")

        # Step 3: Process all PDFs
        log_data["steps"].append({"step": "process_pdfs", "status": "started"})
        all_chunks = []
        all_ids = []
        all_metadatas = []
        total_chunks = 0

        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(pdf_path, folder_path).replace(".pdf", "").replace(os.sep, "_").replace(" ", "_")
                    try:
                        pages = extract_text_from_pdf(pdf_path)
                        for page_num, page in enumerate(pages):
                            if not page.strip():
                                continue
                            page_chunks = semantic_chunk(page)
                            for chunk_idx, chunk in enumerate(page_chunks):
                                all_chunks.append(chunk)
                                chunk_id = f"{rel_path}_page_{page_num}_chunk_{chunk_idx}"
                                all_ids.append(chunk_id)
                                all_metadatas.append({
                                    "page_number": page_num + 1,
                                    "chunk_index": chunk_idx,
                                    "source": os.path.relpath(pdf_path, folder_path)
                                })
                                total_chunks += 1
                    except Exception as e:
                        print(f"Error processing {pdf_path}: {e}")
                        continue

        # Step 4: Add all chunks in batch
        if all_chunks:
            collection.add(
                documents=all_chunks,
                ids=all_ids,
                metadatas=all_metadatas
            )

        log_data["steps"][-1].update(
            {
                "status": "completed",
                "total_chunks_added": total_chunks,
            }
        )

        # Overall success
        log_data.update(
            {"status": "completed", "success": True, "total_chunks": total_chunks}
        )

        # Save logs
        log_dir = os.path.join(os.path.dirname(__file__), "logs", "embeddings")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"folder_embeddings_log_{timestamp}.json")

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"Folder embedding process completed. Log saved to {log_file}")
        return total_chunks

    except Exception as e:
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        log_data.update({"status": "failed", "success": False, "error": error_info})

        log_dir = os.path.join(os.path.dirname(__file__), "logs", "embeddings")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"folder_embeddings_error_{timestamp}.json")

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"Folder embedding process failed. Error log saved to {log_file}")
        raise


def process_folder_and_create_embeddings(folder_path):
    """Process the folder of documents and create embeddings."""
    try:
        total_pages = create_embeddings_from_folder(folder_path)
        print(
            f"Created embeddings from {total_pages} pages across multiple PDF documents"
        )
        return {
            "success": True,
            "message": f"Successfully created embeddings from {total_pages} pages across multiple documents.",
        }
    except Exception as e:
        print(f"Error creating folder embeddings: {e}")
        return {
            "success": False,
            "message": f"Error creating folder embeddings: {str(e)}",
        }


def test_search(query, initial_n=20, final_n=5):
    """Test the search functionality by retrieving and printing reranked chunks for a query."""
    try:
        chunks = search_similar_chunks(query, initial_n=initial_n, final_n=final_n)
        
        print(f"\nTest Search for query: '{query}'")
        print(f"Retrieved {len(chunks)} chunks after reranking (from initial {initial_n}):")
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            preview = chunk[:300].replace('\n', ' ')  # Clean up for printing
            formatted_chunks.append(f"Chunk {i+1} (preview): {preview}...")
            print(f"\nChunk {i+1} (preview): {preview}...")
        
        return formatted_chunks
    except Exception as e:
        print(f"Error during test search: {str(e)}")
        return []

if __name__ == "__main__":
    # Example usage - replace with your test query
    test_query = "What are the symptoms of diabetes?"
    test_search(test_query)
