import os
import tempfile
import json
import datetime
import openai
import chromadb
import traceback
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pypdf


#use pydf instead of fitz
# Load environment variables from .env file
load_dotenv()

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
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)

def log_interaction(user_question, context_chunks, prompt, response):
    """Log the interaction details to a file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_directory, f"interaction_{timestamp}.json")
    
    log_data = {
        "timestamp": timestamp,
        "user_question": user_question,
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

def create_embeddings(pdf_path):
    """Extract text from PDF page by page and store embeddings in ChromaDB."""
    log_data = {
        "document": pdf_path,
        "status": "started",
        "steps": []
    }
    
    try:
        # Step 1: Extract text from PDF document, page by page
        log_data["steps"].append({"step": "extract_text", "status": "started"})
        pages = extract_text_from_pdf(pdf_path)
        log_data["steps"][-1].update({
            "status": "completed", 
            "num_pages": len(pages),
            "first_page_preview": pages[0][:100] if pages else "None"
        })
        
        # Step 2: Create collection
        log_data["steps"].append({"step": "create_collection", "status": "started"})
        collection = chroma_client.get_or_create_collection(
            name="diabetes_docs", 
            embedding_function=openai_ef
        )
        log_data["steps"][-1].update({"status": "completed", "collection_name": "diabetes_docs"})
        
        # Step 3: Clear existing data
        log_data["steps"].append({"step": "clear_existing_data", "status": "started"})
        try:
            # Get all existing IDs
            existing_ids = collection.get()["ids"]
            if existing_ids:
                # Delete by IDs instead of using where clause
                collection.delete(ids=existing_ids)
            log_data["steps"][-1].update({"status": "completed", "ids_deleted": len(existing_ids) if existing_ids else 0})
        except Exception as delete_error:
            log_data["steps"][-1].update({
                "status": "failed", 
                "error": str(delete_error),
                "note": "Failed to delete existing data, continuing with adding new pages"
            })
            print(f"Warning: Failed to delete existing data: {delete_error}")
        
        # Step 4: Add pages to collection
        log_data["steps"].append({"step": "add_pages", "status": "started"})
        ids = [f"page_{i}" for i in range(len(pages))]
        
        # To avoid potential memory issues, add pages one by one
        for i, (page, page_id) in enumerate(zip(pages, ids)):
            collection.add(
                documents=[page],
                ids=[page_id],
                metadatas=[{"page_number": i+1, "source": os.path.basename(pdf_path)}]
            )
            
        log_data["steps"][-1].update({
            "status": "completed", 
            "num_pages_added": len(ids),
            "first_id": ids[0] if ids else "None"
        })
        
        # Overall success
        log_data.update({
            "status": "completed",
            "success": True,
            "num_pages": len(pages)
        })
        
        # Save logs to file
        log_dir = os.path.join(os.path.dirname(__file__), "logs", "embeddings")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"embeddings_log_{timestamp}.json")
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Embedding process completed successfully. Log saved to {log_file}")
        return len(pages)
        
    except Exception as e:
        # Log the error
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        log_data.update({
            "status": "failed",
            "success": False,
            "error": error_info
        })
        
        # Save error logs
        log_dir = os.path.join(os.path.dirname(__file__), "logs", "embeddings")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"embeddings_error_{timestamp}.json")
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Embedding process failed. Error log saved to {log_file}")
        raise

def search_similar_chunks(query, n_results=5):
    """Search for pages most similar to the query."""
    collection = chroma_client.get_or_create_collection(
        name="diabetes_docs", 
        embedding_function=openai_ef
    )
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results["documents"][0]

def generate_response(query, context_chunks):
    """Generate a response from OpenAI with the context pages."""
    prompt = f"""
    You are a helpful medical assistant specializing in diabetes care and management.
    You are answering questions from patients who use the Viedial mobile application for diabetes management.

    I'll provide you with some relevant information about diabetes extracted from reliable medical documents,
    and a question from a user. Please answer their question based on the information provided.

    IF A QUESTION FROM A USER DOES NOT PERTAIN TO DIABETES PLEASE RESPOND THAT YOU DO NOT HAVE AN ANSWER TO THAT.
    IF THE PROVIDED INFORMATION DOES NOT CONTAIN SOMETHING RELEVANT TO THE USER QUESTION, ACKNOWLEDGE THAT AND RESPOND THAT YOU DO NOT HAVE
    AN ANSWER TO THAT!
    IF THE CONTEXT INFORMATION IS EMPTY, PLEASE RESPOND WITH:
    "I apologize, but I don't have enough reliable medical information in my database to properly answer your question about diabetes.

    Here are some follow-up questions you may want to ask your healthcare provider:
    1. What are the most reliable sources for diabetes information?
    2. Can you recommend any diabetes education programs?
    3. What specific aspects of diabetes would you like me to learn more about?"

    When responding:
    1. Be compassionate and conversational, but accurate and precise.
    2. Only use information from the provided context.
    3. If you don't have enough information to answer confidently, acknowledge that.
    4. Don't include citations or references to the source documents.
    5. Provide 2-3 suggested follow-up questions at the end of your response that the user might want to ask next.

    CONTEXT INFORMATION:
    {' '.join(context_chunks)}

    USER QUESTION:
    {query}

    RESPONSE:
    """

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful, accurate, and compassionate diabetes care assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
   
    response_content = response.choices[0].message.content
    
    # Log the interaction
    log_interaction(query, context_chunks, prompt, response_content)
    
    return response_content

def process_document_and_create_embeddings(pdf_file_path):
    """Process the document and create embeddings."""
    try:
        num_pages = create_embeddings(pdf_file_path)
        print(f"Created embeddings from {num_pages} pages of PDF document")
        return {"success": True, "message": f"Successfully created embeddings from {num_pages} document pages."}
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return {"success": False, "message": f"Error creating embeddings: {str(e)}"}


def correct_grammar_and_spellings(question):
    """Correct the grammar and spellings of a question."""
    prompt = f"""
    You are a helpful assistant that corrects the grammar and spellings of a question.
    I'll provide you with a question. Please correct the grammar and spellings of the question and return only the corrected question.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that corrects the grammar and spellings of a question."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    response_content = response.choices[0].message.content
    return response_content

def auto_generate_chat_title(question):
    """Auto generate a chat title based on the question."""
    prompt = f"""
    You are a helpful medical assistant specializing in diabetes care and management.
    You are answering questions from patients who use the Viedial mobile application for diabetes management.

    I'll provide you with a question from a user. Please generate a title for a chat based on the question and return only the title and keep it short.

    USER QUESTION:
    {question}

    RESPONSE (ONLY THE TITLE):
    """
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful, accurate, and compassionate diabetes care assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    response_content = response.choices[0].message.content
    return {"success": True, "title": response_content}

def get_answer_for_question(question):
    """Get answer for a user question using retrieval augmented generation."""
    try:
        # make the question gramaticaly correct
        # question = correct_grammar_and_spellings(question)
        # Search for relevant pages
        context_pages = search_similar_chunks(question)
        
        # Generate response using OpenAI
        response = generate_response(question, context_pages)
       
        
        return {
            "success": True,
            "answer": response,
            "sources": len(context_pages)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 


def create_embeddings_from_folder(folder_path):
    """Extract text from all PDFs in a folder and store embeddings in ChromaDB."""
    log_data = {
        "folder": folder_path,
        "status": "started",
        "steps": []
    }
    
    try:
        # Step 1: Create collection
        log_data["steps"].append({"step": "create_collection", "status": "started"})
        collection = chroma_client.get_or_create_collection(
            name="diabetes_docs", 
            embedding_function=openai_ef
        )
        log_data["steps"][-1].update({"status": "completed", "collection_name": "diabetes_docs"})
        
        # Step 2: Clear existing data
        log_data["steps"].append({"step": "clear_existing_data", "status": "started"})
        try:
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)
            log_data["steps"][-1].update({"status": "completed", "ids_deleted": len(existing_ids) if existing_ids else 0})
        except Exception as delete_error:
            log_data["steps"][-1].update({
                "status": "failed", 
                "error": str(delete_error),
                "note": "Failed to delete existing data, continuing with adding new pages"
            })
            print(f"Warning: Failed to delete existing data: {delete_error}")
        
        # Step 3: Process all PDFs
        log_data["steps"].append({"step": "process_pdfs", "status": "started"})
        all_documents = []
        all_ids = []
        all_metadatas = []
        total_pages = 0
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(folder_path, filename)
                try:
                    pages = extract_text_from_pdf(pdf_path)
                    doc_name = os.path.basename(filename).replace(' ', '_').replace('.pdf', '')
                    for i, page in enumerate(pages):
                        if page.strip():
                            all_documents.append(page)
                            all_ids.append(f"{doc_name}_page_{i}")
                            all_metadatas.append({"page_number": i+1, "source": filename})
                    total_pages += len(pages)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue  # Continue with next PDF
        
        # Step 4: Add all pages in batch
        if all_documents:
            collection.add(
                documents=all_documents,
                ids=all_ids,
                metadatas=all_metadatas
            )
        
        log_data["steps"][-1].update({
            "status": "completed", 
            "num_pdfs_processed": len(all_documents) // total_pages if total_pages else 0,  # Approximate
            "total_pages_added": len(all_ids),
            "total_pages_extracted": total_pages
        })
        
        # Overall success
        log_data.update({
            "status": "completed",
            "success": True,
            "total_pages": total_pages
        })
        
        # Save logs
        log_dir = os.path.join(os.path.dirname(__file__), "logs", "embeddings")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"folder_embeddings_log_{timestamp}.json")
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Folder embedding process completed. Log saved to {log_file}")
        return total_pages
        
    except Exception as e:
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        log_data.update({
            "status": "failed",
            "success": False,
            "error": error_info
        })
        
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
        print(f"Created embeddings from {total_pages} pages across multiple PDF documents")
        return {"success": True, "message": f"Successfully created embeddings from {total_pages} pages across multiple documents."}
    except Exception as e:
        print(f"Error creating folder embeddings: {e}")
        return {"success": False, "message": f"Error creating folder embeddings: {str(e)}"} 