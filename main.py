from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
import os
import utils

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


class Chat(BaseModel):
    question: str
    history: list | None = None
    
    
class Topic(BaseModel):
    topic: str
    
@app.post("/api/auto_generate_chat_title")
def auto_generate_chat_title(chat: Chat):
    """Auto generate a chat title based on the question."""
    question = chat.question
    result = utils.auto_generate_chat_title(question)
    return result



@app.post("/api/ask")
def ask_question(chat: Chat):
    """Answer a question based on the document."""

    
    if not chat.question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="a question is required"
        )
    
    question = chat.question
    history = chat.history or []
    
    
    # Get answer
    result = utils.get_answer_for_question(question, history)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=result
    )
    
    
@app.post("/api/generate_question_query")
def generate_question_query(topic: Topic):
    """Receive a topic and automatically generate a search
    query which will be used to get a first chat.
    """
    
    if not topic.topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="a topic is required"
        )
        
    generated_query = utils.auto_generate_question_query(topic.topic)

    if "success" in generated_query and  generated_query["success"]:
        question = generated_query["query"]
        chat_title =  utils.auto_generate_chat_title(question)
        response = utils.get_answer_for_question(question,[])
        response["title"] = chat_title["title"]
        response["generated_question"] = question
        
        if "success" in response and response["success"]:
             return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=response
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="could not generate new search query"
        )
    
    
    
    
@app.post("/api/test")
def test(chat: Chat):
    """Test the search functionality."""
    question = chat.question
   
   
    result = utils.test_search(question)
    return result