from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import process_pdf, ask_question
from pydantic import BaseModel
import shutil
import os 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return{"message":"DevDoc chat API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    #process the PDF into vector database
    num_chunks = process_pdf(file_path)
    
    return{
        "message":f"{file.filename} uploaded successfully",
        "chunks" : num_chunks
        }  

class Question(BaseModel):
    query:str
    
@app.post("/ask")
def ask(question:Question):
    answer = ask_question(question.query)
    return{"answer": answer}  
    

