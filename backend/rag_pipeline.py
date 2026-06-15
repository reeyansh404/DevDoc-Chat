import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DIR = "chroma_db"

def process_pdf(file_path):
    #1. load pdf
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    #2. Split into chunks 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = splitter.split_documents(documents)
    
    #3. Embed and Store in ChromaDB
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = CHROMA_DIR
    )
    
    return len(chunks)


def ask_question(question):
    # 1. Connect to the existing vector database
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory = CHROMA_DIR,
        embedding_function = embeddings
    )
    
    # 2. Find the 3 most relevant chunks
    relevant_chunks = vectorstore.similarity_search(question, k=3)
    
    # 3. Combine those chunk into one block of context text
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)
    
    # 4. Build a prompt that gives GPT the context + question
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below \n\n"
        "Context: \n{context}\n"
        "Question: {question}"
    )
    
    # 5. Send it to GPT and get the answer
    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return response.content 