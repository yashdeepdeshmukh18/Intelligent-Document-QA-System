# ============================================================
#          Voice-Based RAG PDF Assistant using Llama3
# ============================================================
#
# Author       : Mangesh Bedre
# Date         : 16/06/2026
#
# Description  :
# AI-powered voice assistant that performs question
# answering over PDF documents using Retrieval
# Augmented Generation (RAG).
#
# Features:
# 1. PDF Text Extraction
# 2. Semantic Search using FAISS
# 3. Sentence Transformer Embeddings
# 4. Llama3 Integration via Ollama
# 5. Speech-to-Text
# 6. Text-to-Speech using Piper
# 7. Context-Aware Question Answering
#
# Technologies:
# Python
# FAISS
# Sentence Transformers
# Ollama
# Llama3
# PyPDF2
# SpeechRecognition
# Piper TTS
#
# ============================================================

import numpy as np 
import PyPDF2
import faiss
import requests
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
import subprocess
import sounddevice as sd
import soundfile as sf
import os

PIPER_MODEL = "en_US-lessac-medium.onnx"

def pdf_text_extractor(pdf_file_name):
    """
Function Name : pdf_text_extractor

Description   :
Reads the PDF document and extracts text from all pages.

Parameters    :
    pdf_file_name : Name or path of PDF file

Return Value  :
    Returns complete extracted text as string
"""
    try:
        data = PyPDF2.PdfReader(pdf_file_name)

        text = ""

        for page_no,page_data in enumerate(data.pages):
            page_text = page_data.extract_text()

            if page_data:
                text+= page_text + "\n"

    except Exception as e:
        raise IOError(f"Couldn't read the pdf file : {e}")

    return text 

def chuncks_from_text(text,chunck_size = 500,overlap = 100):
    """ 
    Function Name : chuncks_from_text 

    Description : 
        Splits large text into smaller overlapping chunks for embedding generation and semantic search.

    Parameters : 

        text : Input text 

        chunck_size : Size of each chunk 

        overlap : Number of overlapping characters 

    Return Value : 

        Returns list of text chunks 

    """
    chuncks = []
    start = 0

    while start < len(text):
        stop = start + chunck_size

        chunck = text[start:stop]

        if chunck.strip():
            chuncks.append(chunck)

        start = stop - overlap

    return chuncks

def create_vector_db(embedding_model,chuncks):

    """ 
    Function Name : create_vector_db 
    
    Description : 
        Generates embeddings for text chunks and stores them inside a FAISS vector database. 
    
    Parameters : 
        embedding_model : SentenceTransformer model 
        chuncks : List of text chunks 
        
    Return Value :
      Returns: 1. FAISS index 
               2. Embedding matrix 
                
    """
    embeddings = embedding_model.encode(chuncks)            # Convert text chunks into semantic embedding vectors

    embeddings = np.array(embeddings).astype(np.float32)    

    dimension = embeddings.shape[1]                         # getting the shape to create such vector space for search

    index = faiss.IndexFlatL2(dimension)                    # Builds an index of size dimension and uses L2 Distance formula for search

    index.add(embeddings)                                   # Added the embedding to the vector space

    return index,embeddings

def semantic_search(question,chuncks,index,embedding_model,top_k=3):

    """ 
    Function Name : semantic_search 
    
    Description : 
        Converts the user question into embeddings and retrieves the most relevant chunks using FAISS. 
    
    Parameters : 
        question : User question 

        chuncks : List of text chunks 

        index : FAISS index 

        embedding_model : SentenceTransformer model 
        
        top_k : Number of relevant chunks 
        
    Return Value : 

        Returns list of relevant text chunks """
    embedded_que = embedding_model.encode([question])

    embedded_que = np.array(embedded_que).astype("float32")

    distances,indexes = index.search(embedded_que,top_k)

    relevant_chunck = []

    for i in indexes[0]:
        if i < len(chuncks):
            relevant_chunck.append(chuncks[i])

    return relevant_chunck


def get_ans_from_llm(que,context):
    """ 
    Function Name : get_ans_from_llm 
    
    Description : Sends retrieved context and user question to Ollama Llama3 model and generates an answer. 
    
    Parameters : 
        que : User question 
        context : Retrieved relevant chunks 
        
    Return Value : 
        Returns generated answer from LLM """
    prompt = f'''
You are an expert knowledge assistant. Your job is to answer questions strictly based on the provided context.

## Rules
- Answer only from the given context. Do not use outside knowledge.
- If the answer is not in the context, say: "This information is not available in the provided context."
- Be concise, accurate, and direct.
- Do not hallucinate or assume missing details.

## Output Format
- Use bullet points for all answers
- dont give any bold character for output just plain simple text
- Bold key terms or important phrases
- Keep each bullet focused on one idea
- If steps are involved, use numbered list instead


Context: {context}
Question: {que}

Answer : 
'''
    url = "http://localhost:11434/api/generate"

    payload = {
        "model" : "llama3",
        "prompt" : prompt,
        "stream" : False
    }

    try:
        response = requests.post(url,json=payload)
        response.raise_for_status()

        result = response.json()

        return result["response"]
    
    except requests.exceptions.ConnectionError as c:
        raise ConnectionError("Couldn't connect to the model,check if ollama is running the model")

    except Exception as e:
        return f"Error while communicating with LLM: {e}"
    

def audio_to_text():

    """ 
    Function Name : audio_to_text 
    
    Description : Records audio from microphone and converts speech into text using Google Speech Recognition. 
    
    Parameters : 
        None 
    
    Return Value : 
        Returns recognized text string 
        
    """

    r = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as mic:
                print("Listening...")
                r.adjust_for_ambient_noise(mic, duration=0.2)
                audio = r.listen(mic)
                text = r.recognize_google(audio).lower()

                if "exit" in text:
                    print("Exiting program...")
                    raise SystemExit(0)  

                return text

        except SystemExit:
            raise  

        except KeyboardInterrupt:
            raise  

        except sr.RequestError as e:
            print(f"Could not request results: {e}")

        except sr.UnknownValueError:
            print("Could not understand audio, please try again.")

def text_to_audio(text):
    """
    Function Name : text_to_audio

    Description :
    Converts text into speech using Piper TTS.
    """

    try:
        print("\n🔊 Generating speech...")
        output_file = "answer.wav"

        result = subprocess.run(
            ["piper", "--model", PIPER_MODEL, "--output_file", output_file],
            input=text.encode("utf-8"),  
            check=True
        )

        print("▶️ Playing response...")
        data, samplerate = sf.read(output_file)
        sd.play(data, samplerate)
        sd.wait()

    except Exception as e:
        print(f"❌ TTS Error: {e}")

    finally:
        if os.path.exists(output_file):
            os.remove(output_file) 


def main():

    """ 
    Function Name : main 
    
    Description : Entry point of the Voice-Based RAG PDF Assistant. 
    
    Workflow: 
        1. Load PDF 
        2. Extract text 
        3. Create chunks 
        4. Generate embeddings 
        5. Build FAISS vector database 
        6. Accept voice query 
        7. Perform semantic search 
        8. Generate answer using LLM 
        9. Convert answer to speech 
        
        Parameters : 
            None 
        
        Return Value : 
            None 
"""

    print("\n" + "=" * 70)
    print("🎙️ Voice-Based RAG PDF Assistant")
    print("=" * 70)

    file_name = "Case_Study.pdf"

    print("\n📄 Loading PDF...")
    data = pdf_text_extractor(file_name)

    print("✅ PDF Loaded Successfully")

    print("\n✂️ Creating Chunks...")
    chunks = chuncks_from_text(data)

    print(f"✅ Total Chunks Created : {len(chunks)}")

    print("\n🧠 Loading Embedding Model...")
    transformer = SentenceTransformer(
        "all-mpnet-base-v2"
    )

    print("✅ Embedding Model Loaded")

    print("\n📦 Creating Vector Database...")
    index, embeddings = create_vector_db(
        transformer,
        chunks
    )

    print("✅ Vector Database Ready")
    print("\n🎤 Assistant Ready!")
    print("Say 'exit' to quit.\n")


    while True:
        
        print("\n" + "-" * 70)
        print("🎤 Please ask your question...")
        print("-" * 70)

        try:
            question = audio_to_text()

        except KeyboardInterrupt:
            print("\n👋 Program terminated by user")
            return
        
        if not question:
            continue

        print(f"\n🗣️ You Asked : {question}")

        if question.lower() == "exit":
            print("👋 Goodbye!")
            return

        print("\n🔍 Searching relevant context...")

        relevant_chunks = semantic_search(
            question,
            chunks,
            index,
            transformer
        )

        print(
            f"✅ Found {len(relevant_chunks)} relevant chunks"
        )

        print("\n🤖 Generating answer using Llama3...")

        answer = get_ans_from_llm(
            question,
            relevant_chunks
        )

        print("\n" + "=" * 70)
        print("📢 Assistant Response")
        print("=" * 70)
        print(answer)

        text_to_audio(answer)


if __name__ == "__main__":
    main()