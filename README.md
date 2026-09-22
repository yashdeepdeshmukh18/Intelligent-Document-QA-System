# Voice-Based RAG PDF Assistant using Llama3

An AI-powered voice assistant that answers questions from PDF documents using Retrieval-Augmented Generation (RAG). The project extracts text from a PDF, splits it into chunks, embeds those chunks with Sentence Transformers, stores them in a FAISS index, retrieves the most relevant context for a spoken query, sends that context to Llama3 through Ollama, and reads the answer back using Piper TTS.

## Project Description

This project combines document retrieval, local LLM inference, speech recognition, and text-to-speech into a single interactive assistant. A user loads a PDF, asks a question through the microphone, and receives a context-grounded answer based only on the document contents.

The pipeline is designed around a standard RAG flow:
- Extract text from PDF using PyPDF2
- Split text into overlapping chunks
- Generate embeddings using `all-mpnet-base-v2`
- Store embeddings in a FAISS vector index
- Convert speech to text using SpeechRecognition
- Retrieve top relevant chunks for the question
- Generate an answer with Llama3 via Ollama
- Convert the answer to speech using Piper

## Features

- PDF text extraction from multi-page documents
- Chunking with overlap for better retrieval quality
- Semantic search with FAISS
- Sentence Transformer embeddings
- Local Llama3 inference through Ollama
- Voice input using microphone
- Spoken response using Piper TTS
- Context-restricted answering to reduce hallucination

## Tech Stack

- Python
- PyPDF2
- FAISS
- Sentence Transformers
- Ollama
- Llama3
- SpeechRecognition
- Piper TTS
- sounddevice
- soundfile

## How It Works

1. The application loads a PDF file.
2. Text is extracted from all pages.
3. The extracted text is split into overlapping chunks.
4. Each chunk is converted into embeddings using a Sentence Transformer model.
5. Embeddings are stored in a FAISS vector database.
6. The user asks a question through the microphone.
7. The spoken query is converted into text.
8. The query embedding is matched against the vector database.
9. The most relevant chunks are sent to Llama3 through Ollama.
10. The generated answer is printed and spoken aloud.

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

You also need:
- Ollama installed and running locally
- Llama3 model available in Ollama
- Piper installed and accessible from the command line
- A Piper voice model file such as `en_US-lessac-medium.onnx`
- Working microphone and audio output device

## Run

```bash
python your_script_name.py
```

Before running, update the PDF filename in the script if needed:

```python
file_name = "Case_Study.pdf"
```

## Notes

- The assistant answers only from retrieved PDF context.
- If Ollama is not running, the script raises a connection error.
- Saying `exit` is intended to close the assistant.
- Pressing `Ctrl+C` should terminate the application from the terminal.
- Piper model files should be excluded through `.gitignore` because they are large local assets.

## Suggested `.gitignore`

```gitignore
en_US-lessac-medium.onnx
en_US-lessac-medium.onnx.json
answer.wav
__pycache__/
*.pyc
```

## Future Improvements

- Replace character-based chunking with token-aware chunking
- Persist FAISS index to disk
- Add support for multiple PDFs
- Add better prompt formatting and answer post-processing
- Use offline speech-to-text for a fully local pipeline
- Add a simple GUI or web interface

## Author

Mangesh Bedre
