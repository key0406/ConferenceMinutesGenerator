# Conference Minutes Generator
An AI-powered web application that automatically generates structured meeting minutes from long audio recordings.
This project combines speech-to-text processing using Whisper with multi-stage summarization powered by large language models (LLMs), while handling long audio inputs through segmentation and token-aware chunking.

---

## Features

- Long audio processing with automatic segmentation (10-minute chunks)
- Speech-to-text transcription using OpenAI Whisper
- Token-aware text chunking to handle LLM context limits
- Multi-stage summarization using large language models
- Structured meeting minutes generation (committee meetings, study sessions, etc.)
- Web-based interface built with Flask and Jinja2

---

## Tech Stack

- **Language**: Python  
- **Web Framework**: Flask  
- **Speech-to-Text**: OpenAI Whisper  
- **LLM Integration**: LangChain + Together API (Llama 3 series)  
- **Audio Processing**: pydub  
- **Tokenization**: Hugging Face Transformers (GPT-2 tokenizer)  

---

## How It Works

1. An audio file is uploaded via the web interface.
2. The audio is split into fixed-length segments.
3. Each segment is transcribed in memory using Whisper.
4. The transcribed text is split into token-safe chunks.
5. Each chunk is summarized individually using an LLM.
6. All partial summaries are combined and reformatted into structured meeting minutes.

No transcription or summary data is permanently stored.

---

Speaker diarization

Asynchronous processing for long audio files

Deployment using Docker or cloud platforms


