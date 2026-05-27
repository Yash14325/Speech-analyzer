# TEAM-5 Speech Analysis Pipeline

An advanced AI-powered speech analysis system that records audio, transcribes speech, analyzes communication patterns, and generates personalized feedback reports using NVIDIA NIM as the primary LLM, Ollama as a local fallback, and RAG (Retrieval Augmented Generation).

## Features

- 🎙️ **Audio Recording**: Record audio directly from your microphone
- 🔧 **Audio Preprocessing**: Automatic noise reduction and normalization
- 📝 **Speech-to-Text**: Accurate transcription using Faster-Whisper
- 📊 **Speech Analysis**: Comprehensive acoustic feature extraction (pitch, energy, pauses, etc.)
- 🧠 **AI Agents**: Specialized agents for communication, confidence, and personality analysis
- 🔍 **RAG System**: Knowledge-enhanced reports using vector database retrieval
- 🛡️ **Guardrails**: Input/output validation and safety checks
- ✨ **Final Report**: AI-generated personalized feedback and recommendations

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 8GB recommended
- **Storage**: ~5GB for models and dependencies
- **Microphone**: Required for audio recording

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GSMPRANEETH/TEAM-5.git
cd TEAM-5
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure NVIDIA NIM Primary LLM

The backend uses NVIDIA NIM first when `NVIDIA_API_KEY` is available, then
falls back to Ollama if the hosted request fails.

Create `backend/.env` using `backend/.env.example` as the template:

```bash
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_TIMEOUT=45
```

### 5. Install Ollama (Fallback LLM)

Ollama is used for local fallback inference. Install it from [ollama.ai](https://ollama.ai/):

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
1. Download and install from [ollama.ai/download](https://ollama.ai/download)
2. **Important**: Add Ollama to your system PATH environment variable:
   - The installer typically installs Ollama to `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama`
   - **Note**: Use `%USERNAME%` exactly as shown - Windows will automatically replace it with your actual username
   - **Option 1 - Add to User Environment Variables:**
     1. Open "Edit environment variables for your account" from Start menu
     2. Under "User variables", select "Path" and click "Edit"
     3. Click "New" and add the path exactly: `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama`
     4. Click "OK" on all dialogs
   - **Option 2 - Add to System Environment Variables (requires admin):**
     1. Open "Edit the system environment variables" from Start menu
     2. Click "Environment Variables"
     3. Under "System variables", select "Path" and click "Edit"
     4. Click "New" and add the path exactly: `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama`
     5. Click "OK" on all dialogs
3. **Restart your terminal/command prompt** after adding to PATH
4. Verify installation: Open a new terminal and run `ollama --version`

### 6. Pull the LLM Model

After installing Ollama, pull the required model:

```bash
ollama pull llama3.2:3b
```

The default model is `llama3.2:3b`. You can change this in `llm1/llm_config.py`.

### 7. (Optional) Install GuardrailsAI for Enhanced Safety

GuardrailsAI provides additional input/output validation:

```bash
pip install guardrails-ai
guardrails configure
```

Install Hub validators (optional but recommended):

```bash
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/profanity_free
guardrails hub install hub://guardrails/detect_pii
```

## Configuration

### LLM Configuration

Use environment variables or `backend/.env` to customize LLM settings:

```bash
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
OLLAMA_MODEL=llama3:8b
OLLAMA_TEMPERATURE=0.2
OLLAMA_MAX_TOKENS=512
```

### RAG Configuration

Edit `rag/config.py` to customize retrieval settings:

```python
CHROMA_PERSIST_DIR = "./chroma_db"  # Vector database storage
TOP_K_RESULTS = 3                    # Number of documents to retrieve
```

### Recording Settings

Edit the configuration in `main.py`:

```python
DURATION = 45        # Recording duration in seconds
SAMPLE_RATE = 16000  # Audio sample rate (required for Whisper)
CHANNELS = 1         # Mono audio
```

## Usage

### Basic Usage - Full Pipeline

Run the complete pipeline (record, transcribe, analyze, and report):

```bash
python main.py
```

This will:
1. Record audio for 45 seconds (speak into your microphone)
2. Preprocess and clean the audio
3. Transcribe speech to text
4. Analyze speech features
5. Run AI agents for analysis
6. Generate a personalized AI report

### Testing Individual Components

#### Test Speech-to-Text Only

```bash
# Requires clean_audio.wav file
python speech_to_text.py
```

#### Test Speech Analysis Only

```bash
# Requires clean_audio.wav file
python -c "from pipeline import get_pipeline_output; print(get_pipeline_output('clean_audio.wav'))"
```

#### Test Agent System

```bash
python agent.py
```

#### Test RAG System

```bash
python test_rag.py
```

#### Test LLM Connection

```bash
python test_llm_step5.py
```

### Processing Existing Audio

To analyze an existing audio file instead of recording:

1. Place your audio file as `raw_audio.wav` in the project directory
2. Comment out the `record_audio()` call in `main.py`
3. Run `python main.py`

Or use the pipeline module directly:

```python
from pipeline import get_pipeline_output

# Analyze your audio file
result = get_pipeline_output("your_audio.wav")
print(result)
```

## Project Structure

```
TEAM-5/
├── main.py                 # Main entry point - full pipeline
├── agent.py                # Agent orchestrator
├── speech_to_text.py       # Whisper transcription
├── speech_features.py      # Acoustic feature extraction
├── pipeline.py             # Pipeline utilities
├── llm_helper.py           # LLM loader helper
├── guardrails_config.py    # Input/output validation
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── agents/                # Specialized AI agents
│   ├── communication_agent.py
│   ├── confidence_agent.py
│   └── personality_agent.py
│
├── llm/                   # LLM wrapper (for agents)
│   └── local_llm.py
│
├── llm1/                  # LLM configuration & report generation
│   ├── llm_config.py
│   ├── local_llm.py
│   ├── prompt_templates.py
│   └── report_generator.py
│
├── rag/                   # RAG system
│   ├── config.py
│   ├── retriever.py
│   ├── knowledge_base.py
│   ├── rag_pipeline.py
│   └── documents/         # Knowledge base documents
│       ├── communication_rules.md
│       ├── confidence_psychology.md
│       └── personality_traits.md
│
└── utils/                 # Utilities
    └── parser.py          # JSON parsing utilities
```

## Dependencies

### Core Dependencies

- **torch** & **torchaudio**: Deep learning framework for audio processing
- **faster-whisper**: Efficient speech-to-text transcription
- **librosa**: Audio analysis and processing
- **sounddevice** & **soundfile**: Audio recording and I/O
- **opensmile**: Acoustic feature extraction
- **pyannote.audio**: Speaker diarization and audio analysis

### LLM & Agent Framework

- **langchain**: LLM orchestration framework
- **langchain-ollama**: Ollama integration for LangChain
- **ollama**: Local LLM runtime

### RAG System

- **chromadb**: Vector database for semantic search
- **transformers**: HuggingFace models for embeddings
- **sentence-transformers**: Efficient sentence embeddings

### Validation & Safety

- **guardrails-ai**: Input/output validation framework
- **pydantic**: Data validation

See `requirements.txt` for complete dependency list with versions.

## Troubleshooting

### Ollama Connection Issues

**Error**: `Ollama not available` or `ollama: command not found`

**Solution**:
1. Ensure Ollama is installed and running: `ollama serve`
2. Check if the model is pulled: `ollama list`
3. Pull the model if missing: `ollama pull llama3.2:3b`
4. **Windows users**: If you get `command not found`, verify Ollama is in your PATH:
   - Open a new terminal and run `ollama --version`
   - If it fails, add Ollama to PATH (see installation step 4 above)
   - Use the path with %USERNAME% variable: `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama`
   - Windows will automatically expand %USERNAME% to your actual username
   - Restart your terminal after adding to PATH
5. **Linux/macOS users**: If Ollama isn't in PATH, it may be installed in a custom location:
   - Try: `which ollama` to find the installation path
   - Add it to PATH: `export PATH=$PATH:/path/to/ollama`
   - Add to `~/.bashrc` or `~/.zshrc` to make permanent

The system will fall back to a stub LLM if Ollama is unavailable, which returns mock data for testing.

### Microphone Issues

**Error**: Audio recording fails

**Solution**:
1. Check microphone permissions for your terminal/Python
2. List available audio devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
3. Update device ID in recording code if needed

### ChromaDB Issues

**Error**: ChromaDB initialization fails

**Solution**:
1. Ensure ChromaDB is installed: `pip install chromadb`
2. Clear the database: `rm -rf chroma_db/`
3. The system will fall back to keyword-based retrieval if ChromaDB is unavailable

### Import Errors

**Error**: Module not found

**Solution**:
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.8+)

### Memory Issues

**Error**: Out of memory during processing

**Solution**:
1. Use a smaller LLM model in `llm1/llm_config.py`
2. Reduce `MAX_TOKENS` in the configuration
3. Close other memory-intensive applications

## Advanced Configuration

### Using a Different LLM Model

Edit `llm1/llm_config.py`:

```python
# Available models (example):
LLM_MODEL_NAME = "llama3.2:3b"     # 3B parameters (fastest)
# LLM_MODEL_NAME = "llama3.2:7b"   # 7B parameters (balanced)
# LLM_MODEL_NAME = "llama3.2:13b"  # 13B parameters (most accurate)
```

Pull the new model:
```bash
ollama pull llama3.2:7b
```

### Customizing Knowledge Base

Add your own documents to the RAG system:

1. Create markdown files in `rag/documents/`
2. Restart the system to re-index

### Adjusting Recording Duration

Edit `main.py`:

```python
DURATION = 60  # Record for 60 seconds instead of 45
```

## Performance Tips

1. **Use GPU acceleration**: If available, install PyTorch with CUDA support
2. **Optimize Whisper**: Use smaller model sizes for faster transcription
3. **Reduce context length**: Decrease `TOP_K_RESULTS` in RAG config for faster retrieval
4. **Pre-download models**: Download all models before first run to avoid delays

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add license information here]

## Acknowledgments

- **Faster-Whisper**: OpenAI Whisper implementation by Guillaume Klein
- **Ollama**: Local LLM runtime by Ollama Team
- **LangChain**: LLM application framework
- **ChromaDB**: Vector database for AI applications
- **GuardrailsAI**: LLM output validation framework

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact the development team

## Changelog

### Current Version
- Full speech analysis pipeline
- RAG-enhanced AI reporting
- Multi-agent analysis system
- GuardrailsAI integration
- Local LLM support via Ollama
