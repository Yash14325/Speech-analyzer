# 🎙️ TEAM-5 Speech Analysis Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.2+-61DAFB.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An advanced AI-powered speech analysis system that provides real-time personality insights and communication feedback using local LLMs, RAG, and multi-agent AI architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Testing & Evaluation](#testing--evaluation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

TEAM-5 is a comprehensive speech analysis pipeline that combines state-of-the-art speech processing, natural language understanding, and multi-agent AI systems to provide detailed personality insights and communication analysis. The system uses local LLMs (via Ollama) and Retrieval-Augmented Generation (RAG) to deliver personalized, actionable feedback.

### Key Capabilities

- **Real-time Speech Analysis**: Process audio through advanced speech-to-text and acoustic feature extraction
- **Multi-Agent AI System**: Specialized agents for communication, confidence, and personality analysis
- **RAG-Enhanced Reports**: Knowledge-augmented insights using vector database retrieval
- **Quality Assurance**: Built-in evaluation framework using LangChain evaluators
- **Full-Stack Solution**: FastAPI backend + React frontend for seamless user experience

## ✨ Features

### Backend Features
- 🎤 **Audio Recording & Processing**: Multi-format audio support with noise reduction
- 📝 **Speech-to-Text**: High-accuracy transcription using Faster-Whisper
- 📊 **Acoustic Analysis**: Comprehensive feature extraction (pitch, energy, pauses, speech rate)
- 🧠 **Multi-Agent AI**: Specialized agents for different analysis aspects
- 🔍 **RAG System**: ChromaDB-powered knowledge retrieval
- 🛡️ **Guardrails AI**: Input/output validation and safety checks
- 📈 **Evaluation Framework**: Quality assessment using LangChain evaluators
- 🚀 **REST API**: FastAPI-powered endpoints for frontend integration

### Frontend Features
- ⚡ **React + TypeScript**: Modern, type-safe UI development
- 🎨 **Responsive Design**: Works on desktop and mobile devices
- 📤 **File Upload**: Support for various audio formats
- 📊 **Results Visualization**: Interactive display of analysis results
- ⏱️ **Real-time Feedback**: Instant processing status updates

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│                    (React + TypeScript)                      │
│                 Vite Dev Server / Build                      │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      Backend API                             │
│                     (FastAPI + Uvicorn)                      │
├──────────────────────────────────────────────────────────────┤
│                    Speech Processing Pipeline                │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐ │
│  │  Audio   │→ │  Speech   │→ │  Feature │→ │   Multi-   │ │
│  │ Recording│  │  to Text  │  │Extraction│  │Agent System│ │
│  └──────────┘  └───────────┘  └──────────┘  └─────┬──────┘ │
│                                                      │        │
│  ┌──────────────────────────────────────────────────▼──────┐ │
│  │            RAG System (ChromaDB + Ollama)              │ │
│  │  - Communication Knowledge    - Confidence Psychology  │ │
│  │  - Personality Traits         - Improvement Tips       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Evaluation & Guardrails                      │ │
│  │  - LangChain Evaluators  - Input/Output Validation    │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │    Ollama    │
                  │ Local LLM    │
                  │  (mistral)   │
                  └──────────────┘
```

### Component Breakdown

1. **Speech Processing**: Audio recording, preprocessing, and transcription
2. **Feature Extraction**: Acoustic analysis (openSMILE, PyAnnote)
3. **Multi-Agent System**: 
   - Communication Agent: Analyzes clarity, fluency, structure
   - Confidence Agent: Evaluates vocal confidence and emotional tone
   - Personality Agent: Maps communication patterns to personality traits
4. **RAG System**: Retrieves relevant expert knowledge for enhanced insights
5. **Report Generation**: LLM-powered personalized feedback reports
6. **Evaluation Framework**: Quality assessment and validation

## 💻 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: ~5GB for models and dependencies
- **Node.js**: 18.x or higher (for frontend)
- **Microphone**: Required for audio recording

### Recommended Requirements
- **RAM**: 16GB or more
- **GPU**: CUDA-compatible GPU (optional, for faster inference)
- **Storage**: SSD with 10GB+ free space

## 🚀 Quick Start

### Option 1: Full Stack Development

```bash
# Clone the repository
git clone https://github.com/GSMPRANEETH/TEAM-5.git
cd TEAM-5

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install and start Ollama
ollama pull mistral

# Start backend API
uvicorn api:app --reload --port 8000

# In a new terminal - Frontend setup
cd ../frontend
npm install
npm run dev
```

### Option 2: Backend Only

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py  # Run standalone pipeline
```

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/GSMPRANEETH/TEAM-5.git
cd TEAM-5
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
cd backend
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama (Required for LLM)

Ollama provides local LLM inference.

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
1. Download from [ollama.ai/download](https://ollama.ai/download)
2. Run the installer
3. Add Ollama to PATH:
   - Open "Edit environment variables for your account"
   - Add `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama` to PATH
   - Restart terminal

**Verify Installation:**
```bash
ollama --version
```

#### Pull LLM Model

```bash
ollama pull mistral
```

### 4. Frontend Setup (Optional)

```bash
cd ../frontend
npm install
```

### 5. Configuration

Edit backend configuration files as needed:

- `backend/llm1/llm_config.py` - LLM settings (model, temperature, max tokens)
- `backend/rag/config.py` - RAG system configuration
- `backend/evals/eval_config.py` - Evaluation criteria

## 🎯 Usage

### Running the Full Stack

#### Start Backend API

```bash
cd backend
source venv/bin/activate
uvicorn api:app --reload --port 8000
```

API will be available at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

#### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend will be available at: `http://localhost:5173`

### Running Backend Standalone

```bash
cd backend
python main.py
```

This will:
1. Record 45 seconds of audio
2. Process and analyze speech
3. Generate comprehensive report
4. Display results in terminal

### Using the API

```python
import requests

# Upload audio file
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f}
    )

result = response.json()
print(result)
```

### Running Tests

```bash
cd backend

# Test LLM connection
python test_llm_step5.py

# Test RAG system
python test_rag.py

# Run evaluations
python -m evals.test_evals
```

## 📁 Project Structure

```
TEAM-5/
├── backend/                      # Python backend
│   ├── api.py                   # FastAPI application
│   ├── main.py                  # Standalone pipeline
│   ├── link.py                  # Pipeline orchestration
│   ├── requirements.txt         # Python dependencies
│   ├── README.md                # Backend documentation
│   │
│   ├── agents/                  # Multi-agent system
│   │   ├── communication_agent.py
│   │   ├── confidence_agent.py
│   │   └── personality_agent.py
│   │
│   ├── llm/                     # LLM wrapper (agents)
│   │   └── local_llm.py
│   │
│   ├── llm1/                    # LLM config & reporting
│   │   ├── llm_config.py
│   │   ├── local_llm.py
│   │   ├── prompt_templates.py
│   │   └── report_generator.py
│   │
│   ├── rag/                     # RAG system
│   │   ├── config.py
│   │   ├── retriever.py
│   │   ├── knowledge_base.py
│   │   ├── rag_pipeline.py
│   │   └── documents/
│   │
│   ├── evals/                   # Evaluation framework
│   │   ├── eval_config.py
│   │   ├── eval_runner.py
│   │   ├── eval_refinement.py
│   │   └── test_evals.py
│   │
│   ├── utils/                   # Utilities
│   │   ├── parser.py
│   │   └── feature_scoring.py
│   │
│   ├── speech_to_text.py        # Whisper transcription
│   ├── speech_features.py       # Acoustic analysis
│   ├── record_audio.py          # Audio recording
│   ├── preprocess_audio.py      # Audio preprocessing
│   ├── guardrails_config.py     # Safety & validation
│   └── agent.py                 # Agent orchestrator
│
├── frontend/                     # React frontend
│   ├── src/                     # Source code
│   │   ├── App.tsx
│   │   └── components/
│   ├── public/                  # Static assets
│   ├── package.json
│   ├── vite.config.ts
│   └── README.md
│
└── README.md                     # This file
```

## 📚 API Documentation

### Endpoints

#### `POST /analyze`
Analyze uploaded audio file.

**Request:**
```http
POST /analyze HTTP/1.1
Content-Type: multipart/form-data

file: <audio_file>
```

**Response:**
```json
{
  "transcript": "...",
  "audio_features": { ... },
  "communication_analysis": { ... },
  "confidence_emotion_analysis": { ... },
  "personality_analysis": { ... },
  "final_report": "..."
}
```

### Interactive API Docs

When the backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## ⚙️ Configuration

### LLM Configuration (`backend/llm1/llm_config.py`)

```python
LLM_MODEL_NAME = "mistral"  # Change model
TEMPERATURE = 0.3            # Creativity (0.0-1.0)
MAX_TOKENS = 512            # Max response length
```

### RAG Configuration (`backend/rag/config.py`)

```python
CHROMA_PERSIST_DIR = "./chroma_db"  # Vector DB storage
TOP_K_RESULTS = 3                    # Documents to retrieve
```

### Recording Configuration (`backend/main.py`)

```python
DURATION = 45        # Recording duration (seconds)
SAMPLE_RATE = 16000  # Required for Whisper
CHANNELS = 1         # Mono audio
```

## 🛠️ Development

### Backend Development

```bash
cd backend
source venv/bin/activate

# Run with auto-reload
uvicorn api:app --reload

# Run tests
python -m pytest

# Format code
black .
flake8 .
```

### Frontend Development

```bash
cd frontend

# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint
npm run lint
```

## 🧪 Testing & Evaluation

### Built-in Evaluations

The system includes a comprehensive evaluation framework:

```bash
cd backend
python -m evals.test_evals
```

**Evaluation Criteria:**
- Helpfulness, Relevance, Coherence
- Actionability, Specificity, Accuracy
- Completeness, Constructiveness

### Manual Testing

```bash
# Test LLM connection
python test_llm_step5.py

# Test RAG retrieval
python test_rag.py

# Test full pipeline
python main.py
```

## 🔧 Troubleshooting

### Ollama Connection Issues

**Error:** `Ollama not available`

**Solution:**
1. Ensure Ollama is running: `ollama serve`
2. Check model is pulled: `ollama list`
3. Pull model if needed: `ollama pull mistral`
4. **Windows**: Verify Ollama is in PATH
5. **Linux/macOS**: Check `which ollama`

### Import Errors

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Memory Issues

**Solution:**
- Use smaller LLM model
- Reduce `MAX_TOKENS` in configuration
- Close other applications
- Upgrade to 16GB+ RAM

### ChromaDB Issues

**Solution:**
```bash
# Clear database
rm -rf backend/chroma_db/

# Restart backend
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend code
- Add tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Faster-Whisper**: OpenAI Whisper implementation
- **Ollama**: Local LLM runtime
- **LangChain**: LLM application framework
- **ChromaDB**: Vector database for AI
- **FastAPI**: Modern Python web framework
- **React**: UI library

## 📞 Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/GSMPRANEETH/TEAM-5/issues)
- Contact: [Project Team]

## 🌟 Fixes & Updates
- Resolved issue where the front end report was not being generated correctly by ensuring backend mock LLM schema strictly follows the expected prompt schema output patterns.
- Added safeguards in local_llm wrappers when API keys are not supplied.

## 🗺️ Roadmap

- [ ] Support for additional LLM providers
- [ ] Multi-language support
- [ ] Real-time streaming analysis
- [ ] Advanced visualization dashboards
- [ ] Mobile app
- [ ] Docker containerization
- [ ] Cloud deployment guide

---

**Built with ❤️ by TEAM-5**
(chatgpt)[https://chatgpt.com/g/g-p-693cf63b3d608191a200ca21f1c5f7e2-tts/project]
(perplexity)[https://www.perplexity.ai/spaces/tts-1gnsM.HoSV.NHB6HbWS31g#0]
