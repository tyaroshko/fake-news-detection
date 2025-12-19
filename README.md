# News Classification App

A comprehensive web application for classifying news articles as real or fake using advanced AI models including BERT-like transformers and Large Language Models (LLMs).

## Features

- 📰 **Text Classification**: Paste news articles directly for classification
- 📁 **File Upload**: Upload text files (.txt, .md, .csv) for batch classification
- 🤖 **Multiple Models**: Choose from BERT-like models and LLMs through LiteLLM
- 💾 **Database Storage**: All classifications are stored in SQLite database
- 📊 **History Tracking**: View classification history with statistics
- ⚡ **Async Processing**: Fast, non-blocking classification using async/await
- 🎨 **Modern UI**: Clean Streamlit interface with real-time results

## Architecture

- **Backend**: FastAPI with async endpoints, SQLAlchemy database, HuggingFace transformers, LiteLLM
- **Frontend**: Streamlit web application
- **Database**: SQLite with SQLAlchemy ORM
- **Models**: BERT-like models from HuggingFace + LLMs through LiteLLM

## Available Models

### BERT-like Models
- `martin-ha/toxic-comment-model`: Fine-tuned for toxic content detection
- `cardiffnlp/twitter-roberta-base-sentiment`: RoBERTa for sentiment analysis
- `facebook/bart-large-mnli`: BART for natural language inference

### LLM Models (via LiteLLM)
- `gpt-5-mini`: OpenAI GPT-3.5 Turbo
- `claude-3-haiku`: Anthropic Claude 3 Haiku
- `gemini-pro`: Google Gemini Pro

## Installation

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
pip install -r requirements.txt
```

## Running the Application

### Start Backend (Terminal 1)

```bash
cd backend
python -m app.main
```

The API will be available at `http://localhost:8000`

### Start Frontend (Terminal 2)

```bash
cd frontend
streamlit run app.py
```

The web app will be available at `http://localhost:8501`

### Viewing Logs

The application includes comprehensive logging. You'll see log messages for:
- Application startup and database initialization
- API requests and responses
- Model loading and classification processes
- LLM API calls and responses

Logs appear in the terminal where the backend server is running.

## API Endpoints

- `POST /api/classify/text` - Classify text content (returns ClassificationResponse or LLMClassificationResponse)
- `POST /api/classify/file` - Classify uploaded file (returns ClassificationResponse or LLMClassificationResponse)
- `GET /api/models` - Get available models
- `GET /api/results` - Get classification history
- `GET /health` - Health check

## Response Models

### BERT Models Response
```json
{
  "id": 1,
  "text_content": "News article text...",
  "file_name": null,
  "model_used": "facebook/bart-large-mnli",
  "prediction": "real",
  "confidence": 0.87,
  "created_at": "2024-12-14T10:30:00Z"
}
```

### LLM Models Response
```json
{
  "id": 2,
  "text_content": "News article text...",
  "file_name": null,
  "model_used": "gpt-5-mini",
  "classification": {
    "result": "fake",
    "confidence": 0.92
  },
  "created_at": "2024-12-14T10:35:00Z"
}
```

## Frontend Updates

The frontend has been updated to handle both BERT and LLM response formats automatically:

- **Real-time Classification**: Displays results from both model types seamlessly
- **History View**: Shows mixed classification history with proper formatting
- **Statistics**: Correctly calculates metrics for both response types
- **Data Normalization**: Automatically handles the different JSON structures

The frontend detects the response format and normalizes the data for consistent display and processing.

## Database Schema

```sql
CREATE TABLE classification_results (
    id INTEGER PRIMARY KEY,
    text_content TEXT NOT NULL,
    file_name TEXT,
    model_used TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

### Environment Variables

For LLM models, you may need to set API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

## Usage

1. **Text Classification**:
   - Go to "Text Classification" tab
   - Paste news article text
   - Select a model from the sidebar
   - Click "Classify Text"

2. **File Upload**:
   - Go to "File Upload" tab
   - Upload a text file
   - Select a model
   - Click "Classify File"

3. **View History**:
   - Go to "History" tab
   - View past classifications
   - See statistics and trends

## Development

### Backend Development

```bash
cd backend
pip install -r requirements.txt
# Install additional dev dependencies if needed
pip install pytest black isort
```

### Frontend Development

```bash
cd frontend
pip install -r requirements.txt
# Run with auto-reload
streamlit run app.py --server.headless true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details
