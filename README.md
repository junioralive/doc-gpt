# 🤖 AI Document Assistant

An intelligent document analysis platform that transforms various document formats into interactive knowledge hubs. Chat with your documents, generate summaries, create study materials, and extract insights using state-of-the-art AI technology.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 📚 **Multi-Format Support**
- **PDF Documents** - Research papers, academic articles, textbooks
- **Word Documents** - DOCX and DOC files with full text extraction
- **Excel Spreadsheets** - XLSX and XLS files with table data analysis
- **CSV Files** - Structured data analysis with encoding detection
- **Text Files** - Plain text documents with smart encoding handling

### 🧠 **AI-Powered Intelligence**
- **Document Chat** - Ask questions about your uploaded documents
- **Smart Summarization** - Generate comprehensive document summaries
- **Question Generation** - Create study questions from content
- **MCQ Creation** - Generate multiple-choice questions for testing
- **Study Notes** - Create organized, structured notes from documents
- **Content Search** - Semantic search across all uploaded documents

### 🚀 **Advanced Technology Stack**
- **Google Gemini 2.0 Flash** - Latest AI model for superior understanding
- **FAISS Vector Database** - Fast similarity search and retrieval
- **LangChain Framework** - Advanced document processing pipeline
- **Sentence Transformers** - High-quality text embeddings
- **Streamlit Interface** - Modern, responsive web application

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Google API Key (for Gemini AI)

### Quick Setup

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/junioralive/doc-gpt
   cd doc-gpt
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   
   To get your Google API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy and paste it into your `.env` file

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Open in Browser**
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. **Upload Documents**
- Click on "Choose Files" in the sidebar
- Select one or multiple files (PDF, DOCX, XLSX, CSV, TXT)
- Supported formats are automatically detected

### 2. **Process Documents**
- Click "⚙️ Process Documents" button
- Wait for the processing to complete
- Your documents are now ready for analysis

### 3. **Interact with Your Documents**

#### 💬 **Chat with Documents**
- Type questions in the text input field
- Get instant answers based on your document content
- Ask follow-up questions for deeper understanding

#### 📋 **Generate Summaries**
- Click "📋 Summarize Documents"
- Get comprehensive overviews of your content
- Perfect for quick document reviews

#### ❓ **Create Study Questions**
- Click "❓ Generate Questions"
- Get thoughtful questions to test understanding
- Each question can be answered instantly

#### 📝 **Generate MCQs**
- Click "📝 Create MCQs"
- Multiple-choice questions with correct answers
- Ideal for exam preparation

#### 📚 **Create Study Notes**
- Click "📚 Generate Notes"
- Well-organized, structured notes
- Bullet points and clear formatting

## 🔧 Configuration

### Environment Variables
```env
GOOGLE_API_KEY=your_api_key_here
```

### Supported File Types
- **PDF**: `.pdf`
- **Word**: `.docx`, `.doc`
- **Excel**: `.xlsx`, `.xls`
- **CSV**: `.csv`
- **Text**: `.txt`

### Processing Limits
- **Chunk Size**: 10,000 characters per chunk
- **Chunk Overlap**: 1,000 characters
- **Max File Size**: Limited by available system memory
- **Concurrent Files**: No limit (processed sequentially)

## 🏗️ Technical Architecture

### Core Components
```
├── Document Processing Pipeline
│   ├── Multi-format text extraction
│   ├── Text chunking and preprocessing
│   └── Vector embedding generation
├── AI Integration
│   ├── Google Gemini 2.0 Flash model
│   ├── LangChain conversation chains
│   └── Contextual prompt engineering
├── Vector Database
│   ├── FAISS similarity search
│   ├── Persistent storage
│   └── Fast retrieval system
└── Web Interface
    ├── Streamlit frontend
    ├── Responsive design
    └── Real-time processing
```

### Data Flow
1. **Upload** → Multiple file formats accepted
2. **Extract** → Text extraction with format-specific handlers
3. **Process** → Text chunking and cleaning
4. **Embed** → Vector representation generation
5. **Store** → FAISS vector database storage
6. **Query** → Semantic search and AI response generation

## 🔒 Privacy & Security

- **Local Processing**: Documents processed locally on your machine
- **No Data Storage**: Files are not permanently stored on servers
- **Secure API**: Google API calls use encrypted connections
- **Environment Variables**: API keys stored securely in `.env` file
- **Session-Based**: Data exists only during your session

## 🚨 Troubleshooting

### Common Issues

**"GOOGLE_API_KEY not found"**
- Ensure `.env` file exists in project root
- Check that your API key is correctly formatted
- Verify the API key is valid and active

**"Could not extract text from file"**
- Ensure file is not corrupted
- Check if file format is supported
- Try with a different file

**"No relevant information found"**
- Try rephrasing your question
- Ensure documents were processed successfully
- Check if your question relates to the uploaded content

**Import/Module Errors**
- Run `pip install -r requirements.txt` again
- Check Python version compatibility
- Ensure all dependencies are properly installed

### Performance Tips

- **Large Files**: Process large files individually for better performance
- **Multiple Files**: Upload related documents together for better context
- **Question Clarity**: Use specific, clear questions for better results
- **Memory Usage**: Close and restart if processing many large files

## 📊 Dependencies

### Core Libraries
- `streamlit` - Web application framework
- `langchain` - LLM application framework
- `google-generativeai` - Google AI integration
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embeddings

### Document Processing
- `PyPDF2` - PDF text extraction
- `python-docx` - Word document processing
- `pandas` - Excel and CSV handling
- `openpyxl` - Excel file support
- `chardet` - Character encoding detection

### Utilities
- `python-dotenv` - Environment variable management
- `xlrd` - Legacy Excel support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Junior Alive**
- GitHub: [@JuniorAlive](https://github.com/junioralive)

## 🙏 Acknowledgments

- Google AI for the powerful Gemini model
- LangChain community for the excellent framework
- Streamlit team for the amazing web framework
- All open-source contributors who made this project possible

### Upcoming Features
- [ ] Support for PowerPoint presentations (PPT/PPTX)
- [ ] Image text extraction (OCR support)
- [ ] Multi-language document support
- [ ] Document comparison features
- [ ] Export generated content (PDF, Word)
- [ ] Batch processing for multiple files
- [ ] Custom AI model integration
- [ ] Advanced search filters
- [ ] Document annotation features
- [ ] Collaborative features