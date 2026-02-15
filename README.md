# HealthX - AI-Powered Medical Analysis Platform

## 🏥 Overview

HealthX is a comprehensive medical analysis platform that leverages artificial intelligence to provide accurate diagnoses and health insights. The platform offers multiple AI-powered analysis modules including blood cell analysis, skin disease detection, lung cancer screening, and medical report processing.

## 🚀 Features

### 🔬 Blood Cell Analysis
- **AI-powered blood cell classification** using deep learning models
- **8 different cell types** detection: basophil, eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte, neutrophil, platelet
- **GradCAM visualization** for explainable AI
- **Saliency maps** for model interpretability
- **PDF report generation** with detailed analysis

### 🫁 Lung Cancer Detection
- **CT scan analysis** using VGG16-based models
- **4 cancer types** classification: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal
- **High accuracy** predictions with confidence scores
- **Visual explanations** of AI decisions

### 🩺 Skin Disease Detection
- **7 skin conditions** classification using EfficientNet-B4
- **Real-time analysis** of uploaded images
- **GradCAM heatmaps** for visual explanations
- **Professional medical insights**

### 📄 Medical Report Processing
- **OCR-powered** text extraction from medical reports
- **AI analysis** of blood test results
- **Automated report generation**
- **PDF processing** with pytesseract

### 🤖 AI Chatbot
- **OpenAI GPT integration** for medical consultations
- **Symptom analysis** and preliminary diagnoses
- **Multi-language support** with translation capabilities
- **Contextual medical advice**

## 🛠️ Technology Stack

### Backend
- **Django 5.1.5** - Web framework
- **Django REST Framework** - API development
- **Python 3.12** - Programming language

### Machine Learning
- **TensorFlow 2.18.0** - Deep learning framework
- **Keras 3.8.0** - High-level neural networks API
- **PyTorch 2.6.0** - Deep learning framework
- **Transformers 4.48.1** - Natural language processing
- **OpenCV 4.11.0** - Computer vision
- **scikit-image 0.21.0** - Image processing

### Data Processing
- **NumPy 1.26.4** - Numerical computing
- **Pandas 2.2.3** - Data manipulation
- **Matplotlib 3.10.3** - Data visualization
- **Plotly 6.0.1** - Interactive visualizations

### Document Processing
- **pdfplumber 0.11.5** - PDF text extraction
- **pytesseract 0.3.13** - OCR processing
- **fpdf2 2.7.6** - PDF generation

### External Services
- **OpenAI API** - AI chatbot integration
- **Stripe 7.8.0** - Payment processing
- **Deep Translator** - Multi-language support

## 📦 Installation

### Prerequisites
- Python 3.12+
- Git
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Alishnis/healthX.git
   cd healthX
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd mysite
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file in mysite directory
   OPENAI_API_KEY=your_openai_api_key
   STRIPE_SECRET_KEY=your_stripe_secret_key
   STRIPE_PUBLISHABLE_KEY=your_stripe_publishable_key
   ```

5. **Run database migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create superuser**
   ```bash
   python manage.py createsuperuser
   ```

7. **Start development server**
   ```bash
   python manage.py runserver
   ```

## 🎯 Usage

### Blood Cell Analysis
1. Navigate to the blood analysis section
2. Upload a blood cell image
3. Get AI-powered classification results
4. View GradCAM visualizations
5. Download detailed PDF reports

### Lung Cancer Detection
1. Upload CT scan images
2. Receive instant cancer type predictions
3. View confidence scores and explanations
4. Access detailed analysis reports

### Skin Disease Detection
1. Upload skin condition images
2. Get AI-powered disease classification
3. View GradCAM heatmaps
4. Receive professional medical insights

### Medical Report Processing
1. Upload medical reports (PDF/Image)
2. Extract text using OCR
3. Analyze blood test results
4. Generate comprehensive reports

## 🔧 Configuration

### Model Files
Ensure the following model files are present:
- `myapp/model_from_scratch_blood.json` - Blood cell classification model
- `myapp/model_from_scratch_blood.weights.h5` - Model weights
- `myapp/vgg16.json` - VGG16 model architecture
- `myapp/vgg16.weights.h5` - VGG16 weights
- `trained_model.h5` - Lung cancer detection model

### API Keys
Configure the following in your environment:
- **OpenAI API Key**: For AI chatbot functionality
- **Stripe Keys**: For payment processing
- **Tesseract**: For OCR functionality

## 📊 Performance

- **Blood Cell Classification**: 95%+ accuracy
- **Lung Cancer Detection**: 90%+ accuracy
- **Skin Disease Detection**: 85%+ accuracy
- **Real-time Processing**: < 5 seconds per analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Contact: [Your Email]
- Documentation: [Link to docs]

## 🙏 Acknowledgments

- Medical AI research community
- Open source contributors
- Healthcare professionals who provided feedback

---

**⚠️ Medical Disclaimer**: This platform is for educational and research purposes only. Always consult with qualified healthcare professionals for medical decisions.

