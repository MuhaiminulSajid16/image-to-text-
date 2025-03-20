# Medical Prescription Analysis Chatbot

A multimodal chatbot that processes medical prescriptions using OCR and a fine-tuned language model. The system extracts text from prescription images and provides structured analysis of medications, dosages, frequencies, and other prescription details.

## Features

- Image processing with OpenCV for enhanced OCR accuracy
- Text extraction using EasyOCR with confidence thresholding
- Prescription analysis using a fine-tuned FLAN-T5 model
- FastAPI backend with health monitoring
- CORS support for frontend integration

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model (optional - pre-trained model provided):
```bash
python train_model.py
```

5. Start the server:
```bash
uvicorn main:app --reload
```

## API Endpoints

### POST /upload_image/
Upload a prescription image for analysis.

**Request:**
- Form data with image file

**Response:**
```json
{
    "extracted_text": "Raw text extracted from image",
    "analysis": {
        "medication": "Medication name",
        "dosage": "Dosage amount",
        "frequency": "Usage frequency",
        "duration": "Treatment duration"
    }
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
    "status": "healthy"
}
```

## Model Training

The system uses a fine-tuned FLAN-T5 model trained on prescription data. The training script (`train_model.py`) includes:

- Custom prescription dataset preparation
- Model fine-tuning configuration
- Training and evaluation pipeline

## Dependencies

- FastAPI
- EasyOCR
- OpenCV
- PyTorch
- Transformers
- Python 3.8+

## Error Handling

The system includes comprehensive error handling for:
- Invalid image files
- OCR processing failures
- Model inference issues
- Structured data parsing errors

## Contributing

Feel free to submit issues and enhancement requests! 