# Freshness Detection and Inventory Management

This project is an AI-powered system that detects the freshness of fruits or vegetables from images and performs Optical Character Recognition (OCR) to extract product label information. It also includes basic inventory management features based on item counts.

## Features

- Freshness Detection: Classifies items as "Fresh" or "Rotten" using a fine-tuned MobileNetV2 model.
- OCR Integration: Extracts text from product labels using Tesseract OCR.
- Web Interface: User-friendly Flask-based web application for image upload and real-time webcam integration.
- Inventory Management: Counts items in uploaded images based on contour detection.

## Tech Stack

- Python Libraries:
  - TensorFlow/Keras for deep learning
  - OpenCV for image processing
  - NumPy for numerical operations
  - Flask for web application
  - PyTesseract for OCR
- Deep Learning Model: MobileNetV2 (Transfer Learning)
- Web Server Framework: Flask

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow
- OpenCV
- Flask
- Tesseract OCR (Ensure it is installed and accessible on your system)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/freshness-detection.git
   cd freshness-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - For Windows:
     Download and install Tesseract OCR from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki).
   - For Linux:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - For macOS:
     ```bash
     brew install tesseract
     ```

4. Update the Tesseract executable path in `pytesseract.pytesseract.tesseract_cmd` in the code if necessary.

5. Run the application:
   ```bash
   python app.py
   ```

6. Open the application in your browser at `http://127.0.0.1:5000/`.

## Usage

### Upload Images
- Upload an image through the web interface.
- The model will classify the item as "Fresh" or "Rotten" and display the result along with extracted text.

### Real-Time Capture
- Use the "Capture" feature to detect freshness and extract text in real-time using your webcam.

### Inventory Count
- Upload an image containing multiple items. The system will count the number of items based on image processing techniques.

## Directory Structure

```
├── dataset/
│   ├── Train_data/
│       ├── fresh/
│       └── rotten/
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── (Static files like CSS or JS if any)
├── app.py
├── requirements.txt
├── freshness_model.h5
└── README.md
```

- **dataset/Train_data/**: Contains training data categorized into `fresh` and `rotten` folders.
- **templates/**: HTML templates for the Flask app.
- **static/**: Folder for static files (e.g., CSS, JS).
- **app.py**: Main Flask application script.
- **freshness_model.h5**: Trained deep learning model.
- **requirements.txt**: List of Python dependencies.

## Model Training

If you wish to retrain the model:

1. Prepare your dataset in the `dataset/Train_data/` directory.
2. Update the paths in `app.py` if necessary.
3. Run the script to train the model:
   ```bash
   python app.py
   ```
4. The trained model will be saved as `freshness_model.h5`.

## Acknowledgments

- **MobileNetV2:** Pre-trained on ImageNet for transfer learning.
- **Tesseract OCR:** For text extraction.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

Feel free to contribute to the project by submitting issues or pull requests. Happy coding!

