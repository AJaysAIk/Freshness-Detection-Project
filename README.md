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

