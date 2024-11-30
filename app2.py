import os
import cv2 # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from flask import Flask, render_template, request, redirect, url_for, jsonify   
import pytesseract # type: ignore

# Define data paths
DATA_DIR = 'dataset/Train_data'
IMG_SIZE = (224, 224)

# Load images function
def load_images(data_dir):
    images = []
    labels = []
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
        # Determine label based on category name
        label = 0 if 'rotten' in category.lower() else 1
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:  # Avoid NoneType error if the image cannot be read
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)


# Load images from train subfolders
images, labels = load_images(DATA_DIR)
images = images / 255.0  # Normalize

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Model Architecture
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save('freshness_model.h5')

# Flask App
app = Flask(__name__)
model = load_model('freshness_model.h5')

# Tesseract OCR configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img_normalized = img / 255.0
        prediction = model.predict(np.expand_dims(img_normalized, axis=0))[0][0]
        freshness = 'Fresh' if prediction > 0.5 else 'Rotten'
        
        # OCR to extract text from product labels
        text = pytesseract.image_to_string(img)
        
        return render_template('result.html', freshness=freshness, ocr_text=text)

@app.route('/capture')
def capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    freshness = ""
    ocr_text = ""
    if ret:
        img = cv2.resize(frame, (224, 224))
        img_normalized = img / 255.0
        prediction = model.predict(np.expand_dims(img_normalized, axis=0))[0][0]
        freshness = 'Fresh' if prediction > 0.5 else 'Rotten'
        
        # OCR to extract text from product labels
        ocr_text = pytesseract.image_to_string(frame)
    cap.release()
    return render_template('result.html', freshness=freshness, ocr_text=ocr_text)

@app.route('/inventory', methods=['POST'])
def inventory():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Simple product counting logic (assuming distinct items are present)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        item_count = len(contours)
        return jsonify({'item_count': item_count})

if __name__ == '__main__':
    app.run(debug=True)

