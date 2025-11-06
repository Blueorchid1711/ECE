🩺 COVID-19 X-Ray Image Classification System

A Deep Learning–Based Medical Imaging Project using CNN and Streamlit

📘 Overview

This project presents an AI-powered system that automatically classifies chest X-ray images as either COVID-19 Positive or Normal using Convolutional Neural Networks (CNN) and Transfer Learning (MobileNetV2).

The model is trained on publicly available Kaggle COVID-19 chest X-ray datasets and deployed as a Streamlit web application for real-time predictions.
Users can upload an X-ray image, and the system instantly returns the diagnosis along with a confidence score and visual probability chart.

🚀 Key Features

🧠 Deep Learning Model: CNN built with TensorFlow + Keras using MobileNetV2 backbone.

📈 High Accuracy: Achieved 88–90% accuracy on validation data.

⚙️ Transfer Learning: Efficient feature extraction from ImageNet pre-trained weights.

💻 Interactive Web App: Built using Streamlit with real-time image upload and prediction.

🔍 Data Augmentation: Rotation, zooming, and flipping to improve generalization.

☁️ Deployed on Streamlit Cloud: Accessible through any browser, no installation required.

📊 Performance Visualization: Confusion matrix, accuracy plots, and probability charts.

🧬 Tech Stack
Category	Tools & Libraries
Programming Language	Python 3.10
Deep Learning Framework	TensorFlow / Keras
Model Architecture	MobileNetV2 (Transfer Learning)
Image Processing	PIL, OpenCV, NumPy
Web Framework	Streamlit
Visualization	Matplotlib, Pandas
Deployment Platform	Streamlit Cloud
🗂️ Project Structure
📦 covid19-xray-classification
├── 📁 dataset/                 # X-ray image dataset (COVID, Normal)
├── 📁 models/                  # Trained and saved models
│   └── final_model.keras
├── 📁 notebooks/               # Google Colab training notebook
│   └── covid_19.ipynb
├── 📁 static/                  # Sample images / figures
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── report.tex                  # Full LaTeX project report

🧠 Model Architecture

Base Model: MobileNetV2 (ImageNet pretrained)
Added Layers:

GlobalAveragePooling2D()

Dense(128, activation='relu')

Dropout(0.3)

Dense(2, activation='softmax')

Training Configuration:

Optimizer: Adam (lr=1e-4)

Epochs: 10

Batch Size: 16

Loss: Categorical Crossentropy

Accuracy: ~88.8% on validation set

📊 Evaluation Metrics
Metric	Value
Training Accuracy	90.0%
Validation Accuracy	88.8%
Test Accuracy	87.9%
Precision (macro avg)	0.89
Recall (macro avg)	0.87
F1-score (macro avg)	0.88

The confusion matrix shows robust classification performance with minimal overlap between classes.

💻 Running the Project Locally
🔹 Step 1: Clone the Repository
git clone https://github.com/<your-username>/covid19-xray-classification.git
cd covid19-xray-classification

🔹 Step 2: Install Dependencies
pip install -r requirements.txt

🔹 Step 3: Run the Application
streamlit run app.py

🔹 Step 4: Upload a Chest X-Ray Image

Open your browser at the URL shown in the terminal (usually http://localhost:8501) and upload an image (JPG/JPEG/PNG).
The model will predict whether it is COVID-19 or Normal, showing a confidence score and class probability chart.

🌐 Deployment on Streamlit Cloud
✅ Steps:

Push your project to GitHub.

Go to Streamlit Cloud
.

Sign in with GitHub and select your repository.

Set app.py as the entry point.

Streamlit automatically installs dependencies from requirements.txt.

Once deployed, share your public app URL (e.g. https://username-covid19-app.streamlit.app).

📘 Research Highlights

Implemented a lightweight CNN with MobileNetV2, optimized for real-time inference.

Demonstrated strong generalization despite small dataset using data augmentation.

Validated model with accuracy–loss plots and confusion matrix.

Fully deployed model accessible online using Streamlit Cloud.

Designed for educational, research, and healthcare screening purposes.

🧩 Future Enhancements

🔍 Integrate Grad-CAM for visual model explainability.

🧠 Extend to multi-disease classification (e.g., Pneumonia, Tuberculosis).

☁️ Deploy as a REST API (Flask/FastAPI) for hospital integration.

📱 Develop a mobile-friendly version using Streamlit Mobile or Flutter.

🏥 Validate using larger, clinically verified datasets.

🧑‍💻 Contributors

Team ECE Hackathon 2025
Department of Computer Science and Engineering, SRM University AP
Specialization: Machine Learning and Artificial Intelligence

⚠️ Disclaimer

This project is developed for research and educational purposes only.
It is not intended for clinical diagnosis or medical decision-making.
Always consult a licensed medical professional for diagnosis or treatment.

📎 References

TensorFlow Documentation: https://www.tensorflow.org

Keras MobileNetV2 API: https://keras.io/api/applications/mobilenetv2/

Streamlit Docs: https://docs.streamlit.io

Kaggle COVID-19 Dataset: https://www.kaggle.com

⭐ If you found this project useful, don’t forget to star the repository!
