🍂 Potato Disease Classification
A deep learning-based project for detecting potato leaf diseases using a trained machine learning model. The model classifies images of potato leaves into healthy or diseased categories.

📌 Features
✅ Image Classification – Detects common potato diseases from leaf images.
✅ Deep Learning Model – Uses CNN-based architecture (TensorFlow/Keras).
✅ Web & Mobile Support – Deployable as a Flask/FastAPI API, Android app (TensorFlow Lite), or TensorFlow.js for web.
✅ Cloud Deployment – Host on Google Cloud, AWS, or Firebase.

🖼️ Dataset
The dataset is from the PlantVillage dataset and includes:

Healthy Potato Leaves
Diseased Potato Leaves (Late Blight, Early Blight)
📌 Dataset Source: PlantVillage Dataset

🛠️ Tech Stack
Python 3.11
TensorFlow/Keras (for model training)
Flask/FastAPI (for API deployment)
OpenCV/PIL (for image preprocessing)
NumPy, Pandas, Matplotlib (for data analysis)
TensorFlow Lite (TFLite) (for mobile app deployment)
Google Cloud/AWS (for cloud hosting)
🚀 Installation & Setup
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/Abena-3565/Potato_Disease-Classification.git
cd Potato_Disease-Classification
2️⃣ Create Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate  # For Windows
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Train the Model (Optional)
If you want to retrain the model, run:

bash
Copy
Edit
python train.py
This will save the model as potato_model.h5.

📡 Deployment Options
🌐 Web API (FastAPI)
Run the API locally:

bash
Copy
Edit
uvicorn app:app --host 0.0.0.0 --port 8000
Then send a test request:

bash
Copy
Edit
curl -X POST -F "file=@test_leaf.jpg" http://127.0.0.1:8000/predict/
📱 Android App (TensorFlow Lite)
Convert the model to TFLite format:

bash
Copy
Edit
tflite_convert --saved_model_dir=potato_model/ --output_file=potato_model.tflite
Integrate it into an Android app using ML Kit.

☁️ Cloud Deployment
Google Cloud Run (for scalable API hosting)
AWS Lambda + API Gateway (serverless API)
Firebase Hosting (for web app)
📌 Example Usage
Using Python
python
Copy
Edit
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("potato_model.h5")

def predict_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))
    prediction = model.predict(img_array)
    return prediction

print(predict_image("test_leaf.jpg"))
📷 Screenshots
Healthy	Late Blight	Early Blight
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repo
Create a new branch (feature-new-idea)
Commit your changes
Submit a Pull Request
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

💡 Acknowledgments
PlantVillage Dataset
TensorFlow/Keras Community
OpenAI & Deep Learning Research

📩 Contact
For questions or suggestions, reach out:
📧 Email: abenezeralz659gmail.com
