🔍 Overview


This project implements a real-time facial recognition system using deep learning techniques. It leverages powerful models like MTCNN for face detection and InceptionResnetV1 for face recognition, providing accurate and efficient performance.


🚀 Features


* Real-Time Face Detection: Utilizes MTCNN to detect faces in live video streams or images.
* Face Recognition: Employs InceptionResnetV1 to recognize and verify identities.
* Image and Video Support: Processes both static images and live video feeds.
* Embeddings Management: Stores and updates facial embeddings for known individuals.


🛠️ Installation

1- Clone the repository
git clone https://github.com/OmarAbdelhedi/Facial-Recognition-Project.git
cd Facial-Recognition-Project

2- Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


3- Install Dependencies:
pip install -r requirements.txt


📁 Project Structure


Facial-Recognition-Project/
├── face_recognition_using_a_path.py
├── live_face_recognition.py
├── deleting_identities.ipynb
├── preprocessing.py
├── server.py
├── face_embeddings.pkl
├── requirements.txt
└── README.md

* face_recognition_using_a_path.py: Script to recognize faces from a given image path.
* live_face_recognition.py: Script to perform real-time face recognition using a webcam.
* deleting_identities.ipynb: Notebook to manage and delete stored identities.
* preprocessing.py: Contains preprocessing functions for face images to create a face embeddings on your own dataset.
* server.py: Optional server script for deploying the application.
* face_embeddings.pkl: Serialized file storing facial embeddings and identities.
* requirements.txt: List of required Python packages.

💡 Future Improvements

* Integrate a graphical user interface (GUI) for enhanced user experience.
* Implement face recognition in video files.
* Enhance the database management system for storing embeddings.




