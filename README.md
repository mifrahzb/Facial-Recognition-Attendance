# Facial Recognition Attendance System

A real-time attendance tracking system that uses facial recognition technology to automatically identify and mark student attendance. Built with a Python Flask backend for facial recognition processing and a React frontend for real-time attendance visualization.

## 🚀 Features

- **Real-Time Face Detection**: Uses OpenCV's DNN model for accurate face detection from webcam feed
- **Facial Recognition**: Employs PyTorch-based deep learning models to generate and match face embeddings
- **Automated Attendance Marking**: Automatically marks attendance when a registered face is detected
- **Live Updates**: WebSocket integration provides real-time attendance updates to the frontend
- **Duplicate Prevention**: Intelligent caching system prevents duplicate attendance entries within a 10-second window
- **PostgreSQL Database**: Stores student information, face embeddings, and attendance records
- **Quality Checks**: Validates face detection quality, centering, and size before processing
- **High Accuracy**: Uses cosine similarity matching with configurable threshold (default: 0.89)

## 🏗️ Architecture

### Backend (`/backend`)
- **Flask API**: RESTful endpoints for student and attendance management
- **Face Detection**: Pre-trained Caffe DNN model (ResNet-based SSD)
- **Face Recognition**: PyTorch-based embedding generation
- **Database**: PostgreSQL with psycopg2 adapter
- **Real-time Communication**: Flask-SocketIO for WebSocket connections

### Frontend (`/front_end`)
- **React**: Modern UI for displaying students and attendance records
- **Socket.io Client**: Real-time updates for newly marked attendance
- **Responsive Design**: Clean interface for monitoring attendance

## 📋 Prerequisites

- Python 3.8+
- Node.js 14+
- PostgreSQL 12+
- Webcam

## 🔧 Installation

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mifrahzb/Facial-Recognition-Attendance.git
   cd Facial-Recognition-Attendance/backend
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Additional dependencies** (for OpenCV and capture functionality)
   ```bash
   pip install opencv-python flask flask-cors flask-socketio requests
   ```

4. **Download Face Detection Model**
   - Place the following files in `backend/model/`:
     - `deploy.prototxt`
     - `res10_300x300_ssd_iter_140000.caffemodel`
   - Download from: [OpenCV Face Detection Model](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

5. **Configure Database**
   - Create PostgreSQL database:
     ```sql
     CREATE DATABASE student_attendance;
     ```
   - Update database credentials in `app.py`, `capture.py`, and `database.py`:
     ```python
     db = Database(
         db_name="student_attendance",
         user="your_username",
         password="your_password"
     )
     ```

6. **Initialize Database Tables**
   ```bash
   python database.py
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd ../front_end
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

## 🚀 Usage

### 1. Start the Backend Server

```bash
cd backend
python app.py
```

The Flask server will start on `http://127.0.0.1:5000`

### 2. Start the Frontend Application

```bash
cd front_end
npm start
```

The React app will open on `http://localhost:3000`

### 3. Run the Face Recognition System

In a separate terminal:

```bash
cd backend
python capture.py
```

This will:
- Activate your webcam
- Detect faces in real-time
- Match detected faces against stored embeddings
- Automatically mark attendance when a match is found

### 4. Register New Students

Use the `InsertData.py` script to register new students:

```bash
cd backend
python InsertData.py
```

This will:
- Capture student information
- Take their photo
- Generate face embeddings
- Store everything in the database

## 📡 API Endpoints

### GET `/students`
Retrieve all registered students
```json
[
  {"id": 1, "name": "John Doe"}
]
```

### GET `/attendance`
Retrieve all attendance records
```json
[
  {
    "id": 1,
    "name": "John Doe",
    "timestamp": "Fri, 29 Nov 2024 10:30:00 GMT"
  }
]
```

### POST `/mark_attendance`
Mark attendance for a student
```json
{
  "student_id": 1
}
```

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework
- **OpenCV**: Face detection and image processing
- **PyTorch & Torchvision**: Deep learning for face embeddings
- **NumPy**: Numerical computations
- **Pillow**: Image manipulation
- **psycopg2**: PostgreSQL adapter
- **Flask-SocketIO**: WebSocket support
- **Flask-CORS**: Cross-origin resource sharing

### Frontend
- **React 18**: UI framework
- **Socket.io-client**: WebSocket client
- **React Scripts**: Build tooling

### Database
- **PostgreSQL**: Relational database for storing:
  - Student information
  - Face embeddings (as FLOAT8 arrays)
  - Attendance records with timestamps

## 📁 Project Structure

```
Facial-Recognition-Attendance/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── capture.py                # Face detection & recognition
│   ├── database.py               # Database connection & operations
│   ├── preprocessing.py          # Image preprocessing
│   ├── ToEmbeddings.py          # Face embedding generation
│   ├── cosine_similarity.py     # Similarity matching
│   ├── InsertData.py            # Student registration
│   ├── requirements.txt          # Python dependencies
│   └── model/                    # Face detection models
│       ├── deploy.prototxt
│       └── res10_300x300_ssd_iter_140000.caffemodel
├── front_end/
│   ├── src/                      # React source files
│   ├── public/                   # Static assets
│   ├── package.json              # Node dependencies
│   └── README.md                 # React documentation
└── README.md                     # This file
```

## 🔒 Database Schema

### Students Table
```sql
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    student_id VARCHAR(20) NOT NULL UNIQUE,
    gender VARCHAR(10),
    department VARCHAR(50),
    image BYTEA,
    embedding FLOAT8[]
);
```

### Attendance Table
```sql
CREATE TABLE attendance (
    id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## ⚙️ Configuration

### Adjustable Parameters

**Face Detection (`capture.py`)**:
- `confidence`: Detection threshold (default: 0.7)
- `face_area`: Minimum face size (default: 5% of frame)
- `buffer_time`: Time between captures (default: 2 seconds)

**Face Matching (`cosine_similarity.py`)**:
- `threshold`: Similarity threshold for match (default: 0.89)

**Duplicate Prevention (`capture.py`)**:
- `recently_marked`: Cache timeout (default: 10 seconds)

## 🐛 Troubleshooting

### Camera Not Opening
- Ensure webcam is connected
- Check if another application is using the camera
- Try changing camera index in `cv2.VideoCapture(0)` to `1` or `2`

### Face Not Detected
- Ensure good lighting
- Position face directly in front of camera
- Face should be centered and occupy at least 5% of frame

### Database Connection Error
- Verify PostgreSQL is running
- Check database credentials
- Ensure database exists

### No Match Found
- Verify student is registered in database
- Check similarity threshold (lower if needed)
- Ensure good quality face capture during registration
