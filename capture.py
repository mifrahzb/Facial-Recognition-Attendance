import cv2
import time
import numpy as np
import preprocessing  # Import the preprocessing module
import ToEmbeddings  # Import the embeddings module
from database import Database
from cosine_similarity import match_embedding

# Load the pre-trained DNN model for face detection
model = cv2.dnn.readNetFromCaffe(
    "./model/deploy.prototxt",
    "./model/res10_300x300_ssd_iter_140000.caffemodel"
)

# Initialize webcam capture
cap = cv2.VideoCapture(1)

# Database connection
db = Database(db_name="student_attendance", user="postgres", password="arslanbtw123")
db.create_table()  # Ensure table exists

# Fetch stored embeddings from the database
db.cursor.execute("SELECT id, name, embedding FROM students")
stored_data = db.cursor.fetchall()
stored_embeddings = [np.array(record[2]) for record in stored_data]  # Convert embeddings to numpy arrays
student_ids = [record[0] for record in stored_data]
student_names = [record[1] for record in stored_data]

# Buffer time setup
last_capture_time = time.time()
buffer_time = 2  # Buffer time in seconds to avoid duplicate captures

while True:
    # Capture the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for the model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))  # Larger input size
    model.setInput(blob)
    detections = model.forward()

    # Quality check variables
    frame_height, frame_width = frame.shape[:2]
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Apply a high confidence threshold to ensure it's a strong detection
        if confidence > 0.7:  # Adjusted confidence threshold
            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * [frame_width, frame_height, frame_width, frame_height]
            (x, y, x1, y1) = box.astype("int")

            # Calculate the area of the detected face
            face_area = (x1 - x) * (y1 - y)

            # Quality check: Ensure the face is sufficiently large
            if face_area > 0.1 * frame_width * frame_height:  # Ensure the face covers at least 10% of the frame
                face_boxes.append((x, y, x1, y1))

    # Process and crop each face
    for (x, y, x1, y1) in face_boxes:
        # Check if the bounding box is valid (non-zero area)
        if (x1 - x) > 0 and (y1 - y) > 0:
            cropped_face = frame[y:y1, x:x1]  # Crop face
            preprocessed_face = preprocessing.process_image(cropped_face)  # Preprocess the cropped face
            new_embedding = ToEmbeddings.get_face_embedding(preprocessed_face)  # Generate embedding
            # print("Embedding generated for one face:", new_embedding)
            
            # Match embedding with stored embeddings
            matches = match_embedding(new_embedding, stored_embeddings, threshold=0.8)

            if matches:
                for idx, similarity in matches:
                    print(f"Match found: Student ID: {student_ids[idx]}, Name: {student_names[idx]}, Similarity: {similarity:.2f}")
            else:
                print("No match found for this face.")


    # Show the frame with bounding boxes
    for (x, y, x1, y1) in face_boxes:
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)  # Draw bounding box

    cv2.imshow("Face Detection and Processing", frame)

    # Press 'q' to manually exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
