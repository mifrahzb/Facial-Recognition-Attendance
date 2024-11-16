import preprocessing
import ToEmbeddings
from database import Database
import cv2

# Database connection
db = Database(db_name="student_attendance", user="postgres", password="arslanbtw123")

# Create the students table (if it doesn't already exist)
db.create_table()

# Example: Insert student data
name = "Arslan"
student_id = "CS1234"
gender = "Male"
department = "Computer Science"
image_path = "./arslan.png"

# Preprocess the image and generate embedding
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")
preprocessed_face = preprocessing.process_image(image)
embedding = ToEmbeddings.get_face_embedding(preprocessed_face)

# Read the image file as binary
with open(image_path, "rb") as file:
    image_binary = file.read()

# Insert data into the database
db.insert_student(name, student_id, gender, department, image_binary, embedding)

# Close the database connection
db.close()
