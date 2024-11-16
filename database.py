import psycopg2
import numpy as np

class Database:
    def __init__(self, db_name, user, password, host="localhost", port=5432):
        """
        Initialize the database connection and cursor.
        """
        self.connection = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cursor = self.connection.cursor()

    def create_table(self):
        """
        Create the 'students' table if it doesn't already exist.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            student_id VARCHAR(20) NOT NULL UNIQUE,
            gender VARCHAR(10),
            department VARCHAR(50),
            image BYTEA,
            embedding FLOAT8[]
        );
        """
        self.cursor.execute(create_table_query)
        self.connection.commit()

    def insert_student(self, name, student_id, gender, department, image, embedding):
        """
        Insert a student into the database.

        Parameters:
        - name (str): Name of the student.
        - student_id (str): ID of the student.
        - gender (str): Gender of the student.
        - department (str): Department of the student.
        - image (bytes): Binary image data.
        - embedding (numpy.ndarray): Vector embedding of the student's face.
        """
        try:
            # Convert the embedding from numpy array to a list of Python floats
            embedding_list = embedding.astype(float).tolist()
            
            # SQL query to insert student data
            query = """
            INSERT INTO students (name, student_id, gender, department, image, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (name, student_id, gender, department, image, embedding_list))
            
            # Commit the transaction
            self.connection.commit()
            print("Student inserted successfully!")
        except Exception as e:
            print("Error inserting student:", e)

    def close(self):
        """
        Close the database connection and cursor.
        """
        self.cursor.close()
        self.connection.close()
