from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from database import Database

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")  # Enable CORS for frontend
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSocket
db = Database(db_name="student_attendance", user="postgres", password="arslanbtw123")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('test_event', {'message': 'Hello from Flask!'})

# Fetch all students
@app.route("/students", methods=["GET"])
def get_students():
    db.cursor.execute("SELECT id, name FROM students")
    students = db.cursor.fetchall()
    return jsonify(students)

# Fetch attendance records
@app.route("/attendance", methods=["GET"])
def get_attendance():
    db.cursor.execute("""
        SELECT a.id, s.name, a.timestamp 
        FROM attendance a 
        JOIN students s ON a.student_id = s.id
        ORDER BY a.timestamp DESC
    """)
    attendance_records = db.cursor.fetchall()
    return jsonify(attendance_records)

@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():
    data = request.json
    student_id = data.get("student_id")

    if not student_id:
        return jsonify({"error": "Student ID is required"}), 400

    try:
        # Use context manager for the cursor
        with db.connection.cursor() as cursor:
            # Check if attendance is already marked
            cursor.execute("""
                SELECT COUNT(*) FROM attendance
                WHERE student_id = %s AND DATE(timestamp) = CURRENT_DATE
            """, (student_id,))
            if cursor.fetchone()[0] > 0:
                return jsonify({"status": "duplicate", "message": "Attendance already marked"}), 200

            # Mark attendance
            cursor.execute(
                "INSERT INTO attendance (student_id) VALUES (%s) RETURNING id, timestamp",
                (student_id,)
            )
            attendance = cursor.fetchone()
            cursor.connection.commit()

            # Convert timestamp to the desired format
            formatted_timestamp = attendance[1].strftime('%a, %d %b %Y %H:%M:%S GMT')  # Example: Fri, 29 Nov 2024 00:00:51 GMT
            
            # Query to get student's name
            cursor.execute("""
                SELECT name FROM students
                WHERE id = %s
            """, (student_id,))
            name = cursor.fetchone()

            attendance_data = {
                "student_id": attendance[0],
                "student_name": name[0],
                "timestamp": formatted_timestamp
            }
            
            # Emit real-time update to clients
            socketio.emit("new_attendance", {
                "data":attendance_data,
            
            })

        return jsonify({"message": "Attendance marked successfully"}), 201
    except Exception as e:
        # Rollback on error to maintain consistency
        db.connection.rollback()
        return jsonify({"error": str(e)}), 500


# Run Flask with SocketIO
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, debug=True)
