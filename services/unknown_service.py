import sqlite3
import face_recognition
import json
import numpy as np

DB_PATH = 'storage/amor_memory.db'

def init_db():
    """Run this once when the backend starts to ensure the table exists."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS unknown_faces (
            unknown_id TEXT PRIMARY KEY,
            encoding JSON,
            times_seen INTEGER DEFAULT 1,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_new_unknown(unknown_id: str, encoding: np.ndarray):
    """Saves a brand new unknown face to the database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    # Convert numpy array -> list -> JSON string
    encoding_json = json.dumps(encoding.tolist())
    
    cursor.execute('''
        INSERT INTO unknown_faces (unknown_id, encoding) 
        VALUES (?, ?)
    ''', (unknown_id, encoding_json))
    
    conn.commit()
    conn.close()

def get_all_unknowns():
    """Pulls all unknown faces from the DB to compare against a new frame."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('SELECT unknown_id, encoding FROM unknown_faces')
    rows = cursor.fetchall()
    conn.close()
    
    unknown_data = []
    for row in rows:
        uid = row[0]
        enc_array = np.array(json.loads(row[1]))
        unknown_data.append({"id": uid, "encoding": enc_array})
        
    return unknown_data

def update_unknown_stats(unknown_id: str):
    """If we see the same unknown person again, increase their view count."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE unknown_faces 
        SET times_seen = times_seen + 1, last_seen = CURRENT_TIMESTAMP 
        WHERE unknown_id = ?
    ''', (unknown_id,))
    
    conn.commit()
    conn.close()

def find_matching_unknown(new_encoding: np.ndarray, tolerance: float = 0.5):
    """
    Pulls all faces from the DB and checks if the new face matches any of them.
    Returns the unknown_id if a match is found, otherwise returns None.
    """
    stored_unknowns = get_all_unknowns() # The function we built last time!
    
    if not stored_unknowns:
        return None # Database is empty, so it's definitely a new person
        
    # Extract just the math arrays (encodings) from our database records
    known_encodings = [item["encoding"] for item in stored_unknowns]
    
    # Compare the new face against everyone in the database
    matches = face_recognition.compare_faces(known_encodings, new_encoding, tolerance=tolerance)
    
    # If we get a True, find out exactly who it matched with
    if True in matches:
        first_match_index = matches.index(True)
        matched_id = stored_unknowns[first_match_index]["id"]
        return matched_id
        
    return None