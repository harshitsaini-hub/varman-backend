import sqlite3
import json
import numpy as np

def setup_test_db():
    
    conn = sqlite3.connect('test_amor_memory.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS unknown_faces (
            unknown_id TEXT PRIMARY KEY,
            encoding JSON
        )
    ''')

    fake_encoding = np.random.rand(128)
    encoding_json = json.dumps(fake_encoding.tolist())
    
    cursor.execute('''
        INSERT OR REPLACE INTO unknown_faces (unknown_id, encoding) 
        VALUES (?, ?)
    ''', ("U-TEST-001", encoding_json))
    
    conn.commit()

    cursor.execute('SELECT * FROM unknown_faces WHERE unknown_id = "U-TEST-001"')
    row = cursor.fetchone()
    
    print(f"Success! Retrieved ID: {row[0]}")
    print(f"Data stored safely as: {type(row[1])} (Array Length: {len(json.loads(row[1]))})")
    
    conn.close()

if __name__ == "__main__":
    setup_test_db()