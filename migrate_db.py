import sqlite3
import os

db_path = 'varman.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    try:
        cur.execute('ALTER TABLE users ADD COLUMN vault_salt VARCHAR(64) NOT NULL DEFAULT ""')
        print('Added vault_salt to users.')
    except Exception as e:
        print('Error altering users:', e)
        
    try:
        cur.execute('ALTER TABLE protected_images ADD COLUMN vault_sealed BOOLEAN NOT NULL DEFAULT 0')
        print('Added vault_sealed to protected_images.')
    except Exception as e:
        print('Error altering protected_images:', e)
        
    conn.commit()
    conn.close()
    print("Migration complete.")
else:
    print('Database app.db not found.')
