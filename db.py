from services.db_service import get_db_connection

def get_db_session():
    return get_db_connection()
