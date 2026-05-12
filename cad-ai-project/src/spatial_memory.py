import sqlite3
import os
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List

# Set default SQLite database path to storage/cad_memory.db
# Assuming this script is run from the project root or its directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")

# Ensure storage directory exists
if not os.path.exists(STORAGE_DIR):
    try:
        os.makedirs(STORAGE_DIR)
    except Exception:
        pass

DB_PATH = os.getenv("CAD_MEMORY_DB", os.path.join(STORAGE_DIR, "cad_memory.db"))

class MemoryEntry(BaseModel):
    id: Optional[int] = None
    file_path: str
    query_keyword: str
    matched_text: str
    classification: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    parent_keyword: Optional[str] = None
    parent_min_x: Optional[float] = None
    parent_min_y: Optional[float] = None
    parent_max_x: Optional[float] = None
    parent_max_y: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

def get_connection(db_path=DB_PATH):
    """Returns a SQLite connection with performance optimizations enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def init_db(db_path=DB_PATH):
    """Initializes the SQLite database with the required schema and indexes."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cad_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            query_keyword TEXT,
            matched_text TEXT,
            classification TEXT,
            min_x REAL,
            min_y REAL,
            max_x REAL,
            max_y REAL,
            created_at DATETIME
        )
    ''')
    
    # Check if new columns exist, add them if not (for schema migration)
    cursor.execute("PRAGMA table_info(cad_memory)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'parent_keyword' not in columns:
        cursor.execute("ALTER TABLE cad_memory ADD COLUMN parent_keyword TEXT")
    if 'parent_min_x' not in columns:
        cursor.execute("ALTER TABLE cad_memory ADD COLUMN parent_min_x REAL")
    if 'parent_min_y' not in columns:
        cursor.execute("ALTER TABLE cad_memory ADD COLUMN parent_min_y REAL")
    if 'parent_max_x' not in columns:
        cursor.execute("ALTER TABLE cad_memory ADD COLUMN parent_max_x REAL")
    if 'parent_max_y' not in columns:
        cursor.execute("ALTER TABLE cad_memory ADD COLUMN parent_max_y REAL")
        
    # Create indexes for efficient retrieval
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON cad_memory(file_path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_keyword ON cad_memory(query_keyword)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_classification ON cad_memory(classification)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_class ON cad_memory(file_path, classification)")
        
    conn.commit()
    conn.close()

def clear_memories(file_path: str, query_keyword: str, db_path=DB_PATH):
    """Clears previous memories for a specific file and query to prevent duplicates."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM cad_memory 
        WHERE file_path = ? AND query_keyword = ?
    ''', (file_path, query_keyword))
    conn.commit()
    conn.close()

def save_memory(entry: MemoryEntry, db_path=DB_PATH) -> MemoryEntry:
    """Saves a new spatial memory entry to the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO cad_memory 
        (file_path, query_keyword, matched_text, classification, min_x, min_y, max_x, max_y, parent_keyword, parent_min_x, parent_min_y, parent_max_x, parent_max_y, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        entry.file_path, 
        entry.query_keyword, 
        entry.matched_text, 
        entry.classification, 
        entry.min_x, 
        entry.min_y, 
        entry.max_x, 
        entry.max_y, 
        entry.parent_keyword,
        entry.parent_min_x,
        entry.parent_min_y,
        entry.parent_max_x,
        entry.parent_max_y,
        entry.created_at.isoformat()
    ))
    conn.commit()
    entry.id = cursor.lastrowid
    conn.close()
    return entry

def get_memories(file_path: str, query_keyword: str, classification: Optional[str] = None, db_path=DB_PATH, time_window_seconds: int = 60) -> List[MemoryEntry]:
    """Retrieves all memory entries matching the criteria created around the same time."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # First get the most recent timestamp for this query
    latest_query = """
        SELECT created_at 
        FROM cad_memory 
        WHERE file_path = ? AND query_keyword = ?
    """
    params = [file_path, query_keyword]
    
    if classification:
        latest_query += " AND classification = ?"
        params.append(classification)
        
    latest_query += " ORDER BY created_at DESC LIMIT 1"
    
    cursor.execute(latest_query, tuple(params))
    latest_row = cursor.fetchone()
    
    if not latest_row:
        conn.close()
        return []
        
    latest_time_str = latest_row[0]
    
    # Now get all entries within a small time window of the latest query
    # (since multiple boxes for the same query are saved back-to-back)
    query = """
        SELECT id, file_path, query_keyword, matched_text, classification, 
               min_x, min_y, max_x, max_y, parent_keyword, parent_min_x, parent_min_y, parent_max_x, parent_max_y, created_at 
        FROM cad_memory 
        WHERE file_path = ? AND query_keyword = ?
    """
    
    cursor.execute(query, tuple([file_path, query_keyword]))
    rows = cursor.fetchall()
    conn.close()
    
    # Filter by classification and time window in Python for simplicity
    latest_dt = datetime.fromisoformat(latest_time_str) if isinstance(latest_time_str, str) else latest_time_str
    
    results = []
    for row in rows:
        row_class = row[4]
        if classification and row_class != classification:
            continue
            
        row_time = row[14]
        row_dt = datetime.fromisoformat(row_time) if isinstance(row_time, str) else row_time
        
        # Check if it was created within the time window (e.g. same batch of saves)
        diff = abs((latest_dt - row_dt).total_seconds())
        if diff <= time_window_seconds:
            results.append(MemoryEntry(
                id=row[0],
                file_path=row[1],
                query_keyword=row[2],
                matched_text=row[3],
                classification=row[4],
                min_x=row[5],
                min_y=row[6],
                max_x=row[7],
                max_y=row[8],
                parent_keyword=row[9],
                parent_min_x=row[10],
                parent_min_y=row[11],
                parent_max_x=row[12],
                parent_max_y=row[13],
                created_at=row_dt
            ))
            
    return results

def get_all_parents(file_path: str, db_path=DB_PATH) -> List[MemoryEntry]:
    """Retrieves all memory entries that are general drawings (parents) for a file."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    query = """
        SELECT id, file_path, query_keyword, matched_text, classification, 
               min_x, min_y, max_x, max_y, parent_keyword, parent_min_x, parent_min_y, parent_max_x, parent_max_y, created_at 
        FROM cad_memory 
        WHERE file_path = ? AND classification = 'general drawing'
    """
    
    cursor.execute(query, (file_path,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append(MemoryEntry(
            id=row[0],
            file_path=row[1],
            query_keyword=row[2],
            matched_text=row[3],
            classification=row[4],
            min_x=row[5],
            min_y=row[6],
            max_x=row[7],
            max_y=row[8],
            parent_keyword=row[9],
            parent_min_x=row[10],
            parent_min_y=row[11],
            parent_max_x=row[12],
            parent_max_y=row[13],
            created_at=datetime.fromisoformat(row[14]) if isinstance(row[14], str) else row[14]
        ))
        
    return results
