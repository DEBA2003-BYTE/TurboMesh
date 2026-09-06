import sqlite3

DATABASE = "turbomesh.db"


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS gpu_hosts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            gpu_name TEXT,
            vram TEXT,
            status TEXT NOT NULL DEFAULT 'OFFLINE',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS access_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            host_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (host_id) REFERENCES gpu_hosts(id)
        )
    """)
    connection.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        host_id INTEGER NOT NULL,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'QUEUED',
        result TEXT,
        image_path TEXT,
        image_path_2 TEXT,
        output_path TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (host_id) REFERENCES gpu_hosts(id)
    )
    """)

    connection.commit()
    connection.close()