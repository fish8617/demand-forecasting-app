import sqlite3
from datetime import datetime, timedelta

DB_NAME = "alpha_db.sqlite"

def connect_db():
    conn = sqlite3.connect(DB_NAME)
    return conn

def init_db():
    conn = connect_db()
    c = conn.cursor()

    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            department TEXT NOT NULL
        )
    ''')

    # Create sales table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            submitted_by TEXT,
            FOREIGN KEY(submitted_by) REFERENCES users(username)
        )
    ''')

    # Create production table
    c.execute('''
        CREATE TABLE IF NOT EXISTS production (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            submitted_by TEXT,
            FOREIGN KEY(submitted_by) REFERENCES users(username)
        )
    ''')

    # Create stock table
    c.execute('''
        CREATE TABLE IF NOT EXISTS stock (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            production_date TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            expiry_date TEXT NOT NULL
        )
    ''')

    # Create approvals table
    c.execute('''
        CREATE TABLE IF NOT EXISTS approvals (
            report_date TEXT PRIMARY KEY,
            predicted_sales INTEGER,
            actual_sales INTEGER,
            recommended_production INTEGER,
            status TEXT DEFAULT 'pending',
            approved_by TEXT,
            approved_at TEXT,
            notes TEXT
        )
    ''')

    conn.commit()
    conn.close()

def create_usertable():
    # Users table is created in init_db, so just ensure DB is initialized
    init_db()

def add_userdata(username, password, department):
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (username, password, department) VALUES (?, ?, ?)",
              (username, password, department))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    data = c.fetchone()
    conn.close()
    return data is not None

def get_user_department(username):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT department FROM users WHERE username = ?", (username,))
    data = c.fetchone()
    conn.close()
    if data:
        return data[0]
    return None

def add_sales_data(date_str, quantity, username):
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT INTO sales (date, quantity, submitted_by) VALUES (?, ?, ?)",
              (date_str, quantity, username))
    conn.commit()
    conn.close()

def add_production_data(date_str, quantity, username):
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT INTO production (date, quantity, submitted_by) VALUES (?, ?, ?)",
              (date_str, quantity, username))
    conn.commit()
    conn.close()

def update_stock(production_date, quantity_change):
    conn = connect_db()
    c = conn.cursor()

    # Check if stock entry exists for the production_date
    c.execute("SELECT id, quantity, expiry_date FROM stock WHERE production_date = ?", (production_date,))
    row = c.fetchone()

    # For simplicity, assume expiry date is 30 days after production date if new entry
    expiry_date = (datetime.strptime(production_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")

    if row:
        stock_id, current_quantity, current_expiry = row
        new_quantity = current_quantity + quantity_change
        if new_quantity <= 0:
            # Remove stock entry if quantity zero or less
            c.execute("DELETE FROM stock WHERE id = ?", (stock_id,))
        else:
            c.execute("UPDATE stock SET quantity = ? WHERE id = ?", (new_quantity, stock_id))
    else:
        if quantity_change > 0:
            c.execute("INSERT INTO stock (production_date, quantity, expiry_date) VALUES (?, ?, ?)",
                      (production_date, quantity_change, expiry_date))
        # If quantity_change negative and no stock entry, do nothing

    conn.commit()
    conn.close()

def submit_report_for_approval(report_date, predicted_sales, actual_sales, recommended_production):
    conn = connect_db()
    c = conn.cursor()
    # Insert or replace report
    c.execute('''
        INSERT OR REPLACE INTO approvals
        (report_date, predicted_sales, actual_sales, recommended_production, status)
        VALUES (?, ?, ?, ?, 'pending')
    ''', (report_date, predicted_sales, actual_sales, recommended_production))
    conn.commit()
    conn.close()

def get_stock_level():
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT SUM(quantity) FROM stock")
    result = c.fetchone()
    conn.close()
    return result[0] if result[0] is not None else 0

def get_expiring_stock(days_threshold=7):
    conn = connect_db()
    c = conn.cursor()
    threshold_date = (datetime.now() + timedelta(days=days_threshold)).strftime("%Y-%m-%d")
    c.execute("SELECT production_date, quantity, expiry_date FROM stock WHERE expiry_date <= ?", (threshold_date,))
    results = c.fetchall()
    conn.close()
    return results

def get_report_history(limit=10):
    conn = connect_db()
    c = conn.cursor()
    c.execute('''
        SELECT report_date, predicted_sales, actual_sales, recommended_production, status, approved_by, approved_at, notes
        FROM approvals
        ORDER BY report_date DESC
        LIMIT ?
    ''', (limit,))
    results = c.fetchall()
    conn.close()
    return results

def get_pending_reports():
    conn = connect_db()
    c = conn.cursor()
    c.execute('''
        SELECT report_date, predicted_sales, actual_sales, recommended_production, status, approved_by, approved_at, notes
        FROM approvals
        WHERE status = 'pending'
        ORDER BY report_date DESC
    ''')
    results = c.fetchall()
    conn.close()
    return results

def approve_report(report_date, username, status, notes):
    conn = connect_db()
    c = conn.cursor()
    approved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        UPDATE approvals
        SET status = ?, approved_by = ?, approved_at = ?, notes = ?
        WHERE report_date = ?
    ''', (status, username, approved_at, notes, report_date))
    conn.commit()
    conn.close()

def get_sales_data(start_date=None, end_date=None):
    conn = connect_db()
    c = conn.cursor()
    query = "SELECT date, quantity FROM sales"
    params = []
    if start_date and end_date:
        query += " WHERE date BETWEEN ? AND ?"
        params.extend([start_date, end_date])
    query += " ORDER BY date ASC"
    c.execute(query, params)
    results = c.fetchall()
    conn.close()
    return results

def get_production_data(start_date=None, end_date=None):
    conn = connect_db()
    c = conn.cursor()
    query = "SELECT date, quantity FROM production"
    params = []
    if start_date and end_date:
        query += " WHERE date BETWEEN ? AND ?"
        params.extend([start_date, end_date])
    query += " ORDER BY date ASC"
    c.execute(query, params)
    results = c.fetchall()
    conn.close()
    return results
