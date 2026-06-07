#Initializes and resets the weekly attendance tracking matrices in the database.

import sqlite3
import datetime
import os
from pathlib import Path

db_path = os.path.join(os.path.dirname(__file__), "..", "face_db", "uniguard.db")

# Connect Database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop existing table first to rebuild with new schema
cursor.execute("DROP TABLE IF EXISTS weekly_attendance")

# Ensure table structure exists with MON - SUN column names
cursor.execute("""
CREATE TABLE IF NOT EXISTS weekly_attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    register_number TEXT,
    week_name TEXT,
    MON TEXT,
    TUE TEXT,
    WED TEXT,
    THU TEXT,
    FRI TEXT,
    SAT TEXT,
    SUN TEXT,
    total_present INTEGER,
    attendance_percentage REAL,
    FOREIGN KEY (register_number) REFERENCES students(roll_no)
)
""")

# Clear existing records
cursor.execute("DELETE FROM weekly_attendance")

# Find Mondays strictly within May 2026
year = 2026
month = 5

mondays = []
for day in range(1, 32):
    try:
        d = datetime.date(year, month, day)
        if d.weekday() == 0:  # Monday
            mondays.append(d)
    except ValueError:
        break
mondays.sort()

# Fetch all student records to build real database-driven rows
cursor.execute("SELECT roll_no, name FROM students ORDER BY roll_no ASC")
students = cursor.fetchall()

for offset, monday in enumerate(mondays[:4]):
    week_name = f"Week {offset + 1}"
    
    for roll_no, name in students:
        days_vals = []
        present_count = 0
        
        # Determine status for each day (1 to 7) based on real 'attendance' log entries
        for day_offset in range(7):
            target_date = monday + datetime.timedelta(days=day_offset)
            target_date_str = target_date.strftime("%Y-%m-%d")
            
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND date(timestamp) = ? AND status = 'PRESENT'
            """, (roll_no, target_date_str))
            
            if cursor.fetchone():
                days_vals.append("P")
                present_count += 1
            else:
                days_vals.append("A")
                
        # In the uniguard.db database schema, percentage is calculated over all 7 columns
        att_pct = round(present_count / 7 * 100, 2)
        
        cursor.execute("""
            INSERT INTO weekly_attendance (
                register_number, week_name,
                MON, TUE, WED, THU, FRI, SAT, SUN,
                total_present, attendance_percentage
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (roll_no, week_name, *days_vals, present_count, att_pct))

conn.commit()
conn.close()

print("Successfully synchronized and rebuilt weekly_attendance table dynamically from real attendance logs!")
