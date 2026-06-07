#A command-line tool to inspect the tables, schemas, and current contents of the student database.

import sqlite3
from pathlib import Path

def inspect_database(db_path='face_db/uniguard.db'):
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"[Error] Database file not found at {db_path.resolve()}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "="*60)
    print("           UNI-GUARD DATABASE INSPECTOR          ")
    print("="*60)

    # 1. Database Stats
    cursor.execute("SELECT COUNT(*) FROM students")
    student_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM attendance")
    attendance_count = cursor.fetchone()[0]

    print(f"  • SQLite File:      {db_path.name}")
    print(f"  • Registered Students: {student_count}")
    print(f"  • Scanned Attendance:  {attendance_count}")
    print("="*60)

    # 2. Print Students
    print("\n[Table: students] Registered Student Profiles")
    print("-" * 88)
    print(f"{'Roll No':<10} | {'Name':<22} | {'Gender':<8} | {'Year':<5} | {'Target Tag':<10}")
    print("-" * 88)
    
    cursor.execute("SELECT roll_no, name, gender, batch_year, tag_color FROM students")
    for row in cursor.fetchall():
        roll_no, name, gender, year, tag = row
        print(f"{roll_no:<10} | {name:<22} | {gender:<8} | {year:<5} | {tag:<10}")
    print("-" * 88)

    # 3. Print Attendance
    print("\n[Table: attendance] Recent Scan Events")
    if attendance_count == 0:
        print("  (No attendance scans have been logged yet. Launch the dashboard to scan faces!)")
    else:
        print("-" * 90)
        print(f"{'ID':<4} | {'Student Name':<20} | {'Timestamp':<20} | {'Status':<10} | {'Violation Type':<15}")
        print("-" * 90)
        
        cursor.execute("""
            SELECT a.id, s.name, a.timestamp, a.status, COALESCE(a.violation_type, 'None')
            FROM attendance a
            JOIN students s ON a.student_id = s.roll_no
            ORDER BY a.timestamp DESC
            LIMIT 10
        """)
        for row in cursor.fetchall():
            id_val, name, ts, status, violation = row
            # Trim name if too long for layout
            name_str = name[:18] + ".." if len(name) > 18 else name
            print(f"{id_val:<4} | {name_str:<20} | {ts:<20} | {status:<10} | {violation:<15}")
        print("-" * 90)

    conn.close()

if __name__ == "__main__":
    inspect_database()
