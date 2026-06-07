#Handles all SQLite operations, table creations, auto-migrations, and logs daily and weekly attendance records.

import sqlite3
from pathlib import Path
from datetime import datetime

class DatabaseManager:
    """Manages SQLite database for students and attendance with strict, immediate connection disposal."""
    
    def __init__(self, db_path='uniguard.db'):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize database tables with automatic migration from older schemas containing the 'id' column."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                
                # Check if students table exists and has 'id' column
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(students)")
                    cols = [info[1] for info in cursor.fetchall()]
                    if "id" in cols:
                        print("[Migration] Detected old schema with 'id' column. Migrating to new schema...")
                        
                        # Disable foreign keys temporarily
                        cursor.execute("PRAGMA foreign_keys = OFF")
                        
                        # Create temporary tables with new schema
                        cursor.execute('''
                            CREATE TABLE students_new (
                                roll_no TEXT PRIMARY KEY,
                                name TEXT NOT NULL,
                                gender TEXT NOT NULL,
                                batch_year INTEGER,
                                tag_color TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            ) WITHOUT ROWID
                        ''')
                        
                        # Create temporary table for attendance
                        cursor.execute('''
                            CREATE TABLE attendance_new (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                student_id TEXT NOT NULL,
                                roll_no TEXT,
                                name TEXT,
                                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                status TEXT NOT NULL,
                                tag_color_verified TEXT,
                                violation_type TEXT,
                                FOREIGN KEY (student_id) REFERENCES students_new (roll_no)
                            )
                        ''')
                        
                        # Copy student data
                        cursor.execute('''
                            INSERT OR IGNORE INTO students_new (roll_no, name, gender, batch_year, tag_color, created_at)
                            SELECT roll_no, name, gender, batch_year, tag_color, created_at FROM students
                        ''')
                        
                        # Copy attendance data mapping integer student_id to roll_no
                        cursor.execute('''
                            SELECT name FROM sqlite_master WHERE type='table' AND name='attendance'
                        ''')
                        if cursor.fetchone():
                            cursor.execute('''
                                INSERT OR IGNORE INTO attendance_new (id, student_id, roll_no, name, timestamp, status, tag_color_verified, violation_type)
                                SELECT a.id, s.roll_no, a.roll_no, a.name, a.timestamp, a.status, a.tag_color_verified, a.violation_type
                                FROM attendance a
                                JOIN students s ON a.student_id = s.id
                            ''')
                            cursor.execute("DROP TABLE attendance")
                            
                        cursor.execute("DROP TABLE students")
                        cursor.execute("ALTER TABLE students_new RENAME TO students")
                        cursor.execute("ALTER TABLE attendance_new RENAME TO attendance")
                        
                        cursor.execute("PRAGMA foreign_keys = ON")
                        print("[Migration] Migration completed successfully.")
                    else:
                        # Check if students table is WITHOUT ROWID
                        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='students'")
                        row = cursor.fetchone()
                        if row:
                            sql = row[0]
                            if "WITHOUT ROWID" not in sql.upper():
                                print("[Migration] Detected students table with ROWID. Migrating to WITHOUT ROWID...")
                                cursor.execute("PRAGMA foreign_keys = OFF")
                                
                                # 1. Create temporary table with new schema
                                cursor.execute('''
                                    CREATE TABLE students_new (
                                        roll_no TEXT PRIMARY KEY,
                                        name TEXT NOT NULL,
                                        gender TEXT NOT NULL,
                                        batch_year INTEGER,
                                        tag_color TEXT,
                                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                    ) WITHOUT ROWID
                                ''')
                                
                                # 2. Copy data
                                cursor.execute('''
                                    INSERT OR IGNORE INTO students_new (roll_no, name, gender, batch_year, tag_color, created_at)
                                    SELECT roll_no, name, gender, batch_year, tag_color, created_at FROM students
                                ''')
                                
                                # 3. Drop old and rename new
                                cursor.execute("DROP TABLE students")
                                cursor.execute("ALTER TABLE students_new RENAME TO students")
                                
                                cursor.execute("PRAGMA foreign_keys = ON")
                                print("[Migration] Recreated students table as WITHOUT ROWID successfully.")
                
                # Create tables (will not do anything if they were already renamed or created during migration)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS students (
                        roll_no TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        gender TEXT NOT NULL,
                        batch_year INTEGER,
                        tag_color TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) WITHOUT ROWID
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        roll_no TEXT,
                        name TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT NOT NULL,
                        tag_color_verified TEXT,
                        violation_type TEXT,
                        FOREIGN KEY (student_id) REFERENCES students (roll_no)
                    )
                ''')
        finally:
            conn.close()

    def add_student(self, roll_no, name, gender, year, tag_color):
        """Add or update a student record."""
        if hasattr(self, '_student_cache') and roll_no in self._student_cache:
            del self._student_cache[roll_no]
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO students (roll_no, name, gender, batch_year, tag_color)
                    VALUES (?, ?, ?, ?, ?)
                ''', (roll_no, name, gender.lower(), year, tag_color.lower()))
                return True
        except Exception as e:
            print(f"[DBManager] Error adding student {roll_no}: {e}")
            return False
        finally:
            conn.close()

    def get_student_by_roll(self, roll_no):
        """Fetch student details by roll number with caching."""
        if not hasattr(self, '_student_cache'):
            self._student_cache = {}
            
        if roll_no in self._student_cache:
            return self._student_cache[roll_no]
            
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM students WHERE roll_no = ?', (roll_no,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                self._student_cache[roll_no] = data
                return data
            return None
        finally:
            conn.close()

    def clear_student_cache(self):
        """Clear the cached student records to force a fresh SQLite reload."""
        self._student_cache = {}

    def log_attendance(self, roll_no, status, tag_color=None, violation=None):
        """Log attendance with Priority State Override (Self-Correction & Downgrade Prevention)
        to guarantee exactly one resolved row per student per day."""
        student = self.get_student_by_roll(roll_no)
        if not student:
            print(f"[DBManager] person {roll_no} not found in database.")
            return False
            
        today_date = datetime.now().strftime('%Y-%m-%d')
        local_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                
                # Check if there is already an attendance entry for this student today
                cursor.execute('''
                    SELECT id, status FROM attendance 
                    WHERE student_id = ? AND date(timestamp) = ?
                ''', (roll_no, today_date))
                existing = cursor.fetchone()
                
                if existing:
                    existing_id, existing_status = existing
                    
                    # 1. Once a student is marked PRESENT today, we NEVER downgrade them back to VIOLATION
                    if existing_status == "PRESENT":
                        return True
                    
                    # 2. If they were marked VIOLATION, but are now PRESENT, upgrade them in-place (Self-Correction!)
                    if existing_status == "VIOLATION" and status == "PRESENT":
                        cursor.execute('''
                            UPDATE attendance 
                            SET timestamp = ?, status = ?, tag_color_verified = ?, violation_type = ?, roll_no = ?, name = ?
                            WHERE id = ?
                        ''', (local_time_str, status, tag_color, violation, roll_no, student['name'], existing_id))
                        print(f"[DBManager] Upgraded {roll_no} to PRESENT via Self-Correction!")
                        
                        # Real-time synchronization with weekly_attendance
                        self._sync_weekly_attendance(cursor, roll_no, local_time_str, status)
                        return True
                    
                    # 3. If both are VIOLATION, keep the initial record or refresh the timestamp
                    return True
                else:
                    # No entry today yet, create a fresh entry
                    cursor.execute('''
                        INSERT INTO attendance (student_id, roll_no, name, timestamp, status, tag_color_verified, violation_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (roll_no, roll_no, student['name'], local_time_str, status, tag_color, violation))
                    
                    # Real-time synchronization with weekly_attendance
                    self._sync_weekly_attendance(cursor, roll_no, local_time_str, status)
                    return True
        except Exception as e:
            print(f"[DBManager] Error logging attendance for {roll_no}: {e}")
            return False
        finally:
            conn.close()

    def _sync_weekly_attendance(self, cursor, roll_no, timestamp_str, status):
        """Automatically synchronize changes to the attendance table with weekly_attendance table."""
        try:
            # Parse the timestamp to find the date
            date_part = timestamp_str.split()[0]
            dt_date = datetime.strptime(date_part, "%Y-%m-%d").date()
            
            # Check if this is May 2026
            if dt_date.year != 2026 or dt_date.month != 5:
                return # Only tracking May 2026 in the weekly table

            # Find all Mondays of May 2026
            mondays = []
            for day in range(1, 32):
                try:
                    d = datetime(2026, 5, day).date()
                    if d.weekday() == 0:  # Monday
                        mondays.append(d)
                except ValueError:
                    break
            mondays.sort()
            
            # Determine which week name and day column this date corresponds to
            target_week_name = None
            target_col_name = None
            day_offset = None
            
            for offset, m in enumerate(mondays[:4]):
                diff = (dt_date - m).days
                if 0 <= diff < 7:
                    target_week_name = f"Week {offset + 1}"
                    day_offset = diff
                    break
                    
            if target_week_name is None or day_offset is None:
                return # Not within any of the 4 weeks of May 2026
                
            cols_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}
            target_col_name = cols_map.get(day_offset)
            
            # If the log status is PRESENT, mark 'P', else 'A'
            val = "P" if status == "PRESENT" else "A"
            
            # Check if weekly_attendance record exists for this student and week
            cursor.execute("""
                SELECT id FROM weekly_attendance 
                WHERE register_number = ? AND week_name = ?
            """, (roll_no, target_week_name))
            w_row = cursor.fetchone()
            
            if w_row:
                # Update this specific day
                cursor.execute(f"""
                    UPDATE weekly_attendance 
                    SET {target_col_name} = ? 
                    WHERE id = ?
                """, (val, w_row[0]))
                
                # Recalculate total_present and attendance_percentage
                cursor.execute("""
                    SELECT MON, TUE, WED, THU, FRI, SAT, SUN 
                    FROM weekly_attendance 
                    WHERE id = ?
                """, (w_row[0],))
                day_row = cursor.fetchone()
                if day_row:
                    present_count = sum(1 for d in day_row if str(d).upper() == "P")
                    att_pct = round(present_count / 7 * 100, 2)
                    cursor.execute("""
                        UPDATE weekly_attendance 
                        SET total_present = ?, attendance_percentage = ? 
                        WHERE id = ?
                    """, (present_count, att_pct, w_row[0]))
            else:
                # Create a new row if it doesn't exist
                days_vals = ["A"] * 7
                days_vals[day_offset] = val
                present_count = 1 if val == "P" else 0
                att_pct = round(present_count / 7 * 100, 2)
                cursor.execute("""
                    INSERT INTO weekly_attendance (
                        register_number, week_name, MON, TUE, WED, THU, FRI, SAT, SUN,
                        total_present, attendance_percentage
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (roll_no, target_week_name, *days_vals, present_count, att_pct))
                
        except Exception as e:
            print(f"[DBManager] Error syncing weekly attendance: {e}")

    def get_daily_report(self, date=None):
        """Fetch attendance report for a specific date (YYYY-MM-DD)."""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
            
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.name, s.roll_no, a.timestamp, a.status, a.violation_type
                FROM attendance a
                JOIN students s ON a.student_id = s.roll_no
                WHERE date(a.timestamp) = ?
                ORDER BY a.timestamp DESC
            ''', (date,))
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()
