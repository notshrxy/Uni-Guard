#Exports attendance logs from the SQLite database to formatted Excel sheets or CSV files.


import sqlite3
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

def export_attendance_to_excel(db_path='face_db/uniguard.db', output_xlsx='attendance_report.xlsx'):
    """
    Connects to the Uni-Guard SQLite database and exports attendance records 
    into a beautifully styled, professional Excel spreadsheet.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"[Error] SQLite database not found at: {db_path.resolve()}")
        print("Please run the Uni-Guard dashboard and log some scans first!")
        return

    print(f"[Export] Reading database: {db_path.resolve()}...")
    
    # 1. Connect and query SQLite Database
    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            a.timestamp AS [Scan Timestamp],
            s.roll_no AS [Roll No],
            s.name AS [Student Full Name],
            s.gender AS [Gender],
            s.batch_year AS [Year of Study],
            s.tag_color AS [Target Tag],
            a.tag_color_verified AS [Detected Tag],
            a.status AS [Compliance Status],
            COALESCE(a.violation_type, 'None') AS [Violation Details]
        FROM attendance a
        JOIN students s ON a.student_id = s.roll_no
        ORDER BY a.timestamp DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("[Warning] No attendance logs found in the database. Creating an empty styled sheet.")
        # Create dummy columns to preserve structure
        df = pd.DataFrame(columns=[
            'Scan Timestamp', 'Roll No', 'Student Full Name', 'Gender', 
            'Year of Study', 'Target Tag', 'Detected Tag', 'Compliance Status', 'Violation Details'
        ])

    # Convert Timestamp string to proper DateTime format if rows exist
    if not df.empty:
        df['Scan Timestamp'] = pd.to_datetime(df['Scan Timestamp'])
        # Capitalize status and gender for aesthetics
        df['Compliance Status'] = df['Compliance Status'].str.title()
        df['Gender'] = df['Gender'].str.title()
        df['Target Tag'] = df['Target Tag'].str.title()
        if 'Detected Tag' in df.columns and df['Detected Tag'] is not None:
            df['Detected Tag'] = df['Detected Tag'].str.title()

    # 2. Write and Style with openpyxl
    print(f"[Export] Generating styled Excel spreadsheet: {output_xlsx}...")
    
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Daily Attendance', index=False)
        
        # Access sheet for styling
        workbook = writer.book
        worksheet = writer.sheets['Daily Attendance']
        
        # Import openpyxl formatting elements
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter
        
        # Color Palettes
        NAVY_HEADER = "2C3E50"  # Deep slate navy
        WHITE = "FFFFFF"
        LIGHT_GRAY = "F8F9FA"   # Alternating row color
        SUCCESS_GREEN = "D4EDDA" # Light pastel green for permitted
        SUCCESS_TEXT = "155724"
        DANGER_RED = "F8D7DA"    # Light pastel red for violations
        DANGER_TEXT = "721C24"
        BORDER_GRAY = "E2E8F0"
        
        # Fonts & Alignments
        header_font = Font(name='Segoe UI', size=11, bold=True, color=WHITE)
        regular_font = Font(name='Segoe UI', size=11, bold=False)
        bold_font = Font(name='Segoe UI', size=11, bold=True)
        
        header_fill = PatternFill(start_color=NAVY_HEADER, end_color=NAVY_HEADER, fill_type="solid")
        alt_fill = PatternFill(start_color=LIGHT_GRAY, end_color=LIGHT_GRAY, fill_type="solid")
        success_fill = PatternFill(start_color=SUCCESS_GREEN, end_color=SUCCESS_GREEN, fill_type="solid")
        danger_fill = PatternFill(start_color=DANGER_RED, end_color=DANGER_RED, fill_type="solid")
        
        thin_border_side = Side(border_style="thin", color=BORDER_GRAY)
        thin_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
        
        center_align = Alignment(horizontal="center", vertical="center")
        left_align = Alignment(horizontal="left", vertical="center")
        
        # Style Header Row (Row 1)
        worksheet.row_dimensions[1].height = 28
        for col_idx in range(1, len(df.columns) + 1):
            cell = worksheet.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
            cell.border = thin_border
            
        # Style Data Rows
        for row_idx in range(2, worksheet.max_row + 1):
            worksheet.row_dimensions[row_idx].height = 22
            is_alt = (row_idx % 2 == 0)
            
            # Fetch status for row coloring
            status_cell = worksheet.cell(row=row_idx, column=df.columns.get_loc('Compliance Status') + 1)
            status_val = str(status_cell.value).lower()
            
            row_fill = alt_fill if is_alt else None
            
            # Highlight Permitted and Violations
            if "permitted" in status_val:
                status_cell.fill = success_fill
                status_cell.font = Font(name='Segoe UI', size=11, bold=True, color=SUCCESS_TEXT)
            elif "violation" in status_val:
                status_cell.fill = danger_fill
                status_cell.font = Font(name='Segoe UI', size=11, bold=True, color=DANGER_TEXT)
            
            for col_idx in range(1, len(df.columns) + 1):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                
                # Apply base font
                if cell != status_cell:
                    cell.font = regular_font
                    if row_fill:
                        cell.fill = row_fill
                        
                cell.border = thin_border
                
                # Alignments
                col_name = df.columns[col_idx - 1]
                if col_name in ['Scan Timestamp', 'Roll No', 'Gender', 'Year of Study', 'Target Tag', 'Detected Tag', 'Compliance Status']:
                    cell.alignment = center_align
                else:
                    cell.alignment = left_align
                    
        # Auto-fit Column Widths with some breathing room
        for col in worksheet.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.value is not None:
                    # Format DateTime string length properly
                    val_str = cell.value.strftime('%Y-%m-%d %H:%M:%S') if isinstance(cell.value, datetime) else str(cell.value)
                    max_len = max(max_len, len(val_str))
            worksheet.column_dimensions[col_letter].width = max(max_len + 4, 12)
            
        # Enable grid lines visibility explicitly
        worksheet.views.sheetView[0].showGridLines = True
        
    print(f"[Success] Attendance database successfully exported to: {Path(output_xlsx).resolve()}")

if __name__ == "__main__":
    # Ensure database folder exists
    export_attendance_to_excel()
