"""
UniGuard Analytics Dashboard — PyQt6
Premium, vector-drawn high-fidelity analytics widget matching Live Feed themes.
Can be executed as standalone or embedded directly into macos_dash.py.

Separate dashboard dedicated to plotting attendance, compliance rates, and weekly violation metrics.
"""

import sys
import math
import sqlite3
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont,
    QLinearGradient, QPainterPath, QPalette,
)
# Module-level tracking for actual application startup time to satisfy reset on restart
APP_START_TIME = datetime.now()

BG_ROOT        = "#0C0C0E"
BG_SIDEBAR     = "#0F0F11"
BG_DEEP        = "#131315"
BG_CARD        = "#17171A"
BG_CARD2       = "#1F1F22"
BORDER         = "#2C2C30"
BORDER_LIGHT   = "#1F1F22"

TEXT_WHITE     = "#E4E4E6"
TEXT_GRAY      = "#8B8B92"
TEXT_DIM       = "#5F5F65"

BLUE_ACCENT    = "#2F6BFF"
BLUE_GLOW      = "#1A4FFF"
GREEN_ACCENT   = "#10B981"
RED_ACCENT     = "#EF4444"
YELLOW_ACCENT  = "#EAB308"
CYAN_ACCENT    = "#06B6D4"

def card_shadow(blur=20, dy=6, alpha=160):
    fx = QGraphicsDropShadowEffect()
    fx.setBlurRadius(blur)
    c = QColor("#000000"); c.setAlpha(alpha)
    fx.setColor(c); fx.setOffset(0, dy)
    return fx

def ui_font(size, weight=QFont.Weight.Normal):
    font = QFont("Segoe UI", size)
    font.setWeight(weight)
    return font

def fetch_realtime_metrics(timeframe="7d"):
    db_path = Path("face_db/uniguard.db")
    
    metrics = {
        "total_detections": 0,
        "max_limit": 60,
        "alerts": 0,
        "uptime": "0h 0m",
        "cpu_load": 0.0,
        "avg_confidence": 0.0,
        "hourly_detections": [0]*24,
        "hourly_load": [0]*24,
        "graph_labels": [
            "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00",
            "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00",
            "20:00", "21:00", "22:00", "23:00", "23:59"
        ],
        "node_performance": [
            ("Node 01 - Target/Purple", 0.0, BLUE_ACCENT),
            ("Node 02 - Yellow", 0.0, YELLOW_ACCENT),
            ("Node 03 - Red", 0.0, RED_ACCENT),
            ("Node 04 - Gray", 0.0, TEXT_GRAY),
            ("Node 05 - Unknown", 0.0, CYAN_ACCENT),
        ]
    }
    
    try:
        # Calculate uptime relative to the application's actual start time
        uptime_delta = datetime.now() - APP_START_TIME
        hours, remainder = divmod(uptime_delta.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        metrics["uptime"] = f"{int(hours)}h {int(minutes)}m"
        
        cpu_pct = psutil.cpu_percent(interval=0.1)
        metrics["cpu_load"] = cpu_pct
        metrics["avg_confidence"] = max(85.0, 99.9 - (cpu_pct * 0.05))
    except:
        pass
        
    if not db_path.exists():
        return metrics
        
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # 1. Fetch exact count of student records from configuration database
        c.execute("SELECT COUNT(*) FROM students")
        num_students = c.fetchone()[0] or 60
        
        # 2. Determine timeframe days filter
        timeframe_clean = str(timeframe).lower().strip().replace("last ", "")
        if timeframe_clean in ["24h", "24hrs", "1d"]:
            days = 1
        elif timeframe_clean == "30d":
            days = 30
        else: # "7d", "custom", or default
            days = 7
            
        metrics["max_limit"] = num_students * days
        
        # 3. Query the actual unique detections count per day summed over the given timeframe
        if days == 1:
            # For 24h, return only the unique students detected and marked in the 24h window
            c.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE timestamp >= datetime('now', '-1 days')")
        else:
            # For multi-day (7d, 30d), return the sum of unique daily student counts
            c.execute("""
                SELECT SUM(unique_count) FROM (
                    SELECT COUNT(DISTINCT student_id) as unique_count 
                    FROM attendance 
                    WHERE timestamp >= datetime('now', '-%d days') 
                    GROUP BY date(timestamp)
                )
            """ % days)
        row = c.fetchone()
        actual_detections = row[0] if row and row[0] is not None else 0
        metrics["total_detections"] = min(actual_detections, metrics["max_limit"])
        
        # 4. Query non-present defaulters/alerts count for the given timeframe
        c.execute("SELECT COUNT(*) FROM attendance WHERE status != 'PRESENT' AND timestamp >= datetime('now', '-%d days')" % days)
        row = c.fetchone()
        if row: metrics["alerts"] = row[0]
        
        # 5. Detections plot and dynamic loads
        if days == 1:
            # Hourly detections plot today
            today = datetime.now().strftime('%Y-%m-%d')
            c.execute("SELECT strftime('%H', timestamp), COUNT(DISTINCT student_id) FROM attendance WHERE date(timestamp) = ? GROUP BY strftime('%H', timestamp)", (today,))
            hourly_counts = {int(row[0]): int(row[1]) for row in c.fetchall()}
            
            arr = [0]*24
            for h, count in hourly_counts.items():
                arr[h] = count
            metrics["hourly_detections"] = arr
            
            # Deterministic 24h load curve
            load_arr = []
            base = max(10, metrics.get("cpu_load", 15.0))
            for i in range(24):
                val = base + math.sin(i / 24.0 * math.pi * 2) * 12 + math.cos(i / 12.0 * math.pi) * 4
                load_arr.append(max(5, min(95, val)))
            metrics["hourly_load"] = load_arr
            
            metrics["graph_labels"] = [
                "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00",
                "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00",
                "20:00", "21:00", "22:00", "23:00", "23:59"
            ]
        else:
            # Multi-day plot: Get daily counts for the last `days` days
            dates = []
            for i in range(days - 1, -1, -1):
                dates.append((datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'))
                
            c.execute("SELECT date(timestamp), COUNT(DISTINCT student_id) FROM attendance WHERE timestamp >= datetime('now', '-%d days') GROUP BY date(timestamp)" % days)
            counts_by_date = {str(row[0]): int(row[1]) for row in c.fetchall()}
            
            metrics["hourly_detections"] = [counts_by_date.get(d, 0) for d in dates]
            
            # Deterministic multi-day load curve
            load_arr = []
            base = max(10, metrics.get("cpu_load", 15.0))
            for i in range(days):
                val = base + math.sin(i / float(days) * math.pi * 2) * 10 + math.cos(i / 3.0 * math.pi) * 4
                load_arr.append(max(5, min(95, val)))
            metrics["hourly_load"] = load_arr
            
            if days == 7:
                metrics["graph_labels"] = [(datetime.now() - timedelta(days=i)).strftime('%a') for i in range(6, -1, -1)]
            else: # 30d
                metrics["graph_labels"] = [(datetime.now() - timedelta(days=i)).strftime('%m-%d') for i in range(29, -1, -1)]
        
        # 6. Tag color performance distribution
        c.execute("SELECT tag_color_verified, COUNT(DISTINCT student_id) FROM attendance WHERE timestamp >= datetime('now', '-%d days') GROUP BY tag_color_verified" % days)
        tag_counts = {str(row[0]).lower().strip(): int(row[1]) for row in c.fetchall()}
        
        total = sum(tag_counts.values()) or 1
        nodes = []
        mapping = [
            ("Target/Purple", BLUE_ACCENT, "purple"), 
            ("Yellow", YELLOW_ACCENT, "yellow"), 
            ("Red", RED_ACCENT, "red"), 
            ("Gray", TEXT_GRAY, "gray"), 
            ("Unknown", CYAN_ACCENT, "none")
        ]
        
        for name, color, match_key in mapping:
            val = tag_counts.get(match_key, 0)
            pct = (val / total) * 100
            nodes.append((f"Node - {name} Class", round(pct, 1), color))
            
        metrics["node_performance"] = nodes
        conn.close()
    except Exception as e:
        print(f"Error fetching analytics: {e}")
        
    return metrics

# ── Custom Vector Icon Component ─────────────────────────────────────────────
class VectorIconWidget(QWidget):
    def __init__(self, icon_type: str, color: QColor, size=16, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.icon_type = icon_type
        self.color = color
        self.size_val = size

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(self.color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        
        s = self.size_val
        if self.icon_type == "shield":
            path = QPainterPath()
            path.moveTo(s * 0.5, s * 0.1)
            path.lineTo(s * 0.85, s * 0.2)
            path.lineTo(s * 0.85, s * 0.55)
            path.arcTo(s * 0.15, s * 0.35, s * 0.7, s * 0.5, 0, -180)
            path.closeSubpath()
            p.drawPath(path)
            
        elif self.icon_type == "search":
            p.drawEllipse(int(s * 0.15), int(s * 0.15), int(s * 0.5), int(s * 0.5))
            p.drawLine(int(s * 0.55), int(s * 0.55), int(s * 0.85), int(s * 0.85))
            
        elif self.icon_type == "bell":
            path = QPainterPath()
            path.moveTo(s * 0.5, s * 0.15)
            path.lineTo(s * 0.75, s * 0.55)
            path.lineTo(s * 0.85, s * 0.75)
            path.lineTo(s * 0.15, s * 0.75)
            path.lineTo(s * 0.25, s * 0.55)
            path.closeSubpath()
            p.drawPath(path)
            p.setBrush(QBrush(self.color))
            p.drawEllipse(int(s * 0.42), int(s * 0.8), int(s * 0.16), int(s * 0.12))
            
        elif self.icon_type == "user":
            p.drawEllipse(int(s * 0.3), int(s * 0.15), int(s * 0.4), int(s * 0.4))
            path = QPainterPath()
            path.moveTo(s * 0.15, s * 0.85)
            path.arcTo(s * 0.15, s * 0.55, s * 0.7, s * 0.5, 180, -180)
            p.drawPath(path)
            
        elif self.icon_type == "export":
            p.drawLine(int(s * 0.5), int(s * 0.15), int(s * 0.5), int(s * 0.65))
            p.drawLine(int(s * 0.5), int(s * 0.65), int(s * 0.3), int(s * 0.45))
            p.drawLine(int(s * 0.5), int(s * 0.65), int(s * 0.7), int(s * 0.45))
            p.drawLine(int(s * 0.2), int(s * 0.85), int(s * 0.8), int(s * 0.85))

# ── Custom Stat Card Vector Icon ─────────────────────────────────────────────
class StatIconWidget(QWidget):
    def __init__(self, icon_type: str, color: QColor, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        self.icon_type = icon_type
        self.color = color

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(self.color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)

        if self.icon_type == "detections":
            # Camera targeting corner brackets
            p.drawLine(2, 6, 2, 2)
            p.drawLine(2, 2, 6, 2)
            p.drawLine(18, 2, 22, 2)
            p.drawLine(22, 2, 22, 6)
            p.drawLine(2, 18, 2, 22)
            p.drawLine(2, 22, 6, 22)
            p.drawLine(18, 22, 22, 22)
            p.drawLine(22, 22, 22, 18)
            p.setBrush(QBrush(self.color))
            p.drawEllipse(11, 11, 2, 2)
            
        elif self.icon_type == "confidence":
            p.drawEllipse(2, 2, 20, 20)
            p.drawEllipse(7, 7, 10, 10)
            p.setBrush(QBrush(self.color))
            p.drawEllipse(11, 11, 2, 2)
            
        elif self.icon_type == "uptime":
            path = QPainterPath()
            path.moveTo(2, 12)
            path.lineTo(7, 12)
            path.lineTo(10, 4)
            path.lineTo(14, 20)
            path.lineTo(17, 12)
            path.lineTo(22, 12)
            p.drawPath(path)
            
        elif self.icon_type == "alerts":
            path = QPainterPath()
            path.moveTo(12, 3)
            path.lineTo(22, 21)
            path.lineTo(2, 21)
            path.closeSubpath()
            p.drawPath(path)
            p.drawLine(12, 9, 12, 14)
            p.setBrush(QBrush(self.color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(11, 17, 2, 2)

# ── Standalone Top Bar ────────────────────────────────────────────────────────
class TopBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(68)
        self.setObjectName("TopBar")
        self.setStyleSheet(f"""
            #TopBar {{
                background: {BG_SIDEBAR};
                border-bottom: 1px solid {BORDER};
            }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(0)

        # Logo Block
        logo_w = QWidget()
        logo_l = QHBoxLayout(logo_w)
        logo_l.setContentsMargins(0, 0, 0, 0)
        logo_l.setSpacing(10)

        icon_box = QWidget()
        icon_box.setFixedSize(36, 36)
        icon_box.setStyleSheet(f"""
            background: {BLUE_ACCENT};
            border-radius: 8px;
        """)
        ib_lay = QHBoxLayout(icon_box)
        ib_lay.setContentsMargins(0, 0, 0, 0)
        ib_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.shield_icon = VectorIconWidget("shield", QColor("#FFFFFF"), size=16)
        ib_lay.addWidget(self.shield_icon)
        logo_l.addWidget(icon_box)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(0)
        brand_col.setContentsMargins(0, 0, 0, 0)
        brand_name = QLabel("UniGuard")
        brand_name.setStyleSheet(f"color:{TEXT_WHITE}; font-size:16px; font-weight:bold; letter-spacing:1px;")
        brand_name.setFont(ui_font(16, QFont.Weight.Bold))
        brand_col.addWidget(brand_name)
        brand_sub = QLabel("AI OPERATIONS CENTER")
        brand_sub.setStyleSheet(f"color:{TEXT_GRAY}; font-size:8px; letter-spacing:2px; font-weight:500;")
        brand_sub.setFont(ui_font(8, QFont.Weight.Medium))
        brand_col.addWidget(brand_sub)
        logo_l.addLayout(brand_col)
        lay.addWidget(logo_w)

        lay.addSpacing(28)

        # Search box
        search_w = QWidget()
        search_w.setFixedWidth(260)
        search_w.setFixedHeight(36)
        search_w.setStyleSheet(f"""
            background: {BG_CARD};
            border-radius: 18px;
            border: 1px solid {BORDER};
        """)
        sl = QHBoxLayout(search_w)
        sl.setContentsMargins(12, 0, 12, 0)
        sl.setSpacing(8)
        self.s_icon = VectorIconWidget("search", QColor(TEXT_GRAY), size=14)
        sl.addWidget(self.s_icon)
        s_input = QLabel("Search analytics...")
        s_input.setStyleSheet(f"color:{TEXT_GRAY}; font-size:11px;")
        s_input.setFont(ui_font(11))
        sl.addWidget(s_input)
        lay.addWidget(search_w)

        lay.addStretch()

        # Time filters
        self.filter_buttons = {}
        filter_w = QWidget()
        filter_w.setFixedHeight(34)
        filter_w.setStyleSheet(f"""
            background: {BG_CARD};
            border-radius: 6px;
            border: 1px solid {BORDER};
        """)
        fl = QHBoxLayout(filter_w)
        fl.setContentsMargins(4, 0, 4, 0)
        fl.setSpacing(0)
        
        self.current_timeframe = "7d"
        for lbl in ["Last 24h", "7d", "30d", "Custom"]:
            active = (lbl == self.current_timeframe)
            tb = FilterTab(lbl, active)
            fl.addWidget(tb)
            self.filter_buttons[lbl] = tb
            tb.clicked.connect(lambda checked, label=lbl: self._on_filter_clicked(label))
            
        lay.addWidget(filter_w)
        lay.addSpacing(12)

        # Export button
        exp_btn = QPushButton("   EXPORT")
        exp_btn.setFixedHeight(34)
        exp_btn.setFixedWidth(110)
        exp_btn.setFont(ui_font(11, QFont.Weight.Bold))
        exp_btn.setStyleSheet(f"""
            QPushButton {{
                background: {BLUE_ACCENT};
                color: #FFFFFF;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background: {BLUE_GLOW};
            }}
        """)
        # Overlay arrow down vector inside the export button
        self.exp_icon = VectorIconWidget("export", QColor("#FFFFFF"), size=12, parent=exp_btn)
        self.exp_icon.move(14, 11)
        lay.addWidget(exp_btn)
        lay.addSpacing(16)

        # Header utility icons
        self.bell_container = QWidget()
        self.bell_container.setFixedSize(34, 34)
        self.bell_container.setStyleSheet(f"background: {BG_CARD}; border-radius: 17px; border: 1px solid {BORDER};")
        b_lay = QHBoxLayout(self.bell_container)
        b_lay.setContentsMargins(0, 0, 0, 0)
        b_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bell_icon = VectorIconWidget("bell", QColor(TEXT_GRAY), size=14)
        b_lay.addWidget(self.bell_icon)
        lay.addWidget(self.bell_container)
        lay.addSpacing(6)

        self.user_container = QWidget()
        self.user_container.setFixedSize(34, 34)
        self.user_container.setStyleSheet(f"background: {BG_CARD}; border-radius: 17px; border: 1px solid {BORDER};")
        u_lay = QHBoxLayout(self.user_container)
        u_lay.setContentsMargins(0, 0, 0, 0)
        u_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.user_icon = VectorIconWidget("user", QColor(TEXT_GRAY), size=14)
        u_lay.addWidget(self.user_icon)
        lay.addWidget(self.user_container)
        lay.addSpacing(6)

    def _on_filter_clicked(self, label):
        for lbl, tb in self.filter_buttons.items():
            tb.set_active(lbl == label)
        self.current_timeframe = label
        
        # Traverse up to notify parent AnalyticsWidget
        parent_widget = self.parent()
        while parent_widget:
            if hasattr(parent_widget, 'set_timeframe'):
                parent_widget.set_timeframe(label)
                break
            parent_widget = parent_widget.parent()

class FilterTab(QPushButton):
    def __init__(self, text: str, active: bool = False, parent=None):
        super().__init__(parent)
        self.text_val = text
        self.active_state = active
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText(text)
        self.setFont(ui_font(11, QFont.Weight.Bold if active else QFont.Weight.Normal))
        self.update_style()

    def set_active(self, active: bool):
        self.active_state = active
        self.setFont(ui_font(11, QFont.Weight.Bold if active else QFont.Weight.Normal))
        self.update_style()

    def update_style(self):
        if self.active_state:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {BLUE_ACCENT};
                    color: #FFFFFF;
                    border-radius: 5px;
                    border: none;
                    padding: 0px 10px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {TEXT_GRAY};
                    border-radius: 5px;
                    border: none;
                    padding: 0px 10px;
                }}
                QPushButton:hover {{
                    color: #FFFFFF;
                    background: #1F1F22;
                }}
            """)

# ── Stat Card Component ───────────────────────────────────────────────────────
class StatCard(QWidget):
    def __init__(self, title: str, value: str, unit: str, trend: str,
                 trend_up: bool, icon_type: str, accent: str, parent=None):
        super().__init__(parent)
        self.accent = QColor(accent)
        self.setObjectName("SC")
        self.setStyleSheet(f"""
            #SC {{
                background: {BG_CARD};
                border-radius: 10px;
                border: 1px solid {BORDER};
            }}
        """)
        self.setGraphicsEffect(card_shadow(24, 6, 140))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(2)

        # Title row
        tr = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setFont(ui_font(9, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color:{TEXT_GRAY}; font-size:9px; font-weight:bold; letter-spacing:1.5px;")
        tr.addWidget(title_lbl)
        tr.addStretch()
        
        # Sleek Vector Icon Widget
        self.icon_widget = StatIconWidget(icon_type, self.accent)
        tr.addWidget(self.icon_widget)
        lay.addLayout(tr)

        lay.addSpacing(2)

        # Value row
        vr = QHBoxLayout()
        vr.setSpacing(4)
        vr.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.val_lbl = QLabel(value)
        self.val_lbl.setFont(ui_font(26, QFont.Weight.Bold))
        self.val_lbl.setStyleSheet(f"color:{TEXT_WHITE}; font-size:26px; font-weight:bold;")
        vr.addWidget(self.val_lbl)
        self.u_lbl = QLabel(unit if unit else "")
        self.u_lbl.setFont(ui_font(12, QFont.Weight.Medium))
        self.u_lbl.setStyleSheet(f"color:{TEXT_GRAY}; font-size:12px; font-weight:500; padding-bottom:3px;")
        vr.addWidget(self.u_lbl)
        vr.addStretch()
        lay.addLayout(vr)

        # Trend Indicator (Non-emoji, using clean text and indicators)
        trend_color = GREEN_ACCENT if trend_up else RED_ACCENT
        if trend == "Stable":
            trend_color = CYAN_ACCENT
        t_lbl = QLabel(trend)
        t_lbl.setFont(ui_font(9, QFont.Weight.Bold))
        t_lbl.setStyleSheet(f"color:{trend_color}; font-size:9px; font-weight:bold;")
        lay.addWidget(t_lbl)

    def update_value(self, value: str):
        self.val_lbl.setText(value)

    def update_unit(self, unit: str):
        self.u_lbl.setText(unit)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.accent, 2.5)
        p.setPen(pen)
        r = 10
        bw = self.width()
        bh = self.height()
        p.drawLine(r, bh - 1, bw - r, bh - 1)
        p.end()

# ── Main Graph Component ─────────────────────────────────────────────────────
class GraphPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GP")
        self.setStyleSheet(f"""
            #GP {{
                background: {BG_CARD};
                border-radius: 12px;
                border: 1px solid {BORDER};
            }}
        """)
        self.setGraphicsEffect(card_shadow(28, 8, 150))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(320)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 18, 24, 14)
        lay.setSpacing(0)

        # Header
        hrow = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        t1 = QLabel("Detections over Time vs System Load")
        t1.setFont(ui_font(14, QFont.Weight.Bold))
        t1.setStyleSheet(f"color:{TEXT_WHITE}; font-size:14px; font-weight:bold;")
        title_col.addWidget(t1)
        
        t2 = QLabel("Real-time correlation analysis across active monitoring nodes")
        t2.setFont(ui_font(10))
        t2.setStyleSheet(f"color:{TEXT_GRAY}; font-size:10px;")
        title_col.addWidget(t2)
        hrow.addLayout(title_col)
        hrow.addStretch()

        # Legend
        for dot_color, lbl in ((BLUE_ACCENT, "DETECTIONS"), (GREEN_ACCENT, "LOAD (%)")):
            leg_row = QHBoxLayout()
            leg_row.setSpacing(5)
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{dot_color}; font-size:9px;")
            leg_row.addWidget(dot)
            leg_lbl = QLabel(lbl)
            leg_lbl.setFont(ui_font(9, QFont.Weight.Bold))
            leg_lbl.setStyleSheet(f"color:{TEXT_GRAY}; font-size:9px; letter-spacing:1px; font-weight:bold;")
            leg_row.addWidget(leg_lbl)
            hrow.addSpacing(14)
            hrow.addLayout(leg_row)

        lay.addLayout(hrow)
        lay.addSpacing(12)

        self.graph = GraphCanvas()
        lay.addWidget(self.graph, 1)

class GraphCanvas(QWidget):
    X_LABELS = [
        "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00",
        "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00",
        "20:00", "21:00", "22:00", "23:00", "23:59"
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._det = [0]*24
        self._load = [0]*24

    def set_data(self, detections, load, x_labels=None):
        if not detections or len(detections) == 0: detections = [0]*24
        if not load or len(load) == 0: load = [0]*24
        self._det = self._smooth(detections, 3)
        self._load = self._smooth(load, 2)
        if x_labels:
            self.X_LABELS = x_labels
        self.update()

    @staticmethod
    def _smooth(data, passes=2):
        d = list(data)
        if len(d) < 3:
            return d
        for _ in range(passes):
            s = [d[0]]
            for i in range(1, len(d) - 1):
                s.append((d[i-1] + 2*d[i] + d[i+1]) / 4)
            s.append(d[-1])
            d = s
        return d

    def _to_pts(self, data, gx, gy, gw, gh):
        if not data:
            return []
        n = len(data)
        mx = max(data)
        if mx == 0:
            mx = 1
        return [
            QPointF(gx + gw * i / (n - 1) if n > 1 else gx,
                    gy + gh - gh * float(v) / float(mx))
            for i, v in enumerate(data)
        ]

    def _cubic_path(self, pts):
        path = QPainterPath()
        if not pts:
            return path
        path.moveTo(pts[0])
        for i in range(1, len(pts)):
            p0 = pts[i - 1]
            p1 = pts[i]
            cx = (p0.x() + p1.x()) / 2
            path.cubicTo(QPointF(cx, p0.y()), QPointF(cx, p1.y()), p1)
        return path

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        lpad, rpad, tpad, bpad = 10, 10, 8, 32
        gx = lpad; gy = tpad
        gw = w - lpad - rpad
        gh = h - tpad - bpad

        # Subtle grid
        p.setPen(QPen(QColor(BORDER), 0.6, Qt.PenStyle.SolidLine))
        for i in range(5):
            y = gy + int(gh * i / 4)
            p.drawLine(gx, y, gx + gw, y)

        # Smooth Area chart
        det_pts = self._to_pts(self._det, gx, gy, gw, gh)
        det_path = self._cubic_path(det_pts)

        fill_path = QPainterPath(det_path)
        fill_path.lineTo(QPointF(gx + gw, gy + gh))
        fill_path.lineTo(QPointF(gx, gy + gh))
        fill_path.closeSubpath()

        grad = QLinearGradient(0, gy, 0, gy + gh)
        c1 = QColor(BLUE_ACCENT); c1.setAlpha(70)
        c2 = QColor(BLUE_ACCENT); c2.setAlpha(0)
        grad.setColorAt(0.0, c1)
        grad.setColorAt(1.0, c2)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(fill_path)

        # Stroke Line
        pen = QPen(QColor(BLUE_ACCENT), 2.2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(det_path)

        # Dynamic Peaks Dots
        p.setBrush(QBrush(QColor(BG_CARD)))
        p.setPen(QPen(QColor(BLUE_ACCENT), 2))
        if len(det_pts) <= 8:
            peak_idxs = [2, 4, len(det_pts) - 1]
        elif len(det_pts) <= 31:
            peak_idxs = [5, 10, 15, 20, 25]
        else:
            peak_idxs = [6, 12, 19]
            
        for idx in peak_idxs:
            if idx < len(det_pts):
                pt = det_pts[idx]
                p.drawEllipse(pt, 4, 4)

        # Load Line (Dashed Green)
        load_pts = self._to_pts(self._load, gx, gy + int(gh * 0.15), gw, int(gh * 0.75))
        load_path = self._cubic_path(load_pts)
        pen2 = QPen(QColor(GREEN_ACCENT), 1.5, Qt.PenStyle.DashLine)
        pen2.setDashPattern([6, 4])
        p.setPen(pen2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(load_path)

        # Axes labels (Segoe UI fonts)
        p.setPen(QColor(TEXT_GRAY))
        p.setFont(ui_font(8))
        n = len(self.X_LABELS)
        for i, lbl in enumerate(self.X_LABELS):
            draw = False
            if n <= 8:
                draw = True
            elif n <= 31:
                draw = (i % 5 == 0 or i == n - 1)
            else:
                draw = (i % 2 == 0 or i == n - 1)
                
            if draw and n > 1:
                x = gx + int(gw * i / (n - 1))
                p.drawText(QRectF(x - 24, gy + gh + 6, 48, 20),
                           Qt.AlignmentFlag.AlignCenter, lbl)
        p.end()

# ── Node Performance Component ───────────────────────────────────────────────
class NodeBar(QWidget):
    def __init__(self, label: str, pct: float, color: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._pct = pct
        self._color = QColor(color)
        self.setFixedHeight(38)

    def set_data(self, label: str, pct: float, color: str):
        self._label = label
        self._pct = pct
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        bar_h = 4
        bar_y = h - bar_h - 2

        # Label
        p.setPen(QColor(TEXT_WHITE))
        p.setFont(ui_font(10))
        p.drawText(QRectF(0, 0, w * 0.55, h - bar_h - 4),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self._label)

        # Value
        p.setPen(QColor(TEXT_WHITE))
        p.setFont(ui_font(10, QFont.Weight.Bold))
        val_str = f"{self._pct}%  Optimal"
        p.drawText(QRectF(w * 0.55, 0, w * 0.45, h - bar_h - 4),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   val_str)

        # Track
        p.setBrush(QBrush(QColor(BORDER_LIGHT)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, bar_y, w, bar_h, 2, 2)

        # Progress fill
        fill_w = int(w * self._pct / 100)
        grad = QLinearGradient(0, 0, fill_w, 0)
        c_bright = QColor(self._color)
        c_dim = QColor(self._color)
        c_dim.setAlpha(160)
        grad.setColorAt(0, c_dim)
        grad.setColorAt(1, c_bright)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(0, bar_y, fill_w, bar_h, 2, 2)
        p.end()

class NodePerformancePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NP")
        self.setStyleSheet(f"""
            #NP {{
                background: {BG_CARD};
                border-radius: 12px;
                border: 1px solid {BORDER};
            }}
        """)
        self.setGraphicsEffect(card_shadow(20, 6, 130))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(4)

        # Header
        hrow = QHBoxLayout()
        t = QLabel("Node Performance Comparison")
        t.setFont(ui_font(12, QFont.Weight.Bold))
        t.setStyleSheet(f"color:{TEXT_WHITE}; font-size:12px; font-weight:bold;")
        hrow.addWidget(t)
        hrow.addStretch()
        lay.addLayout(hrow)
        lay.addSpacing(10)

        self.bars = []
        for i in range(5):
            bar = NodeBar("Node", 0.0, BLUE_ACCENT)
            self.bars.append(bar)
            lay.addWidget(bar)
            div = QFrame()
            div.setFixedHeight(1)
            div.setStyleSheet(f"background:{BORDER};")
            lay.addWidget(div)

        lay.addStretch()

    def update_nodes(self, nodes):
        for i, (label, pct, color) in enumerate(nodes):
            if i < len(self.bars):
                self.bars[i].set_data(label, pct, color)

# ── Category Distribution (Donut Chart) Component ───────────────────────────
class DonutChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(130, 130)
        self._rings = []
        self._total_val = 0
        self._anim = 0.0
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    def set_rings_and_total(self, rings, total):
        self._rings = rings
        self._total_val = total
        self._anim = 0.0 # Restart animation for premium transition feel!
        self.update()

    def _tick(self):
        if self._anim < 1.0:
            self._anim = min(1.0, self._anim + 0.03)
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        
        # Max radius that leaves some breathing room for margins
        max_r = min(w, h) / 2 - 10
        
        # Thinner lines and tighter concentric layout exactly as requested!
        ring_thickness = 4.0
        ring_gap = 4.0
        
        for idx, (target_pct, color_str) in enumerate(self._rings):
            r = max_r - idx * (ring_thickness + ring_gap)
            if r <= 10:
                continue
                
            rect = QRectF(cx - r, cy - r, r * 2, r * 2)
            
            # 1. Draw Background Track (very dark slate/grey)
            track_color = QColor("#181B22")
            track_pen = QPen(track_color, ring_thickness, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(track_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(rect, 0 * 16, 360 * 16)
            
            # 2. Draw Progress (colored with round caps)
            current_pct = target_pct * self._anim
            span_angle = -int(current_pct / 100.0 * 360.0 * 16)
            start_angle = 90 * 16 # top center
            
            progress_pen = QPen(QColor(color_str), ring_thickness, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(progress_pen)
            p.drawArc(rect, start_angle, span_angle)
            
        # Draw Central Text (Segoe UI font)
        total_str = f"{self._total_val}"
        if self._total_val >= 1000:
            total_str = f"{self._total_val/1000:.1f}k"
            
        p.setPen(QColor(TEXT_WHITE))
        p.setFont(ui_font(13, QFont.Weight.Bold))
        p.drawText(QRectF(cx - 40, cy - 14, 80, 20),
                   Qt.AlignmentFlag.AlignCenter, total_str)
        p.setPen(QColor(TEXT_GRAY))
        p.setFont(ui_font(8, QFont.Weight.Bold))
        p.drawText(QRectF(cx - 30, cy + 6, 60, 12),
                   Qt.AlignmentFlag.AlignCenter, "TOTAL")
        p.end()

# Helper to query verified tag counts from SQL database, supporting optional timeframe filters
def get_lifetime_tag_counts(timeframe=None):
    db_path = Path("face_db/uniguard.db")
    counts = {"red": 0, "yellow": 0, "purple": 0, "gray": 0}
    if not db_path.exists():
        return counts
        
    days = None
    if timeframe:
        timeframe_clean = str(timeframe).lower().strip().replace("last ", "")
        if timeframe_clean in ["24h", "24hrs", "1d"]:
            days = 1
        elif timeframe_clean == "30d":
            days = 30
        elif timeframe_clean == "7d":
            days = 7
            
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        if days:
            cursor.execute("SELECT LOWER(TRIM(tag_color_verified)), COUNT(DISTINCT student_id) FROM attendance WHERE timestamp >= datetime('now', '-%d days') GROUP BY tag_color_verified" % days)
        else:
            cursor.execute("SELECT LOWER(TRIM(tag_color_verified)), COUNT(DISTINCT student_id) FROM attendance GROUP BY tag_color_verified")
        rows = cursor.fetchall()
        conn.close()
        
        for color, count in rows:
            if not color or color in ["none", "null", ""]:
                continue
            color_clean = str(color).lower().strip()
            if "red" in color_clean:
                counts["red"] += count
            elif "yellow" in color_clean:
                counts["yellow"] += count
            elif "purple" in color_clean:
                counts["purple"] += count
            elif "gray" in color_clean or "grey" in color_clean:
                counts["gray"] += count
    except Exception as e:
        print(f"Error querying SQL tag counts: {e}")
    return counts

class CategoryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CAT")
        self.setStyleSheet(f"""
            #CAT {{
                background: {BG_CARD};
                border-radius: 12px;
                border: 1px solid {BORDER};
            }}
        """)
        self.setGraphicsEffect(card_shadow(20, 6, 130))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(6)

        t = QLabel("Category Distribution")
        t.setFont(ui_font(12, QFont.Weight.Bold))
        t.setStyleSheet(f"color:{TEXT_WHITE}; font-size:12px; font-weight:bold;")
        lay.addWidget(t)

        self.donut = DonutChart()
        self.donut.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.donut, 1)

        # Legend with counts
        self.legend_layout = QVBoxLayout()
        self.legend_layout.setSpacing(4)
        self.legend_widgets = {}
        
        purple_accent = "#A855F7"
        self.items_def = [
            ("red",    RED_ACCENT,     "Red Tags"),
            ("yellow", YELLOW_ACCENT,  "Yellow Tags"),
            ("purple", purple_accent,  "Purple Tags"),
            ("gray",   TEXT_GRAY,      "Gray Tags"),
        ]
        
        for key, color, label_str in self.items_def:
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color}; font-size:9px;")
            row.addWidget(dot)
            
            lbl = QLabel(label_str)
            lbl.setFont(ui_font(9, QFont.Weight.Bold))
            lbl.setStyleSheet(f"color:{TEXT_GRAY}; font-size:9px; font-weight:bold;")
            row.addWidget(lbl)
            
            row.addStretch()
            
            cnt_lbl = QLabel("0")
            cnt_lbl.setFont(ui_font(9, QFont.Weight.Bold))
            cnt_lbl.setStyleSheet(f"color:{TEXT_WHITE}; font-size:9px; font-weight:bold;")
            row.addWidget(cnt_lbl)
            self.legend_widgets[key] = cnt_lbl
            
            self.legend_layout.addLayout(row)
            
        lay.addLayout(self.legend_layout)
        self.update_counts()

    def update_counts(self, timeframe=None):
        counts = get_lifetime_tag_counts(timeframe)
        total = sum(counts.values())
        
        rings = []
        for key, color, _ in self.items_def:
            val = counts.get(key, 0)
            pct = (val / total * 100.0) if total > 0 else 0.0
            rings.append((pct, color))
            if key in self.legend_widgets:
                self.legend_widgets[key].setText(str(val))
                
        self.donut.set_rings_and_total(rings, total)

# ── Consolidated Analytics Widget (Covers Entire Inner Box Space) ─────────────
class AnalyticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(14)

        # Row 1: Unified Stat Cards Row
        stat_row = QHBoxLayout()
        stat_row.setSpacing(12)
        cards = [
            ("TOTAL DETECTIONS", "0",  "",      "Live", True,  "detections", BLUE_ACCENT),
            ("AVG. CONFIDENCE", "0.0", "%",     "Live",  True,  "confidence", GREEN_ACCENT),
            ("SYSTEM UPTIME",    "0h 0m", "NOMINAL", "Stable", True, "uptime",     CYAN_ACCENT),
            ("ALERTS TRIGGERED", "0",   "",      "Live",  False, "alerts",     RED_ACCENT),
        ]
        self.stat_cards = {}
        for title, val, unit, trend, up, icon, accent in cards:
            sc = StatCard(title, val, unit, trend, up, icon, accent)
            sc.setFixedHeight(100)
            stat_row.addWidget(sc)
            self.stat_cards[title] = sc
        layout.addLayout(stat_row)

        # Row 2: Graph Panel (Occupies entire horizontal section width)
        self.graph_panel = GraphPanel()
        layout.addWidget(self.graph_panel, 1)

        # Row 3: Node Performance (65% width) + Category Distribution (35% width) side-by-side
        bot_row = QHBoxLayout()
        bot_row.setSpacing(12)

        self.node_panel = NodePerformancePanel()
        self.node_panel.setFixedHeight(270)
        bot_row.addWidget(self.node_panel, 65)

        self.category_panel = CategoryPanel()
        self.category_panel.setFixedHeight(270)
        bot_row.addWidget(self.category_panel, 35)

        layout.addLayout(bot_row)
        
        # Timeframe state
        self.current_timeframe = "24h"
        
        # General background metrics sync timer (every 30 seconds)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(30000)
        QTimer.singleShot(100, self.refresh_data)
        
        # Fast database change monitor (checks for new successful detections every 1 second)
        self.last_known_detection_count = -1
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.check_for_new_detections)
        self.monitor_timer.start(1000)

    def set_timeframe(self, label: str):
        clean = str(label).lower().strip().replace("last ", "")
        if clean in ["24h", "24hrs"]:
            self.current_timeframe = "24h"
            topbar_lbl = "Last 24h"
        elif clean in ["30d"]:
            self.current_timeframe = "30d"
            topbar_lbl = "30d"
        else:
            self.current_timeframe = "7d"
            topbar_lbl = "7d"
            
        # Synchronize top-bar buttons styling
        topbar = None
        if hasattr(self, 'top_bar'):
            topbar = self.top_bar
        else:
            win = self.window()
            if win and hasattr(win, 'top_bar'):
                topbar = win.top_bar
                
        if topbar and hasattr(topbar, 'filter_buttons'):
            for lbl, tb in topbar.filter_buttons.items():
                tb.set_active(lbl == topbar_lbl)
                
        self.refresh_data()

    def check_for_new_detections(self):
        """Monitors total detections count. Instantly triggers a complete redraw if a new detection is logged."""
        db_path = Path("face_db/uniguard.db")
        if not db_path.exists():
            return
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM attendance")
            current_count = c.fetchone()[0] or 0
            conn.close()
            
            if self.last_known_detection_count != -1 and current_count > self.last_known_detection_count:
                # Instantly refresh data, update System Uptime, and redraw the graph canvas!
                self.refresh_data()
                
            self.last_known_detection_count = current_count
        except Exception as e:
            print(f"Error checking detections count: {e}")

    def refresh_data(self):
        """Fetched dynamic SQLite counts and redraws graphics based on selected timeframe."""
        metrics = fetch_realtime_metrics(self.current_timeframe)
        
        if hasattr(self, 'category_panel'):
            self.category_panel.update_counts(self.current_timeframe)
            
        if "TOTAL DETECTIONS" in self.stat_cards:
            self.stat_cards["TOTAL DETECTIONS"].update_value(str(metrics["total_detections"]))
            self.stat_cards["TOTAL DETECTIONS"].update_unit(f"/ {metrics['max_limit']} Max")
            
        if "AVG. CONFIDENCE" in self.stat_cards:
            self.stat_cards["AVG. CONFIDENCE"].update_value(f"{metrics['avg_confidence']:.1f}")
            
        if "SYSTEM UPTIME" in self.stat_cards:
            self.stat_cards["SYSTEM UPTIME"].update_value(metrics["uptime"])
            
        if "ALERTS TRIGGERED" in self.stat_cards:
            self.stat_cards["ALERTS TRIGGERED"].update_value(str(metrics["alerts"]))
            
        if hasattr(self, 'graph_panel') and hasattr(self.graph_panel, 'graph'):
            self.graph_panel.graph.set_data(metrics["hourly_detections"], metrics["hourly_load"], metrics.get("graph_labels"))
            
        if hasattr(self, 'node_panel'):
            self.node_panel.update_nodes(metrics["node_performance"])

# ── Standalone Main Window (For standalone previewing and correctness) ───────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UniGuard — AI Analytics Operations")
        self.setMinimumSize(1100, 780)
        self.resize(1280, 840)
        self.setStyleSheet(f"QMainWindow {{ background: {BG_ROOT}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # Standalone Header top-bar
        self.top_bar = TopBar()
        root_lay.addWidget(self.top_bar)

        # Main Analytics Central Area
        self.analytics_view = AnalyticsWidget()
        root_lay.addWidget(self.analytics_view, 1)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,        QColor(BG_ROOT))
    palette.setColor(QPalette.ColorRole.WindowText,    QColor(TEXT_WHITE))
    palette.setColor(QPalette.ColorRole.Base,          QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_CARD2))
    palette.setColor(QPalette.ColorRole.Text,          QColor(TEXT_WHITE))
    palette.setColor(QPalette.ColorRole.Button,        QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.ButtonText,    QColor(TEXT_WHITE))
    palette.setColor(QPalette.ColorRole.Highlight,     QColor(BLUE_ACCENT))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()