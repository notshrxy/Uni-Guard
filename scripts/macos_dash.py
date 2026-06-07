"""
UniGuard Security Dashboard
A production-grade PyQt6 desktop application replicating the UniGuard UI.
Integrated with AI Vision Pipelines, pyqtgraph, and psutil.

Dashboard variant customized with premium styling and layouts tailored for macOS window configurations.
"""

import sys
import math
import random
import os

# Suppress harmless Qt screen/monitor driver warnings on Windows
os.environ["QT_LOGGING_RULES"] = "qt.qpa.screen=false"

from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np
import psutil
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QScrollArea, QFrame,
    QProgressBar, QSizePolicy, QSpacerItem, QGraphicsDropShadowEffect,
    QStackedWidget, QComboBox, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QMessageBox, QLineEdit
)
from PyQt6.QtCore import (
    Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve,
    QRect, pyqtSignal, QThread, QPoint
)
from PyQt6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QFontDatabase,
    QPixmap, QPainterPath, QLinearGradient, QRadialGradient,
    QIcon, QPalette, QPolygonF, QImage
)
from PyQt6.QtCore import QPointF
from pathlib import Path

# Add project root to path so we can import model_inference
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import YOLO Pipeline
from model_inference.pipeliner import SmartIDPipeline
from scripts.analytics_dash import AnalyticsWidget

# ─────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────
class Theme:
    BG_DEEP       = "#131315"
    BG_PANEL      = "#1F1F1F"
    BG_CARD       = "#131315"
    BG_SIDEBAR    = "#181818"
    BG_HOVER      = "#3A3A3A"
    BG_ACTIVE     = "#1A4FBE"
    BG_DOCK       = "#28282a"

    BORDER        = "#252528"
    BORDER_LIGHT  = "#2C2C30"

    TEXT_PRIMARY   = "#E8EDF5"
    TEXT_SECONDARY = "#6B7A99"
    TEXT_MUTED     = "#3D4A63"
    TEXT_ACCENT    = "#4A8EFF"

    GREEN         = "#22C55E"
    YELLOW        = "#F59E0B"
    RED           = "#EF4444"
    BLUE          = "#3B82F6"
    ORANGE        = "#F97316"
    PURPLE        = "#8B5CF6"

    TAG_CRITICAL  = ("#EF4444", "#3D1515")
    TAG_WATCH     = ("#FACC15", "#3D330A")
    TAG_VIP       = ("#8B5CF6", "#1E1535")
    TAG_IGNORE    = ("#9CA3AF", "#2E2E2E")

    CPU_COLOR     = "#3B82F6"
    MEM_COLOR     = "#22C55E"
    GPU_COLOR     = "#F59E0B"

    GRAPH_A       = "#4A8EFF"
    GRAPH_B       = "#6B7A99"
    GRAPH_GRID    = "#1A2035"

    RADIUS        = 12
    RADIUS_SM     = 8
    RADIUS_LG     = 16

# ─────────────────────────────────────────────
#  FONT HELPERS
# ─────────────────────────────────────────────
def mono_font(size=11, weight=QFont.Weight.Normal):
    f = QFont("JetBrains Mono")
    if not QFontDatabase.families().__contains__("JetBrains Mono"):
        f = QFont("Courier New")
    f.setPointSize(size)
    f.setWeight(weight)
    return f

def ui_font(size=11, weight=QFont.Weight.Normal):
    for name in ["Geist", "Geist Variable", "SF Pro Display", "Segoe UI", "Arial"]:
        f = QFont(name)
        f.setPointSize(size)
        f.setWeight(weight)
        return f

def topic_font(size=11):
    f = QFont("Lato")
    if not QFontDatabase.families().__contains__("Lato"):
        f = QFont("Arial")
    f.setPointSize(size)
    f.setWeight(QFont.Weight.Bold)
    return f

def label(text, size=11, color=Theme.TEXT_PRIMARY, weight=QFont.Weight.Normal, mono=False, topic=False):
    lbl = QLabel(text)
    if topic:
        lbl.setFont(topic_font(size))
    elif mono:
        lbl.setFont(mono_font(size, weight))
    else:
        lbl.setFont(ui_font(size, weight))
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    return lbl

# ─────────────────────────────────────────────
#  AI WORKER THREAD
# ─────────────────────────────────────────────
class VideoWorker(QThread):
    frame_ready = pyqtSignal(QPixmap)
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_path, source="0"):
        super().__init__()
        self.model_path = model_path
        self.source = int(source) if source.isdigit() else source
        self.running = True
        self.target_tag = "purple"
        self.pipeline = None
        self.paused = False
        
    def set_paused(self, paused):
        self.paused = paused
        
    def run(self):
        try:
            self.pipeline = SmartIDPipeline(model_path=self.model_path)
            cap = cv2.VideoCapture(self.source)
            if isinstance(self.source, int):
                # Set HD 720p capture resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception as e:
            print(f"[VideoWorker] Failed to initialize: {e}")
            self.error_occurred.emit(f"AI Pipeline Error:\n{str(e)}")
            return
            
        while self.running and cap.isOpened():
            if self.paused:
                self.msleep(100)
                continue
                
            ret, frame = cap.read()
            if not ret:
                # Loop video if it's a file
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                
            results = self.pipeline.process_frame(frame, target_tag=self.target_tag)
            
            # Convert annotated frame to QPixmap
            annotated = results['annotated_frame']
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            self.frame_ready.emit(pixmap)
            self.data_ready.emit(results)
            
            # Cap FPS to ~30 to avoid UI freezing
            self.msleep(30)
            
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class CardOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.hide()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        radius = getattr(self.parent(), '_radius', 12)
        path.addRoundedRect(0, 0, self.width(), self.height(), radius, radius)
        p.fillPath(path, QColor(10, 10, 10, 160))
        
    def mousePressEvent(self, event):
        event.accept()
    def mouseReleaseEvent(self, event):
        event.accept()
    def wheelEvent(self, event):
        event.accept()

# ─────────────────────────────────────────────
#  CARD BASE WIDGET
# ─────────────────────────────────────────────
class Card(QFrame):
    def __init__(self, parent=None, radius=Theme.RADIUS):
        super().__init__(parent)
        self._radius = radius
        self.setStyleSheet(f"Card {{ background: {Theme.BG_CARD}; border: 1px solid {Theme.BORDER}; border-radius: {radius}px; }}")
        self.disabled_overlay = CardOverlay(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.disabled_overlay.resize(self.size())

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        if enabled:
            self.disabled_overlay.hide()
        else:
            self.disabled_overlay.show()
            self.disabled_overlay.raise_()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self._radius, self._radius)
        painter.fillPath(path, QColor(Theme.BG_CARD))
        pen = QPen(QColor(Theme.BORDER))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawPath(path)

# ─────────────────────────────────────────────
#  ANIMATED PROGRESS BAR
# ─────────────────────────────────────────────
class MetricBar(QWidget):
    def __init__(self, label_text, color, value=0, parent=None):
        super().__init__(parent)
        self._color = color
        self._value = value
        self._label_text = label_text
        self.setFixedHeight(24)

    def set_value(self, v):
        self._value = max(0, min(100, v))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        lbl_w = 36
        val_w = 38
        
        # Metric Label
        p.setPen(QColor(Theme.TEXT_SECONDARY))
        p.setFont(mono_font(9, QFont.Weight.Bold))
        p.drawText(0, 0, w, 14, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, self._label_text)
        
        # Value Label
        p.setPen(QColor(Theme.TEXT_PRIMARY))
        p.setFont(mono_font(9, QFont.Weight.Bold))
        p.drawText(0, 0, w, 14, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, f"{int(self._value)}%")

        bar_x = 0
        bar_w = w
        bar_h = 4
        bar_y = 16

        track = QPainterPath()
        track.addRoundedRect(bar_x, bar_y, bar_w, bar_h, 2, 2)
        p.fillPath(track, QColor(Theme.BORDER_LIGHT))

        fill_w = int(bar_w * self._value / 100)
        if fill_w > 0:
            fill = QPainterPath()
            fill.addRoundedRect(bar_x, bar_y, fill_w, bar_h, 2, 2)
            grad = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
            c = QColor(self._color)
            grad.setColorAt(0, c.lighter(110))
            grad.setColorAt(1, c)
            p.fillPath(fill, grad)

# ─────────────────────────────────────────────
#  CAMERA FEED PANEL
# ─────────────────────────────────────────────
class CameraFeedWidget(QWidget):
    def __init__(self, parent=None, lbl="LIVE", color=Theme.BLUE):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        self._lbl = label(lbl, 10, QColor(color).lighter(150).name(), QFont.Weight.Bold, mono=True)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = QLabel("Initializing Video Feed...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 14px; background: #0D0F14;")
        self.layout.addWidget(self.image_label)

    def update_frame(self, pixmap):
        scaled = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)

# ─────────────────────────────────────────────
#  PLAY / PAUSE BUTTON
# ─────────────────────────────────────────────
class PlayPauseButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(False) # False = playing, True = paused
        self.setFixedSize(28, 28)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 14px;
            }}
            QPushButton:hover {{
                background: {Theme.BG_HOVER};
            }}
        """)
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        # Determine color (bright on hover)
        color = QColor(Theme.TEXT_PRIMARY if self.underMouse() else Theme.TEXT_SECONDARY)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        
        cx, cy = w // 2, h // 2
        
        if self.isChecked():
            # Paused -> Show Play Icon (sideways triangle)
            poly = QPolygonF([
                QPointF(cx - 4, cy - 6),
                QPointF(cx - 4, cy + 6),
                QPointF(cx + 6, cy)
            ])
            p.drawPolygon(poly)
        else:
            # Playing -> Show Pause Icon (two vertical rounded bars)
            p.drawRoundedRect(cx - 5, cy - 6, 3, 12, 1, 1)
            p.drawRoundedRect(cx + 2, cy - 6, 3, 12, 1, 1)

# ─────────────────────────────────────────────
#  TOP CAMERA HEADER
# ─────────────────────────────────────────────
class CameraHeaderWidget(QWidget):
    source_changed = pyqtSignal(str)
    play_pause_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(38)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)

        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background: {Theme.RED}; border-radius: 5px;")
        layout.addWidget(dot)

        cam_lbl = label("Live Video Feed", 10, Theme.TEXT_PRIMARY, topic=True)
        layout.addWidget(cam_lbl)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Live Webcam (0)", "Recorded Video File..."])
        self.source_combo.setStyleSheet(f"background: {Theme.BG_CARD}; color: {Theme.TEXT_PRIMARY}; border: 1px solid {Theme.BORDER}; padding: 2px 8px; border-radius: 4px;")
        self.source_combo.currentIndexChanged.connect(self._on_combo)
        layout.addWidget(self.source_combo)
        
        layout.addStretch()
        
        self.play_pause_btn = PlayPauseButton()
        self.play_pause_btn.toggled.connect(self.play_pause_toggled.emit)
        layout.addWidget(self.play_pause_btn)
        
        self.setStyleSheet(f"background: {Theme.BG_DEEP}; border-bottom: 1px solid {Theme.BORDER};")

    def _on_combo(self, idx):
        if idx == 0:
            self.source_changed.emit("0")
        elif idx == 1:
            file, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
            if file:
                self.source_changed.emit(file)
            else:
                self.source_combo.setCurrentIndex(0)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
class NavItem(QPushButton):
    def __init__(self, name, active=False, parent=None):
        super().__init__(parent)
        self._name = name
        self._active = active
        self.setFixedHeight(42)
        self.setCheckable(True)
        self.setChecked(active)
        self._update_style()
        self.toggled.connect(self._on_toggle)
    def text(self):
        return self._name

    def update_alert_state(self, show_alert):
        self._show_alert = show_alert
        self.update()

    def _on_toggle(self, checked):
        self._active = checked
        self._update_style()

    def _update_style(self):
        bg = Theme.BG_ACTIVE if self._active else "transparent"
        self.setStyleSheet(f"QPushButton {{ background: {bg}; border: none; border-radius: {Theme.RADIUS_SM}px; }} QPushButton:hover {{ background: {'#2455C0' if self._active else Theme.BG_HOVER}; }}")
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        base_color = Theme.TEXT_PRIMARY if self._active else Theme.TEXT_SECONDARY
        color = QColor(base_color)
        if not self.isEnabled():
            color.setAlpha(60) # beautifully grayed out (24% opacity)
        
        # Draw Icon
        p.setPen(QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        ix, iy = 20, h // 2
        name = self._name
        
        if name == "Attendance":
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(ix - 7, iy - 6, 5, 5)
            p.drawRect(ix, iy - 6, 7, 5)
            p.drawRect(ix - 7, iy + 1, 5, 6)
            p.drawRect(ix, iy + 1, 7, 6)
        elif name == "Live Feed":
            p.drawRoundedRect(ix - 7, iy - 5, 10, 10, 1, 1)
            poly = QPolygonF([QPointF(ix + 3, iy - 3), QPointF(ix + 7, iy - 5), QPointF(ix + 7, iy + 5), QPointF(ix + 3, iy + 3)])
            p.drawPolygon(poly)
        elif name == "Analytics":
            p.drawRect(ix - 7, iy + 1, 3, 5)
            p.drawRect(ix - 2, iy - 2, 3, 8)
            p.drawRect(ix + 3, iy - 5, 3, 11)
            p.drawPolyline(QPolygonF([QPointF(ix - 5, iy + 3), QPointF(ix - 1, iy - 3), QPointF(ix + 4, iy - 7), QPointF(ix + 6, iy - 7)]))
        elif name == "Database":
            p.drawRoundedRect(ix - 6, iy - 7, 12, 14, 1, 1)
            p.drawLine(ix - 3, iy - 7, ix + 3, iy - 7)
            p.drawLine(ix - 3, iy - 3, ix + 3, iy - 3)
            p.drawLine(ix - 3, iy, ix + 3, iy)
            p.drawLine(ix - 3, iy + 3, ix + 3, iy + 3)
        elif name == "Configuration":
            p.drawLine(ix - 5, iy - 6, ix - 5, iy + 6)
            p.drawLine(ix, iy - 6, ix, iy + 6)
            p.drawLine(ix + 5, iy - 6, ix + 5, iy + 6)
            bg_color = QColor(Theme.BG_ACTIVE if self._active else Theme.BG_SIDEBAR)
            p.setBrush(bg_color)
            p.drawRoundedRect(ix - 7, iy - 2, 4, 4, 1, 1)
            p.drawRoundedRect(ix - 2, iy + 2, 4, 4, 1, 1)
            p.drawRoundedRect(ix + 3, iy - 4, 4, 4, 1, 1)
        elif name == "Tag Detection":
            path = QPainterPath()
            path.moveTo(ix, iy - 7)
            path.cubicTo(ix + 5, iy - 2, ix + 6, iy + 2, ix + 5, iy + 5)
            path.cubicTo(ix + 3, iy + 8, ix - 3, iy + 8, ix - 5, iy + 5)
            path.cubicTo(ix - 6, iy + 2, ix - 5, iy - 2, ix, iy - 7)
            p.drawPath(path)

        # Draw Text (Body text greyed out if inactive)
        p.setPen(color)
        p.setFont(ui_font(11, QFont.Weight.DemiBold if self._active else QFont.Weight.Normal))
        p.drawText(44, 0, w - 44, h, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)

        # Draw red alert exclamation badge if enabled
        if getattr(self, '_show_alert', False) and self._name != "Configuration":
            p.save()
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            badge_r = 7
            bx = w - 24
            by = h // 2
            
            # Glow
            glow = QColor("#EF4444")
            glow.setAlpha(60)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(glow)
            p.drawEllipse(QPointF(bx, by), badge_r + 2, badge_r + 2)
            
            # Red circle
            p.setBrush(QColor("#EF4444"))
            p.drawEllipse(QPointF(bx, by), badge_r, badge_r)
            
            # Text '!'
            p.setPen(QColor("#FFFFFF"))
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            p.drawText(QRect(bx - badge_r, by - badge_r, badge_r * 2, badge_r * 2), Qt.AlignmentFlag.AlignCenter, "!")
            p.restore()

class ArrowButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("QPushButton { background: transparent; border: none; }")
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Sleek vector arrow (->) pointing right
        color = QColor(Theme.TEXT_ACCENT if self.underMouse() else Theme.TEXT_SECONDARY)
        p.setPen(QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        
        p.translate(12, 12)
        
        # Draw horizontal stem and arrowhead pointing right
        p.drawLine(-6, 0, 6, 0) # stem
        p.drawLine(6, 0, 1, -4) # upper arrowhead
        p.drawLine(6, 0, 1, 4)  # lower arrowhead

class SwitchUserDialog(QDialog):
    def __init__(self, can_close=True, parent=None):
        super().__init__(parent)
        self.role = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.card = QFrame(self)
        self.card.setObjectName("LoginCard")
        self.card.setFixedSize(180, 148)
        self.card.setStyleSheet(f"""
            QFrame#LoginCard {{
                background-color: {Theme.BG_CARD};
                border: 2px solid {Theme.BORDER_LIGHT};
                border-radius: 12px;
            }}
        """)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 4)
        self.card.setGraphicsEffect(shadow)
        
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(5)
        
        header_layout = QHBoxLayout()
        title_lbl = QLabel("Login")
        title_lbl.setStyleSheet(f"color: #FFFFFF; font-family: 'Segoe UI', Arial; font-size: 13px; font-weight: bold;")
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(16, 16)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                border: none;
                background: transparent;
                color: {Theme.TEXT_SECONDARY};
                font-family: 'Segoe UI', Arial;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: #FF5F56;
            }}
        """)
        close_btn.clicked.connect(self.reject)
        if not can_close:
            close_btn.hide()
        
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(close_btn)
        card_layout.addLayout(header_layout)
        
        self.error_lbl = QLabel("")
        self.error_lbl.setStyleSheet(f"color: {Theme.RED}; font-family: 'Segoe UI', Arial; font-size: 9px; font-weight: bold;")
        self.error_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_lbl.setWordWrap(True)
        self.error_lbl.setVisible(False) # Hide initially to eliminate empty layout space
        card_layout.addWidget(self.error_lbl)
        
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Username")
        self.user_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #1F1F21;
                border: 1px solid {Theme.BORDER_LIGHT};
                border-radius: 6px;
                color: #FFFFFF;
                padding: 4px 8px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 1px solid {Theme.BG_ACTIVE};
            }}
        """)
        card_layout.addWidget(self.user_input)
        
        self.pass_input = QLineEdit()
        self.pass_input.setPlaceholderText("Password")
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.pass_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #1F1F21;
                border: 1px solid {Theme.BORDER_LIGHT};
                border-radius: 6px;
                color: #FFFFFF;
                padding: 4px 8px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 1px solid {Theme.BG_ACTIVE};
            }}
        """)
        card_layout.addWidget(self.pass_input)
        
        card_layout.addSpacing(2)
        
        submit_btn = QPushButton("Confirm")
        submit_btn.setFixedHeight(26)
        submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        submit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_ACTIVE};
                border: none;
                color: #FFFFFF;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2455C0;
            }}
        """)
        submit_btn.clicked.connect(self.handle_submit)
        card_layout.addWidget(submit_btn)
        
    def handle_submit(self):
        username = self.user_input.text().strip()
        password = self.pass_input.text().strip()
        
        admin_pass = "Admin@SIST"
        if self.parent() and hasattr(self.parent(), 'admin_password'):
            admin_pass = self.parent().admin_password
            
        if username == "Admin" and password == admin_pass:
            self.role = "Admin"
            self.accept()
        elif username == "Student" and password == "Student@SIST":
            self.role = "Student"
            self.accept()
        else:
            self.error_lbl.setText("Invalid Username or Password")
            self.error_lbl.setVisible(True)
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 12, 210))
        
    def showEvent(self, event):
        super().showEvent(event)
        self._reposition_card()
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_card()

    def _reposition_card(self):
        if self.parent():
            self.setGeometry(self.parent().rect())
            if hasattr(self.parent(), 'sidebar') and self.parent().sidebar:
                sidebar = self.parent().sidebar
                sidebar_pos = sidebar.mapTo(self.parent(), QPoint(0, 0))
                # Sidebar width is 224px, card width is 180px. Offset = (224 - 180) // 2 = 22
                x = sidebar_pos.x() + 22
                # Place directly above footer components inside sidebar
                y = sidebar_pos.y() + sidebar.height() - 148 - 100
                self.card.move(int(x), int(y))
            else:
                h = self.parent().height()
                self.card.move(22, h - 148 - 100)

class ModifyPasswordDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.card = QFrame(self)
        self.card.setObjectName("ModifyCard")
        self.card.setFixedSize(180, 148)
        self.card.setStyleSheet(f"""
            QFrame#ModifyCard {{
                background-color: {Theme.BG_CARD};
                border: 2px solid {Theme.BORDER_LIGHT};
                border-radius: 12px;
            }}
        """)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 4)
        self.card.setGraphicsEffect(shadow)
        
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(5)
        
        header_layout = QHBoxLayout()
        title_lbl = QLabel("Modify Pass")
        title_lbl.setStyleSheet(f"color: #FFFFFF; font-family: 'Segoe UI', Arial; font-size: 12px; font-weight: bold;")
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(16, 16)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                border: none;
                background: transparent;
                color: {Theme.TEXT_SECONDARY};
                font-family: 'Segoe UI', Arial;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: #FF5F56;
            }}
        """)
        close_btn.clicked.connect(self.reject)
        
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(close_btn)
        card_layout.addLayout(header_layout)
        
        self.error_lbl = QLabel("")
        self.error_lbl.setStyleSheet(f"color: {Theme.RED}; font-family: 'Segoe UI', Arial; font-size: 9px; font-weight: bold;")
        self.error_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_lbl.setWordWrap(True)
        self.error_lbl.setVisible(False) # Hide initially to eliminate empty layout space
        card_layout.addWidget(self.error_lbl)
        
        self.curr_input = QLineEdit()
        self.curr_input.setPlaceholderText("Current Password")
        self.curr_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.curr_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #1F1F21;
                border: 1px solid {Theme.BORDER_LIGHT};
                border-radius: 6px;
                color: #FFFFFF;
                padding: 4px 8px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 1px solid {Theme.BG_ACTIVE};
            }}
        """)
        card_layout.addWidget(self.curr_input)
        
        self.new_input = QLineEdit()
        self.new_input.setPlaceholderText("New Password")
        self.new_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.new_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #1F1F21;
                border: 1px solid {Theme.BORDER_LIGHT};
                border-radius: 6px;
                color: #FFFFFF;
                padding: 4px 8px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 1px solid {Theme.BG_ACTIVE};
            }}
        """)
        card_layout.addWidget(self.new_input)
        
        card_layout.addSpacing(2)
        
        submit_btn = QPushButton("Confirm")
        submit_btn.setFixedHeight(26)
        submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        submit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_ACTIVE};
                border: none;
                color: #FFFFFF;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2455C0;
            }}
        """)
        submit_btn.clicked.connect(self.handle_submit)
        card_layout.addWidget(submit_btn)
        
    def handle_submit(self):
        curr_pass = self.curr_input.text().strip()
        new_pass = self.new_input.text().strip()
        
        admin_pass = "Admin@SIST"
        if self.parent() and hasattr(self.parent(), 'admin_password'):
            admin_pass = self.parent().admin_password
            
        if curr_pass != admin_pass:
            self.error_lbl.setText("Invalid Current Password")
            self.error_lbl.setVisible(True)
        elif not new_pass:
            self.error_lbl.setText("New Password cannot be empty")
            self.error_lbl.setVisible(True)
        else:
            if self.parent():
                self.parent()._save_admin_password(new_pass)
                self.parent()._log_audit_action("Admin updated password successfully")
            self.accept()
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 12, 210))
        
    def showEvent(self, event):
        super().showEvent(event)
        self._reposition_card()
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_card()

    def _reposition_card(self):
        if self.parent():
            self.setGeometry(self.parent().rect())
            if hasattr(self.parent(), 'sidebar') and self.parent().sidebar:
                sidebar = self.parent().sidebar
                sidebar_pos = sidebar.mapTo(self.parent(), QPoint(0, 0))
                x = sidebar_pos.x() + 22
                y = sidebar_pos.y() + sidebar.height() - 148 - 100
                self.card.move(int(x), int(y))
            else:
                h = self.parent().height()
                self.card.move(22, h - 148 - 100)

class SidebarWidget(QWidget):
    nav_changed = pyqtSignal(str)
    config_cancelled = pyqtSignal()
    config_database_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(224)
        self.setStyleSheet(f"SidebarWidget {{ background: {Theme.BG_SIDEBAR}; border-right: 1px solid {Theme.BORDER}; border-top-left-radius: 18px; border-bottom-left-radius: 18px; }}")
        self.current_active_tab = "Live Feed"
        self._setup_ui()
        self._setup_config_connections()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 40, 12, 40)
        
        layout.addWidget(label("UniGuard", 18, "#b4cae4", QFont.Weight.Bold))
        layout.addWidget(label("System Node 01", 10, Theme.TEXT_SECONDARY))

        status_row = QWidget()
        sr_layout = QHBoxLayout(status_row)
        sr_layout.setContentsMargins(0, 4, 0, 0)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background: {Theme.GREEN}; border-radius: 4px;")
        sr_layout.addWidget(dot)
        sr_layout.addWidget(label("Active / Low Latency", 9, Theme.GREEN))
        sr_layout.addStretch()
        layout.addWidget(status_row)
        layout.addSpacerItem(QSpacerItem(0, 20))

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {Theme.BORDER};")
        layout.addWidget(sep)
        layout.addSpacerItem(QSpacerItem(0, 10))

        nav_items = ["Attendance", "Live Feed", "Analytics", "Database", "Configuration", "Tag Detection"]
        self._nav_group = []
        self.config_nav_btn = None
        for i, name in enumerate(nav_items):
            active = (name == "Live Feed")
            btn = NavItem(name, active)
            btn.clicked.connect(lambda checked, n=name, b=btn: self._on_nav(n, b))
            layout.addWidget(btn)
            self._nav_group.append(btn)
            if name == "Configuration":
                self.config_nav_btn = btn

        # --- Nested Sidebar Config Panel ---
        self.config_panel = QFrame()
        self.config_panel.setObjectName("ConfigPanel")
        self.config_panel.setFrameShape(QFrame.Shape.NoFrame)
        self.config_panel.setStyleSheet(f"""
            #ConfigPanel {{
                background-color: #121214;
                border: 1px solid {Theme.BORDER_LIGHT};
                border-radius: 10px;
                margin-top: 4px;
                margin-right: 12px;
            }}
        """)
        self.config_panel.setVisible(False)
        
        cp_layout = QVBoxLayout(self.config_panel)
        cp_layout.setContentsMargins(12, 10, 12, 10)
        cp_layout.setSpacing(6)
        
        # Header "Student List"
        cp_layout.addWidget(label("Student List", 11, Theme.TEXT_PRIMARY, QFont.Weight.Bold))
        
        # File selector row
        file_row = QWidget()
        fr_layout = QHBoxLayout(file_row)
        fr_layout.setContentsMargins(0, 0, 0, 0)
        fr_layout.setSpacing(6)
        
        lbl_select = label("Select file", 10, Theme.TEXT_SECONDARY)
        fr_layout.addWidget(lbl_select)
        fr_layout.addStretch()
        
        self.attach_btn = ArrowButton()
        fr_layout.addWidget(self.attach_btn)
        cp_layout.addWidget(file_row)
        
        # Button row
        btn_row = QWidget()
        br_layout = QHBoxLayout(btn_row)
        br_layout.setContentsMargins(0, 4, 0, 0)
        br_layout.setSpacing(6)
        
        self.config_cancel_btn = QPushButton("Cancel")
        self.config_cancel_btn.setFixedSize(64, 24)
        self.config_cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {Theme.BORDER_LIGHT};
                color: {Theme.TEXT_SECONDARY};
                border-radius: 5px;
                font-family: 'Segoe UI', Arial;
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: {Theme.TEXT_PRIMARY};
                background: #1F1F21;
            }}
        """)
        
        self.config_change_btn = QPushButton("Change")
        self.config_change_btn.setFixedSize(64, 24)
        self.config_change_btn.setEnabled(False)
        self.config_change_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_ACTIVE};
                border: none;
                color: #FFFFFF;
                border-radius: 5px;
                font-family: 'Segoe UI', Arial;
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2455C0;
            }}
            QPushButton:disabled {{
                background-color: #1F1F21;
                color: #4D4D50;
            }}
        """)
        
        br_layout.addWidget(self.config_cancel_btn)
        br_layout.addWidget(self.config_change_btn)
        cp_layout.addWidget(btn_row)
        
        layout.addWidget(self.config_panel)

        # File name status label (placed OUTSIDE the component, below Cancel/Change)
        import json
        import os
        config_path = "face_db/config.json"
        init_file = "No file selected"
        init_style = "color: #6B7A99; margin-top: 4px; margin-left: 2px;"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                    p_file = data.get("committed_config_file")
                    if p_file:
                        init_file = Path(p_file).name
                        init_style = f"color: {Theme.TEXT_ACCENT}; font-weight: bold; margin-top: 4px; margin-left: 2px;"
            except:
                pass

        self.file_name_lbl = label(init_file, 9, Theme.TEXT_SECONDARY)
        self.file_name_lbl.setStyleSheet(init_style)
        self.file_name_lbl.setWordWrap(True)
        self.file_name_lbl.setVisible(False)
        layout.addWidget(self.file_name_lbl)

        layout.addStretch()

        # --- Nested Sidebar Analytics Controls Widget (No border/background container, transparent) ---
        self.analytics_panel = QWidget()
        self.analytics_panel.setObjectName("AnalyticsControls")
        self.analytics_panel.setStyleSheet("background: transparent;")
        self.analytics_panel.setVisible(False)
        
        ap_layout = QVBoxLayout(self.analytics_panel)
        ap_layout.setContentsMargins(0, 0, 12, 12)
        ap_layout.setSpacing(10)
        
        # Compact sidebar filter buttons
        filter_row = QWidget()
        fr_lay = QHBoxLayout(filter_row)
        fr_lay.setContentsMargins(0, 0, 0, 0)
        fr_lay.setSpacing(4)
        
        self.sidebar_filters = []
        for lbl in ["24h", "7d", "30d", "Custom"]:
            btn = QPushButton(lbl)
            btn.setFixedHeight(22)
            btn.setCheckable(True)
            if lbl == "24h":
                btn.setChecked(True)
            
            active_style = f"""
                QPushButton {{
                    background: {Theme.BG_ACTIVE};
                    color: #FFFFFF;
                    border-radius: 4px;
                    border: none;
                    font-family: 'Segoe UI', Arial;
                    font-size: 9px;
                    font-weight: bold;
                }}
            """
            inactive_style = f"""
                QPushButton {{
                    background: transparent;
                    color: {Theme.TEXT_SECONDARY};
                    border-radius: 4px;
                    border: 1px solid {Theme.BORDER_LIGHT};
                    font-family: 'Segoe UI', Arial;
                    font-size: 9px;
                }}
                QPushButton:hover {{
                    color: {Theme.TEXT_PRIMARY};
                    background: #1F1F21;
                }}
            """
            btn.setStyleSheet(active_style if btn.isChecked() else inactive_style)
            btn.toggled.connect(lambda checked, b=btn, a_st=active_style, ia_st=inactive_style: b.setStyleSheet(a_st if checked else ia_st))
            
            # Closure for radio behavior
            def make_radio_handler(lbl_clicked):
                def handler(*args):
                    for b in self.sidebar_filters:
                        if b.text() != lbl_clicked:
                            b.setChecked(False)
                        else:
                            b.setChecked(True)
                    
                    # Notify analytics page of the timeframe change (using self-healing parent traversal)
                    analytics_page = None
                    if hasattr(self, 'analytics_page') and self.analytics_page:
                        analytics_page = self.analytics_page
                    else:
                        p = self.parent()
                        while p:
                            if hasattr(p, 'analytics_page') and p.analytics_page:
                                analytics_page = p.analytics_page
                                break
                            p = p.parent()
                    
                    if analytics_page:
                        analytics_page.set_timeframe(lbl_clicked)
                return handler
                
            btn.clicked.connect(make_radio_handler(lbl))
            
            fr_lay.addWidget(btn)
            self.sidebar_filters.append(btn)
            
        ap_layout.addWidget(filter_row)
        
        # Export Button (Sleek blue button with vector download icon)
        self.sidebar_export_btn = QPushButton("   EXPORT REPORT")
        self.sidebar_export_btn.setFixedHeight(28)
        self.sidebar_export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_ACTIVE};
                border: none;
                color: #FFFFFF;
                border-radius: 5px;
                font-family: 'Segoe UI', Arial;
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2455C0;
            }}
        """)
        
        from scripts.analytics_dash import VectorIconWidget
        self.sidebar_export_icon = VectorIconWidget("export", QColor("#FFFFFF"), size=10, parent=self.sidebar_export_btn)
        self.sidebar_export_icon.move(14, 9)
        
        ap_layout.addWidget(self.sidebar_export_btn)
        layout.addWidget(self.analytics_panel)

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background: {Theme.BORDER};")
        layout.addWidget(sep2)
        layout.addSpacerItem(QSpacerItem(0, 10))

        user_row = QWidget()
        ur_layout = QHBoxLayout(user_row)
        ur_layout.setContentsMargins(0, 0, 0, 0)
        avatar = AvatarWidget(size=36)
        ur_layout.addWidget(avatar)

        info = QVBoxLayout()
        info.setSpacing(2)
        
        self.username_lbl = QLabel("Unauthorized")
        self.username_lbl.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-family: 'Segoe UI', Arial;
                font-size: 10px;
                font-weight: bold;
                background: transparent;
            }}
        """)
        info.addWidget(self.username_lbl)
        
        self.actions_row = QWidget()
        self.actions_row.setStyleSheet("background: transparent;")
        actions_layout = QHBoxLayout(self.actions_row)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)
        
        link_style = f"""
            QPushButton {{
                border: none;
                background: transparent;
                color: {Theme.TEXT_SECONDARY};
                font-family: 'Segoe UI', Arial;
                font-size: 9px;
                padding: 0;
            }}
            QPushButton:hover {{
                color: #FFFFFF;
                text-decoration: underline;
            }}
        """
        
        self.switch_btn = QPushButton("Switch")
        self.switch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.switch_btn.setStyleSheet(link_style)
        
        self.divider = QLabel("|")
        self.divider.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-family: 'Segoe UI', Arial; font-size: 9px; background: transparent;")
        
        self.logout_btn = QPushButton("Logout")
        self.logout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.logout_btn.setStyleSheet(link_style)
        
        actions_layout.addWidget(self.switch_btn)
        actions_layout.addWidget(self.divider)
        actions_layout.addWidget(self.logout_btn)
        actions_layout.addStretch()
        
        self.actions_row.setVisible(False)
        info.addWidget(self.actions_row)

        self.login_btn = QPushButton("Login")
        self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_btn.setStyleSheet(link_style)
        self.login_btn.setVisible(True)
        info.addWidget(self.login_btn)
        ur_layout.addLayout(info)
        ur_layout.addStretch()
        layout.addWidget(user_row)

    def _setup_config_connections(self):
        self.attach_btn.clicked.connect(self._on_config_browse)
        self.config_cancel_btn.clicked.connect(self._on_config_cancel)
        self.config_change_btn.clicked.connect(self._on_config_change)
        self.selected_config_file = None
        
        # Load persisted committed config file from disk!
        import json
        import os
        config_path = "face_db/config.json"
        self.committed_config_file = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                    self.committed_config_file = data.get("committed_config_file")
            except:
                pass
        
    def _on_config_browse(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Student Record File", "", 
            "Student Files (*.csv *.xlsx)"
        )
        if file_path:
            self.selected_config_file = file_path
            self.file_name_lbl.setText(Path(file_path).name)
            self.file_name_lbl.setStyleSheet(f"color: {Theme.TEXT_ACCENT}; font-weight: bold; margin-top: 4px; margin-left: 2px;")
            self.config_change_btn.setEnabled(True)
            
    def _on_config_cancel(self):
        self.selected_config_file = None
        if self.committed_config_file:
            filename = Path(self.committed_config_file).name
            self.file_name_lbl.setText(filename)
            self.file_name_lbl.setStyleSheet(f"color: {Theme.TEXT_ACCENT}; font-weight: bold; margin-top: 4px; margin-left: 2px;")
        else:
            self.file_name_lbl.setText("No file selected")
            self.file_name_lbl.setStyleSheet("color: #6B7A99; margin-top: 4px; margin-left: 2px;")
            
        self.config_change_btn.setEnabled(False)
        self.config_panel.setVisible(False)
        self.file_name_lbl.setVisible(False)
        self.config_cancelled.emit()
        
    def _on_config_change(self):
        if self.selected_config_file:
            self.config_database_changed.emit(self.selected_config_file)
            self.committed_config_file = self.selected_config_file
            self.selected_config_file = None
            
            # Persist committed config file to disk!
            import json
            import os
            config_path = "face_db/config.json"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            try:
                with open(config_path, "w") as f:
                    json.dump({"committed_config_file": self.committed_config_file}, f)
            except:
                pass
            
            filename = Path(self.committed_config_file).name
            self.file_name_lbl.setText(filename)
            self.file_name_lbl.setStyleSheet(f"color: {Theme.TEXT_ACCENT}; font-weight: bold; margin-top: 4px; margin-left: 2px;")
            self.config_change_btn.setEnabled(False)
            self.config_panel.setVisible(False)
            self.file_name_lbl.setVisible(False)

    def _on_nav(self, name, clicked_btn):
        # Requirement 2: Clicking on Configuration when it is already active will toggle it off,
        # collapsing the Student File panel and restoring focus back to the core active screen.
        if name == "Configuration" and self.config_panel.isVisible():
            self.select_tab(self.current_active_tab)
            return

        for btn in self._nav_group:
            if btn != clicked_btn:
                btn.setChecked(False)
                btn._active = False
                btn._update_style()
        clicked_btn._active = True
        clicked_btn._update_style()
        
        if name == "Configuration":
            self.config_panel.setVisible(True)
            self.file_name_lbl.setVisible(True)
            if self.committed_config_file:
                filename = Path(self.committed_config_file).name
                self.file_name_lbl.setText(filename)
                self.file_name_lbl.setStyleSheet(f"color: {Theme.TEXT_ACCENT}; font-weight: bold; margin-top: 4px; margin-left: 2px;")
            else:
                self.file_name_lbl.setText("No file selected")
                self.file_name_lbl.setStyleSheet("color: #6B7A99; margin-top: 4px; margin-left: 2px;")
            self.analytics_panel.setVisible(self.current_active_tab == "Analytics")
        elif name == "Analytics":
            self.current_active_tab = "Analytics"
            self.config_panel.setVisible(False)
            self.file_name_lbl.setVisible(False)
            self.analytics_panel.setVisible(True)
        else:
            self.current_active_tab = name
            self.config_panel.setVisible(False)
            self.file_name_lbl.setVisible(False)
            self.analytics_panel.setVisible(False)
            
        self.nav_changed.emit(name)

    def select_tab(self, name):
        for btn in self._nav_group:
            if btn._name == name:
                btn.setChecked(True)
                btn._active = True
                btn._update_style()
                self._on_nav(name, btn)

# ─────────────────────────────────────────────
#  AVATAR WIDGET
# ─────────────────────────────────────────────
class AvatarWidget(QWidget):
    def __init__(self, size=40, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self._size
        path = QPainterPath()
        path.addEllipse(0, 0, s, s)
        grad = QLinearGradient(0, 0, s, s)
        grad.setColorAt(0, QColor("#2A3A5C"))
        grad.setColorAt(1, QColor("#1A2540"))
        p.fillPath(path, grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#4A6080"))
        head_r = int(s * 0.22)
        p.drawEllipse(s // 2 - head_r, int(s * 0.18), head_r * 2, head_r * 2)
        body_w = int(s * 0.55)
        body_h = int(s * 0.32)
        p.drawEllipse(s // 2 - body_w // 2, int(s * 0.58), body_w, body_h)

# ─────────────────────────────────────────────
#  SYSTEM METRICS PANEL
# ─────────────────────────────────────────────
class SystemMetricsPanel(Card):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        
        hdr = QHBoxLayout()
        hdr.addWidget(label("System Metrics", 10, Theme.TEXT_PRIMARY, topic=True))
        hdr.addStretch()
        layout.addLayout(hdr)

        bars_layout = QHBoxLayout()
        bars_layout.setSpacing(16)
        self._cpu_bar = MetricBar("CPU", Theme.CPU_COLOR, 0)
        self._mem_bar = MetricBar("MEM", Theme.MEM_COLOR, 0)
        self._gpu_bar = MetricBar("GPU", Theme.GPU_COLOR, 0)
        bars_layout.addWidget(self._cpu_bar)
        bars_layout.addWidget(self._mem_bar)
        bars_layout.addWidget(self._gpu_bar)
        layout.addLayout(bars_layout)

    def update_metrics(self, cpu, mem, gpu):
        self._cpu_bar.set_value(cpu)
        self._mem_bar.set_value(mem)
        self._gpu_bar.set_value(gpu)

# ─────────────────────────────────────────────
#  DETECTION CONFIDENCE (PYQTGRAPH)
# ─────────────────────────────────────────────
class DetectionConfidencePanel(Card):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(180)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)

        hdr = QHBoxLayout()
        hdr.addWidget(label("Detection Confidence", 10, Theme.TEXT_PRIMARY, topic=True))
        hdr.addStretch()
        hdr.addWidget(label("— Model A", 8, Theme.GRAPH_A))
        layout.addLayout(hdr)

        self.graph = pg.PlotWidget()
        self.graph.setBackground(Theme.BG_CARD)
        self.graph.showGrid(x=False, y=True, alpha=0.3)
        self.graph.setYRange(0, 100)
        self.graph.setMouseEnabled(x=False, y=False)
        self.graph.setMenuEnabled(False)
        self.graph.hideButtons()
        
        axis_pen = pg.mkPen(Theme.BORDER)
        self.graph.getAxis('left').setPen(axis_pen)
        self.graph.getAxis('bottom').setPen(axis_pen)
        self.graph.getAxis('left').setTextPen(Theme.TEXT_MUTED)
        self.graph.getAxis('bottom').setTextPen(Theme.TEXT_MUTED)
        
        self.curve_a = self.graph.plot(pen=pg.mkPen(Theme.GRAPH_A, width=2))
        self._data_a = []
        layout.addWidget(self.graph)

    def update_data(self, val_a):
        self._data_a.append(val_a)
        if len(self._data_a) > 60:
            self._data_a.pop(0)
        self.curve_a.setData(self._data_a)

# ─────────────────────────────────────────────
#  LAST ENTITY PANEL
# ─────────────────────────────────────────────
class LastEntityPanel(Card):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)

        hdr = QHBoxLayout()
        hdr.addWidget(label("Last Student", 10, Theme.TEXT_PRIMARY, topic=True))
        hdr.addStretch()
        layout.addLayout(hdr)

        content = QHBoxLayout()
        avatar = AvatarWidget(size=48)
        content.addWidget(avatar)

        info_col = QVBoxLayout()
        
        name_row = QHBoxLayout()
        self.name_lbl = label("-", 11, Theme.TEXT_PRIMARY, QFont.Weight.Bold)
        self.match_lbl = label("0.0%", 10, Theme.GREEN, QFont.Weight.Bold)
        name_row.addWidget(self.name_lbl)
        name_row.addWidget(self.match_lbl)
        name_row.addStretch()
        
        details_row = QHBoxLayout()
        self.reg_lbl = label("Reg: -", 9, Theme.TEXT_MUTED)
        self.time_lbl = label("Time: -", 9, Theme.TEXT_MUTED)
        details_row.addWidget(self.reg_lbl)
        details_row.addWidget(self.time_lbl)
        details_row.addStretch()
        
        info_col.addLayout(name_row)
        info_col.addLayout(details_row)
        
        content.addLayout(info_col)
        content.addStretch()
        layout.addLayout(content)

    def update_info(self, first, last, authorized, match_pct, person_id=None):
        import datetime
        self.name_lbl.setText(f"{first} {last}")
        self.match_lbl.setText(f"{match_pct*100:.1f}%")
        
        # Determine color based on authorization
        if authorized:
            self.match_lbl.setStyleSheet(f"color: {Theme.GREEN}; background: transparent;")
        else:
            self.match_lbl.setStyleSheet(f"color: {Theme.RED}; background: transparent;")
            
        if person_id:
            self.reg_lbl.setText(f"Reg: {person_id}")
        else:
            self.reg_lbl.setText(f"Reg: 21BCE{str(hash(first+last))[-4:].replace('-', '0')}")
        self.time_lbl.setText(f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}")

# ─────────────────────────────────────────────
#  TARGET TAGS PANEL
# ─────────────────────────────────────────────
class TagButton(QPushButton):
    def __init__(self, text, fg, bg, parent=None):
        super().__init__(parent)
        self.setText(f"● {text}")
        self.setCheckable(True)
        self.setFixedHeight(30)
        self._fg = fg
        self._bg = bg
        self._update_style(False)
        self.toggled.connect(self._update_style)

    def _update_style(self, checked):
        self.setStyleSheet(f"QPushButton {{ background: {self._bg}; color: {self._fg}; border: 1px solid {self._fg}; border-radius: 15px; padding: 0 12px; font-size: 10px; font-weight: {'700' if checked else '500'}; }} QPushButton:hover {{ border: 2px solid {self._fg}; }}")

class TargetTagsPanel(Card):
    tag_selected = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        
        hdr = QHBoxLayout()
        hdr.addWidget(label("Target Tags", 10, Theme.TEXT_PRIMARY, topic=True))
        layout.addLayout(hdr)

        tags_layout = QHBoxLayout()
        tags_layout.setSpacing(6)
        tags = [
            ("red", "Red", *Theme.TAG_CRITICAL),
            ("yellow", "Yellow", *Theme.TAG_WATCH),
            ("purple", "Purple", *Theme.TAG_VIP),
            ("gray", "Gray", *Theme.TAG_IGNORE),
        ]
        self.btns = []
        for i, (id_name, disp_name, fg, bg) in enumerate(tags):
            btn = TagButton(disp_name, fg, bg)
            btn.toggled.connect(lambda checked, n=id_name: self._on_btn(n, checked))
            tags_layout.addWidget(btn)
            self.btns.append(btn)
            
        # Set Purple checked by default
        for btn in self.btns:
            if "Purple" in btn.text():
                btn.setChecked(True)
                
        tags_layout.addStretch()
        layout.addLayout(tags_layout)

    def _on_btn(self, name, checked):
        if checked:
            self.tag_selected.emit(name)
            for b in self.btns:
                if b.sender() != b: b.setChecked(False)

# ─────────────────────────────────────────────
#  EVENT LOG PANEL
# ─────────────────────────────────────────────
class EventLogEntry(QWidget):
    SEVERITY_COLORS = {
        "critical": Theme.RED, "warning": Theme.YELLOW, "info": Theme.TEXT_MUTED,
        "permitted": Theme.GREEN, "success": Theme.GREEN,
        "Red": Theme.RED, "Yellow": Theme.YELLOW, "Purple": Theme.PURPLE, "Gray": Theme.TEXT_MUTED
    }
    def __init__(self, severity, message, time_str, detail="", parent=None):
        super().__init__(parent)
        color = self.SEVERITY_COLORS.get(severity, Theme.TEXT_MUTED)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background: {color}; border-radius: 4px; margin-top: 4px;")
        layout.addWidget(dot, 0, Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        msg_row = QHBoxLayout()
        msg_lbl = QLabel(message)
        msg_lbl.setFont(ui_font(9))
        msg_lbl.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        msg_lbl.setWordWrap(True)
        msg_row.addWidget(msg_lbl)
        time_lbl = label(time_str, 8, Theme.TEXT_MUTED, mono=True)
        msg_row.addWidget(time_lbl, 0, Qt.AlignmentFlag.AlignRight)
        text_col.addLayout(msg_row)

        if detail:
            det_lbl = QLabel(detail)
            det_lbl.setFont(ui_font(9, QFont.Weight.DemiBold))
            det_lbl.setStyleSheet(f"color: {color if severity != 'info' else Theme.TEXT_MUTED};")
            text_col.addWidget(det_lbl)
        layout.addLayout(text_col)

class EventLogPanel(Card):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 6)

        hdr = QHBoxLayout()
        hdr.addWidget(label("Event Log", 10, Theme.TEXT_PRIMARY, topic=True))
        layout.addLayout(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; } QWidget#scrollAreaWidgetContents { background: transparent; }")
        
        self.container = QWidget()
        self.container.setObjectName("scrollAreaWidgetContents")
        self.container.setStyleSheet("background: transparent;")
        self.entries_layout = QVBoxLayout(self.container)
        self.entries_layout.addStretch()
        scroll.setWidget(self.container)
        layout.addWidget(scroll)

    def add_event(self, severity, message, time_str, detail=""):
        entry = EventLogEntry(severity, message, time_str, detail)
        self.entries_layout.insertWidget(0, entry)
        if self.entries_layout.count() > 30:
            item = self.entries_layout.takeAt(30)
            if item and item.widget():
                item.widget().deleteLater()

class TagDetectionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(400, 320)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Drag variables
        self._drag_position = QPoint()
        self._is_dragging = False
        
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        # Layout inside our custom painted boundaries (leaving margin for drop shadows!)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # Header Row
        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        
        icon_lbl = QLabel("🔬")
        icon_lbl.setFont(ui_font(14))
        hdr.addWidget(icon_lbl)
        
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title_box.addWidget(label("Tag Color Analyzer", 11, Theme.TEXT_PRIMARY, QFont.Weight.Bold))
        title_box.addWidget(label("Real-time HSV pixel density tracking", 8, Theme.TEXT_SECONDARY))
        hdr.addLayout(title_box)
        hdr.addStretch()
        
        # Internal circular '×' close button (matches premium Mac-style window controls)
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background: #2E2E32;
                border: 1px solid {Theme.BORDER_LIGHT};
                color: {Theme.TEXT_SECONDARY};
                border-radius: 10px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                font-weight: bold;
                line-height: 20px;
            }}
            QPushButton:hover {{
                background: {Theme.RED};
                color: #FFFFFF;
                border: none;
            }}
        """)
        self.close_btn.clicked.connect(self.hide)
        hdr.addWidget(self.close_btn)
        layout.addLayout(hdr)

        # Horizontal meters for each color
        self.meters = {}
        self.labels = {}
        for color, name, theme_color in [('red', 'Red Spectrum', Theme.RED), 
                                         ('yellow', 'Yellow Spectrum', Theme.YELLOW), 
                                         ('purple', 'Purple Spectrum', Theme.PURPLE)]:
            row = QVBoxLayout()
            row.setSpacing(4)
            
            lbl_row = QHBoxLayout()
            lbl_row.addWidget(label(name, 9, Theme.TEXT_PRIMARY, QFont.Weight.DemiBold))
            lbl_row.addStretch()
            val_lbl = label("0 px", 9, theme_color, mono=True)
            lbl_row.addWidget(val_lbl)
            row.addLayout(lbl_row)
            
            pbar = QProgressBar()
            pbar.setFixedHeight(10)
            pbar.setTextVisible(False)
            pbar.setRange(0, 1500) # Dynamic scaling
            pbar.setStyleSheet(f"""
                QProgressBar {{
                    background: {Theme.BG_DEEP};
                    border: 1px solid {Theme.BORDER};
                    border-radius: 5px;
                }}
                QProgressBar::chunk {{
                    background: {theme_color};
                    border-radius: 4px;
                }}
            """)
            row.addWidget(pbar)
            layout.addLayout(row)
            
            self.meters[color] = pbar
            self.labels[color] = val_lbl

        # Dominant Color badge row
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(2, 0, 2, 0)
        badge_row.addWidget(label("Dominant:", 9, Theme.TEXT_SECONDARY))
        
        self.badge = label("NONE", 9, Theme.TEXT_MUTED, QFont.Weight.Bold)
        self.badge.setStyleSheet("background: #2E2E2E; padding: 2px 8px; border-radius: 4px; border: 1px solid #3E3E3E;")
        badge_row.addWidget(self.badge)
        
        self.status_text = label("No Tag Detected", 9, Theme.TEXT_SECONDARY)
        badge_row.addWidget(self.status_text, 1, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(badge_row)

        # Calibration / Range Box
        range_box = QFrame()
        range_box.setStyleSheet(f"background: {Theme.BG_DEEP}; border: 1px solid {Theme.BORDER}; border-radius: 6px;")
        rb_layout = QVBoxLayout(range_box)
        rb_layout.setContentsMargins(8, 6, 8, 6)
        rb_layout.setSpacing(2)
        
        rb_layout.addWidget(label("🎯 Active HSV Saturation Threshold Gating:", 8, Theme.TEXT_PRIMARY, QFont.Weight.Bold))
        rb_layout.addWidget(label("• Purple:  H[110, 170] | S >= 70  | V >= 30", 8, Theme.PURPLE, mono=True))
        rb_layout.addWidget(label("• Yellow:  H[15, 40]   | S >= 90  | V >= 80", 8, Theme.YELLOW, mono=True))
        rb_layout.addWidget(label("• Red:     H[0, 10]+   | S >= 90  | V >= 50", 8, Theme.RED, mono=True))
        layout.addWidget(range_box)

    def update_data(self, color_debug):
        if not color_debug:
            for color in ['red', 'yellow', 'purple']:
                self.meters[color].setValue(0)
                self.labels[color].setText("0 px")
            self.badge.setText("NONE")
            self.badge.setStyleSheet("background: #2E2E2E; padding: 2px 8px; border-radius: 4px; border: 1px solid #3E3E3E; color: #6B7A99;")
            self.status_text.setText("Searching for lanyard...")
            return

        # Update values and meters
        for color in ['red', 'yellow', 'purple']:
            val = color_debug.get(color, 0)
            self.labels[color].setText(f"{val} px")
            # Auto-scale range to fit large counts gracefully
            if val > self.meters[color].maximum():
                self.meters[color].setMaximum(val + 500)
            self.meters[color].setValue(val)

        # Compute winner
        best_color = max(color_debug, key=color_debug.get)
        best_val = color_debug[best_color]
        
        if best_val < 50:
            self.badge.setText("NONE")
            self.badge.setStyleSheet("background: #2E2E2E; padding: 2px 8px; border-radius: 4px; border: 1px solid #3E3E3E; color: #6B7A99;")
            self.status_text.setText("Density too low (< 50 px)")
        else:
            color_names = {'red': 'RED', 'yellow': 'YELLOW', 'purple': 'PURPLE'}
            color_themes = {'red': Theme.RED, 'yellow': Theme.YELLOW, 'purple': Theme.PURPLE}
            color_bg = {'red': '#3D1515', 'yellow': '#3D330A', 'purple': '#1E1535'}
            
            self.badge.setText(color_names[best_color])
            self.badge.setStyleSheet(f"background: {color_bg[best_color]}; padding: 2px 8px; border-radius: 4px; border: 1px solid {color_themes[best_color]}; color: {color_themes[best_color]}; font-weight: bold;")
            self.status_text.setText(f"Locked onto {color_names[best_color].lower()} lanyard ribbon")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            event.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Draw Soft Drop Shadow
        shadow_color = QColor(0, 0, 0, 80)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(shadow_color)
        p.drawRoundedRect(6, 6, w - 12, h - 12, 16, 16)
        
        # Draw cohesive Dark slate Card background with 1px light border
        card_color = QColor(Theme.BG_PANEL)
        p.setBrush(card_color)
        border_pen = QPen(QColor(Theme.BORDER_LIGHT), 1)
        p.setPen(border_pen)
        p.drawRoundedRect(6, 6, w - 12, h - 12, 14, 14)

# ─────────────────────────────────────────────
#  RIGHT PANEL  (stacked cards)
# ─────────────────────────────────────────────
class RightPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(292)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.metrics    = SystemMetricsPanel()
        self.confidence = DetectionConfidencePanel()
        self.entity     = LastEntityPanel()
        self.tags       = TargetTagsPanel()
        self.event_log  = EventLogPanel()
        
        layout.addWidget(self.metrics)
        layout.addWidget(self.confidence)
        layout.addWidget(self.entity)
        layout.addWidget(self.tags)
        layout.addWidget(self.event_log, 1)

    def set_panel_enabled(self, name: str, enabled: bool):
        panels = {
            "metrics": self.metrics,
            "graphs":  self.confidence,
            "tags":    self.tags,
            "profile": self.entity,
            "logs":    self.event_log,
        }
        if name in panels:
            panels[name].setEnabled(enabled)

# ─────────────────────────────────────────────
#  DOCK BAR (CUSTOM DRAWN ICONS)
# ─────────────────────────────────────────────
class DockButton(QPushButton):
    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        self._label = label_text
        self.setCheckable(True)
        self.setChecked(True)
        self.setFixedSize(60, 68)
        self._update_style()
        self.toggled.connect(self._update_style)

    def _update_style(self):
        self.setStyleSheet(f"QPushButton {{ background: transparent; border: none; border-radius: {Theme.RADIUS_SM}px; }} QPushButton:hover {{ background: {Theme.BG_HOVER}; }}")
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        active = self.isChecked()
        is_eye = getattr(self, 'is_eye', False)

        cy = h // 2 if is_eye else h // 2 - 14
        dot_y = h // 2 + 18
        text_y = h // 2 + 2

        if active and not is_eye:
            p.setBrush(QColor(Theme.BLUE))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(w // 2 - 2, dot_y, 4, 4)

        if is_eye:
            icon_color = QColor(Theme.BLUE if active else Theme.TEXT_SECONDARY)
        else:
            icon_color = QColor(Theme.BLUE if active else Theme.TEXT_SECONDARY)
            
        p.setPen(QPen(icon_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)

        cx = w // 2
        
        # Draw custom vectors from user image
        if self._label == "FEED":
            p.drawRoundedRect(cx - 10, cy - 6, 14, 12, 2, 2)
            poly = QPolygonF([QPointF(cx + 4, cy - 3), QPointF(cx + 10, cy - 6), QPointF(cx + 10, cy + 6), QPointF(cx + 4, cy + 3)])
            p.drawPolygon(poly)
        elif self._label == "METRICS":
            p.drawRect(cx - 8, cy + 2, 4, 6)
            p.drawRect(cx - 2, cy - 2, 4, 10)
            p.drawRect(cx + 4, cy - 6, 4, 14)
            p.setPen(QPen(icon_color, 1.5))
            p.drawPolyline(QPolygonF([QPointF(cx - 6, cy + 2), QPointF(cx, cy - 2), QPointF(cx + 6, cy - 6)]))
        elif self._label == "GRAPHS":
            p.drawPolyline(QPolygonF([QPointF(cx - 8, cy + 6), QPointF(cx - 4, cy + 6), QPointF(cx, cy), QPointF(cx + 4, cy + 4), QPointF(cx + 8, cy - 4)]))
            p.drawLine(cx - 10, cy - 8, cx - 10, cy + 8)
            p.drawLine(cx - 10, cy + 8, cx + 10, cy + 8)
        elif self._label == "TAGS":
            poly = QPolygonF([QPointF(cx - 8, cy - 6), QPointF(cx + 2, cy - 6), QPointF(cx + 8, cy), QPointF(cx + 8, cy + 2), QPointF(cx + 2, cy + 8), QPointF(cx - 8, cy - 2)])
            p.drawPolygon(poly)
            p.drawEllipse(cx - 5, cy - 3, 2, 2)
        elif self._label == "PROFILE":
            p.drawEllipse(cx - 8, cy - 8, 16, 16)
            p.drawEllipse(cx - 3, cy - 5, 6, 6)
            path = QPainterPath()
            path.moveTo(cx - 6, cy + 6)
            path.quadTo(cx, cy + 2, cx + 6, cy + 6)
            p.drawPath(path)
        elif self._label == "LOGS":
            p.drawRoundedRect(cx - 8, cy - 8, 16, 16, 2, 2)
            for i in range(3):
                y = cy - 3 + i * 4
                p.drawLine(cx - 4, y, cx - 3, y)
                p.drawLine(cx - 1, y, cx + 5, y)
        
        if is_eye:
            path = QPainterPath()
            path.moveTo(cx - 10, cy)
            path.quadTo(cx, cy - 8, cx + 10, cy)
            path.quadTo(cx, cy + 8, cx - 10, cy)
            p.drawPath(path)
            p.drawEllipse(cx - 3, cy - 3, 6, 6)
            if active:
                p.drawLine(cx - 12, cy + 10, cx + 12, cy - 10)

        if not is_eye:
            lbl_color = QColor(Theme.TEXT_PRIMARY if active else Theme.TEXT_SECONDARY)
            p.setPen(lbl_color)
            p.setFont(mono_font(9, QFont.Weight.Bold))
            p.drawText(0, text_y, w, 20, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop, self._label)

class DockContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        path = QPainterPath()
        # Perfect pill shape
        radius = h / 2
        path.addRoundedRect(1, 1, w - 2, h - 2, radius, radius)
        p.fillPath(path, QColor(Theme.BG_DOCK))
        p.setPen(QPen(QColor(Theme.BORDER_LIGHT), 1))
        p.drawPath(path)

class DockBar(QWidget):
    panel_toggled = pyqtSignal(str, bool)
    DOCK_ITEMS = ["FEED", "METRICS", "GRAPHS", "TAGS", "PROFILE", "LOGS"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(88)
        
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        
        left_zone = QWidget()
        left_zone.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lz_layout = QHBoxLayout(left_zone)
        lz_layout.setContentsMargins(0, 0, 0, 0)
        lz_layout.addStretch()
        lz_layout.addSpacing(32)
        outer.addWidget(left_zone)

        pill = DockContainer()
        pill_layout = QHBoxLayout(pill)
        pill_layout.setContentsMargins(20, 8, 20, 8)
        
        for name in self.DOCK_ITEMS:
            btn = DockButton(name)
            btn.toggled.connect(lambda checked, n=name.lower(): self.panel_toggled.emit(n, checked))
            pill_layout.addWidget(btn)
        outer.addWidget(pill, 0, Qt.AlignmentFlag.AlignCenter)
        
        right_zone = QWidget()
        right_zone.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        rz_layout = QHBoxLayout(right_zone)
        rz_layout.setContentsMargins(20, 0, 0, 0)
        rz_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        self.eye_btn = DockButton("")
        self.eye_btn.is_eye = True
        self.eye_btn.setFixedSize(60, 68)
        self.eye_btn.setChecked(False)
        self.eye_btn.toggled.connect(lambda checked: self.panel_toggled.emit("focus", checked))
        
        rz_layout.addWidget(self.eye_btn)
        outer.addWidget(right_zone)

class MacTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(38)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(8)

        self.close_btn = self._make_btn("#FF5F56", "#E0443E")
        self.min_btn = self._make_btn("#FFBD2E", "#DEA123")
        self.max_btn = self._make_btn("#27C93F", "#1AAB29")

        layout.addWidget(self.close_btn)
        layout.addWidget(self.min_btn)
        layout.addWidget(self.max_btn)
        layout.addStretch()
        
        self.close_btn.clicked.connect(self.window().close)
        self.min_btn.clicked.connect(self.window().showMinimized)
        self.max_btn.clicked.connect(self._toggle_fullscreen)

        self._is_dragging = False
        self._drag_pos = QPoint()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setPen(QColor(Theme.TEXT_SECONDARY))
        p.setFont(ui_font(10, QFont.Weight.DemiBold))
        p.drawText(0, 0, self.width(), self.height(), Qt.AlignmentFlag.AlignCenter, self.window().windowTitle())

    def _make_btn(self, color, hover_color):
        btn = QPushButton()
        btn.setFixedSize(12, 12)
        btn.setStyleSheet(f"QPushButton {{ background-color: {color}; border-radius: 6px; border: none; }} QPushButton:hover {{ background-color: {hover_color}; }}")
        return btn

    def _toggle_fullscreen(self):
        w = self.window()
        if w.isFullScreen():
            w.showNormal()
        else:
            w.showFullScreen()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._is_dragging = False

class DotBackgroundWidget(QWidget):
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 16, 16)
        p.fillPath(path, QColor("#18181A"))
        
        p.setClipPath(path)
        p.setPen(QColor("#2A2A30"))
        w, h = self.width(), self.height()
        spacing = 20
        for x in range(0, w, spacing):
            for y in range(0, h, spacing):
                p.drawPoint(x, y)

class InnerContainerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._overlay = None

    def set_overlay(self, overlay):
        self._overlay = overlay

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._overlay:
            self._overlay.resize(self.size())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        path = QPainterPath()
        path.addRoundedRect(3, 3, self.width() - 6, self.height() - 6, 24, 24)
        
        p.fillPath(path, QColor(Theme.BG_DEEP))
        
        pen = QPen(QColor("#9298a5"), 6)
        p.setPen(pen)
        p.drawPath(path)

class OverlayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.hide()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 24, 24)
        p.fillPath(path, QColor(10, 10, 10, 200))
        
    def mousePressEvent(self, event):
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()

    def wheelEvent(self, event):
        event.accept()

class IDBadgeDecoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120) # larger size to make the actual badge visible and high-fidelity
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        path = Path(__file__).resolve().parent / "id_badge.png"
        self.pixmap = QPixmap(str(path))
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self.pixmap.isNull():
            p.drawPixmap(self.rect(), self.pixmap)

class ShoesDecoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 93) # larger size (preserving 1.5:1 ratio) to look incredibly premium
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        path = Path(__file__).resolve().parent / "yellow_shoes.png"
        self.pixmap = QPixmap(str(path))
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self.pixmap.isNull():
            p.drawPixmap(self.rect(), self.pixmap)

# ─────────────────────────────────────────────
#  HAND-DRAWN STICKY NOTE WIDGETS
# ─────────────────────────────────────────────
class HandDrawnStatRow(QWidget):
    def __init__(self, label_text, count_or_val="", parent=None):
        super().__init__(parent)
        self._label_text = label_text
        self._val = count_or_val
        self.setFixedHeight(26) # reduced height! Previously 30.
        self.setFixedWidth(224)
        self.setStyleSheet("background: transparent; border: none;")

    def set_value(self, val):
        self._val = val
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Draw soft blue horizontal notebook ruled lines (above and below the text content)
        # We use a beautiful light pastel blue characteristic of ruled sheets
        p.setPen(QPen(QColor("#A0C5E8"), 1))
        p.drawLine(4, 1, w - 10, 1)      # Top ruled line (above the text)
        p.drawLine(4, h - 2, w - 10, h - 2) # Bottom ruled line (below the text)
        
        # Draw text in rich ink charcoal (#201A06) resting exactly on top of the bottom line
        p.setPen(QColor("#201A06"))
        
        f = QFont("Comic Sans MS", 12)
        p.setFont(f)
        
        # Align bottom with offset of 3px so descenders have room and the baseline rests beautifully on top of the bottom blue line
        p.drawText(4, 0, w - 80, h - 3, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, self._label_text)
        p.drawText(0, 0, w - 10, h - 3, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, str(self._val))

class StickyNoteWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 330)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Drag variables
        self._drag_position = QPoint()
        self._is_dragging = False
        self._has_been_dragged = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 50, 28, 28) # Pushed up to y=50, extremely close to the red pin!
        layout.setSpacing(6) # reduced spacing! Previously 8.
        
        # Stat rows
        self.row_scans = HandDrawnStatRow("Scans", "14")
        self.row_permitted = HandDrawnStatRow("Permitted", "78")
        self.row_violations = HandDrawnStatRow("Violations", "3")
        self.row_presentees = HandDrawnStatRow("Presentees", "0")
        self.row_absentees = HandDrawnStatRow("Absentees", "0")
        self.row_target = HandDrawnStatRow("Target", "Purple") # default Purple!
        
        layout.addWidget(self.row_scans)
        layout.addWidget(self.row_permitted)
        layout.addWidget(self.row_violations)
        layout.addWidget(self.row_presentees)
        layout.addWidget(self.row_absentees)
        layout.addWidget(self.row_target)
        layout.addStretch()
        
        self.hide()

    def update_stats(self, scans, permitted, violations, presentees=0, absentees=0):
        self.row_scans.set_value(scans)
        self.row_permitted.set_value(permitted)
        self.row_violations.set_value(violations)
        self.row_presentees.set_value(presentees)
        self.row_absentees.set_value(absentees)

    def update_target(self, target):
        self.row_target.set_value(target)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._has_been_dragged = True
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            event.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 1. Soft Shadow for the whole stack
        shadow_color = QColor(0, 0, 0, 40)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(shadow_color)
        p.drawRoundedRect(14, 34, w - 28, h - 38, 4, 4)
        
        # 2. Bottom Paper (Rotated at -5 degrees)
        p.save()
        p.translate(w / 2, h / 2)
        p.rotate(-5)
        rect_bottom = QRect(int(-w / 2 + 18), int(-h / 2 + 34), w - 36, h - 44)
        p.setBrush(QColor("#E1C95C")) # Warmer, darker yellow for depth
        p.setPen(QPen(QColor("#C4AD3E"), 1))
        p.drawRect(rect_bottom)
        p.restore()
        
        # 3. Middle Paper (Rotated at 3 degrees)
        p.save()
        p.translate(w / 2, h / 2)
        p.rotate(3)
        rect_middle = QRect(int(-w / 2 + 20), int(-h / 2 + 36), w - 40, h - 46)
        p.setBrush(QColor("#ECCF65")) # Medium warm yellow
        p.setPen(QPen(QColor("#D0B443"), 1))
        p.drawRect(rect_middle)
        p.restore()
        
        # 4. Top Paper (Rotated at 0 degrees - PERFECTLY STRAIGHT!)
        rect_top = QRect(20, 38, w - 40, h - 48)
        
        # Soft yellow gradient for real paper look (perfect type handling to avoid crash)
        grad = QLinearGradient(
            float(rect_top.left()), float(rect_top.top()),
            float(rect_top.right()), float(rect_top.bottom())
        )
        grad.setColorAt(0, QColor("#FFF19D")) # bright yellow top-left
        grad.setColorAt(1, QColor("#F4E071")) # warm yellow bottom-right
        p.setBrush(grad)
        p.setPen(QPen(QColor("#DBC256"), 1))
        p.drawRect(rect_top)
        
        # 5. Draw Red Push Pin on the top paper (centered, straight)
        px, py = w / 2, 30
        
        # Draw Pin Shadow (shifted slightly to bottom-right)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 85))
        p.drawEllipse(int(px + 4), int(py + 12), 10, 5) # Pin head shadow
        p.setPen(QPen(QColor(0, 0, 0, 95), 2))
        p.drawLine(int(px + 1), int(py + 14), int(px + 7), int(py + 19)) # Metal pin shadow
        
        # Draw actual metal pin sticking into the paper
        p.setPen(QPen(QColor("#8C8C8C"), 2))
        p.drawLine(int(px - 1), int(py + 10), int(px - 3), int(py + 17))
        
        # Draw Red Pin Body (3D effect)
        p.setPen(Qt.PenStyle.NoPen)
        
        # Base round cylinder of pin
        p.setBrush(QColor("#9E0000"))
        p.drawRoundedRect(int(px - 5), int(py + 4), 10, 5, 2, 2)
        
        # Middle ring of pin
        p.setBrush(QColor("#E60000"))
        p.drawEllipse(int(px - 7), int(py), 14, 5)
        
        # Top spherical head of pin
        pin_head = QRadialGradient(px - 1, py - 4, 9)
        pin_head.setColorAt(0, QColor("#FF7777")) # specular highlight
        pin_head.setColorAt(0.4, QColor("#FF0000"))
        pin_head.setColorAt(1, QColor("#700000")) # dark red ambient shadow
        p.setBrush(pin_head)
        p.drawEllipse(int(px - 6), int(py - 9), 12, 10)

class ReorderableTableWidget(QTableWidget):
    rows_reordered = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setDragDropMode(QTableWidget.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

    def dragEnterEvent(self, event):
        if event.source() == self:
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.source() == self:
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.source() == self:
            # Get source row
            selected_rows = [idx.row() for idx in self.selectedIndexes() if idx.column() == 0]
            if not selected_rows:
                event.ignore()
                return
            src_row = selected_rows[0]

            # Find destination row index
            drop_pos = event.position().toPoint()
            dest_index = self.indexAt(drop_pos)
            
            if dest_index.isValid():
                dest_row = dest_index.row()
            else:
                dest_row = self.rowCount()

            if src_row == dest_row or src_row == dest_row - 1:
                event.acceptProposedAction()
                return

            event.acceptProposedAction()
            self.blockSignals(True)

            # Store and clone row items
            row_data = []
            for col in range(self.columnCount()):
                item = self.item(src_row, col)
                if item:
                    row_data.append({
                        'text': item.text(),
                        'font': item.font(),
                        'alignment': item.textAlignment(),
                        'foreground': item.foreground(),
                        'background': item.background(),
                        'flags': item.flags(),
                        'user_data': item.data(Qt.ItemDataRole.UserRole)
                    })
                else:
                    row_data.append(None)

            # Insert new row before removing the old one
            self.insertRow(dest_row)
            
            # Recalculate original row position if it shifted
            actual_src_row = src_row
            if dest_row <= src_row:
                actual_src_row += 1

            # Populate the new row
            for col, data in enumerate(row_data):
                if data:
                    new_item = QTableWidgetItem(data['text'])
                    new_item.setFont(data['font'])
                    new_item.setTextAlignment(data['alignment'])
                    new_item.setForeground(data['foreground'])
                    new_item.setBackground(data['background'])
                    new_item.setFlags(data['flags'])
                    if data['user_data'] is not None:
                        new_item.setData(Qt.ItemDataRole.UserRole, data['user_data'])
                    self.setItem(dest_row, col, new_item)

            # Safely remove the original row
            self.removeRow(actual_src_row)

            # Keep row selected
            final_dest_row = dest_row
            if actual_src_row < dest_row:
                final_dest_row -= 1
            self.selectRow(final_dest_row)

            self.blockSignals(False)
            self.rows_reordered.emit()
        else:
            super().dropEvent(event)

class ConfigurationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("UniGuard — System Configuration")
        self.setFixedSize(500, 320)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Outer container with rounded corners and border matching Theme
        self.container = QFrame(self)
        self.container.setGeometry(0, 0, 500, 320)
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: #131315;
                border: 2px solid {Theme.BORDER_LIGHT};
                border-radius: 16px;
            }}
        """)
        
        # Dropshadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 5)
        self.container.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(16)
        
        # Close / Cancel button at top right
        close_btn = QPushButton("×", self.container)
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #6B7A99;
                font-size: 20px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                color: #EF4444;
            }
        """)
        close_btn.clicked.connect(self.reject)
        
        # Position close button
        close_btn.move(456, 16)
        
        # Header title
        header_lbl = label("Student Record File", 15, Theme.TEXT_PRIMARY, QFont.Weight.Bold)
        layout.addWidget(header_lbl)
        
        # Description
        desc_lbl = label("Change the active student database by uploading a new .csv or .xlsx student list. Swapping databases will reset the current records sequentially.", 10, Theme.TEXT_SECONDARY)
        desc_lbl.setWordWrap(True)
        layout.addWidget(desc_lbl)
        
        # File selector input container
        file_container = QHBoxLayout()
        file_container.setSpacing(10)
        
        self.file_path_edit = label("No file selected", 10, Theme.TEXT_SECONDARY)
        self.file_path_edit.setMinimumHeight(38)
        self.file_path_edit.setStyleSheet(f"""
            QLabel {{
                background-color: #1F1F21;
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                padding: 0px 12px;
                color: {Theme.TEXT_SECONDARY};
            }}
        """)
        file_container.addWidget(self.file_path_edit, 1)
        
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedSize(90, 38)
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #1F1F21;
                border: 1px solid {Theme.BORDER_LIGHT};
                color: {Theme.TEXT_PRIMARY};
                border-radius: 8px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2E2F32;
                border-color: {Theme.TEXT_ACCENT};
            }}
        """)
        browse_btn.clicked.connect(self._on_browse)
        file_container.addWidget(browse_btn)
        
        layout.addLayout(file_container)
        
        # Spacer
        layout.addSpacerItem(QSpacerItem(0, 10))
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedSize(100, 38)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {Theme.BORDER_LIGHT};
                color: {Theme.TEXT_SECONDARY};
                border-radius: 8px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: {Theme.TEXT_PRIMARY};
                background: #1F1F21;
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        self.import_btn = QPushButton("Update Database")
        self.import_btn.setFixedSize(140, 38)
        self.import_btn.setEnabled(False)
        self.import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_ACTIVE};
                border: none;
                color: #FFFFFF;
                border-radius: 8px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2455C0;
            }}
            QPushButton:disabled {{
                background-color: #2D2D30;
                color: #5D5D60;
            }}
        """)
        self.import_btn.clicked.connect(self._on_update)
        btn_layout.addWidget(self.import_btn)
        
        layout.addLayout(btn_layout)
        
        self.selected_file = None
        
    def _on_browse(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Student Record File", "", 
            "Student Files (*.csv *.xlsx)"
        )
        if file_path:
            self.selected_file = file_path
            self.file_path_edit.setText(Path(file_path).name)
            self.file_path_edit.setStyleSheet(f"""
                QLabel {{
                    background-color: #1F1F21;
                    border: 1px solid {Theme.TEXT_ACCENT};
                    border-radius: 8px;
                    padding: 0px 12px;
                    color: {Theme.TEXT_PRIMARY};
                }}
            """)
            self.import_btn.setEnabled(True)
            
    def _on_update(self):
        if not self.selected_file:
            return
            
        import pandas as pd
        import sqlite3
        
        try:
            # Read file
            if self.selected_file.endswith('.csv'):
                df = pd.read_csv(self.selected_file)
            else:
                df = pd.read_excel(self.selected_file)
                
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Identify columns
            col_mapping = {}
            for possible_name in ['roll_no', 'rollno', 'roll', 'registration_no', 'roll_number']:
                if possible_name in df.columns:
                    col_mapping['roll_no'] = possible_name
                    break
                    
            for possible_name in ['name', 'student_name', 'student name', 'full_name', 'fullname']:
                if possible_name in df.columns:
                    col_mapping['name'] = possible_name
                    break
                    
            for possible_name in ['gender', 'sex']:
                if possible_name in df.columns:
                    col_mapping['gender'] = possible_name
                    break
                    
            for possible_name in ['year', 'batch_year', 'year of study', 'year_of_study', 'batch', 'study_year']:
                if possible_name in df.columns:
                    col_mapping['batch_year'] = possible_name
                    break
                    
            for possible_name in ['tag_color', 'tag color', 'tag', 'color', 'target_tag', 'tagcolor']:
                if possible_name in df.columns:
                    col_mapping['tag_color'] = possible_name
                    break
                    
            # Check for critical columns
            if 'roll_no' not in col_mapping or 'name' not in col_mapping:
                QMessageBox.critical(
                    self, "Import Error", 
                    "The selected file must contain 'roll_no' (or Roll No) and 'name' (or Student Name) columns."
                )
                return
                
            # Connect to SQLite
            db_path = Path("face_db/uniguard.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Start database operations
            cursor.execute("PRAGMA foreign_keys = OFF")
            cursor.execute("DELETE FROM attendance")
            cursor.execute("DELETE FROM students")
            
            for idx, row in df.iterrows():
                roll_no = str(row.get(col_mapping.get('roll_no'), f"REG_{1000+idx}"))
                name = str(row.get(col_mapping.get('name'), "Unknown Student"))
                gender = str(row.get(col_mapping.get('gender'), "male")).lower()
                
                # Safe parse batch_year
                try:
                    batch_year = int(row.get(col_mapping.get('batch_year'), 3))
                except:
                    batch_year = 3
                    
                tag_color = str(row.get(col_mapping.get('tag_color'), "purple")).lower()
                
                cursor.execute("""
                    INSERT INTO students (roll_no, name, gender, batch_year, tag_color)
                    VALUES (?, ?, ?, ?, ?)
                """, (roll_no, name, gender, batch_year, tag_color))
                
            conn.commit()
            cursor.execute("PRAGMA foreign_keys = ON")
            conn.close()
            
            # Show success box
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"An error occurred while importing: {str(e)}")

from PyQt6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QStyle, QApplication

class TableHighlightDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        bg = index.data(Qt.ItemDataRole.BackgroundRole)
        if bg:
            painter.save()
            painter.fillRect(option.rect, bg)
            new_opt = QStyleOptionViewItem(option)
            self.initStyleOption(new_opt, index)
            new_opt.backgroundBrush = QBrush(Qt.BrushStyle.NoBrush)
            style = option.widget.style() if option.widget else QApplication.style()
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, new_opt, painter, option.widget)
            painter.restore()
        else:
            super().paint(painter, option, index)

class DatabaseExplorerWidget(QWidget):
    back_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_loading_data = False
        self.is_student_role = False
        self._setup_ui()
        self._load_data("attendance") # load attendance by default
        
    def set_role(self, role):
        if role == "Student":
            self.is_student_role = True
            if hasattr(self, 'btn_add_student'):
                self.btn_add_student.hide()
            if hasattr(self, 'table'):
                self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
                self.table.setDragEnabled(False)
                self.table.setAcceptDrops(False)
        else:
            self.is_student_role = False
            if hasattr(self, 'btn_add_student'):
                self.btn_add_student.show()
            if hasattr(self, 'table'):
                self.table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.SelectedClicked)
                self.table.setDragEnabled(True)
                self.table.setAcceptDrops(True)
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # --- Top Header & Segmented Pill Switch Layout (Perfect 3-Column Center Alignment) ---
        top_bar = QGridLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        
        # --- Column 0: Left Title & Circular Back Button ---
        left_layout = QHBoxLayout()
        left_layout.setSpacing(12)
        
        # Circle Back Button to return to dashboard
        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(36, 36)
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.setFont(ui_font(13, QFont.Weight.Bold))
        self.back_btn.setStyleSheet(f"""
            QPushButton {{
                background: #1F2937;
                color: #E5E7EB;
                border: 1px solid {Theme.BORDER};
                border-radius: 18px;
            }}
            QPushButton:hover {{
                background: #374151;
                color: #FFFFFF;
                border: 1px solid #4B5563;
            }}
        """)
        self.back_btn.clicked.connect(self.back_clicked.emit)
        left_layout.addWidget(self.back_btn)
        
        title_v = QVBoxLayout()
        title_v.setSpacing(2)
        title_v.addWidget(label("Databases", 14, Theme.TEXT_PRIMARY, QFont.Weight.Bold))
        title_v.addWidget(label("Live SQLite Records", 9, Theme.TEXT_SECONDARY))
        left_layout.addLayout(title_v)
        left_layout.addStretch()
        
        top_bar.addLayout(left_layout, 0, 0)
        
        # --- Column 1: Centered Capsule Switch Control with Add Student Button ---
        center_container = QWidget()
        center_layout = QHBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        self.switch_container = QWidget()
        self.switch_container.setFixedHeight(36)
        self.switch_container.setStyleSheet(f"background: #1B1B1C; border-radius: 18px; border: 1px solid {Theme.BORDER};")
        switch_layout = QHBoxLayout(self.switch_container)
        switch_layout.setContentsMargins(3, 3, 3, 3)
        switch_layout.setSpacing(0)
        
        self.btn_attendance = QPushButton("Overall")
        self.btn_weekly = QPushButton("Weekly")
        self.btn_students = QPushButton("Student Records")
        
        self.btn_attendance.setCheckable(True)
        self.btn_attendance.setFixedHeight(30)
        self.btn_attendance.setFixedWidth(120)
        self.btn_attendance.setFont(ui_font(11, QFont.Weight.Bold))
        switch_layout.addWidget(self.btn_attendance)
        
        self.btn_weekly.setCheckable(True)
        self.btn_weekly.setFixedHeight(30)
        self.btn_weekly.setFixedWidth(80)
        self.btn_weekly.setFont(ui_font(11, QFont.Weight.Bold))
        switch_layout.addWidget(self.btn_weekly)
        
        self.btn_students.setCheckable(True)
        self.btn_students.setFixedHeight(30)
        self.btn_students.setFixedWidth(120)
        self.btn_students.setFont(ui_font(11, QFont.Weight.Bold))
        switch_layout.addWidget(self.btn_students)
            
        self.btn_attendance.setChecked(True)
        
        self.btn_attendance.clicked.connect(lambda: self._on_switch("attendance"))
        self.btn_weekly.clicked.connect(lambda: self._on_switch("weekly"))
        self.btn_students.clicked.connect(lambda: self._on_switch("students"))
        
        # Elegant circular Add Student "+" button placed to the right of the options inside the capsule
        self.btn_add_student = QPushButton("+")
        self.btn_add_student.setFixedSize(30, 30)
        self.btn_add_student.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_student.setFont(ui_font(14, QFont.Weight.Bold))
        self.btn_add_student.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: #34D399;
                border: none;
                border-radius: 15px;
            }}
            QPushButton:hover {{
                background: #1F2937;
                color: #10B981;
            }}
        """)
        self.btn_add_student.clicked.connect(self._on_add_student_clicked)
        switch_layout.addWidget(self.btn_add_student)
        
        center_layout.addWidget(self.switch_container, 0, Qt.AlignmentFlag.AlignCenter)
        top_bar.addWidget(center_container, 0, 1, Qt.AlignmentFlag.AlignCenter)
        
        # --- Column 2: Right Control Area (Month label with Up/Down arrows & Circular Refresh Button) ---
        right_layout = QHBoxLayout()
        right_layout.setSpacing(12)
        right_layout.addStretch()
        
        # Month Navigation Controller State
        self.current_selected_month_idx = datetime.now().month  # 1-12
        self.current_selected_year = datetime.now().year        # e.g., 2026
        self.MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

        month_nav_layout = QHBoxLayout()
        month_nav_layout.setSpacing(6)
        month_nav_layout.setContentsMargins(0, 0, 0, 0)
        
        # Stacked Up/Down Arrow Buttons (Placed to the left of Month)
        arrow_container = QWidget()
        arrow_container.setFixedSize(14, 30)
        arrow_layout = QVBoxLayout(arrow_container)
        arrow_layout.setContentsMargins(0, 0, 0, 0)
        arrow_layout.setSpacing(0)
        
        btn_arrow_up = QPushButton("▲")
        btn_arrow_up.setFixedSize(14, 15)
        btn_arrow_up.setFont(ui_font(7, QFont.Weight.Bold))
        btn_arrow_up.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_arrow_up.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #94A3B8;
                border: none;
                padding: 0;
            }
            QPushButton:hover {
                color: #10B981;
            }
        """)
        
        btn_arrow_down = QPushButton("▼")
        btn_arrow_down.setFixedSize(14, 15)
        btn_arrow_down.setFont(ui_font(7, QFont.Weight.Bold))
        btn_arrow_down.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_arrow_down.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #94A3B8;
                border: none;
                padding: 0;
            }
            QPushButton:hover {
                color: #F87171;
            }
        """)
        
        arrow_layout.addWidget(btn_arrow_up)
        arrow_layout.addWidget(btn_arrow_down)
        month_nav_layout.addWidget(arrow_container)
        
        # Current Month Label (Like April, May, etc.)
        current_month = self.MONTH_NAMES[self.current_selected_month_idx - 1]
        self.month_lbl = label(current_month, 11, Theme.TEXT_SECONDARY, QFont.Weight.Bold)
        month_nav_layout.addWidget(self.month_lbl)
        
        # Year Label "2026"
        self.year_lbl = label(str(self.current_selected_year), 11, Theme.TEXT_SECONDARY, QFont.Weight.Bold)
        month_nav_layout.addWidget(self.year_lbl)
        
        def _on_month_up():
            self.current_selected_month_idx += 1
            if self.current_selected_month_idx > 12:
                self.current_selected_month_idx = 1
                self.current_selected_year += 1
            self.month_lbl.setText(self.MONTH_NAMES[self.current_selected_month_idx - 1])
            self.year_lbl.setText(str(self.current_selected_year))
            self._refresh_current_table()
            
        def _on_month_down():
            self.current_selected_month_idx -= 1
            if self.current_selected_month_idx < 1:
                self.current_selected_month_idx = 12
                self.current_selected_year -= 1
            self.month_lbl.setText(self.MONTH_NAMES[self.current_selected_month_idx - 1])
            self.year_lbl.setText(str(self.current_selected_year))
            self._refresh_current_table()
            
        btn_arrow_up.clicked.connect(_on_month_up)
        btn_arrow_down.clicked.connect(_on_month_down)
        
        right_layout.addLayout(month_nav_layout)
        
        # Week Selector Combobox
        self.week_combo = QComboBox()
        self.week_combo.addItems(["W1", "W2", "W3", "W4"])
        self.week_combo.setStyleSheet(f"background: #1B1B1C; color: {Theme.TEXT_PRIMARY}; border: 1px solid {Theme.BORDER}; padding: 4px 12px; border-radius: 6px; font-weight: bold;")
        self.week_combo.currentTextChanged.connect(self._refresh_current_table)
        self.week_combo.hide()
        right_layout.addWidget(self.week_combo)
        
        # Circle Refresh Button
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedSize(36, 36)
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.setFont(ui_font(13, QFont.Weight.Bold))
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: #1F2937;
                color: #E5E7EB;
                border: 1px solid {Theme.BORDER};
                border-radius: 18px;
            }}
            QPushButton:hover {{
                background: #374151;
                color: #FFFFFF;
                border: 1px solid #4B5563;
            }}
        """)
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        right_layout.addWidget(self.refresh_btn)
        
        top_bar.addLayout(right_layout, 0, 2)
        
        # Set 3 equal columns to mathematically force center alignment of switches
        top_bar.setColumnStretch(0, 1)
        top_bar.setColumnStretch(1, 1)
        top_bar.setColumnStretch(2, 1)
        
        layout.addLayout(top_bar)
        
        # --- The Table Widget (Sleek Classic Excel Format with Gray Headers and borders) ---
        self.table = ReorderableTableWidget()
        self.table.setItemDelegate(TableHighlightDelegate(self.table))
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.SelectedClicked)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(38)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: #1B1C1E;
                alternate-background-color: #222325;
                color: #FFFFFF;
                border: 1.5px solid #37383A;
                gridline-color: #37383A;
                border-radius: 12px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
            }}
            QTableWidget QLineEdit {{
                background-color: #2E2F31;
                color: #FFFFFF;
                border: 2px solid #10B981;
                border-radius: 4px;
                padding: 0px 6px;
                margin: 0px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                min-height: 28px;
            }}
            QTableWidget::item {{
                padding: 0px 12px;
                border-bottom: 1px solid #37383A;
            }}
            QTableWidget::item:hover {{
                background-color: #2E2F31;
            }}
            QTableWidget::item:selected {{
                background-color: #374151;
                color: #FFFFFF;
                font-weight: bold;
            }}
        """)
        
        # Header Styling - Gray Classic Excel-like Headers
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setStyleSheet(f"""
            QHeaderView::section {{
                background-color: #2E2F31;
                color: #E5E7EB;
                font-weight: bold;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                padding: 10px;
                border: none;
                border-right: 1px solid #37383A;
                border-bottom: 2px solid #4B5563;
            }}
        """)
        
        # --- Summary Section for Attendance Logs ---
        self.summary_widget = QWidget()
        summary_layout = QHBoxLayout(self.summary_widget)
        summary_layout.setContentsMargins(0, 0, 0, 10)
        summary_layout.setSpacing(12)
        
        self.lbl_total_students = label("Total Students: 0", 11, Theme.TEXT_PRIMARY, QFont.Weight.Bold)
        self.lbl_avg_att = label("Average Attendance: 0%", 11, Theme.TEXT_PRIMARY, QFont.Weight.Bold)
        self.lbl_high_att = label("Highest: 0%", 11, Theme.GREEN, QFont.Weight.Bold)
        self.lbl_low_att = label("Lowest: 0%", 11, Theme.RED, QFont.Weight.Bold)
        
        for lbl in [self.lbl_total_students, self.lbl_avg_att, self.lbl_high_att, self.lbl_low_att]:
            lbl.setStyleSheet(f"background: #1B1C1E; padding: 8px 16px; border-radius: 8px; border: 1px solid {Theme.BORDER};")
            summary_layout.addWidget(lbl)
            
        summary_layout.addStretch()
        layout.addWidget(self.summary_widget)
        
        layout.addWidget(self.table)
        
        self.table.itemChanged.connect(self._on_cell_changed)
        self.table.rows_reordered.connect(self._on_rows_reordered)
        self._on_switch("attendance")
        
    def _update_switch_styles(self):
        active_style = f"QPushButton {{ background: #374151; color: #FFFFFF; border: none; border-radius: 15px; font-weight: bold; font-size: 11px; }}"
        inactive_style = f"QPushButton {{ background: transparent; color: #94A3B8; border: none; border-radius: 15px; font-weight: normal; font-size: 11px; }} QPushButton:hover {{ color: #F1F5F9; }}"
        
        self.btn_attendance.setStyleSheet(active_style if self.btn_attendance.isChecked() else inactive_style)
        self.btn_weekly.setStyleSheet(active_style if self.btn_weekly.isChecked() else inactive_style)
        self.btn_students.setStyleSheet(active_style if self.btn_students.isChecked() else inactive_style)
        
    def _on_switch(self, table_name):
        if table_name == "attendance":
            self.btn_attendance.setChecked(True)
            self.btn_weekly.setChecked(False)
            self.btn_students.setChecked(False)
            self.week_combo.show()
            self.summary_widget.show()
        elif table_name == "weekly":
            self.btn_attendance.setChecked(False)
            self.btn_weekly.setChecked(True)
            self.btn_students.setChecked(False)
            self.week_combo.show()
            self.summary_widget.hide()
        else:
            self.btn_attendance.setChecked(False)
            self.btn_weekly.setChecked(False)
            self.btn_students.setChecked(True)
            self.week_combo.hide()
            self.summary_widget.hide()
            
        self._update_switch_styles()
        self.active_table = table_name
        self._load_data(table_name)
        
    def set_missing_embeddings(self, missing_list):
        self._missing_embeddings = {roll_no: reason for roll_no, reason in missing_list}
        # If we are currently viewing the students table, refresh to apply highlight
        if hasattr(self, 'active_table') and self.active_table == "students" and not getattr(self, 'is_loading_data', False):
            # Defer slightly to avoid recursion if currently loading
            QTimer.singleShot(50, self._refresh_current_table)

    def _refresh_current_table(self):
        self._load_data(self.active_table)
        
    def _on_refresh_clicked(self):
        main_win = self.window()
        if main_win and hasattr(main_win, '_scan_for_missing_embeddings'):
            main_win._scan_for_missing_embeddings()
        else:
            self._refresh_current_table()
        
    def _load_data(self, table_name):
        self.is_loading_data = True
        
        import sqlite3
        from pathlib import Path
        db_path = Path("face_db/uniguard.db")
        if not db_path.exists():
            self.table.setRowCount(0)
            self.table.setColumnCount(1)
            self.table.setHorizontalHeaderLabels(["Status"])
            self.table.setItem(0, 0, QTableWidgetItem("Database file face_db/uniguard.db does not exist."))
            self.is_loading_data = False
            return
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if table_name == "students":
            cols = ["Roll No", "Student Name", "Gender", "Year of Study", "Target Tag"]
            
            cursor.execute("SELECT roll_no, name, gender, batch_year, tag_color FROM students ORDER BY roll_no ASC")
            rows = cursor.fetchall()
            
            self.table.setColumnCount(len(cols))
            self.table.setRowCount(len(rows))
            
            # Apply clean bold Excel headers
            for idx, col_name in enumerate(cols):
                item = QTableWidgetItem(col_name)
                item.setForeground(QColor("#E5E7EB"))
                item.setFont(ui_font(11, QFont.Weight.Bold))
                self.table.setHorizontalHeaderItem(idx, item)
                
            for row_idx, row_data in enumerate(rows):
                for col_idx, val in enumerate(row_data):
                    item_str = str(val).capitalize() if col_idx in [2, 4] else str(val)
                    cell_item = QTableWidgetItem(item_str)
                    cell_item.setFont(ui_font(11, QFont.Weight.Normal))
                    cell_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if col_idx != 1 else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    
                    # Store original Roll No in UserRole
                    if col_idx == 0:
                        cell_item.setData(Qt.ItemDataRole.UserRole, val)
                    
                    # Highlight target tag color beautifully with standard text color representations
                    if col_idx == 4:
                        tag_str = str(val).lower()
                        if tag_str == "purple":
                            cell_item.setForeground(QColor("#C084FC"))
                            cell_item.setFont(ui_font(11, QFont.Weight.Bold))
                        elif tag_str == "yellow":
                            cell_item.setForeground(QColor("#FBBF24"))
                            cell_item.setFont(ui_font(11, QFont.Weight.Bold))
                        elif tag_str == "red":
                            cell_item.setForeground(QColor("#F87171"))
                            cell_item.setFont(ui_font(11, QFont.Weight.Bold))
                        elif tag_str == "gray":
                            cell_item.setForeground(QColor("#9CA3AF"))
                            
                    # If this student is missing embeddings, highlight the cell in red and set tooltip description
                    if hasattr(self, '_missing_embeddings') and row_data[0] in self._missing_embeddings:
                        cell_item.setBackground(QBrush(QColor("#4A1D1D")))
                        cell_item.setToolTip(self._missing_embeddings[row_data[0]])
                            
                    self.table.setItem(row_idx, col_idx, cell_item)
                    
        elif table_name == "weekly":
            cols = ["Reg No", "Mon", "Tue", "Wed", "Thu", "Fri", "Total", "%"]
            
            # Map "W1" -> offset 0, "W2" -> 1, etc.
            selected_week = self.week_combo.currentText()
            week_map = {
                "W1": 0, "Week 1": 0,
                "W2": 1, "Week 2": 1,
                "W3": 2, "Week 3": 2,
                "W4": 3, "Week 4": 3
            }
            offset = week_map.get(selected_week, 0)
            
            import datetime
            year = getattr(self, 'current_selected_year', 2026)
            month = getattr(self, 'current_selected_month_idx', 5)
            
            # Find all Mondays in the selected month & year
            mondays = []
            for day in range(1, 32):
                try:
                    d = datetime.date(year, month, day)
                    if d.weekday() == 0:  # Monday
                        mondays.append(d)
                except ValueError:
                    break
            mondays.sort()
            
            # Match offset to index
            if offset < len(mondays):
                target_monday = mondays[offset]
            else:
                target_monday = mondays[-1] if mondays else datetime.date(year, month, 1)
                
            target_friday = target_monday + datetime.timedelta(days=4)
            
            start_date_str = target_monday.strftime("%Y-%m-%d 00:00:00")
            end_date_str = target_friday.strftime("%Y-%m-%d 23:59:59")
            
            # 1. Fetch all students
            cursor.execute("SELECT roll_no, name FROM students ORDER BY roll_no ASC")
            students_list = cursor.fetchall()
            
            # 2. Fetch real attendance for the calculated week
            cursor.execute("""
                SELECT s.roll_no, a.timestamp 
                FROM attendance a
                JOIN students s ON a.student_id = s.roll_no
                WHERE a.timestamp >= ? AND a.timestamp <= ? AND a.status = 'PRESENT'
                ORDER BY a.timestamp ASC
            """, (start_date_str, end_date_str))
            
            attendance_events = cursor.fetchall()
            
            # 3. Map events to days: {roll_no: {0: 'A', 1: 'A', 2: 'A', 3: 'A', 4: 'A'}}
            student_data = {s[0]: {day: 'A' for day in range(5)} for s in students_list}
            for sid, ts_str in attendance_events:
                try:
                    ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    day_idx = ts.weekday()
                    if 0 <= day_idx <= 4:
                        if sid in student_data:
                            student_data[sid][day_idx] = 'P'
                except Exception:
                    pass
            
            self.table.setColumnCount(len(cols))
            self.table.setRowCount(len(students_list))
            
            # Apply clean bold Excel headers
            for idx, col_name in enumerate(cols):
                item = QTableWidgetItem(col_name)
                item.setForeground(QColor("#E5E7EB"))
                item.setFont(ui_font(11, QFont.Weight.Bold))
                self.table.setHorizontalHeaderItem(idx, item)
                
            for row_idx, s in enumerate(students_list):
                reg_no, name = s
                days_dict = student_data.get(reg_no, {d: 'A' for d in range(5)})
                days = [days_dict[i] for i in range(5)]
                
                present = sum(1 for d in days if d == 'P')
                absent = sum(1 for d in days if d == 'A')
                total_days = present + absent
                pct = (present / total_days * 100) if total_days > 0 else 0.0
                
                formatted_row = [
                    (str(reg_no), False), # 0
                    (str(days[0]), True),  # Mon
                    (str(days[1]), True),  # Tue
                    (str(days[2]), True),  # Wed
                    (str(days[3]), True),  # Thu
                    (str(days[4]), True),  # Fri
                    (str(present), False),   # Total
                    (f"{pct:.2f}%", False) # %
                ]
                
                for col_idx, (val_str, is_day) in enumerate(formatted_row):
                    cell_item = QTableWidgetItem(val_str)
                    cell_item.setFont(ui_font(11, QFont.Weight.Normal))
                    cell_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    if col_idx == 0:
                        cell_item.setData(Qt.ItemDataRole.UserRole, reg_no)
                        
                    if not is_day:
                        cell_item.setFlags(cell_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        
                    if is_day:
                        if val_str.upper() == "P":
                            cell_item.setForeground(QColor("#34D399"))
                            cell_item.setFont(ui_font(11, QFont.Weight.Bold))
                        elif val_str.upper() == "A":
                            cell_item.setForeground(QColor("#F87171"))
                            cell_item.setFont(ui_font(11, QFont.Weight.Bold))
                            
                    self.table.setItem(row_idx, col_idx, cell_item)
                    
        else: # attendance (Overall stats for selected week)
            self.table.setSortingEnabled(False)
            cols = ["Reg No", "Student Name", "Total Present", "Total Absent", "Attendance %", "Overall Status", "Risk Level"]
            
            selected_week = self.week_combo.currentText()
            week_map = {
                "W1": 0, "Week 1": 0,
                "W2": 1, "Week 2": 1,
                "W3": 2, "Week 3": 2,
                "W4": 3, "Week 4": 3
            }
            offset = week_map.get(selected_week, 0)
            
            import datetime
            year = getattr(self, 'current_selected_year', 2026)
            month = getattr(self, 'current_selected_month_idx', 5)
            
            # Find all Mondays in the selected month & year
            mondays = []
            for day in range(1, 32):
                try:
                    d = datetime.date(year, month, day)
                    if d.weekday() == 0:  # Monday
                        mondays.append(d)
                except ValueError:
                    break
            mondays.sort()
            
            # Match offset to index
            if offset < len(mondays):
                target_monday = mondays[offset]
            else:
                target_monday = mondays[-1] if mondays else datetime.date(year, month, 1)
                
            target_friday = target_monday + datetime.timedelta(days=4)
            
            start_date_str = target_monday.strftime("%Y-%m-%d 00:00:00")
            end_date_str = target_friday.strftime("%Y-%m-%d 23:59:59")
            
            cursor.execute("SELECT roll_no, name FROM students ORDER BY roll_no ASC")
            students_list = cursor.fetchall()
            
            cursor.execute("""
                SELECT s.roll_no, a.timestamp 
                FROM attendance a
                JOIN students s ON a.student_id = s.roll_no
                WHERE a.timestamp >= ? AND a.timestamp <= ? AND a.status = 'PRESENT'
                ORDER BY a.timestamp ASC
            """, (start_date_str, end_date_str))
            
            attendance_events = cursor.fetchall()
            
            student_data = {s[0]: {day: 'A' for day in range(5)} for s in students_list}
            for sid, ts_str in attendance_events:
                try:
                    ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    day_idx = ts.weekday()
                    if 0 <= day_idx <= 4:
                        if sid in student_data:
                            student_data[sid][day_idx] = 'P'
                except Exception:
                    pass
            
            self.table.setColumnCount(len(cols))
            self.table.setRowCount(len(students_list))
            
            # Apply clean bold Excel headers
            for idx, col_name in enumerate(cols):
                item = QTableWidgetItem(col_name)
                item.setForeground(QColor("#E5E7EB"))
                item.setFont(ui_font(11, QFont.Weight.Bold))
                self.table.setHorizontalHeaderItem(idx, item)
                
            total_students = len(students_list)
            sum_att_pct = 0
            highest_pct = -1.0
            lowest_pct = 101.0
            
            for row_idx, s in enumerate(students_list):
                reg_no, name = s
                days_dict = student_data.get(reg_no, {d: 'A' for d in range(5)})
                days = [days_dict[i] for i in range(5)]
                
                present = sum(1 for d in days if str(d).upper() == 'P')
                absent = sum(1 for d in days if str(d).upper() == 'A')
                total_days = present + absent
                
                pct = (present / total_days * 100) if total_days > 0 else 0.0
                sum_att_pct += pct
                if pct > highest_pct: highest_pct = pct
                if pct < lowest_pct: lowest_pct = pct
                
                if pct >= 90.0:
                    status = "Excellent"
                    risk = "Low"
                    status_color = Theme.GREEN
                elif pct >= 75.0:
                    status = "Good"
                    risk = "Moderate"
                    status_color = Theme.BLUE
                elif pct >= 60.0:
                    status = "Warning"
                    risk = "High"
                    status_color = Theme.ORANGE
                else:
                    status = "Critical"
                    risk = "Severe"
                    status_color = Theme.RED
                    
                row_items = [
                    (str(reg_no), None),
                    (str(name), None),
                    (str(present), present),
                    (str(absent), absent),
                    (f"{pct:.1f}%", pct),
                    (status, None),
                    (risk, None)
                ]
                
                for col_idx, (text, numeric_val) in enumerate(row_items):
                    cell_item = QTableWidgetItem()
                    if numeric_val is not None:
                        # For display we want the text, but for sorting we provide the raw numeric value
                        cell_item.setData(Qt.ItemDataRole.EditRole, numeric_val)
                    if col_idx == 4: # Force text for percentage column to include '%'
                        cell_item.setText(text)
                        
                    if numeric_val is None:
                        cell_item.setText(text)
                        
                    cell_item.setFont(ui_font(11, QFont.Weight.Normal))
                    cell_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if col_idx != 1 else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    
                    # Store DB ID in col 0 UserRole
                    if col_idx == 0:
                        cell_item.setData(Qt.ItemDataRole.UserRole, reg_no)
                        
                    # Disable editing for ALL columns in this summary table
                    cell_item.setFlags(cell_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    
                    if col_idx >= 4:
                        cell_item.setForeground(QColor(status_color))
                        cell_item.setFont(ui_font(11, QFont.Weight.Bold))
                        
                    if pct < 75.0:
                        cell_item.setBackground(QColor(60, 20, 20)) # subtle red glow
                        
                    self.table.setItem(row_idx, col_idx, cell_item)
                    
            if total_students > 0:
                avg_pct = sum_att_pct / total_students
            else:
                avg_pct = 0.0
                highest_pct = 0.0
                lowest_pct = 0.0
                
            self.lbl_total_students.setText(f"Total Students: {total_students}")
            self.lbl_avg_att.setText(f"Average Attendance: {avg_pct:.1f}%")
            self.lbl_high_att.setText(f"Highest: {highest_pct:.1f}%")
            self.lbl_low_att.setText(f"Lowest: {lowest_pct:.1f}%")
            
            self.table.setSortingEnabled(True)
            
        conn.close()
        
        # Set proper column widths
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for idx in range(self.table.columnCount()):
            if table_name == "weekly":
                if idx == 0:  # Reg No
                    self.table.horizontalHeader().setSectionResizeMode(idx, QHeaderView.ResizeMode.Stretch)
                else:
                    self.table.horizontalHeader().setSectionResizeMode(idx, QHeaderView.ResizeMode.ResizeToContents)
            elif table_name == "students":
                if idx == 1:  # Name
                    self.table.horizontalHeader().setSectionResizeMode(idx, QHeaderView.ResizeMode.Stretch)
                else:
                    self.table.horizontalHeader().setSectionResizeMode(idx, QHeaderView.ResizeMode.ResizeToContents)
            else: # attendance
                if idx == 1:  # Name
                    self.table.horizontalHeader().setSectionResizeMode(idx, QHeaderView.ResizeMode.Stretch)
                else:
                    self.table.horizontalHeader().setSectionResizeMode(idx, QHeaderView.ResizeMode.ResizeToContents)
                
        self.is_loading_data = False

    def _on_cell_changed(self, item):
        if getattr(self, 'is_loading_data', False):
            return
            
        row = item.row()
        col = item.column()
        new_val = item.text().strip()
        
        id_item = self.table.item(row, 0)
        if not id_item:
            return
            
        original_db_id = id_item.data(Qt.ItemDataRole.UserRole)
        if original_db_id is None:
            original_db_id = id_item.text()
            
        import sqlite3
        from pathlib import Path
        db_path = Path("face_db/uniguard.db")
        if not db_path.exists():
            return
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            if self.active_table == "students":
                if col == 0: # Roll No
                    # Check for uniqueness constraint
                    cursor.execute("SELECT roll_no FROM students WHERE roll_no = ?", (new_val,))
                    if cursor.fetchone():
                        print(f"[DatabaseExplorerWidget] Save conflict: Roll No {new_val} is already taken!")
                        conn.close()
                        self._refresh_current_table()
                        return
                    # Update students table roll_no and cascade to attendance student_id
                    cursor.execute("UPDATE students SET roll_no = ? WHERE roll_no = ?", (new_val, original_db_id))
                    cursor.execute("UPDATE attendance SET student_id = ? WHERE student_id = ?", (new_val, original_db_id))
                elif col == 1: # Student Name
                    cursor.execute("UPDATE students SET name = ? WHERE roll_no = ?", (new_val, original_db_id))
                elif col == 2: # Gender
                    cursor.execute("UPDATE students SET gender = ? WHERE roll_no = ?", (new_val.lower(), original_db_id))
                elif col == 3: # Year
                    try:
                        val_int = int(new_val)
                        cursor.execute("UPDATE students SET batch_year = ? WHERE roll_no = ?", (val_int, original_db_id))
                    except ValueError:
                        pass
                elif col == 4: # Target Tag
                    cursor.execute("UPDATE students SET tag_color = ? WHERE roll_no = ?", (new_val.lower(), original_db_id))
            elif self.active_table == "weekly":
                if 1 <= col <= 5: # Monday (1) to Friday (5)
                    upper_val = new_val.upper()
                    if upper_val not in ["P", "A"]:
                        upper_val = "A"
                        
                    # Calculate date for the changed day
                    selected_week = self.week_combo.currentText()
                    week_map = {
                        "W1": 0, "Week 1": 0,
                        "W2": 1, "Week 2": 1,
                        "W3": 2, "Week 3": 2,
                        "W4": 3, "Week 4": 3
                    }
                    offset = week_map.get(selected_week, 0)
                    
                    import datetime
                    year = getattr(self, 'current_selected_year', 2026)
                    month = getattr(self, 'current_selected_month_idx', 5)
                    
                    mondays = []
                    for day in range(1, 32):
                        try:
                            d = datetime.date(year, month, day)
                            if d.weekday() == 0:  # Monday
                                mondays.append(d)
                        except ValueError:
                            break
                    mondays.sort()
                    
                    if offset < len(mondays):
                        target_monday = mondays[offset]
                    else:
                        target_monday = mondays[-1] if mondays else datetime.date(year, month, 1)
                        
                    target_date = target_monday + datetime.timedelta(days=col - 1)
                    target_date_str = target_date.strftime("%Y-%m-%d")
                    
                    # 1. Update dynamic 'attendance' table
                    if upper_val == "P":
                        cursor.execute("SELECT id FROM attendance WHERE student_id = ? AND date(timestamp) = ?", (original_db_id, target_date_str))
                        existing_att = cursor.fetchone()
                        if existing_att:
                            cursor.execute("UPDATE attendance SET status = 'PRESENT', violation_type = NULL WHERE id = ?", (existing_att[0],))
                        else:
                            cursor.execute("SELECT name FROM students WHERE roll_no = ?", (original_db_id,))
                            st_info = cursor.fetchone()
                            s_name = st_info[0] if st_info else ""
                            timestamp_str = f"{target_date_str} 09:00:00"
                            cursor.execute("""
                                INSERT INTO attendance (student_id, roll_no, name, timestamp, status, tag_color_verified, violation_type)
                                VALUES (?, ?, ?, ?, 'PRESENT', 'purple', NULL)
                            """, (original_db_id, original_db_id, s_name, timestamp_str))
                    else: # "A"
                        cursor.execute("DELETE FROM attendance WHERE student_id = ? AND date(timestamp) = ?", (original_db_id, target_date_str))
                        
                    # 2. Sync with 'weekly_attendance' table
                    roll_no = original_db_id
                    week_name = f"Week {offset + 1}"
                    cols_map = {1: "MON", 2: "TUE", 3: "WED", 4: "THU", 5: "FRI", 6: "SAT", 7: "SUN"}
                    db_col_name = cols_map.get(col, "MON")
                    
                    cursor.execute("SELECT id FROM weekly_attendance WHERE register_number = ? AND week_name = ?", (roll_no, week_name))
                    w_row = cursor.fetchone()
                    if w_row:
                        cursor.execute(f"UPDATE weekly_attendance SET {db_col_name} = ? WHERE id = ?", (upper_val, w_row[0]))
                        # Recalculate total_present and attendance_percentage
                        cursor.execute("SELECT MON, TUE, WED, THU, FRI, SAT, SUN FROM weekly_attendance WHERE id = ?", (w_row[0],))
                        day_row = cursor.fetchone()
                        if day_row:
                            present_count = sum(1 for d in day_row if str(d).upper() == "P")
                            att_pct = round(present_count / 7 * 100, 2)
                            cursor.execute("UPDATE weekly_attendance SET total_present = ?, attendance_percentage = ? WHERE id = ?", (present_count, att_pct, w_row[0]))
                    else:
                        days_vals = ["A"] * 7
                        days_vals[col - 1] = upper_val
                        present_count = 1 if upper_val == "P" else 0
                        att_pct = round(present_count / 7 * 100, 2)
                        cursor.execute("""
                            INSERT INTO weekly_attendance (register_number, week_name, MON, TUE, WED, THU, FRI, SAT, SUN, total_present, attendance_percentage)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (roll_no, week_name, *days_vals, present_count, att_pct))
            else: # attendance
                pass
            conn.commit()
        except Exception as e:
            print(f"[DatabaseExplorerWidget] Save error: {e}")
        finally:
            conn.close()
            
        # Trigger the scan for missing embeddings on the main dashboard window to update alert states and highlights
        main_win = self.window()
        if main_win and hasattr(main_win, '_scan_for_missing_embeddings'):
            QTimer.singleShot(100, main_win._scan_for_missing_embeddings)
        else:
            QTimer.singleShot(100, self._refresh_current_table)

    def _reorder_students_by_roll_no(self, cursor):
        pass

    def _on_add_student_clicked(self):
        if self.active_table != "students":
            self._on_switch("students")
            
        import sqlite3
        import random
        from pathlib import Path
        db_path = Path("face_db/uniguard.db")
        if not db_path.exists():
            return
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            rand_roll = f"4311{random.randint(1000, 9999)}"
            cursor.execute("""
                INSERT INTO students (roll_no, name, gender, batch_year, tag_color)
                VALUES (?, ?, ?, ?, ?)
            """, (rand_roll, "New Student Name", "male", 3, "purple"))
            
            conn.commit()
        except Exception as e:
            print(f"[DatabaseExplorerWidget] SQLite add student error: {e}")
        finally:
            conn.close()
            
        # Trigger the scan for missing embeddings on the main dashboard window
        main_win = self.window()
        if main_win and hasattr(main_win, '_scan_for_missing_embeddings'):
            main_win._scan_for_missing_embeddings()
        else:
            self._refresh_current_table()

    def _show_context_menu(self, pos):
        if hasattr(self, 'is_student_role') and self.is_student_role:
            return
        item = self.table.itemAt(pos)
        if not item:
            return
            
        row = item.row()
        id_item = self.table.item(row, 0)
        if not id_item:
            return
        record_id = id_item.text()
        
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: #1F2937;
                color: #F3F4F6;
                border: 1px solid #374151;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial;
                font-size: 11px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: #E11D48;
                color: #FFFFFF;
                font-weight: bold;
            }}
        """)
        
        delete_action_text = "Delete Student Profile" if self.active_table == "students" else "Delete Attendance Record"
        delete_action = menu.addAction(delete_action_text)
        
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == delete_action:
            self._delete_record(self.active_table, record_id)

    def _delete_record(self, table_name, record_id):
        import sqlite3
        from pathlib import Path
        db_path = Path("face_db/uniguard.db")
        if not db_path.exists():
            return
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            if table_name == "students":
                cursor.execute("DELETE FROM students WHERE roll_no = ?", (record_id,))
                cursor.execute("DELETE FROM attendance WHERE student_id = ?", (record_id,))
            else:
                cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
            conn.commit()
        except Exception as e:
            print(f"[DatabaseExplorerWidget] Delete error: {e}")
        finally:
            conn.close()
            
        self._refresh_current_table()

    def _on_rows_reordered(self):
        pass

class UniGuardDashboard(QMainWindow):
    def __init__(self, model_path=None):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowSystemMenuHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("UniGuard — Uniform Compliance Dashboard")
        self.setMinimumSize(1200, 780)
        self.resize(1340, 860)
        
        # Dynamically resolve model path if not passed or does not exist
        import os
        if model_path is None or not os.path.exists(model_path):
            model_path = get_default_model_path()
        self.model_path = model_path
        
        # Initialize seed statistics from SQLite database
        self.total_scans = 0
        self.total_permitted = 0
        self.total_violations = 0
        
        try:
            db_path = os.path.join(os.getcwd(), "face_db", "uniguard.db")
            if os.path.exists(db_path):
                import sqlite3
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                
                c.execute("SELECT COUNT() FROM attendance")
                self.total_scans = c.fetchone()[0] or 0
                
                c.execute("SELECT COUNT() FROM attendance WHERE status = 'PRESENT'")
                self.total_permitted = c.fetchone()[0] or 0
                
                c.execute("SELECT COUNT() FROM attendance WHERE status != 'PRESENT'")
                self.total_violations = c.fetchone()[0] or 0
                
                conn.close()
        except Exception as e:
            print(f"Error loading initial sticky note stats: {e}")
        self.last_active_tab = "Live Feed"
        
        # Load persistent admin password
        self.admin_password = "Admin@SIST"
        self._load_admin_password()
        
        self._setup_ui()
        self._setup_logic()

    def _get_attendance_counts(self):
        presentees = 0
        absentees = 0
        try:
            import os
            import sqlite3
            db_path = os.path.join(os.getcwd(), "face_db", "uniguard.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                
                # Total registered students
                c.execute("SELECT COUNT() FROM students")
                total_students = c.fetchone()[0] or 0
                
                # Present today
                today_str = datetime.now().strftime('%Y-%m-%d')
                c.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE status = 'PRESENT' AND date(timestamp) = ?", (today_str,))
                presentees = c.fetchone()[0] or 0
                
                absentees = max(0, total_students - presentees)
                conn.close()
        except Exception as e:
            print(f"Error querying attendance counts: {e}")
        return presentees, absentees

    def _auto_generate_embedding(self, roll_no, name, gender, year, tag_color):
        import cv2
        from pathlib import Path
        try:
            from facial_features.facial_pipeline import FacePipeline
            from facial_features.facial_database import FaceDatabase
        except Exception as e:
            return False, f"Import error: {str(e)}"
        
        photo_dir = Path("student_photos")
        photo_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            p = photo_dir / f"{roll_no}{ext}"
            if p.exists():
                photo_path = p
                break
        
        if not photo_path:
            return False, f"Facial profile photo '{roll_no}.jpg' is missing in the 'student_photos' directory."
            
        try:
            img = cv2.imread(str(photo_path))
            if img is None:
                return False, f"Unable to read image file '{photo_path.name}' in the 'student_photos' directory."
                
            if not hasattr(self, '_face_pipeline') or self._face_pipeline is None:
                self._face_pipeline = FacePipeline()
                
            if self._face_pipeline.app is None:
                return False, "InsightFace model not initialized"
                
            faces = self._face_pipeline.app.get(img)
            if not faces:
                return False, f"No face detected in profile photo '{photo_path.name}'. Ensure a face is clearly visible."
                
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            
            face_db = FaceDatabase()
            face_db.register_face(
                roll_no=roll_no,
                name=name,
                gender=gender,
                year=year,
                tag_color=tag_color,
                embedding=face.normed_embedding,
                face_image=img
            )
            print(f"[AutoEmbed] Successfully generated and registered embedding for {name} ({roll_no})")
            return True, "Success"
        except Exception as e:
            print(f"[AutoEmbed] Error generating embedding for {roll_no}: {e}")
            return False, f"Generation error: {str(e)}"

    def _scan_for_missing_embeddings(self):
        missing = self._get_missing_embeddings_students()
        if hasattr(self, 'sidebar') and self.sidebar and hasattr(self.sidebar, 'config_nav_btn') and self.sidebar.config_nav_btn:
            self.sidebar.config_nav_btn.update_alert_state(False)
            
        if hasattr(self, 'db_explorer') and self.db_explorer:
            self.db_explorer.set_missing_embeddings(missing)

    def _get_missing_embeddings_students(self):
        missing_students = []
        try:
            import os
            import sqlite3
            import numpy as np
            from pathlib import Path
            
            db_path = Path("face_db/uniguard.db")
            ref_dir = Path("face_db/reference_faces")
            
            # Reload embeddings
            from facial_features.facial_database import FaceDatabase
            fdb = FaceDatabase()
            embeddings_db = fdb.embeddings_db
            
            if db_path.exists():
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                c.execute("SELECT roll_no, name, gender, batch_year, tag_color FROM students")
                students = c.fetchall()
                conn.close()
                
                for roll_no, name, gender, year, tag_color in students:
                    has_emb = (roll_no in embeddings_db and len(embeddings_db[roll_no]) > 0)
                    
                    person_dir = ref_dir / roll_no
                    has_img = False
                    if person_dir.exists():
                        images = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')) + list(person_dir.glob('*.jpeg'))
                        if len(images) > 0:
                            has_img = True
                            
                    if not has_emb or not has_img:
                        # Attempt auto-generation if photo exists in student_photos
                        success, reason = self._auto_generate_embedding(roll_no, name, gender, year, tag_color)
                        if success:
                            # Re-load embeddings and verify
                            fdb = FaceDatabase()
                            embeddings_db = fdb.embeddings_db
                            has_emb = True
                            has_img = True
                        else:
                            missing_students.append((roll_no, reason))
        except Exception as e:
            print(f"Error checking missing embeddings: {e}")
        return missing_students

    def _reposition_sticky_note(self):
        if hasattr(self, 'sticky_note') and hasattr(self, 'inner_container'):
            if not self.sticky_note._has_been_dragged:
                geom = self.inner_container.geometry()
                # On initial boot, layouts might not be finalized. Avoid placing it outside!
                if geom.x() > 0:
                    x = geom.x() - self.sticky_note.width() / 2 + 10
                    y = geom.y() + 380
                    self.sticky_note.move(int(x), int(y))
                    
    def _reposition_tag_panel(self):
        if hasattr(self, 'tag_panel') and hasattr(self, 'inner_container'):
            geom = self.inner_container.geometry()
            if geom.x() > 0:
                # Place it floating beautifully on top of the dashboard feed
                x = geom.x() + 180
                y = geom.y() + 80
                self.tag_panel.move(int(x), int(y))

        # Update repositioning for floating shoes, ID badge, and DockBar
        if hasattr(self, 'inner_container'):
            width_inner = self.inner_container.width()
            height_inner = self.inner_container.height()
            
            if width_inner > 0 and height_inner > 0:
                # 1. DockBar Positioning: Center at the bottom of inner_container
                if hasattr(self, 'dock'):
                    self.dock.setGeometry(0, height_inner - 88 - 24, width_inner, 88)
                    self.dock.raise_()
                    if hasattr(self, 'sidebar'):
                        self.sidebar.raise_()
                
        # 2. Decorative PNGs Positioning (Relative to central/entire window)
        if hasattr(self, 'shoes_deco') and hasattr(self, 'id_badge_deco') and hasattr(self, 'centralWidget') and self.centralWidget():
            width = self.centralWidget().width()
            if width > 0:
                # Position ID badge hanging near the top-right of the entire window (unclipped)
                self.id_badge_deco.move(int(width - 190), int(0))
                
                # Position yellow shoes sitting near the top-right of the entire window (unclipped)
                self.shoes_deco.move(int(width - 280), int(-5))
                
                # Hide decos if focus mode is active, otherwise raise and display them
                if hasattr(self, 'overlay') and self.overlay.isVisible():
                    self.shoes_deco.hide()
                    self.id_badge_deco.hide()
                else:
                    if hasattr(self, 'last_active_tab') and self.last_active_tab in ["Live Feed", "Attendance"]:
                        self.shoes_deco.show()
                        self.id_badge_deco.show()
                        self.shoes_deco.raise_()
                        self.id_badge_deco.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_sticky_note()
        self._reposition_tag_panel()

    def showEvent(self, event):
        super().showEvent(event)
        # Force a delayed reposition after Qt has completed initial window layout calculations
        QTimer.singleShot(150, self._reposition_sticky_note)
        QTimer.singleShot(150, self._reposition_tag_panel)
        QTimer.singleShot(200, lambda: self._apply_user_profile("Unauthorized"))
        QTimer.singleShot(200, self._scan_for_missing_embeddings)

    def _setup_ui(self):
        central = DotBackgroundWidget()
        self.setCentralWidget(central)
        
        self.sticky_note = StickyNoteWidget(central)
        pres, abs_ = self._get_attendance_counts()
        self.sticky_note.update_stats(self.total_scans, self.total_permitted, self.total_violations, pres, abs_)
        self.tag_panel = TagDetectionPanel(central)
        
        root_v = QVBoxLayout(central)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)
        
        self.title_bar = MacTitleBar(central)
        root_v.addWidget(self.title_bar)
        
        main_content = QVBoxLayout()
        main_content.setContentsMargins(160, 26, 160, 64)

        self.inner_container = InnerContainerWidget()
        self.inner_container.setObjectName("InnerContainer")
        self.inner_container.setStyleSheet(f"QWidget#InnerContainer {{ background: {Theme.BG_DEEP}; border: 6px solid #9298a5; border-radius: 24px; }}")
        
        self.shoes_deco = ShoesDecoWidget(central)
        self.id_badge_deco = IDBadgeDecoWidget(central)
        self.shoes_deco.show()
        self.id_badge_deco.show()
        
        inner_layout = QVBoxLayout(self.inner_container)
        inner_layout.setContentsMargins(0, 0, 0, 0)

        main_h = QHBoxLayout()
        self.sidebar = SidebarWidget()
        self.sidebar.sidebar_export_btn.clicked.connect(self._export_analytics_screenshot)
        self.sidebar.switch_btn.clicked.connect(self._on_switch_clicked)
        self.sidebar.login_btn.clicked.connect(self._on_switch_clicked)
        self.sidebar.logout_btn.clicked.connect(self._on_logout_clicked)
        main_h.addWidget(self.sidebar)

        # Create Stacked Widget for page switching
        self.main_stack = QStackedWidget()
        
        # --- Page 0: Main Dashboard Page (Live Feed) ---
        self.dashboard_page = QWidget()
        self.dashboard_page.setObjectName("DashboardPage")
        self.dashboard_page.setStyleSheet("background: transparent;")
        
        cw_layout = QHBoxLayout(self.dashboard_page)
        cw_layout.setContentsMargins(12, 40, 32, 40)
        
        self.cam_card = Card()
        cam_layout = QVBoxLayout(self.cam_card)
        cam_layout.setContentsMargins(0,0,0,0)
        self.cam_header = CameraHeaderWidget()
        self.cam_feed = CameraFeedWidget()
        cam_layout.addWidget(self.cam_header)
        cam_layout.addWidget(self.cam_feed, 1)
        
        center_v = QVBoxLayout()
        center_v.setContentsMargins(0, 0, 0, 0)
        center_v.addWidget(self.cam_card, 1)

        self.dock = DockBar(self.inner_container)
        self.dock.panel_toggled.connect(self._on_panel_toggle)
        self.dock.show()
        center_v.addSpacerItem(QSpacerItem(0, 84, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        cw_layout.addLayout(center_v, 1)

        self.right_panel = RightPanel()
        scroll_right = QScrollArea()
        scroll_right.setWidgetResizable(True)
        scroll_right.setWidget(self.right_panel)
        scroll_right.setFixedWidth(304)
        scroll_right.setStyleSheet("QScrollArea { border: none; background: transparent; } QScrollBar:vertical { background: #0A0C10; width: 4px; } QScrollBar::handle:vertical { background: #2A3550; border-radius: 2px; }")
        
        right_v = QVBoxLayout()
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.addWidget(scroll_right, 1)
        right_v.addSpacerItem(QSpacerItem(0, 84, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        
        cw_layout.addLayout(right_v)
        
        self.main_stack.addWidget(self.dashboard_page)
        
        # --- Page 1: Database Explorer Page ---
        self.db_explorer = DatabaseExplorerWidget()
        self.main_stack.addWidget(self.db_explorer)
        
        # --- Page 2: Analytics Page ---
        self.analytics_page = AnalyticsWidget()
        self.sidebar.analytics_page = self.analytics_page
        self.main_stack.addWidget(self.analytics_page)

        main_h.addWidget(self.main_stack, 1)
        inner_layout.addLayout(main_h, 1)
        
        self.overlay = OverlayWidget(self.inner_container)
        self.inner_container.set_overlay(self.overlay)
        
        main_content.addWidget(self.inner_container, 1)
        root_v.addLayout(main_content, 1)

    def _load_admin_password(self):
        import json
        import os
        config_path = os.path.join(str(PROJECT_ROOT), "admin_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.admin_password = data.get("admin_password", "Admin@SIST")
            except Exception as e:
                print(f"Error loading admin password: {e}")

    def _save_admin_password(self, new_password):
        import json
        import os
        self.admin_password = new_password
        config_path = os.path.join(str(PROJECT_ROOT), "admin_config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"admin_password": new_password}, f)
        except Exception as e:
            print(f"Error saving admin password: {e}")

    def _setup_logic(self):
        # Navigation logic
        self.sidebar.nav_changed.connect(self._on_nav_changed)
        self.sidebar.config_cancelled.connect(self._on_config_cancelled)
        self.sidebar.config_database_changed.connect(self._on_config_database_changed)
        self.db_explorer.back_clicked.connect(self._on_db_back)

        # AI Worker (Initialized in dormant standby; starts only after login)
        self.worker = VideoWorker(self.model_path, "0")
        self.worker.frame_ready.connect(self.cam_feed.update_frame)
        self.worker.data_ready.connect(self._on_data_ready)
        self.worker.error_occurred.connect(self._on_video_error)

        self.cam_header.source_changed.connect(self._change_source)
        self.cam_header.play_pause_toggled.connect(self._on_play_pause_toggled)
        self.right_panel.tags.tag_selected.connect(self._on_tag_selected)

    def _on_tag_selected(self, tag):
        setattr(self.worker, 'target_tag', tag)
        self._log_audit_action(f"Changed target compliance tag to: {tag}")
        import datetime
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.right_panel.event_log.add_event(tag, f"Target Tag Color set to {tag}", now, f"System updated to track {tag.lower()} id cards.")
        
        # Dynamic Target text update for the hand-drawn sticky note
        tag_cap = tag.capitalize()
        if hasattr(self, 'sticky_note'):
            self.sticky_note.update_target(tag_cap)

        # Metrics Timer
        self.metric_timer = QTimer()
        self.metric_timer.timeout.connect(self._update_metrics)
        self.metric_timer.start(2000)

        # Graph Timer
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self._update_graph)
        self.graph_timer.start(30000)

    def _export_analytics_screenshot(self):
        if hasattr(self, 'analytics_page') and self.analytics_page:
            pixmap = self.analytics_page.grab()
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Analytics Report",
                "Analytics_Report.png",
                "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)"
            )
            
            if file_path:
                success = pixmap.save(file_path)
                if success:
                    QMessageBox.information(self, "Export Successful", f"Analytics report successfully exported to:\n{file_path}")
                else:
                    QMessageBox.warning(self, "Export Failed", "Failed to save the analytics report screenshot.")

    def _on_switch_clicked(self, can_close=True):
        # Determine if we are logged out (Unauthorized)
        is_unauth = (self.sidebar.username_lbl.text() == "Unauthorized") if hasattr(self, 'sidebar') else False
        if is_unauth:
            can_close = False
            
        dialog = SwitchUserDialog(can_close=can_close, parent=self)
        if not can_close:
            # Re-implement reject as a no-op so they cannot escape!
            dialog.reject = lambda: None
                
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.role:
                username_text = f"{dialog.role} User"
                self.sidebar.username_lbl.setText(username_text)
                self.sidebar.login_btn.hide()
                self.sidebar.actions_row.show()
                self._apply_user_profile(dialog.role)

    def _on_logout_clicked(self):
        if hasattr(self.sidebar, 'logout_btn') and self.sidebar.logout_btn.text() == "Modify":
            dialog = ModifyPasswordDialog(parent=self)
            dialog.exec()
            return

        self._log_audit_action("User initiated logout")
        self.sidebar.username_lbl.setText("Unauthorized")
        self.sidebar.actions_row.hide()
        self.sidebar.login_btn.show()
        self._apply_user_profile("Unauthorized")

    def _apply_user_profile(self, role):
        # 1. Sidebar Nav items locks
        if hasattr(self, 'sidebar') and hasattr(self.sidebar, '_nav_group'):
            for btn in self.sidebar._nav_group:
                btn.show() # Always keep all buttons visible!
                if role == "Admin":
                    btn.setEnabled(True)
                elif role == "Student":
                    btn.setEnabled(btn.text() in ["Live Feed", "Database", "Tag Detection"])
                else:  # Unauthorized
                    btn.setEnabled(False)
                btn.update() # Force visual repaint to apply grayed-out/opacity states immediately
                
            # Hide/Show/Morph logout button & divider based on active profile
            if hasattr(self.sidebar, 'logout_btn') and hasattr(self.sidebar, 'divider'):
                if role == "Student":
                    self.sidebar.logout_btn.hide()
                    self.sidebar.divider.hide()
                elif role == "Admin":
                    self.sidebar.logout_btn.setText("Modify")
                    self.sidebar.logout_btn.show()
                    self.sidebar.divider.show()
                else:
                    self.sidebar.logout_btn.setText("Logout")
                    self.sidebar.logout_btn.show()
                    self.sidebar.divider.show()
                    
        # 2. Database screen edit / add filters lock
        if hasattr(self, 'db_explorer') and self.db_explorer:
            self.db_explorer.set_role(role)
            
        # 3. Target tags panel lock (grays out Target Tags with OpacityEffect for student & unauthorized)
        if hasattr(self, 'right_panel') and self.right_panel and hasattr(self.right_panel, 'tags'):
            self.right_panel.tags.show() # Keep always visible
            self.right_panel.tags.setEnabled(role == "Admin")
            if role == "Admin":
                self.right_panel.tags.setGraphicsEffect(None) # Fully bright
            else:
                from PyQt6.QtWidgets import QGraphicsOpacityEffect
                opacity = QGraphicsOpacityEffect(self.right_panel.tags)
                opacity.setOpacity(0.4) # Beautiful 40% translucent grayed-out look
                self.right_panel.tags.setGraphicsEffect(opacity)
            
        # 4. Authenticated Worker Control
        if role == "Unauthorized":
            if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
                self.worker.stop()
            if hasattr(self, 'cam_feed') and self.cam_feed:
                self.cam_feed.image_label.setText("On Standby.")
                self.cam_feed.image_label.setStyleSheet("color: #6B7A99; font-size: 14px; font-weight: bold; background: #0D0F14; padding: 20px; text-align: center; border-radius: 12px;")
            
            # Auto-trigger login popup
            QTimer.singleShot(100, lambda: self._on_switch_clicked(can_close=False))
        else:
            if hasattr(self, 'worker') and self.worker and not self.worker.isRunning():
                self.worker.start()

    def _log_audit_action(self, action_details):
        import datetime
        import os
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        username = self.sidebar.username_lbl.text() if hasattr(self, 'sidebar') else "Unknown"
        
        log_entry = f"[{timestamp}] User: {username} | Action: {action_details}\n"
        
        try:
            log_path = os.path.join(str(PROJECT_ROOT), "system_audit.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing audit log: {e}")

    def _on_play_pause_toggled(self, is_paused):
        if hasattr(self, 'worker') and self.worker:
            self.worker.set_paused(is_paused)

    def _change_source(self, source):
        self.worker.stop()
        # Reset the play/pause button state to playing (False)
        self.cam_header.play_pause_btn.blockSignals(True)
        self.cam_header.play_pause_btn.setChecked(False)
        self.cam_header.play_pause_btn.blockSignals(False)
        
        self.worker = VideoWorker(self.model_path, source)
        self.worker.frame_ready.connect(self.cam_feed.update_frame)
        self.worker.data_ready.connect(self._on_data_ready)
        self.worker.error_occurred.connect(self._on_video_error)
        self.worker.start()

    def _on_video_error(self, err_msg):
        self.cam_feed.image_label.setText(err_msg)
        self.cam_feed.image_label.setStyleSheet(f"color: {Theme.RED}; font-size: 14px; background: #0D0F14; padding: 20px;")

    def _on_panel_toggle(self, panel_name, visible):
        if panel_name == "focus":
            if visible:
                self.overlay.show()
                self.overlay.raise_()
                self.dock.raise_()
                
                # Dim external overlay items with beautiful 30% opacity!
                from PyQt6.QtWidgets import QGraphicsOpacityEffect
                for widget_name in ['sticky_note', 'shoes_deco', 'id_badge_deco', 'tag_panel']:
                    if hasattr(self, widget_name):
                        widget = getattr(self, widget_name)
                        if widget:
                            eff = QGraphicsOpacityEffect(widget)
                            eff.setOpacity(0.3)
                            widget.setGraphicsEffect(eff)
            else:
                self.overlay.hide()
                if hasattr(self, 'sidebar'):
                    self.sidebar.raise_()
                
                # Restore external overlay items to full brightness!
                for widget_name in ['sticky_note', 'shoes_deco', 'id_badge_deco', 'tag_panel']:
                    if hasattr(self, widget_name):
                        widget = getattr(self, widget_name)
                        if widget:
                            widget.setGraphicsEffect(None)
            return

        if panel_name == "feed":
            self.cam_card.setEnabled(visible)
        else:
            self.right_panel.set_panel_enabled(panel_name, visible)

    def _on_nav_changed(self, name):
        if name == "Tag Detection":
            if hasattr(self, 'tag_panel'):
                self._reposition_tag_panel()
                self.tag_panel.show()
                self.tag_panel.raise_()
            self.sidebar.select_tab(self.last_active_tab)
            return

        if name == "Database":
            self.last_active_tab = "Database"
            if hasattr(self, 'worker') and self.worker:
                self.worker.set_paused(True)
            self.sticky_note.hide()
            if hasattr(self, 'tag_panel'):
                self.tag_panel.hide()
            self.shoes_deco.hide()
            self.id_badge_deco.hide()
            self.dock.hide()
            self.sidebar.hide()
            self.main_stack.setCurrentIndex(1)
            self.db_explorer._refresh_current_table()
        elif name == "Attendance":
            self.last_active_tab = "Attendance"
            if hasattr(self, 'worker') and self.worker:
                # Clear DatabaseManager student cache to propagate database edits/additions
                if hasattr(self.worker, 'pipeline') and self.worker.pipeline:
                    if hasattr(self.worker.pipeline, 'face_db') and self.worker.pipeline.face_db:
                        if hasattr(self.worker.pipeline.face_db, 'sql_db') and self.worker.pipeline.face_db.sql_db:
                            self.worker.pipeline.face_db.sql_db.clear_student_cache()
                is_btn_paused = self.cam_header.play_pause_btn.isChecked() if hasattr(self, 'cam_header') and hasattr(self.cam_header, 'play_pause_btn') else False
                self.worker.set_paused(is_btn_paused)
            self.sidebar.show()
            self.main_stack.setCurrentIndex(0)
            self.shoes_deco.show()
            self.id_badge_deco.show()
            self.dock.show()
            self.sticky_note.show()
            self.sticky_note.raise_()
            if hasattr(self, 'tag_panel') and self.tag_panel.isVisible():
                self.tag_panel.raise_()
            self._reposition_sticky_note()
        elif name == "Analytics":
            self.last_active_tab = "Analytics"
            if hasattr(self, 'worker') and self.worker:
                self.worker.set_paused(True)
            self.sidebar.show()
            self.sticky_note.hide()
            if hasattr(self, 'tag_panel'):
                self.tag_panel.hide()
            self.shoes_deco.hide()
            self.id_badge_deco.hide()
            self.dock.hide()
            self.analytics_page.refresh_data()
            self.main_stack.setCurrentIndex(2)
        elif name == "Configuration":
            # Sidebar expands config_panel inline; wait for config_database_changed or config_cancelled signals.
            pass
        else:
            self.last_active_tab = name
            if hasattr(self, 'worker') and self.worker:
                is_btn_paused = self.cam_header.play_pause_btn.isChecked() if hasattr(self, 'cam_header') and hasattr(self.cam_header, 'play_pause_btn') else False
                self.worker.set_paused(is_btn_paused)
            self.sidebar.show()
            self.main_stack.setCurrentIndex(0)
            self.sticky_note.hide()
            self.shoes_deco.show()
            self.id_badge_deco.show()
            self.dock.show()
            self.shoes_deco.raise_()
            self.id_badge_deco.raise_()

    def _on_config_cancelled(self):
        # Cancelled from the sidebar collapsible! Restore previous active sidebar option.
        self.sidebar.select_tab(self.last_active_tab)
        
    def _on_config_database_changed(self, file_path):
        self._log_audit_action(f"Imported/Modified student configuration database from: {file_path}")
        import pandas as pd
        import sqlite3
        from pathlib import Path
        
        try:
            # Read file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
                
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Identify columns
            col_mapping = {}
            for possible_name in ['roll_no', 'rollno', 'roll', 'registration_no', 'roll_number']:
                if possible_name in df.columns:
                    col_mapping['roll_no'] = possible_name
                    break
                    
            for possible_name in ['name', 'student_name', 'student name', 'full_name', 'fullname']:
                if possible_name in df.columns:
                    col_mapping['name'] = possible_name
                    break
                    
            for possible_name in ['gender', 'sex']:
                if possible_name in df.columns:
                    col_mapping['gender'] = possible_name
                    break
                    
            for possible_name in ['year', 'batch_year', 'year of study', 'year_of_study', 'batch', 'study_year']:
                if possible_name in df.columns:
                    col_mapping['batch_year'] = possible_name
                    break
                    
            for possible_name in ['tag_color', 'tag color', 'tag', 'color', 'target_tag', 'tagcolor']:
                if possible_name in df.columns:
                    col_mapping['tag_color'] = possible_name
                    break
                    
            # Check for critical columns
            if 'roll_no' not in col_mapping or 'name' not in col_mapping:
                QMessageBox.critical(
                    self, "Import Error", 
                    "The selected file must contain 'roll_no' (or Roll No) and 'name' (or Student Name) columns."
                )
                self.sidebar.select_tab(self.last_active_tab)
                return
                
            # Connect to SQLite
            db_path = Path("face_db/uniguard.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Start database operations
            cursor.execute("PRAGMA foreign_keys = OFF")
            cursor.execute("DELETE FROM attendance")
            cursor.execute("DELETE FROM students")
            
            for idx, row in df.iterrows():
                roll_no = str(row.get(col_mapping.get('roll_no'), f"REG_{1000+idx}"))
                name = str(row.get(col_mapping.get('name'), "Unknown Student"))
                gender = str(row.get(col_mapping.get('gender'), "male")).lower()
                
                # Safe parse batch_year
                try:
                    batch_year = int(row.get(col_mapping.get('batch_year'), 3))
                except:
                    batch_year = 3
                    
                tag_color = str(row.get(col_mapping.get('tag_color'), "purple")).lower()
                
                cursor.execute("""
                    INSERT INTO students (roll_no, name, gender, batch_year, tag_color)
                    VALUES (?, ?, ?, ?, ?)
                """, (roll_no, name, gender, batch_year, tag_color))
                
            conn.commit()
            cursor.execute("PRAGMA foreign_keys = ON")
            conn.close()
            
            # Re-scan for missing embeddings
            self._scan_for_missing_embeddings()
            
            # Load new dataset visual records on Database page and focus Database tab
            self.db_explorer._refresh_current_table()
            self.sidebar.select_tab("Database")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"An error occurred while importing: {str(e)}")
            self.sidebar.select_tab(self.last_active_tab)

    def _on_db_back(self):
        self.sidebar.show()
        self.sidebar.select_tab("Live Feed")

    def _update_metrics(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        try: gpu = psutil.sensors_temperatures().get('coretemp', [{}])[0].current
        except: gpu = random.uniform(30, 50)
        self.right_panel.metrics.update_metrics(cpu, mem, gpu)

    def _update_graph(self):
        self.right_panel.confidence.update_data(self._current_conf)

    def _on_data_ready(self, data):
        # Update floating tag spectrum analyzer if active
        if hasattr(self, 'tag_panel') and self.tag_panel.isVisible():
            self.tag_panel.update_data(data.get('color_debug', None))

        # Update logs (Violations)
        for v in data.get('violations', []):
            name = v['identity'].get('name', 'Unknown')
            v_types = ", ".join(v['violation_type'])
            t = datetime.now().strftime("%H:%M:%S")
            self.right_panel.event_log.add_event("critical", f"{name} Marked Absent - {v_types}", t)

        # Update logs (Permitted Scans / Successes)
        for p in data.get('permitted', []):
            name = p['identity'].get('name', 'Unknown')
            t = datetime.now().strftime("%H:%M:%S")
            self.right_panel.event_log.add_event("permitted", f"{name} Marked Present", t)
            
        # Update Profile
        comp = data.get('compliance', [])
        if comp:
            last = comp[-1]
            identity = last['identity']
            name = identity.get('name', 'Unknown')
            if name not in ["Searching...", "Unknown", "unknown", ""]:
                name_parts = name.split()
                first = name_parts[0]
                last_n = name_parts[1] if len(name_parts) > 1 else ""
                person_id = identity.get('person_id', None)
                self.right_panel.entity.update_info(first, last_n, last['compliant'], last['confidence'], person_id)
            
            # Record average confidence for graph
            self._current_conf = sum(c['confidence'] for c in comp) / len(comp) * 100

        # Update stats exactly matching real SQL log writes to avoid rapid increment spam
        violations = data.get('violations', [])
        permitted = data.get('permitted', [])
        if violations or permitted:
            self.total_scans += len(violations) + len(permitted)
            self.total_permitted += len(permitted)
            self.total_violations += len(violations)
            pres, abs_ = self._get_attendance_counts()
            self.sticky_note.update_stats(self.total_scans, self.total_permitted, self.total_violations, pres, abs_)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

def get_default_model_path():
    import os
    candidates = [
        "runs/detect/train/weights/best.pt",
        "yolov8n.pt",
        "models/cayolo.pt"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "yolov8n.pt" # absolute fallback

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=get_default_model_path(), help='Path to YOLO model')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QToolTip {
            background-color: #18181A;
            color: #E8EDF5;
            border: 1px solid #4A8EFF;
            border-radius: 6px;
            padding: 6px 10px;
            font-family: 'Segoe UI', Arial;
            font-size: 11px;
        }
    """)
    
    # Force dark mode palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(Theme.BG_DEEP))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(Theme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(Theme.BG_PANEL))
    palette.setColor(QPalette.ColorRole.Text, QColor(Theme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(Theme.BG_CARD))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(Theme.TEXT_PRIMARY))
    app.setPalette(palette)

    window = UniGuardDashboard(model_path=args.model)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()