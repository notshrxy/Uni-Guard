"""
A quick interface sandbox to prototype Qt Stylesheet selector options and custom QToolTip palette styling overrides.

"""

import sys
from PyQt6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtCore import Qt, QTimer

app = QApplication(sys.argv)

table = QTableWidget(3, 3)
table.setAlternatingRowColors(True)

# Apply QSS similar to the main dashboard
table.setStyleSheet("""
    QTableWidget {
        background-color: #1B1C1E;
        alternate-background-color: #222325;
        color: #FFFFFF;
        gridline-color: #37383A;
    }
    QTableWidget::item {
        padding: 0px 12px;
        border-bottom: 1px solid #37383A;
    }
""")

# Populate items
for r in range(3):
    for c in range(3):
        item = QTableWidgetItem(f"Row {r}, Col {c}")
        if r == 0:
            # Highlight first row in wine-red
            item.setBackground(QBrush(QColor("#4A1D1D")))
        table.setItem(r, c, item)

table.resize(400, 300)
table.show()

# Save screenshot after rendering
def save_screenshot():
    pixmap = table.grab()
    pixmap.save("C:/Users/Shreyas S/.gemini/antigravity-ide/brain/1fd8f85b-04b5-42c1-9ba9-b9690d36abfe/screenshot.png")
    app.quit()

QTimer.singleShot(500, save_screenshot)
sys.exit(app.exec())
