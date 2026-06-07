"""
A sandbox script to test and verify the rendering behavior of custom QTableWidget cell highlights and delegates.

"""


import sys
from PyQt6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QStyledItemDelegate, QStyleOptionViewItem, QStyle
from PyQt6.QtGui import QColor, QBrush, QPainter
from PyQt6.QtCore import Qt, QTimer

class TableHighlightDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        bg = index.data(Qt.ItemDataRole.BackgroundRole)
        if bg:
            # Save painter state
            painter.save()
            # Draw the custom background color
            painter.fillRect(option.rect, bg)
            
            # Draw the text/content of the cell using style
            new_opt = QStyleOptionViewItem(option)
            self.initStyleOption(new_opt, index)
            new_opt.backgroundBrush = QBrush(Qt.BrushStyle.NoBrush)
            
            # Paint it
            style = option.widget.style() if option.widget else QApplication.style()
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, new_opt, painter, option.widget)
            painter.restore()
        else:
            super().paint(painter, option, index)

app = QApplication(sys.argv)

table = QTableWidget(3, 3)
table.setAlternatingRowColors(True)

# Apply QSS
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

# Apply delegate
delegate = TableHighlightDelegate(table)
table.setItemDelegate(delegate)

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
    pixmap.save("C:/Users/Shreyas S/.gemini/antigravity-ide/brain/1fd8f85b-04b5-42c1-9ba9-b9690d36abfe/screenshot_delegate.png")
    app.quit()

QTimer.singleShot(500, save_screenshot)
sys.exit(app.exec())
