from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

class LevelMeter(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0.0  # Linear scale (0.0 - 1.0)
        self.setFixedWidth(8)
        self.setMinimumHeight(200)
        self.update()  # Ensure initial painting

    def set_level(self, level):
        self.level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        try:
            rect = self.rect()

            # Gradient setup (fixed)
            gradient = QtGui.QLinearGradient(
                QtCore.QPointF(rect.left(), rect.bottom()),
                QtCore.QPointF(rect.left(), rect.top())
            )
            gradient.setColorAt(0.0, QtGui.QColor("#0000ff"))  # Blue
            gradient.setColorAt(0.5, QtGui.QColor("#00ff00"))  # Green
            gradient.setColorAt(0.8, QtGui.QColor("#ffff00"))  # Yellow
            gradient.setColorAt(0.95, QtGui.QColor("#ff0000"))  # Red

            # Darkened background
            painter.setOpacity(0.3)
            painter.fillRect(rect, gradient)

            # Lit-up region
            active_height = rect.height() * self.level
            active_rect = QtCore.QRectF(
                rect.left(), rect.bottom() - active_height, rect.width(), active_height
            )
            painter.setOpacity(1.0)
            painter.fillRect(active_rect, gradient)

            # Tick marks
            painter.setPen(QtGui.QPen(QtGui.QColor("grey"), 1))
            for db in range(-60, 1, 6):
                y = rect.bottom() - rect.height() * (10 ** (db / 20.0))
                painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))

            # Threshold line (-1 dB)
            threshold_y = rect.bottom() - rect.height() * (10 ** (-1 / 20.0))
            painter.setPen(QtGui.QPen(QtGui.QColor("grey"), 2))
            painter.drawLine(int(rect.left()), int(threshold_y), int(rect.right()), int(threshold_y))

        finally:
            painter.end()