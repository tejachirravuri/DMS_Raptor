"""DMS-Raptor — Dynamic Model Switching for Real-Time UAV Inspection."""
import sys
import os


def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Windows taskbar icon: must be set BEFORE QApplication
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "DMS-Raptor.v1.8"
            )
        except Exception:
            pass

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QIcon
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("DMS-Raptor")
    app.setApplicationVersion("1.8.0")

    # Application-level icon (taskbar + window title bar)
    icon_path = os.path.join(app_dir, "resources", "logo.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # load dark theme
    qss = os.path.join(app_dir, "resources", "style.qss")
    if os.path.exists(qss):
        with open(qss, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
