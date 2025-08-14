import tkinter as tk
from gui.app import SignalProcessorApp

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()