from tkinter import Tk
from app.backup import App

def main():
    root = Tk()
    app = App(root)
    app.run()

if __name__ == "__main__":
    main()