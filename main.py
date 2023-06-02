from tkinter import Tk
from app.app import App

def main():
    root = Tk()
    app = App(root)
    app.run()

if __name__ == "__main__":
    main()