import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image  # Importa as classes necessárias do Pillow
import subprocess
import threading

class CodeRunner:
    def __init__(self, filename):
        self.filename = filename
        self.process = None

    def start(self):
        self.process = subprocess.Popen(['streamlit', 'run', self.filename])

    def stop(self):
        if self.process:
            self.process.terminate()

class App:
    def __init__(self):
        self.runners = []
        self.root = tk.Tk()
        self.root.title("Learning2Learning")

        # Define a imagem da logo como ícone do aplicativo
        logo_path = "images/logo.png"
        logo_image = Image.open(logo_path)
        logo_photo = ImageTk.PhotoImage(logo_image)
        self.root.iconphoto(True, logo_photo)

        self.create_interface()

    def create_interface(self):
        # Frame superior para o título e a logo
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Título em negrito
        title_style = ttk.Style()
        title_style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        title_label = ttk.Label(top_frame, text="Learning2Learning", style="Title.TLabel")
        title_label.pack()

        # Carrega a imagem da logo
        logo_path = "images/logo.png"  # Caminho para a imagem da logo
        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((150, 150))  # Ajusta o tamanho da imagem conforme necessário
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = ttk.Label(top_frame, image=logo_photo)
        logo_label.image = logo_photo  # Mantém uma referência à imagem
        logo_label.pack()

        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, padx=10)

        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, padx=10)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, pady=10)

        files = ["agents/basicAlgs.py", "agents/continuousAS.py", "agents/discreteAS.py"]

        for i, file in enumerate(files, start=1):
            runner = CodeRunner(file)
            self.runners.append(runner)

            button = ttk.Button(left_frame, text=f"Executar Método {i}", command=runner.start, width=20)
            button.pack(pady=5)

        for i, runner in enumerate(self.runners, start=1):
            stop_button = ttk.Button(right_frame, text=f"Parar Método {i}", command=runner.stop, width=20)
            stop_button.pack(pady=5)

        stop_all_button = ttk.Button(bottom_frame, text="Parar todos", command=self.stop_all)
        stop_all_button.pack(side=tk.RIGHT, padx=5)

        close_button = ttk.Button(bottom_frame, text="Fechar", command=self.root.quit)
        close_button.pack(side=tk.RIGHT, padx=5)

    def stop_all(self):
        for runner in self.runners:
            runner.stop()

    def run(self):
        self.root.mainloop()

app = App()
app.run()
