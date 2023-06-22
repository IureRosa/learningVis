import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image
from app.code_runner import CodeRunner
from app.commonQA import *

class App:
    def __init__(self, root):
        self.runners = []
        self.process = None  # Adicionado o atributo 'process' à classe App
        self.root = root
        self.root.title("Learning2Learning")
        self.root.geometry("800x500")
        self.root.configure(background="#DCDCDC")
        self.root.resizable(False, False)

            # Define a imagem da logo como ícone do aplicativo
        logo_path = "images/icon.png"
        logo_image = Image.open(logo_path)
        logo_photo = ImageTk.PhotoImage(logo_image)
        self.root.iconphoto(True, logo_photo)

        self.create_interface()

    def create_interface(self):
        # Frame superior para o título e a logo
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Cria o menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        filemenu = tk.Menu(menu_bar, tearoff=0)
        filemenu.add_command(label="Save results", command=self.donothing)
        filemenu.add_command(label="Save results as", command=self.donothing)
        filemenu.add_command(label="Preferences", command=self.donothing)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menu_bar, tearoff=0)
        helpmenu.add_command(label="Common Q&A", command=self.commonQA)
        helpmenu.add_command(label="Install Dependencies", command=self.install_libraries_from_file("requirements.txt"))
        helpmenu.add_command(label="About", command=self.donothing)
        menu_bar.add_cascade(label="Help", menu=helpmenu)

        # Carrega a imagem da logo
        logo_path = "images/homev3.png"  # Caminho para a imagem da logo
        logo_image = Image.open(logo_path)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = ttk.Label(top_frame, image=logo_photo)
        logo_label.image = logo_photo  # Mantém uma referência à imagem
        logo_label.pack()

        left_frame = tk.Frame(self.root)
        left_frame.place(x=120,y=300,width=300,height=150)

        right_frame = tk.Frame(self.root)
        right_frame.place(x=370,y=300,width=300,height=150)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.place(x=600,y=450,width=200,height=40)

        files = ["agents/basicAlgs.py", "agents/teste.py", "agents/discreteAS.py"]
        button_names = ["Q-Learning/SARSA", "TD3/DDPG/SAC", "PPO and DQN"]

        labelRun = tk.Label(self.root, text="Run ML Algorithms").place(
            x=120, y=260, width=300, height=30
        )
        labelstop = tk.Label(self.root, text="Stop ML Algorithms").place(
            x=370, y=260, width=300, height=30
        )

        for i, (file, button_name) in enumerate(zip(files, button_names), start=1):
            runner = CodeRunner(file)
            self.runners.append(runner)

            button = ttk.Button(left_frame, text=button_name, command=runner.start, width=20)
            button.pack(pady=5)

        for i, (file, button_name) in enumerate(zip(files, button_names), start=1):
            runner = CodeRunner(file)
            self.runners.append(runner)
            
            stop_button = ttk.Button(right_frame, text=button_name, command=runner.stop, width=20)
            stop_button.pack(pady=5)

        close_button = ttk.Button(bottom_frame, text="Quit", command=self.root.quit)
        close_button.pack(side=tk.RIGHT, padx=5)

        stop_all_button = ttk.Button(bottom_frame, text="Stop All", command=self.stop_all)
        stop_all_button.pack(side=tk.RIGHT, padx=5)

    def stop_all(self):
        stopped_runners = 0
        for runner in self.runners:
            if runner.process and runner.process.poll() is None:
                runner.stop()
                stopped_runners += 1

        if stopped_runners == 0:
            messagebox.showwarning("Aviso", "Nenhum método em execução.")

    def donothing(self):
        messagebox.showwarning("Aviso", "Essa função ainda não foi implementada nesta versão.")

    def commonQA(self):
        executar()


    def install_libraries_from_file(self,file_path):
        try:
            with open(file_path, "r") as file:
                libraries = file.read().splitlines()
                self.install_libraries(*libraries)
        except FileNotFoundError:
            print(f"Arquivo {file_path} não encontrado.")

    def install_libraries(self, *libraries):
        for library in libraries:
            try:
                subprocess.check_call(["pip", "install", library])
                print(f"{library} instalada com sucesso!")
            except subprocess.CalledProcessError:
                print(f"Erro ao instalar {library}.")



    def run(self):
        self.root.mainloop()
