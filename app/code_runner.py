import subprocess
from tkinter import messagebox

class CodeRunner:
    def __init__(self, filename):
        self.filename = filename
        self.process = None

    def start(self):
        try:
            self.process = subprocess.Popen(['streamlit', 'run', self.filename])
        except FileNotFoundError:
            messagebox.showerror("Erro", f"Arquivo '{self.filename}' não encontrado.")
        except subprocess.SubprocessError:
            messagebox.showerror("Erro", f"Erro ao executar o arquivo '{self.filename}'.")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        else:
            messagebox.showwarning("Aviso", "O método não está em execução.")