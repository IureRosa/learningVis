import csv
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import messagebox
from PIL import ImageTk, Image


def executar():
    def exibir_valor():
        linha = entrada_texto.get()
        try:
            linha = int(linha)
            if linha < 1 or linha > 25:
                raise ValueError
            valor = dados[linha - 1][3]
            messagebox.showinfo("Resposta", f"{valor}")
        except ValueError:
            messagebox.showerror("Erro", "Digite um número válido de 1 a 25.")

    def donothing():
        pass


    # Criar janela principal
    root2 = tk.Tk()
    root2.title("Learning2Learning")
    root2.geometry("1200x700")
    root2.configure(background="#DCDCDC")
    root2.resizable(False, False)

    logo_path = "images/logov2.png"
    logo_image = Image.open(logo_path)
    logo_photo = ImageTk.PhotoImage(logo_image)
    root2.iconphoto(True, logo_photo)

    # Frame superior para o título e a logo
    top_frame = tk.Frame(root2)
    top_frame.pack(side=tk.TOP, padx=10, pady=10)

    # Cria o menu bar
    menu_bar = tk.Menu(root2)
    root2.config(menu=menu_bar)

    filemenu = tk.Menu(menu_bar, tearoff=0)
    filemenu.add_command(label="Save results", command=donothing)
    filemenu.add_command(label="Save results as", command=donothing)
    filemenu.add_command(label="Preferences", command=donothing)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root2.quit)
    menu_bar.add_cascade(label="File", menu=filemenu)

    helpmenu = tk.Menu(menu_bar, tearoff=0)
    helpmenu.add_command(label="Common Q&A", command=donothing)
    helpmenu.add_command(label="About", command=donothing)
    menu_bar.add_cascade(label="Help", menu=helpmenu)

    # Carrega a imagem da logo
    logo_path = "images/homev3.png"  # Caminho para a imagem da logo
    logo_image = Image.open(logo_path)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = ttk.Label(top_frame, image=logo_photo)
    logo_label.image = logo_photo  # Mantém uma referência à imagem
    logo_label.pack()

    # Criar tree view
    tree = ttk.Treeview(root2, columns=("col1", "col2"))
    tree.column("#0", width=0, stretch=tk.NO)  # Oculta a primeira coluna
    tree.column("col1", width=30, anchor="w")  # Diminui a largura da coluna 1
    tree.column("col2", width=1100, anchor="w")  # Aumenta a largura da coluna 2
    tree.heading("col1", text="ID")
    tree.heading("col2", text="Questions")

    # Carregar dados do arquivo CSV
    dados = []
    with open("dados.csv", "r", encoding="cp1252") as arquivo_csv:
        leitor = csv.reader(arquivo_csv)
        for linha in leitor:
            dados.append(linha)

    # Inserir dados na tree view
    for linha in dados:
        col1 = linha[1]
        col2 = linha[2]
        tree.insert("", tk.END, values=(col1, col2))


    # Criar entrada de texto
    entrada_texto = ttk.Entry(root2)
    entrada_texto.place(x=500, y=550, width=180, height=30)

    # Criar botão de exibição de valor
    botao_exibir = ttk.Button(root2, text="Ask something (use ID)", command=exibir_valor)
    botao_exibir.place(x=500, y=510, width=180, height=30)

    # Exibir tree view
    tree.pack()

    bottom_frame = tk.Frame(root2)
    bottom_frame.place(x=950,y=600,width=200,height=40)

    quit = ttk.Button(bottom_frame, text="Quit", command=root2.quit)
    quit.pack(side=tk.RIGHT, padx=5)

    # Iniciar loop da janela
    root2.mainloop()

#executar()