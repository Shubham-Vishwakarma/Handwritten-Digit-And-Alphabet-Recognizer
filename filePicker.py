from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename


def OpenFile():
    path = askopenfilename()
    print(path)
    return path