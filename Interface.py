import tkinter
from tkinter import *
from tkinter import messagebox

top = tkinter.Tk()


def helloCallBack():
    messagebox.askokcancel(message='123')


B = tkinter.Button(top, text="点我", command=helloCallBack)

B.pack()
top.mainloop()