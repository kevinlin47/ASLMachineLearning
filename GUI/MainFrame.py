import tkinter
from tkinter import *

top = tkinter.Tk()
top.title("Program B.V-1")
top.resizable(0, 0)
top.geometry("500x500")

upload_button = Button(top, text="one")
cancel_button = Button(top, text="two")

upload_button.pack()
cancel_button.pack()

top.mainloop()
