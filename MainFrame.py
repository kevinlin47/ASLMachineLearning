import tkinter
from tkinter import *

top = tkinter.Tk()  # Create root window
top.title("Program B.V-1")  # Root window title
top.resizable(0, 0)  # Do not allow user to resize the window
top.geometry("500x500")  # Set window size to 500x500 pixels

upload_button = Button(top, text="one")  # Button for video upload
cancel_button = Button(top, text="two")  # Button for canceling request

# Testing tkinter widget layout manager
upload_button.pack()
cancel_button.pack()

top.mainloop()
