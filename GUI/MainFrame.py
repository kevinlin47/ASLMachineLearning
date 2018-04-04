import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename


# Function for uploading button
def import_video():
    file_name = askopenfilename(filetypes=(("MP4 Files", "*.mp4"), ("All files", "*.*")))
    print(file_name)


top = tkinter.Tk()  # Create root window
top.title("Program B.V-1")  # Root window title
top.resizable(0, 0)  # Do not allow user to resize the window
top.geometry("500x500")  # Set window size to 500x500 pixels

import_button = Button(top, text="Import Video", command=import_video)  # Button for video upload
cancel_button = Button(top, text="Cancel")  # Button for canceling request

import_button.config(height=1, width=10)
cancel_button.config(height=1, width=10)

# Testing tkinter widget layout manager
import_button.pack()
cancel_button.pack()

top.mainloop()
