import Tkinter
from Tkinter import *
import FileDialog
from tkFileDialog import askopenfilename


def import_video():  # Function for uploading button
    file_name = askopenfilename(filetypes=(("MP4 Files", "*.mp4"), ("All files", "*.*")))
    video_label.config(text=file_name)

    print(file_name)


def cancel_video():  # Function for cancel button
    # Clear window
    video_label.config(text="")
    print()


top = Tkinter.Tk()  # Create root window
top.title("Program B.V-1")  # Root window title
top.resizable(0, 0)  # Do not allow user to resize the window
top.geometry("500x500")  # Set window size to 500x500 pixels

import_button = Button(top, text="Import Video", command=import_video)  # Button for video upload
cancel_button = Button(top, text="Cancel", command=cancel_video)  # Button for canceling request
exit_button = Button(top, text="Exit", command=top.destroy)
video_label = Label(top, anchor=CENTER, text="");

import_button.config(height=1, width=10)
cancel_button.config(height=1, width=10)
exit_button.config(height=1, width=10)

# Testing tkinter widget layout manager
import_button.pack()
cancel_button.pack()
exit_button.pack()
video_label.pack()

top.mainloop()
