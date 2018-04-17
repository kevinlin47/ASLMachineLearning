import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename


def import_video():  # Function for uploading button
    # set file_name to string of video directory
    file_name = askopenfilename(filetypes=(("MP4 Files", "*.mp4"), ("All files", "*.*")))

    video_label.config(text=file_name)  # set label to show video directory

    print(file_name)  # print test


def cancel_video():  # Function for cancel button
    # Clear window
    video_label.config(text="")


top = tkinter.Tk()  # Create root window
top.title("Program B.V-1")  # Root window title
top.resizable(0, 0)  # Do not allow user to resize the window
top.geometry("500x500")  # Set window size to 500x500 pixels

import_button = Button(top, text="Import Video", command=import_video)  # Button for video upload
cancel_button = Button(top, text="Cancel", command=cancel_video)  # Button for canceling request
exit_button = Button(top, text="Exit", command=top.destroy)  # Button for closing the application
video_label = Label(top, anchor=CENTER, text="")  # text label

import_button.config(height=1, width=10)  # resize the button
cancel_button.config(height=1, width=10)  # resize the button
exit_button.config(height=1, width=10)    # resize the button

# Testing tkinter widget layout manager
import_button.pack()
cancel_button.pack()
exit_button.pack()
video_label.pack()

top.mainloop()
