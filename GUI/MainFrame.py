import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk


def import_video():  # Function for uploading button
    # set file_name to string of video directory
    file_name = askopenfilename(filetypes=(("MP4 Files", "*.mp4"), ("All files", "*.*")))

    video_label.config(text=file_name)  # set label to show video directory
    result_label.config(text="")

    print(file_name)  # print test


def cancel_video():  # Function for cancel button
    # Clear window
    video_label.config(text="")


def start_translation(): # Function for start button
    result = "The model predicts the gesture is: "  # Store predicted result into string variable
    progress.pack()
    bar()
    result_label.config(text=result)  # Set result label to value in result string


def bar():  # Function for loading bar
    import time
    progress['value'] = 20
    top.update_idletasks()
    time.sleep(1)
    progress['value'] = 50
    top.update_idletasks()
    time.sleep(1)
    progress['value'] = 80
    top.update_idletasks()
    time.sleep(1)
    progress['value'] = 100


top = tkinter.Tk()  # Create root window
top.title("Program B.V-1")  # Root window title
top.resizable(0, 0)  # Do not allow user to resize the window
top.geometry("500x500")  # Set window size to 500x500 pixels
progress = ttk.Progressbar(top, orient=HORIZONTAL, length=100, mode='determinate')  # Progress bar

import_button = Button(top, text="Import Video", command=import_video)  # Button for video upload
cancel_button = Button(top, text="Cancel", command=cancel_video)  # Button for canceling request
exit_button = Button(top, text="Exit", command=top.destroy)  # Button for closing the application
start_button = Button(top, text="Start", command=start_translation)  # Button for starting the prediction
video_label = Label(top, anchor=CENTER, text="")   # text label
result_label = Label(top, anchor=CENTER, text="")  # result label

import_button.config(height=1, width=10)  # resize the button
cancel_button.config(height=1, width=10)  # resize the button
exit_button.config(height=1, width=10)    # resize the button
start_button.config(height=1, width=10)   # resize the button

# Testing tkinter widget layout manager
# Place GUI Widgets onto the root window
import_button.pack()
cancel_button.pack()
start_button.pack()
exit_button.pack()
video_label.pack()
result_label.pack()

# Execute root window
top.mainloop()
