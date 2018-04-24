import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from predict_gesture import predict_gesture


def select_video():  # Function for uploading button
    # set file_name to string of video directory
    global file_name
    file_name = askopenfilename(filetypes=(("MP4 Files", "*.mp4"),("MOV Files", "*.mov"), ("All files", "*.*")))
    video_label.config(text=file_name)  # set label to show video directory
    result_label.config(text="")
    print(file_name)  # print test


def cancel_video():  # Function for cancel button
    # Clear window
    video_label.config(text="")


def start_translation(): # Function for start button

    result = predict_gesture(file_name)
    result = "The model predicts the gesture is: " + result  # Store predicted result into string variable
    progress.pack()
    bar()

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


file_name = ''

top = tkinter.Tk()  # Create root window
top.title("Program B.V-1")  # Root window title
top.resizable(0, 0)  # Do not allow user to resize the window
top.geometry("500x500")  # Set window size to 500x500 pixels
progress = ttk.Progressbar(top, orient=HORIZONTAL, length=100, mode='determinate')  # Progress bar

select_button = Button(top, text="Import Video", command=select_video)  # Button for video upload
cancel_button = Button(top, text="Cancel", command=cancel_video)  # Button for canceling request
exit_button = Button(top, text="Exit", command=top.destroy)  # Button for closing the application
start_button = Button(top, text="Start", command=start_translation)  # Button for starting the prediction
video_label = Label(top, anchor=CENTER, text="")   # text label
result_label = Label(top, anchor=CENTER, text="")  # result label

select_button.config(height=1, width=10)  # resize the button
cancel_button.config(height=1, width=10)  # resize the button
exit_button.config(height=1, width=10)    # resize the button
start_button.config(height=1, width=10)   # resize the button

# Testing tkinter widget layout manager
# Place GUI Widgets onto the root window
select_button.pack()
cancel_button.pack()
start_button.pack()
exit_button.pack()
video_label.pack()
result_label.pack()

# Execute root window
top.mainloop()