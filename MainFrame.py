import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from predict_gesture import predict_gesture
import time
import threading


class ProgressBarThread(threading.Thread):
    def __init__(self, label='Working', delay=0.2):
        super(ProgressBarThread, self).__init__()
        self.label = label
        self.delay = delay  # interval between updates
        self.running = False
    def start(self):
        self.running = True
        super(ProgressBarThread, self).start()
    def run(self):
        label = '\r' + self.label + ' '
        while self.running:
            for c in ('-', '\\', '|', '/'):
                sys.stdout.write(label + c)
                sys.stdout.flush()
                time.sleep(self.delay)
    def stop(self):
        self.running = False
        self.join()  # wait for run() method to terminate
        sys.stdout.write('\r' + len(self.label)*' ' + '\r')  # clean-up
        sys.stdout.flush()

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
    result_label.cofig(text="")


def start_translation(): # Function for start button
    pb_thread=ProgressBarThread("Computing")
    pb_thread.start()

    result=predict_gesture(file_name)
    result="Predicted Translation: "+result
    result_label.config(text=result)

    pb_thread.stop()
    print("The work is done!")

file_name = ''

top = tkinter.Tk()  # Create root window
top.title("Program B.V-1")  # Root window title
top.resizable(0, 0)  # Do not allow user to resize the window
top.geometry("500x500")  # Set window size to 500x500 pixels

select_button = Button(top, text="Select Video", command=select_video)  # Button for video selection
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
