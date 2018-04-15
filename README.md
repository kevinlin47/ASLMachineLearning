# ASLMachineLearning

Rutgers Capstone Group 23

Machines Learn Sign Language

## Getting Started


### Prerequisites

You should have python 3.6.4, pip, and virtualenv installed

### Installing

To get all necessary packages:


Make a new virtual environment

```
mkvirtualenv asl
```

Within your virtual environment

```
pip3 install requirements.txt
```

### (Option) Tensorflow GPU users

```
pip3 install requirements-gpu.txt
```


## Testing

### Run

To initialize the neural network

```
python3 asl-keras.py
```


To test out gesture prediction

```
python3 test-nn.py
```

## To-Do-List

- [] Add GUI for demo
- [] Create an OpenCV function to parse an input video and run it through our model to predict the gesture
- [] (Optional) Add additional gestures
