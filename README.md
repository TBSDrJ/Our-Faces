# Our-Faces

Monday 1/30/23:
In order to get the live video feed to work using opencv on a Mac, I needed to `pip install opencv-python-headless` (make sure you get the *same version* as your opencv-python).  To specify a version, use, e.g. `pip install opencv-python-headless==4.4.0.46`.

Later in the day: Sorted out what was up with class names, now class names are read from the dataset at the time of training and stored with the saved model.  The two classify programs in this repository got functionality added to load that data as well as the model.  Also added fps calculator.
