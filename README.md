# Facial-and-Emotion-Recognition

This simple model, to find the emotions of a given person. The dataset fro the model used is [CK+48 dataset](https://www.kaggle.com/datasets/davilsena/ckdataset) and for face recognition we used [haar cascade classifier](https://github.com/opencv/opencv/tree/master/data/haarcascades).

This entire project is created in python it self. For video capture the project uses OpenCV2 library which is present in Video_capture.py file.

For training the model , the entire code is in `Train CK++48.ipynb`, please use jupyter notebook or google collab to run this.

### Pre-requisite
* Python 3.6.0 or above
* Jupyter Notebook or Google Collab

### Pre-requisite libraries required. 
```
pip install numpy
pip install pandas
pip install matplotlib
pip install tensorflow
pip install opencv-python
```

## How to run the project
1. First download the entire project.
2. Please make you have all the pre-requisites installed and working. 
3. Open the `Training CK+48.ipynb`. Train and save the model at the desired location in your system
4. Open the `Video_capture.py` file and the location of the saved model and run the file.

Note : This project is created only for the view of Dayananda Sagar University. This project holds no value outside the Dayananda Sagar University.