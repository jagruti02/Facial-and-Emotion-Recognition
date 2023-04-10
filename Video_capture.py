from cv2 import cv2
import tensorflow as tf
import numpy as np

#importing the haar_code xml file to detect faces.
face_cascade_haar = cv2.CascadeClassifier('haar-cascade-files-master/haarcascade_frontalface_default.xml')

#list of all emotions.
emotions = ('Angry', 'Fear', 'Happy', 'Sad', 'Suprise')



#creating an instance to start the video camera.
capture = cv2.VideoCapture(0)
#processing the caputed video.
while True:
    #extracting each frame of the video and storing in the variable 'frame'.
    _, frame = capture.read()

    #converting each 'frame' ot grey scale to reduce computing power.
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #1.3 is the minsize this defines the minimum possib;e object size.
    #5 is the minNeighbors this defines the number of neighboors each candidate rectangle should have to retain it.
    faces = face_cascade_haar.detectMultiScale(gray_scale, 1.3, 5)
    
    #finding the two diagonals corners of the rectangle.
    for (x, y, w, h) in faces:
        #places the rectangle on the frame.
        rectangle_box = cv2.rectangle(frame, (x, y), (w+x, y+h), (255, 0, 0), 2)
        
        #capturing the faces.
        captured_grey_scale = gray_scale[y:y+h, x:x+w]
        
        #denoising the caputured image and converting the image to 48x48. 
        cv2.fastNlMeansDenoising(captured_grey_scale, captured_grey_scale, 7, 21)
        captured_grey_scale = cv2.resize(captured_grey_scale, (48,48))
        
        #making captured_grey_scale readable in keras and normalizing the array of the image in the of range 0 to 1.
        captured_grey_scale = tf.keras.preprocessing.image.img_to_array(captured_grey_scale)
        captured_grey_scale = np.expand_dims(captured_grey_scale, axis = 0)
        captured_grey_scale /= 225
        
        #loading the trained model using CK+ datatset
        emotion_model = tf.keras.models.load_model('please give the saved model name')
        
        #predicting the model based of the caputed face using haar code.
        predicted_emotion = emotion_model.predict(captured_grey_scale)

        #finding the position of maximum value in the array of predicted value.
        pos = np.argmax(predicted_emotion[0])
        #print(pos)
        
        #places the identified to be displayed with rectangle.
        cv2.putText(rectangle_box, "Face detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        #places the emption to be displayed with rectangle.
        cv2.putText(rectangle_box, emotions[pos], (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    #displays the ractangle on the image.
    cv2.imshow('Trial run', frame)
    
    #pressing 'esc' breaks the entire while loop.
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#webcame stops hear memory gets deallocated.
capture.release()

#all the data collected by the cv2 library gets destroyed if not cleared when closing the video capture. 
cv2.destroyAllWindows()
