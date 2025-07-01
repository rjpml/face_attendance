import cv2 #Image processing
import numpy as np
import face_recognition #For detecting and processing faces

#Load the image using face_recognition library
#This loads the image file from the given path & converts it into a NumPY array
imgAiah = face_recognition.load_image_file("images/BINI_AIAH.png")
#Convert the image from BGR (used by OpenCV) to RGB (used by face_recognition)
imgAiah = cv2.cvtColor(imgAiah,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file("images/BINI_JHOANNA.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#Detect the face location in the image
faceLoc = face_recognition.face_locations(imgAiah)[0]
#Encode the face into a 128-dimensional vector for comparison
encodeAiah = face_recognition.face_encodings(imgAiah)[0]
#Draw a rectangle around the detected face
#(left, top) is the top-left corner; (right, bottom) is the bottom-right corner
#Color is (255, 0, 255) = purple-pink; thickness = 2 pixels
cv2.rectangle(imgAiah,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeAiah],encodeTest)
faceDis = face_recognition.face_distance([encodeAiah],encodeTest)
print(results, faceDis)
cv2.putText(imgTest,f"{results} {round(faceDis[0],2)}",(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

#Display the image in a window using OpenCV
cv2.imshow('BINI_AIAH', imgAiah)
cv2.imshow('BINI_AIAH_TEST', imgTest)
#Keep the image window open until you press any key on your keyboard
cv2.waitKey(0)