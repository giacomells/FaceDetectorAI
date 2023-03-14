import cv2
from random import randrange


# LOADING SOME TRAINED DATA on detecting frontal face
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
#img = cv2.imread('RDJ.png')

#CAPTURE A VIDEO FROM WEBCAM 
webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)

#ITERATE UNTIL THE VIDEO ENDS
while True:
    # Read the current video frame 
    successful_frame_read, frame = webcam.read() # successful_frame_read is a bool that is true if the frame is correct 
    # Turn it in GRAY
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # DETECT FACES
    face_cordinates = trained_face_data.detectMultiScale(grayscaled_frame)
    # DRAW RECTANGLES
    for(x, y, w, h) in face_cordinates: #takes the cordinates and loops for every face [i]
        cv2.rectangle(frame, (x, y), (x+w, y+h) ,(0, 255, 0),  6)



    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1)    

    ###STOP if Q is pressed
    if key==81 or key==113:
        break

### Release the VideoCaptur eobject
webcam.release()



'''
# Take the image and make it GRAY (cvtColor is the converter)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# DETECT FACES (CascadeClassifier::detectMultiScale)-> MultiScale takes all the sizes
#it gives the top-left (x, y) angle of the rectangle and width and height (w, h) of it
face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_cordinates)   

# DRAW RECTANGLES around the face
for(x, y, w, h) in face_cordinates: #takes the cordinates and loops for every face [i]
    cv2.rectangle(img, (x, y), (x+w, y+h) ,(0, 255, 0),  6)


# SHOW IMAGE
#the waitKey() pauses the excecution of the code, 
#let the image appear (type any key to end<-the image must be in focus)

cv2.imshow('Robert Downey Junior face', img)
cv2.waitKey()       

'''


print("Code completed")

