import cv2

#creating an variable to store an video capture
video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["" , "ronaldo"]

#while for reading video from webcamp on frame
while True:
    ret, frame = video.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(grey, 1.3 , 5)

    for(x,y,w,h) in faces:
        serail, conf =  recognizer.predict(grey [y:y+h, x:x+w])
        print(serail)
        if conf>50:
            cv2.putText(frame,name_list[serail], (x,y-40) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        else:
            cv2.putText(frame,"Unknown", (x,y-40) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    #display a frame on screen
    cv2.imshow("Frame" , frame)

    #taking input from keyborad
    k = cv2.waitKey(1)

    #if enter q then it will terminate program
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done .........1")

