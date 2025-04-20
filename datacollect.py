import cv2

#creating an variable to store an video capture
video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Enter Your ID : ")
# id = int(id)
count=0

#while for reading video from webcamp on frame
while True:
    ret, frame = video.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(grey, 1.3 , 5)

    for(x,y,w,h) in faces:
        count = count+1
        cv2.imwrite('datasets/User.'+str(id)+"."+str(count)+".jpg", grey[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    #display a frame on screen
    cv2.imshow("Frame" , frame)

    #taking input from keyborad
    k = cv2.waitKey(1)

    #if enter q then it will terminate program
    if count>500:
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done .........1")

