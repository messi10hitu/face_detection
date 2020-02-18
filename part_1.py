# Before Using This Must Install ==> pip install opencv-contrib-python
# 1 first we create face classifier and then we create face extarctor function by converting it into bgr to gray scale
# 2 after that create detectmultiscale and create the face cropper
# 3 after that video capture and read the frames
# 4 convert it from bgr to gray
# 5 resize the size of window
# 6 give the file name path where the images should save and apply the imwrite function
# 7 then write the cont funtion on it to count the no of images with the help of puttext function
# 8 at last cv2.imshow

# ------START-----------
import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('C:/python 37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # here 1.2 is the scale factor and 3 is the min neighbour it vary b/w (3-6)
    faces = face_classifier.detectMultiScale(gray, 1.2, 3)

    if faces is():
        return None

    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]  # this will crop the face only

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        # this is used to make the size of window
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'C:/Users/hitesh/Downloads/open cv images/faces/user' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('face cropper', face)

    else:
        print("Face Not Found")
        pass
    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print("All Samples Complete")
