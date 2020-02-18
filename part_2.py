import cv2
import numpy as np
from os import listdir  # it is used when we need to fetch data from directory
from os.path import isfile, join

# this path shows us where our images are stored
data_path = 'C:/Users/hitesh/Downloads/open cv images/faces/'
# now we need files that presents in (data_path) location that files are images
only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []  # training data is the set of images we take in part1 that we prepare for dataset images

for i, files in enumerate(only_files):  # it gives us no of iteration as much as we have no of files
    image_path = data_path + only_files[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.IMREAD_GRAYSCALE = 0
    # cv2.IMREAD_COLOR = 1
    # cv2.IMREAD_UNCHANGED = -1

    Training_Data.append(np.asarray(images, dtype=np.uint8))  # bcoz we want or data in array form
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)  # we converted the Labels here in array

model = cv2.face.LBPHFaceRecognizer_create()  # we create our model here LBPHF ==> local binary pattern histogram face

model.train(np.asarray(Training_Data), np.asarray(Labels))  # we train our model here

print("Model Training Complete")

# --------(PART-3)-------------
# first we crete the rectangle on our face
# then we create the percentage value of confidence that it is our face
# then we examine is it Unlocked,Locked,Face Not Found

face_classifier = cv2.CascadeClassifier('C:/python 37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def face_detect(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.2, 6)

    if faces is ():  # if there is no image it will give us the blank frame
        return img, []

    for (x, y, w, h) in faces:
        #  the rectangle have 2 cordinates one is left top-corner and other is right bottom-corner
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]  # it tells us where it make the rectangle
        roi = cv2.resize(roi, (200, 200))  # This is used to resize the rectangle

    return img, roi


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    image, face = face_detect(frame)  # the face detect gives us the 2 things (image,face)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)  # here we predict our face

        if result[1] < 500:
            # this gives us the percentage value of how much is our face matches
            confidence = int(100 * (1 - (result[1]) / 300))
            display_string = str(confidence) + "% confident it is user"

        cv2.putText(image, display_string, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 255), 2)

        if confidence > 75:
            cv2.putText(image, "unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Cropper", image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Face Cropper", image)

    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Face Cropper", image)
        pass

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()