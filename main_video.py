import cv2
from simple_facerec import SimpleFacerec
import datetime


# Encode faces from a folder
sfr = SimpleFacerec()
#sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

sfr.read_stored_files()

cur_faces = []

while True:
    ret, frame = cap.read()
    now = datetime.datetime.now()
    # Detect Faces
    face_locations, face_names, distances = sfr.detect_known_faces(frame)
    if(sorted(face_names) != sorted(cur_faces)):
        cur_faces = face_names
    for face_loc, name, distance in zip(face_locations, face_names, distances):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name + ", var="+  ('%.2f'%distance) ,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        print(name, "distance = ", distance)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
