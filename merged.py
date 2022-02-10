import cv2
from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
#sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(1)
ex = 10       #parameter for extra frame with face detected box
sfr.read_stored_files()

trained_face_data = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x,y,w,h) in face_coordinates:
        frame_roi = frame[y-ex:y+h+ex, x-ex:x+w+ex]
        try:
            # Detect Faces
            face_locations, face_names, distances = sfr.detect_known_faces(frame_roi)
            for face_loc, name, distance in zip(face_locations, face_names, distances):
                (y1, x2, y2, x1) = face_loc
                cv2.putText(frame, name + ", var="+  ('%.2f'%distance) ,(x-ex, y-ex  - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x+x1-ex, y+y1-ex), (x+x2-ex, y+y2-ex), (0, 0, 200), 4)
                print(name, "distance = ", distance)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        except:
            continue
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
