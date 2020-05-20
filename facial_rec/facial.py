import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6                       # higher value will make more matches, higher chance of false positives
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"                         # Convolutional neural network

# first, load in known faces
print('loading known faces')

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}")  #load image
        encoding = face_recognition.face_encodings(image)  # 0th image is image of interest.
        known_faces.append(encoding)
        known_names.append(name)

# now, will iterate over all unknown faces and compare to all known faces
print("processing unknown faces")
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)  # face detection, find location of faces
    encodings = face_recognition.face_encodings(image, locations)
    # location of unknown faces is important in order to go and label those later
    image  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)  # known_faces is list of known faces
          # face encoding is current unknown face
        # results will return a list of booleans which is for each face comparison
        # can check the known names list using the index from return list
        match = None
        if True in results:
            match = known_names[results.index(True)]  # get single identity
            print(f"Match found: {match}")

            # draw rectangle around identified face
            # need top left and bottom right coordinate to draw rectangle
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0, 255, 0]

            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)  # draw rectangle

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)  # shift up

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)  # draw rectangle
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), FONT_THICKNESS)
    cv2.imshow(filename, image) # title, image
    cv2.waitKey(10000)
    # cv2.destroyWindow(filename)





