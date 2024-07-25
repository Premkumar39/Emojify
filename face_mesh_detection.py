import cv2
from cvlearn import FaceMesh
from cvlearn.Utils import findDistance, find_rotation
import numpy as np


def draw_arc(img, x, y, rotation, radius):
    center = (x, y)
    axes = (radius, radius)
    start_angle = 100
    end_angle = 0

    cv2.ellipse(img, center, axes, rotation, start_angle, end_angle, (255, 255, 255), int(radius/2))


# Create FaceMeshDetector object
detector = FaceMesh.FaceMeshDetector()

# Initialize cap
cap = None

# Iterate over camera indices to find the correct one
for i in range(10):  # You can adjust the range as needed
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
    cap.release()

# Check if a camera is found
if cap is None or not cap.isOpened():
    print("No camera found.")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        for face in faces:
            # Define facial feature points
            face_points = {
                'faceUp': 10,
                'faceDown': 152,
                'rightEyeUp': 386,
                'rightEyeDown': 374,
                'leftEyeUp': 159,
                'leftEyeDown': 145,
                'mouthRight': 78,
                'mouthLeft': 308,
                'mouthUp': 13,
                'mouthDown': 14
            }

            # Face ellipse
            faceLength, face_pos = findDistance(face[face_points['faceUp']], face[face_points['faceDown']], img)
            cv2.ellipse(img, (face_pos[0], face_pos[1] - 30),
                        (int(faceLength / 1.5), int(faceLength / 1.5) + 10),
                        find_rotation(face[face_points['faceUp']], face[face_points['faceDown']]),
                        0, 360, (52, 225, 255), -1)

            # Right eye circle
            Right_eye_length, R_eye_pos = findDistance(face[face_points['rightEyeUp']], face[face_points['rightEyeDown']], img)
            cv2.circle(img, (R_eye_pos[0], R_eye_pos[1] - 30), int(Right_eye_length / 1.3), (65, 71, 100), -1)

            draw_arc(img, R_eye_pos[0]-3, R_eye_pos[1]-33, find_rotation(face[face_points['rightEyeUp']],
                                                                        face[face_points['rightEyeDown']]),
                     int(Right_eye_length/3))

            # Left eye circle
            Left_eye_length, L_eye_pos = findDistance(face[face_points['leftEyeUp']], face[face_points['leftEyeDown']], img)
            cv2.circle(img, (L_eye_pos[0], L_eye_pos[1] - 30), int(Left_eye_length / 1.3), (65, 71, 100), -1)

            draw_arc(img, L_eye_pos[0]-3, L_eye_pos[1]-33, find_rotation(face[face_points['leftEyeUp']],
                                                                        face[face_points['leftEyeDown']]),
                     int(Left_eye_length/3))

            # Mouth ellipse
            MouthLength1, mouth_pos = findDistance(face[face_points['mouthRight']], face[face_points['mouthLeft']], img)
            MouthLength2, _ = findDistance(face[face_points['mouthUp']], face[face_points['mouthDown']], img)
            cv2.ellipse(img, (mouth_pos[0], mouth_pos[1] - 30),
                        (int(MouthLength1 / 1.6), int(MouthLength2)),
                        find_rotation(face[face_points['mouthRight']], face[face_points['mouthLeft']]),
                        0, 360, (0, 60, 255), -1)

            # Right eyebrow
            right_eyebrow = [
                [face[336][0], face[336][1] - 40],
                [face[296][0], face[296][1] - 40],
                [face[334][0], face[334][1] - 40],
                [face[293][0], face[293][1] - 40],
                [face[300][0], face[300][1] - 40],
                [face[283][0], face[283][1] - 40],
                [face[282][0], face[282][1] - 40],
                [face[295][0], face[295][1] - 40],
                [face[285][0], face[285][1] - 40]
            ]

            right_eyebrow_polygon = np.array([right_eyebrow], np.int32)
            cv2.fillPoly(img, pts=[right_eyebrow_polygon], color=(0, 0, 0))

            # Left eyebrow
            left_eyebrow = [
                [face[70][0], face[70][1] - 40],
                [face[63][0], face[63][1] - 40],
                [face[105][0], face[105][1] - 40],
                [face[66][0], face[66][1] - 40],
                [face[107][0], face[107][1] - 40],
                [face[55][0], face[55][1] - 40],
                [face[52][0], face[52][1] - 40],
                [face[65][0], face[65][1] - 40],
                [face[53][0], face[53][1] - 40]
            ]

            left_eyebrow_polygon = np.array([left_eyebrow], np.int32)
            cv2.fillPoly(img, pts=[left_eyebrow_polygon], color=(0, 0, 0))

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop when 'q' is pressed
        break

cap.release()  # Release the VideoCapture object
cv2.destroyAllWindows()  # Close all OpenCV windows