import cv2
import mediapipe as mp
import time

# init cam
cap = cv2.VideoCapture(1)

cTime = 0
pTime = 0

# Draw face
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=100000)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)



    ### FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    #FPS text
    cv2.putText(img, f"FPS:{int(fps)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    #Camera show
    cv2.imshow("output", img)
    cv2.waitKey(1)