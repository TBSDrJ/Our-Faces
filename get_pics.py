import cv2
from time import sleep

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera.")
    quit()
else:
    print("Opened camera.")

sleep(1)
print("Make sure your face appears in front of the webcam.  Taking 100 photos...")

for i in range(100):
    _, frame = cap.read()
    cv2.imwrite(f"images/Dr_J/Dr_J_{i}.jpg", frame)

print("Done taking face photos.")
print("Now, move your face out of the view of the webcam.")
print("Press <Enter> when you are ready.")
input()
print("Taking 100 photos")

for i in range(100):
    _, frame = cap.read()
    cv2.imwrite(f"images/no_face/no_face_{i}.jpg", frame)

print("All done gathering images.")
