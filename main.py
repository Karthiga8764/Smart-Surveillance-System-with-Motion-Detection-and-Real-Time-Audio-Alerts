import cv2
import winsound

# Open the camera
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()

    # Calculate difference between two frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert the difference to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to get binary image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the image to fill in the gaps
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours from the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 5000:  # Ignore small movements
            continue

        # Draw rectangle around the detected object
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Play alert sound when motion is detected (unknown person enters)
        winsound.PlaySound('alert.wav', winsound.SND_ASYNC)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) == ord('q'):
        break

    # Show the camera feed
    cv2.imshow('Smart Surveillance System with Motion Detection and Real-Time Audio Alerts', frame1)

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
