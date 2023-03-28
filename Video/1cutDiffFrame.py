import cv2
from skimage.metrics import mean_squared_error
import numpy as np

# Open the video file
cap = cv2.VideoCapture('testVideo.mp4')

# Read the first frame
ret, frame_prev = cap.read()

# Loop through the rest of the frames
while True:
    # Read the next frame
    ret, frame_curr = cap.read()

    # If there are no more frames, break out of the loop
    if not ret:
        break

    # Convert the frames to grayscale
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

    # Compute the mean squared error between the frames
    mse = mean_squared_error(gray_prev, gray_curr)

    # If the mse is greater than the threshold, print a message
    if mse > 100:
        print(
            f"Frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} is significantly different from the previous frame with MSE {mse}")

    # Display the current frame
    cv2.imshow('frame', frame_curr)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Set the current frame as the previous frame for the next iteration
    frame_prev = frame_curr

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

# import cv2
# from skimage.metrics import mean_squared_error
# import numpy as np
#
# # Open the video file
# cap = cv2.VideoCapture('testVideo.mp4')
#
# # Read the first frame
# ret, frame_prev = cap.read()
#
# # Loop through the rest of the frames
# while True:
#     # Read the next frame
#     ret, frame_curr = cap.read()
#
#     # If there are no more frames, break out of the loop
#     if not ret:
#         break
#
#     # Convert the frames to grayscale
#     gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
#     gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
#
#     # Compute the mean squared error between the frames
#     mse = mean_squared_error(gray_prev, gray_curr)
#
#     # If the mse is greater than the threshold, mark the current frame as significantly different
#     if mse > 1:
#         # Mark the current frame as significantly different by drawing a red rectangle around it
#         frame_curr = cv2.rectangle(frame_curr, (0, 0), (frame_curr.shape[1], frame_curr.shape[0]), (0, 0, 255),
#                                    thickness=5)
#
#     # Display the current frame
#     cv2.imshow('frame', frame_curr)
#
#     # Wait for a key press to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     # Set the current frame as the previous frame for the next iteration
#     frame_prev = frame_curr
#
# # Release the video capture and close the window
# cap.release()
# cv2.destroyAllWindows()
