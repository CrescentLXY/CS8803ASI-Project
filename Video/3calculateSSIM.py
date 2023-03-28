import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Open the video file
cap = cv2.VideoCapture('testVideo.mp4')

# Get the frame dimensions and FPS of the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the video codec and output file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('3calculateSSIM.mp4', fourcc, fps, (width, height))

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

    # Compute the structural similarity index between the frames
    ssim_score, _ = ssim(gray_prev, gray_curr, full=True)

    # If the ssim is lower than the threshold, mark the current frame with the frame ID
    if ssim_score < 0.95:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 50)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        cv2.putText(frame_curr, f"Frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}",
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    # Write the current frame to the output video
    out.write(frame_curr)

    # Display the current frame
    cv2.imshow('frame', frame_curr)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Set the current frame as the previous frame for the next iteration
    frame_prev = frame_curr

# Release the video capture, output video, and close the window
cap.release()
out.release()
cv2.destroyAllWindows()
