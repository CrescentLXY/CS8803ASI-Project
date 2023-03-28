import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

cap = cv2.VideoCapture('testVideo.mp4')

# initialize variables
prev_frame = None
prev_frame_id = None

while (cap.isOpened()):
    # get user input for frame id
    user_input = input("Enter a frame id to compare with its previous frame (or 'q' to quit): ")
    if user_input == 'q':
        break
    try:
        current_frame_id = int(user_input)
    except ValueError:
        print("Invalid input. Please enter a number or 'q' to quit.")
        continue

    # read video until desired frame is reached
    while (cap.get(cv2.CAP_PROP_POS_FRAMES) < current_frame_id):
        ret, frame = cap.read()
        if ret == False:
            break

    if ret == False:
        print("Invalid frame id. Please enter a number between 1 and", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        continue

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate mse with previous frame
    if prev_frame is not None:
        mse = np.mean((gray - prev_frame) ** 2)
        # calculate ssim with previous frame
        ssim_score = ssim(gray, prev_frame, multichannel=False)
        print(f"Frame {prev_frame_id} vs frame {current_frame_id}: mse={mse}, ssim={ssim_score}")

    # update previous frame and frame id
    prev_frame = gray
    prev_frame_id = current_frame_id

cap.release()
cv2.destroyAllWindows()