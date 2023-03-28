# ===a working version===
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

cap = cv2.VideoCapture('testVideo.mp4')

# initialize variables
prev_frame = None
prev_frame_id = None

# get user input for frame id to compare with previous frame
frame_id = int(input("Enter frame id: "))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate mse with previous frame
    if prev_frame is not None and prev_frame_id == frame_id - 1:
        mse = np.mean((gray - prev_frame) ** 2)
        # calculate ssim with previous frame
        ssim_score = ssim(gray, prev_frame, multichannel=False)
        print(f"Frame {prev_frame_id} vs frame {prev_frame_id + 1}: mse={mse}, ssim={ssim_score}")
        # break
        # cap.release()
        # cv2.destroyAllWindows()

    # update previous frame and frame id
    prev_frame = gray
    prev_frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)

cap.release()
cv2.destroyAllWindows()

# import cv2
# from skimage.metrics import compare_ssim
# import numpy as np
#
# cap = cv2.VideoCapture('path/to/video/file')
#
# # initialize variables
# prev_frame = None
# prev_frame_id = None
#
# # get user input for frame id to compare with previous frame
# frame_id = int(input("Enter frame id: "))
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#
#     # convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # calculate mse with previous frame
#     if prev_frame is not None and prev_frame_id == frame_id - 1:
#         mse = np.mean((gray - prev_frame) ** 2)
#         # calculate ssim with previous frame
#         ssim_score = compare_ssim(gray, prev_frame)
#         if ssim_score < 100:  # change threshold as needed
#             print(f"Frame {prev_frame_id} vs frame {prev_frame_id + 1}: mse={mse}, ssim={ssim_score}")
#
#     # update previous frame and frame id
#     prev_frame = gray
#     prev_frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
#
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# from skimage.metrics import structural_similarity as ssim
# import numpy as np
#
# cap = cv2.VideoCapture('testVideo.mp4')
#
# # initialize variables
# prev_frame = None
# prev_frame_id = None
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#
#     # convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # calculate mse with previous frame
#     if prev_frame is not None:
#         mse = np.mean((gray - prev_frame) ** 2)
#         # calculate ssim with previous frame
#         ssim_score = ssim(gray, prev_frame, multichannel=False)
#         print(f"Frame {prev_frame_id} vs frame {prev_frame_id + 1}: mse={mse}, ssim={ssim_score}")
#
#     # update previous frame and frame id
#     prev_frame = gray
#     prev_frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
#
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from skimage.measure import compare_ssim
#
# # Load video file
# cap = cv2.VideoCapture('testVideo.mp4')
#
# # Read first frame
# ret, frame1 = cap.read()
# frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#
# # Loop over frames
# while True:
#     # Read current frame
#     ret, frame2 = cap.read()
#     if not ret:
#         break
#     frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#
#     # Calculate SSIM between current frame and previous frame
#     ssim = compare_ssim(frame1_gray, frame2_gray, full=True)[0]
#
#     # Display frames and SSIM value
#     cv2.imshow('frame', frame2)
#     print(f'SSIM: {ssim}')
#
#     # Check if user wants to compare with a specific frame
#     user_input = input('Enter frame number to compare with (or "q" to quit): ')
#     if user_input == 'q':
#         break
#     frame_id = int(user_input)
#     if frame_id > 0:
#         # Calculate SSIM between specified frame and previous frame
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#         ret, specified_frame = cap.read()
#         specified_frame_gray = cv2.cvtColor(specified_frame, cv2.COLOR_BGR2GRAY)
#         ssim = compare_ssim(specified_frame_gray, frame1_gray, full=True)[0]
#
#         # Display specified frame and SSIM value
#         cv2.imshow('specified_frame', specified_frame)
#         print(f'SSIM between frame {frame_id} and its previous frame: {ssim}')
#
#     # Update previous frame
#     frame1_gray = frame2_gray
#
#     # Wait for key press and check if user wants to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
