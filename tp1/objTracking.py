from Detector import detect
from KalmanFilter import KalmanFilter

import cv2

def get_square(xk, radius=15):
    x0 = int(xk[0][0] - radius)
    y0 = int(xk[1][0] - radius)
    x1 = int(xk[0][0] + radius)
    y1 = int(xk[1][0] + radius)
    return x0, y0, x1, y1
    

kalmanfilter = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)


video_path = 'randomball.avi'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
#video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))

# Initialize a counter
frame_count = 0
trajectory = []

while True:
    if not ret:
        break


    # print('Frame: ', frame_count)

    # Detect the object in the frame
    centers = detect(frame)

    if len(centers) != 0:
        

        # draw the circle in the frame
        cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 5, (0, 255, 0), -1)

        # Predict the next state
        kalmanfilter.predict()

        # draw the predicted rectangle in the predicted object position (blue)
        #cv2.rectangle(frame, (int(kalmanfilter.xk[0]-kalmanfilter.xk[2]/2), int(kalmanfilter.xk[1]-kalmanfilter.xk[3]/2)),
        #                 (int(kalmanfilter.xk[0]+kalmanfilter.xk[2]/2), int(kalmanfilter.xk[1]+kalmanfilter.xk[3]/2)), (255, 0, 0), 2)7

        x0,y0,x1,y1 = get_square(kalmanfilter.xk)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

        # Update the state with the detected object
        kalmanfilter.update(centers[0])

        # draw a red rectangle as the estimated object position
        # cv2.rectangle(frame, (int(kalmanfilter.xk[0]-kalmanfilter.xk[2]/2), int(kalmanfilter.xk[1]-kalmanfilter.xk[3]/2)),
        #                (int(kalmanfilter.xk[0]+kalmanfilter.xk[2]/2), int(kalmanfilter.xk[1]+kalmanfilter.xk[3]/2)), (0, 0, 255), 2)
        x0,y0,x1,y1 = get_square(kalmanfilter.xk)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)

        # Draw the trajectory
        trajectory.append((int(kalmanfilter.xk[0][0]), int(kalmanfilter.xk[1][0])))
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

        # create video output
        cv2.imshow('Object Tracking', frame)
        #video.write(frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read a frame
    ret, frame = cap.read()
    frame_count += 1    

cap.release()
cv2.destroyAllWindows()