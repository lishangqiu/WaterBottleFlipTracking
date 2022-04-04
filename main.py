import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter

LOWER_BLUE_BOUND = np.array([95, 140, 90])
UPPER_BLUE_BOUND = np.array([105, 260, 246])

START_FRAME = 246
END_FRAME = 370

DISPLAY = False


def diff_angle(angle1, angle2):
    angle_dif = angle1 - angle2
    return (angle_dif + 180) % 360 - 180


def dist_points(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


vid = cv2.VideoCapture("./IMG_8205.MOV")
FRAME_RATE = vid.get(cv2.CAP_PROP_FPS)

i = 0
angles = []
pointA_locs = []
pointB_locs = []

pbar = tqdm(total=END_FRAME - START_FRAME)
while vid.isOpened():
    i += 1
    ret, frame = vid.read()
    if not ret or i >= END_FRAME:
        break

    if i < START_FRAME:
        continue

    if not DISPLAY:
        pbar.update(1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE_BOUND, UPPER_BLUE_BOUND)
    mask[:, :200] = 0
    dilation = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

    processed = cv2.bitwise_and(frame, frame, mask=mask)
    mask = cv2.medianBlur(mask, 11)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        ys = []
        xs = []
        if cv2.contourArea(contours[1]) > 2000:
            for x in [0, 1]:
                c = contours[x]
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                ys.append(cY)
                xs.append(cX)

            if len(pointA_locs) != 0:
                if dist_points((xs[0], ys[0]), pointA_locs[-1]) > dist_points((xs[1], ys[1]), pointA_locs[-1]):
                    ys[0], ys[1] = ys[1], ys[0]
                    xs[0], xs[1] = xs[1], xs[0]
            pointA_locs.append((xs[0], ys[0]))
            pointB_locs.append((xs[1], ys[1]))

            angle = math.degrees(math.atan2(ys[1] - ys[0], xs[1] - xs[0])) + 90
            angles.append(angle % 360)

cv2.destroyAllWindows()
pbar.close()
vid.release()

angular_diffs = []
for i in range(len(angles)):
    if i == 0:
        continue
    diff = diff_angle(angles[i], angles[i - 1])

    angular_diffs.append(diff * FRAME_RATE)

plt.ion()

replay_vid = cv2.VideoCapture("./IMG_8205.MOV")
for i in range(START_FRAME):
    _ = replay_vid.read()

plt.plot(angular_diffs)
plt.plot(savgol_filter(angular_diffs, 51, 3))
plt.ylabel("Angular Velocity (degrees/second)")
plt.xlabel("Frame (60fps)")
line = plt.axvline(0, color='purple')

for i in range(END_FRAME-START_FRAME):
    ret, frame = replay_vid.read()

    cv2.circle(frame, (pointA_locs[i][0], pointA_locs[i][1]), 7, (0, 255, 0), -1)
    cv2.circle(frame, (pointB_locs[i][0], pointB_locs[i][1]), 7, (255, 0, 0), -1)

    cv2.line(frame, (pointA_locs[i][0], pointA_locs[i][1]), (pointB_locs[i][0], pointB_locs[i][1]),
             (255, 0, 255), thickness=3)
    cv2.putText(frame, str(i), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    cv2.imshow('frame', cv2.resize(frame, (720, 1280)))
    cv2.waitKey(25)
    line.set_xdata(i)
    plt.pause(0.0001)

cv2.destroyAllWindows()
plt.show(block=True)
input()
