import cv2
import numpy as np

#print(cv2.__version__)
#im = cv2.imread('7.png')
im = cv2.imread('114.jpg')
im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gauss = cv2.GaussianBlur(imgray, (5, 5), cv2.BORDER_DEFAULT)

ret, thresh = cv2.threshold(im_gauss, 210, 255, cv2.THRESH_BINARY)
#thresh = cv2.Canny(im_gauss, 10, 50)
thresh = cv2.Canny(thresh, 50, 100)
cv2.imshow('thresh',thresh)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180 # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 200  # minimum number of pixels making up a line
max_line_gap = 100  # maximum gap in pixels between connectable line segments
line_image = np.copy(im) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(thresh, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
angles = []
for line in lines:
    for x1,y1,x2,y2 in line:
        length = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        #angle = np.arccos(abs(x2-x1)/length)
        angle = np.arctan((y1-y2)/(x1-x2))
        flag = True
        if len(angles) > 0:
            for th in angles:
                if abs((angle - th)/th) < 0.05:
                    flag = False
        if length > 300 and flag:
            angles.append(angle)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
lines_edges = cv2.addWeighted(im, 0.8, line_image, 1, 0)
cv2.imshow('lines', lines_edges)
print('angles = ', abs(angles[1]-angles[0])*180/np.pi)

if cv2.waitKey(0):
    cv2.destroyAllWindows()
