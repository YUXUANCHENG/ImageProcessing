import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import os
import pandas
import csv
from scipy.optimize import minimize
import copy

debug = 1
loss_mode = 0

class datapair:
    def __init__(self, m, u):
        self.mean = m
        self.uncertainty = u

class simVSexp:
    def __init__(self, imageName = './14.jpg', exp_angle = 18, rotateAngle = 0):
        self.x_pos = []
        self.y_pos = []
        self.exp_x = []
        self.exp_y = []
        self.sim_angle = 0
        self.sim_area = 0
        self.sim_calA = 0
        self.exp_area = 0
        self.exp_angle = exp_angle
        self.imageName = imageName
        self.exp_height = 0
        self.sim_height = 0
        self.rotateAngle = rotateAngle
        self.nPoints = 64

        self.contours_cirles = []
        self.contours_area = []
        self.contours = []
        self.calA = []
        self.sim_con = []
        self.exp_con = []
        self.im = []
        self.debug_file = open('./debug.txt','w+')
        #'14.jpg'
    def rotate_image(self, angle):
        image_center = tuple(np.array(self.im.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        self.im = cv2.warpAffine(self.im, rot_mat, self.im.shape[1::-1], flags=cv2.INTER_LINEAR)


    def imageProcessing(self):
        #print(cv2.__version__)
        # im = cv2.imread('4.png')
        self.im = cv2.imread(self.imageName)
        self.rotate_image(self.rotateAngle)
        self.im = cv2.resize(self.im, (0, 0), fx=0.5, fy=0.5)
        imgray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        # # make the bright spot dark
        # brightnessThreash = 0.6 * np.max(imgray)
        # for rowIndex in range(np.shape(imgray)[0]):
        #     imgray[rowIndex] = [x if x < brightnessThreash else 0 for x in imgray[rowIndex]]
        
        im_gauss = cv2.GaussianBlur(imgray, (5, 5), cv2.BORDER_DEFAULT)
        #test = np.max(imgray)
        #im_gauss = imgray
        #ret, thresh = cv2.threshold(im_gauss, 90, 255, cv2.THRESH_BINARY)
        ret, thresh = cv2.threshold(im_gauss, 110, 255, cv2.THRESH_BINARY)
        #thresh = cv2.Canny(im_gauss, 70, 100)
        cv2.imshow('thresh',thresh)
        if debug and cv2.waitKey(0):
            cv2.destroyAllWindows()

        # get contours
        self.contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.filterByArea()
        self.filterByCalA()

    def filterByArea(self):
        # calculate area and filter into new array
        for con in self.contours:
            #if (cv2.isContourConvex(con)):
                area = cv2.contourArea(con)
                if 1000 < area < 1000000:
                    con = self.interpolate(con, self.nPoints*5, self.nPoints)
                    self.contours_area.append(con)

    def interpolate(self, con, ss, n):
        x,y = con.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x,y], u=None, s=ss, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), n)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        return np.asarray(res_array, dtype=np.int32)

    def evaluateVDist(self):
        x,y = self.exp_con.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        MSD = 0
        for x_p, y_p in zip(list(self.x_pos),list(self.y_pos)):
            idx = list(np.argsort(abs(x - x_p))[:4])
            idy = np.argsort(abs([y[i] for i in idx] - y_p))[0]
            idy = idx[idy]
            MSD += (x_p - x[idx[0]])**2 + (y_p - y[idy])**2
        return MSD

                
    def filterByCalA(self):
        # check if contour is of circular shape
        for con in self.contours_area:
            calAtemp = []
            n_corners = self.nPoints
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            circularity = perimeter*perimeter/ (4 * np.pi * area)
            x,y = con.T
            sorted_y = np.sort(y)
            bottom_pos = np.mean(sorted_y[0,:int(n_corners*0.2)])
            verheight = np.max(y) - bottom_pos

            # n_iter = 0
            # max_iter = 1000
            # lb = 0.0001
            # ub = 0.01
            # n_corners = 64
            # while True:
            #     n_iter += 1
            #     if n_iter > max_iter:
            #         break

            #     k = (lb + ub)/2.
            #     eps = k*cv2.arcLength(con, True)
            #     appcon = cv2.approxPolyDP(con, eps, True)
            #     perimeter = cv2.arcLength(appcon, True)
            #     area = cv2.contourArea(appcon)
            #     if area == 0:
            #         break
            #     circularity = perimeter*perimeter/ (4 * 3.14 * area)
            #     x,y = appcon.T
            #     sorted_y = np.sort(y)
            #     bottom_pos = np.mean(sorted_y[0,:int(n_corners*0.2)])
            #     verheight = np.max(y) - bottom_pos
            #     if len(appcon) > n_corners:
            #         lb = (lb + ub)/2.
            #     elif len(appcon) < n_corners:
            #         ub = (lb + ub)/2.
            #     else: 
            #         break
            # if area == 0:
            #         continue
            meanCalA = circularity
            # for i in range(6):
            #     epsilon = (0.001 + 0.001 * i)*cv2.arcLength(con, True)
            #     appcon = cv2.approxPolyDP(con, epsilon, True)
            #     perimeter = cv2.arcLength(appcon, True)
            #     area = cv2.contourArea(appcon)
            #     if area == 0:
            #         continue
            #     circularity = perimeter*perimeter/ (4 * 3.14 * area)
            #     calAtemp.append(circularity)
            # if len(calAtemp) == 0:
            #     continue
            # meanCalA = np.mean(calAtemp)

            #if 1 <= meanCalA <= 1.1:
            if 1 <= meanCalA <= 1.3:
                self.exp_height = verheight
                appcon = con
                self.exp_con = con
                self.contours_cirles.append(appcon)
                self.exp_x,self.exp_y = appcon.T
                self.exp_area = area
                self.calA.append(datapair(meanCalA, 2 * np.std(calAtemp)))
   
    def calRotateAngle(self):
        self.imageProcessing()
        con = self.contours_cirles[-1]
        x,y = con.T
        bottom_index = np.argsort(y)[0,:int(64*0.2)]
        x_to_fit = x[0,bottom_index]
        y_to_fit = y[0,bottom_index]
        m, _ = np.polyfit(x_to_fit, y_to_fit, 1)
        angle = np.arctan(m)
        return 180 * angle / np.pi


    def readPosition(self, n_cell):
        df = pandas.read_csv('./jam_01.txt', header = None)
        #data = list(csv.reader(csvfile))
        x = df.loc[:,0].to_numpy()
        y = df.loc[:,1].to_numpy()
        x = x[::-1]
        y = y[::-1]
        self.x_pos = x[:n_cell]
        self.y_pos = y[:n_cell]

    def readContactAngle(self):
        with open('./deformation.txt') as f:
            data = list(csv.reader(f, delimiter=','))
            self.sim_angle = float(data[0][1])

    def runSim(self,para):
        arg = ' '.join(str(x) for x in para)
        command = './jamming.o 0 1 ' + arg
        os.system(command)
        n_cell = 64
        self.readPosition(n_cell)
        self.sim_hight = np.max(self.y_pos) - np.min(self.y_pos)
        self.x_pos *= self.exp_height/self.sim_hight
        self.y_pos *= self.exp_height/self.sim_hight

        x_shift = np.max(self.exp_x) - np.max(self.x_pos)
        y_shift = np.max(self.exp_y) - np.max(self.y_pos)

        self.x_pos += x_shift
        self.y_pos += y_shift
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(self.x_pos,self.y_pos)]
        self.sim_con = np.asarray(res_array, dtype=np.int32)
        self.sim_area = cv2.contourArea(self.sim_con)
        perimeter = cv2.arcLength(self.sim_con, True)
        self.sim_calA = perimeter*perimeter/ (4 * np.pi * self.sim_area)
        self.readContactAngle()
        #self.contours_cirles.append(self.sim_con)

    def score(self, para):
        self.runSim(para)
        angelDev = abs(self.exp_angle - self.sim_angle)/self.exp_angle
        areaDev = abs(self.exp_area - self.sim_area)/self.exp_area
        self.debug_file.write('angleDev = ' + str(angelDev) + ' areDev = ' + str(areaDev) + '\n')
        self.debug_file.flush()
        if loss_mode:
            current_score = angelDev + 100*areaDev
            return current_score
        else:
            return self.evaluateVDist()

    def plotCon(self):
        #self.contours_cirles.append(self.sim_con)
        copy_im = copy.copy(self.im)
        cv2.drawContours(copy_im, self.contours_cirles, -1, (0,255,0), 3)
        cv2.imshow('1',copy_im)
        cv2.drawContours(self.im, [self.sim_con], -1, (0,255,0), 3)
        cv2.imshow('2',self.im)
        cv2.drawContours(self.im, self.contours_cirles, -1, (255,0,0), 2)
        cv2.imshow('3',self.im)
        #print("mean calA = ", np.mean([x.mean for x in self.calA]))
        print("exp calA = ", np.mean([x.mean for x in self.calA]))
        print("sim calA = ", self.sim_calA)
        #print("uncertainty = ", np.sqrt(np.sum([x.uncertainty**2 for x in calA]))/len(calA))

        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    
    def parameterSearch(self):
        #x0 = np.array([1, 0.1, 0.15, 0.3])
        x0 = np.array([0.2, 0.01, 0.02, 0.3])
        if debug:
            self.score(x0)
        else:
            res = minimize(self.score, x0, method='nelder-mead', 
                        bounds=((0.1, 3), (0.01, 0.2), (-0.1, 0.3), (0.05, 0.3)),
                        options={'xatol': 1e-5, 'disp': True})
            print(res.x)

    def run(self):
        self.imageProcessing()
        #self.calRotateAngle()
        #if (not debug):
        self.parameterSearch()
        self.plotCon()
        self.debug_file.close()
        print('MSD = ' + str(self.evaluateVDist()))


if __name__ == '__main__':
    #file = '14.jpg'
    file = '9.png'
    #cAngle = 20
    #cAngle = 22
    cAngle = 17

    # angleObj = simVSexp(file, cAngle)
    # rAngle = angleObj.calRotateAngle()
    #rAngle = 7
    rAngle = 0
    processobj = simVSexp(file, cAngle, rAngle)
    processobj.run()