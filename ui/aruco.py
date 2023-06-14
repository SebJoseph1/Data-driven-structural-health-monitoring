import cv2
from PIL import Image
from IPython.display import display, HTML
import math


class Aruco():

    def __init__(self,arucoDict=cv2.aruco.DICT_6X6_250):
        self.arucoDict = arucoDict

    def predict(self,filepath,squareLength=0.00341,markerLength = 0.0025):
        # Detecting the chAruco and calculating pixel-to-mm ratio
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(self.arucoDict)
        arucoParams = cv2.aruco.DetectorParameters()
        # Detect ArUco markers in the image
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
        if len(corners) == 0:
            return None
        # Detect CharUco diamonds using detected markers
        diamondCorners, diamondIds = cv2.aruco.detectCharucoDiamond(gray, corners, ids, squareLength/markerLength)
        if len(diamondCorners) == 0:
            return None
        else:
            dis = 0
            for i in range(0,len(diamondCorners[0])):
                for j in range (i,len(diamondCorners[0])):
                    new_dis = math.dist(diamondCorners[0][i][0],diamondCorners[0][j][0])
                    if dis < new_dis:
                        dis = new_dis
            # Calculate the pixel-to-mm ratio and return it along with diamond corners
            return float(squareLength/dis) * 1000,diamondCorners[0]
        


