from pickletools import uint1, uint2, uint8
from xmlrpc.client import Boolean, boolean
import cv2
import numpy as np
def onislem(path):
    image=cv2.imread(path)
    harris=image.copy()

    gry=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ret1,tresh=cv2.threshold(gry,120,255,cv2.THRESH_BINARY)

    gry2=cv2.cvtColor(tresh, cv2.COLOR_BAYER_GB2GRAY)

    canny = cv2.Canny(gry2,100,200)
    edge_count = np.count_nonzero(canny)

    bitwise=np.zeros((gry2.shape[0],gry2.shape[1]))

    for i in range(gry2.shape[0]):
        for j in range(gry2.shape[1]):
            bitwise[i][j]= 0 if gry2[i][j]<127 else 255

    sobel_horizontal = cv2.Sobel(bitwise, cv2.CV_64F, 1, 0, ksize=7)
    area_count = np.count_nonzero(sobel_horizontal)

    operatedImage = np.float32(sobel_horizontal)
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
    dest = cv2.dilate(dest, None)
    harris[dest > 0.1 * dest.max()]=[0, 0, 255]

    num_of_corner=0
    pixel_sayisi_dikey=0
    for i in range(harris.shape[0]):
        pixel_sayisi_dikey+=1
        for j in range(harris.shape[1]):
            if harris[i][j][2]==255:
                num_of_corner+=1

    print(pixel_sayisi_dikey)
    print(num_of_corner)
    print(num_of_corner/pixel_sayisi_dikey)
    print(edge_count/pixel_sayisi_dikey)
    print(area_count/pixel_sayisi_dikey)

    cv2.imshow("orjinal",image)
    cv2.imshow("gray",gry)
    cv2.imshow("treshold",tresh)
    cv2.imshow("gray2",gry2)
    cv2.imshow('canny', canny)
    cv2.imshow("bitwise",bitwise)
    cv2.imshow('Sobelhorizontal', sobel_horizontal)
    cv2.imshow('ImagewithBorders', harris)
    """cv2.imwrite("/home/orka/Desktop/croped/WristFracture/cikti/orjinal.jpg",image)
    cv2.imwrite("/home/orka/Desktop/croped/WristFracture/cikti/gray.jpg",gry)
    cv2.imwrite("/home/orka/Desktop/croped/WristFracture/cikti/treshold.jpg",tresh)
    cv2.imwrite("/home/orka/Desktop/croped/WristFracture/cikti/gray2.jpg",gry2)
    cv2.imwrite("/home/orka/Desktop/croped/WristFracture/cikti/canny.jpg",image)
    cv2.imwrite("/home/orka/Desktop/croped/WristFracture/cikti/bitwise.jpg",image)
    cv2.imwrite("/home/orka/Desktop/croped/WristFracture/cikti/Sobelhorizontal.jpg",image)
    cv2.imwrite("/home/orka/Desktop/croped/WristFracture/cikti/ImagewithBorders.jpg",image)"""

    cv2.waitKey(0)
    cv2.destroyAllWindows()
for orca in range(60,61):
    x="/home/orka/Desktop/croped/WristFracture/kirik/"+str(orca)+".jpg"
    onislem(x)