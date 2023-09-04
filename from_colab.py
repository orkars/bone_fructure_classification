import cv2
import numpy as np
import math
import pandas as pd

def onislem(path):
    image=cv2.imread(path)
    harris=image#.copy()
    #line=image.copy()
    cv2.imshow("x",image)
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

    cdst = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(canny,1,np.pi/180,50, None, 50, 10)
    
    if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
         
    linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 5)
    
    uzunluk=[]
    egimler=[]
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            egm=(l[2]-l[0])/(l[3]-l[1]) if (l[3]-l[1])!=0 else 0
            uzunluk.append(math.sqrt(((l[2]-l[0])**2)+((l[3]-l[1])**2)))
            egm=0 if egm>100 else egm
            egm=0 if egm<-100 else egm
            egimler.append(egm)
    max_egim=max(egimler)
    min_egim=min(egimler)

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
    aci=((math.atan(max_egim)-math.atan(min_egim))/(math.pi))*180
    a1=num_of_corner/pixel_sayisi_dikey
    a2=edge_count/pixel_sayisi_dikey
    a3=area_count/pixel_sayisi_dikey
    a4=aci
    a5=len(linesP)
    a6=np.average(uzunluk)/pixel_sayisi_dikey
    a7=np.average(uzunluk)
    return [a1,a2,a3,a4,a5,a6,a7,num_of_corner]

seri=[]

for orca in range(71):
    x="/home/orka/Desktop/croped_bone_project/croped/kirik/"+str(orca)+".jpg"
    satir=onislem(x)
    satir.append(1)
    seri.append(satir)
    print(orca,satir)

for orca in range(105):
    x="/home/orka/Desktop/croped_bone_project/croped/kirik/"+str(orca)+".jpg"
    satir=onislem(x)
    satir.append(0)
    seri.append(satir)
    print(orca,satir)
df2=pd.DataFrame(seri,columns=["kose","cevre","alan","aci","lines","uzunluk","ort_uzunluk","toplam_kose","kirik"])