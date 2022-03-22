import cv2
from glob import glob

imgs = glob('./utils/captcha/img/*.png')
for img in imgs:
    captcha = cv2.imread(img)
    filename = img.split('\\')[1].split('.')[0]
    ch1 = captcha[:,3:23]
    ch2 = captcha[:,23:44]
    ch3 = captcha[:,44:64]
    ch4 = captcha[:,64:85]
    cv2.imwrite('./utils/captcha/img/ch/'+filename+'-1.png',ch1)
    cv2.imwrite('./utils/captcha/img/ch/'+filename+'-2.png',ch2)
    cv2.imwrite('./utils/captcha/img/ch/'+filename+'-3.png',ch3)
    cv2.imwrite('./utils/captcha/img/ch/'+filename+'-4.png',ch4)