import cv2
#import matplotlib.pyplot as plt
       
vc = cv2.VideoCapture(0) #读入视频文件
c=1
 
if vc.isOpened(): #判断是否正常打开
    rval , frame = vc.read()
else:
    rval = False
 
timeF = 30  #视频帧计数间隔频率,目前為一秒(30)一張, 要改成一分鐘一張要改成1800
 
while rval:   #循环读取视频帧
    rval, frame = vc.read()
    if(c%timeF == 0): #每隔timeF帧进行存储操作
        #cv2.imwrite('image/'+str(c) + '.jpg',frame) #存储为图像(記得先在當目錄下建立一個image資料夾')
        #plt.imshow(frame)
        print(frame) #把這裡改成output到kafka
    c = c + 1
    
    #按q跳出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
