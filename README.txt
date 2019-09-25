1. 球員登錄(照片輸入版本)
請用command line 輸入以下指令:

python face_detect_image.py --image face_data/Test/test.jpg --output test0910.jpg

此指令會輸出兩個檔案:
(a)test0910.jpg
(b)player_list.csv

2. 新球員上場(照片輸入版本)
因沒有完整的player_list.csv可以用, 暫用假的data(player_info.csv)
請把60行指令改成
df = pd.read_csv('player_info.csv', names =('Team','Num','Name','R','G','B'))  

請用command line 輸入以下指令:

python project_image.py --image 2.jpg --output test2.jpg

3. webcam.py for 新上場球員
-更改每幾秒截取一次畫面請改:
timeF = 30  #目前為一秒(30)一張, 要改成一分鐘一張要改成1800

-要改成output到kafka, 目前為單純print out該frame而已
print(frame) 

-結束webcam請按q跳出迴圈

請用command line 輸入以下指令:
python webcam.py
