1. �y���n��(�Ӥ���J����)
�Х�command line ��J�H�U���O:

python face_detect_image.py --image face_data/Test/test.jpg --output test0910.jpg

�����O�|��X����ɮ�:
(a)test0910.jpg
(b)player_list.csv

2. �s�y���W��(�Ӥ���J����)
�]�S�����㪺player_list.csv�i�H��, �ȥΰ���data(player_info.csv)
�Ч�60����O�令
df = pd.read_csv('player_info.csv', names =('Team','Num','Name','R','G','B'))  

�Х�command line ��J�H�U���O:

python project_image.py --image 2.jpg --output test2.jpg

3. webcam.py for �s�W���y��
-���C�X��I���@���e���Ч�:
timeF = 30  #�ثe���@��(30)�@�i, �n�令�@�����@�i�n�令1800

-�n�令output��kafka, �ثe�����print out��frame�Ӥw
print(frame) 

-����webcam�Ы�q���X�j��

�Х�command line ��J�H�U���O:
python webcam.py
