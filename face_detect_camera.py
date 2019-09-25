import os
import sys
import argparse
import cv2
import math
import time
import numpy as np
import openpose.util as util
from openpose.config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model.cmu_model import get_testing_model

import matplotlib.pyplot as plt
import signal
from IPython import display
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from imageio import imread
from skimage.transform import resize
from keras.models import load_model
import scipy
from scipy.spatial import distance

import csv
from sklearn.cluster import KMeans
import color.utils as utils

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_align_images(filepaths, margin):
    cascade = cv2.CascadeClassifier(cascade_path)
    
    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)

        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)
        (x, y, w, h) = faces[0]
        cropped = img[y-margin//2:y+h+margin//2,
                      x-margin//2:x+w+margin//2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
            
    return np.array(aligned_images)

def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(filepaths, margin))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

def train(dir_basepath, names, max_num_img=10):
    labels = []
    embs = []
    for name in names:
        dirpath = os.path.abspath(dir_basepath + name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = calc_embs(filepaths)    
        labels.extend([name] * len(embs_))
        embs.append(embs_)
        
    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    clf = SVC(kernel='linear', probability=True).fit(embs, y)
    return le, clf

# for face-detection
def calc_embs_camera(imgs, margin, batch_size):
    aligned_images = prewhiten(imgs)
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

class FaceDemo(object):
    def __init__(self, cascade_path):
        self.vc = None
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.margin = 10
        self.batch_size = 1
        self.n_img_per_person = 10
        self.is_interrupted = False
        self.data = {}
        self.le = le
        self.clf = clf
        
    def _signal_handler(self, signal, frame):
        self.is_interrupted = True
    
    def infer(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.cascade.detectMultiScale(frame,
                                     scaleFactor=1.1,
                                     minNeighbors=3,
                                     minSize=(100, 100))
        pred = None
        if len(faces) != 0:
            face = faces[0]
            (x, y, w, h) = face
            left = x - self.margin // 2
            right = x + w + self.margin // 2
            bottom = y - self.margin // 2
            top = y + h + self.margin // 2
            img = resize(frame[bottom:top, left:right, :],
                         (160, 160), mode='reflect')
            new_embs = calc_embs_camera(img[np.newaxis], self.margin, 1)
            clf_pred=clf.predict(new_embs)
            clf_prob=clf.predict_proba(new_embs)[0]
            pred = le.inverse_transform(clf_pred)

            distance_list=[]
            for n in range(3):
                distance_list.append(distance.euclidean(new_embs, data[pred[0]+str(n)]['emb']))

            if scipy.mean(distance_list) < 1:
                cv2.rectangle(frame,
                          (left-1, bottom-1),
                          (right+1, top+1),
                          (255, 0, 0), thickness=2)
                result = pred
                      
            else:
                result = 'Newplayer'
        else:
            result = 'None'
            
        return frame, result




def process(input_image, params, model_params):
    # load openpose model
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights('model/keras/model.h5')


    # for openpose
    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    # the middle joints heatmap correpondence
    mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
              [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
              [55, 56], [37, 38], [45, 46]]

    # visualize
    '''
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
              [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
              [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
              '''
    
    oriImg = input_image  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    '''
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(input_image, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    stickwidth = 4
    '''
    for n in range(len(subset)):
        right_shoulder_X = int(candidate[int(subset[n][2]), 0])
        right_shoulder_Y = int(candidate[int(subset[n][2]), 1])
        left_shoulder_X = int(candidate[int(subset[n][5]), 0])  
        left_shoulder_Y = int(candidate[int(subset[n][5]), 1])
        if abs(right_shoulder_X - left_shoulder_X) >= 100:
            body_ori = input_image[right_shoulder_Y:, right_shoulder_X:left_shoulder_X]
            body_RGB = cv2.cvtColor(body_ori, cv2.COLOR_BGR2RGB)
            # reshape the image to be a list of pixels
            body_reshape = body_RGB.reshape((body_RGB.shape[0] * body_RGB.shape[1], 3))
            # cluster the pixel intensities
            clt = KMeans(n_clusters = 5)
            # reshape the image to be a list of pixels
            clt.fit(body_reshape)
            # 找出各色之比重
            hist = utils.centroid_histogram(clt)
            #找出顏色比重最重的hist index
            color_idx = np.where(hist==np.max(hist))
            #主色RDB為: clt.cluster_centers_[color_idx]
            color = clt.cluster_centers_[color_idx].flatten().tolist()
        else:
            continue

    return color[0], color[1], color[2]

if __name__ == '__main__':
    tic = time.time()
    print('start training...')

    # train face_data
    cascade_path = './model/cv2/haarcascade_frontalface_alt2.xml'

    image_dir_basepath = './face_data/'
    names = ['PENRITE_1_Tony', 'PENRITE_2_Timmy', 'PENRITE_5_Kaylens', 'ABC_3_Tim', 'ABC_7_Sophie']
    image_size = 160

    model_path = './model/keras/facenet_keras.h5'
    model = load_model(model_path)
    
    for name in names:
        dirpath = os.path.abspath(image_dir_basepath + name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:10]
    
    for filepath in filepaths:
        cascade = cv2.CascadeClassifier(cascade_path)
        img = imread(filepath)
        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)

    labels = []
    embs = []
    data = {}
    for name in names:
        dirpath = os.path.abspath(image_dir_basepath + name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:10]
        embs_ = calc_embs(filepaths)    
        labels.extend([name] * len(embs_))
        embs.append(embs_)
        for i in range(len(filepaths)):
            data['{}{}'.format(name, i)] = {'image_filepath' : filepaths[i],
                                            'emb' : embs_[i]}

    le, clf = train(image_dir_basepath, names)
  
    # for face detection
    
    
    print('start processing...')
    vc = cv2.VideoCapture(0) #读入视频文件
    c=0
    f = FaceDemo(cascade_path)   
    timeF = 30  #视频帧计数间隔频率,目前為一秒(30)一張, 要改成一分鐘一張要改成1800
    player_list=[]
    #循环读取视频帧
    while (vc.isOpened()):   
        rval, frame = vc.read()
        #每隔timeF帧进行存储操作
        if(c%timeF == 0): 
            print(c)
            # run face-detection 
            img, result = f.infer(frame)
            print(result)
            if result in ['Newplayer','None']:
                continue
            if result not in player_list:
                player_list.append(result)
                # load config
                params, model_params = config_reader()
                # generate image with body parts
                R,G,B = process(frame, params, model_params)                              
                team, num, name = result[0].split('_')
                
                with open('player_list.csv', 'a', newline='') as csvFile:   
                    writer = csv.writer(csvFile)
                    writer.writerow([team,num,name,R,G,B])
                
                #cv2.imwrite('image/'+str(c//30) +name+ '.jpg', img)
                cv2.imshow('img', img)
            
        c = c + 1

        #按q跳出迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()

    toc = time.time()
    print ('processing time is %.5f' % (toc - tic))





