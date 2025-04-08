# coding=gb2312
import cv2
import numpy as np

def ComputeMinLevel(hist, pnum):
    index = np.add.accumulate(hist)
    return np.argwhere(index > pnum * 8.3 * 0.01)[0][0]

def ComputeMaxLevel(hist, pnum):
    hist_0 = hist[::-1]
    Iter_sum = np.add.accumulate(hist_0)
    index = np.argwhere(Iter_sum > (pnum * 2.2 * 0.01))[0][0]
    return 255 - index

def LinearMap(minlevel, maxlevel):
    if minlevel >= maxlevel:
        return np.array([])
    else:
        index = np.array(list(range(256)))
        screenNum = np.where(index < minlevel, 0, index)
        screenNum = np.where(screenNum > maxlevel, 255, screenNum)
        for i in range(len(screenNum)):
            if screenNum[i] > 0 and screenNum[i] < 255:
                screenNum[i] = (i - minlevel) / (maxlevel - minlevel) * 255
        return screenNum

def CreateNewFrame(frame):
    h, w, d = frame.shape
    new_frame = np.zeros([h, w, d], dtype=np.uint8)
    for i in range(d):
        imghist = np.bincount(frame[:, :, i].reshape(-1), minlength=256)
        minlevel = ComputeMinLevel(imghist, h * w)
        maxlevel = ComputeMaxLevel(imghist, h * w)
        screenNum = LinearMap(minlevel, maxlevel)
        if screenNum.size == 0:
            continue
        new_frame[:, :, i] = screenNum[frame[:, :, i]]
    return new_frame

def process_video_levels(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        adjusted_frame = CreateNewFrame(frame)
        out.write(adjusted_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video_levels('video(1).mp4', '自动色阶.mp4')
