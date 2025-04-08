# coding=gb2312
import cv2
import numpy as np

def zmMinFilterGray(src, r=5):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1), dtype=src.dtype))

def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b

def Defog(m, r, eps, w, maxV1):
    '''处理视频中的每帧图像'''
    V1 = np.min(m, axis=2)
    Dark_Channel = zmMinFilterGray(V1, r)
    V1 = guidedfilter(V1, Dark_Channel, r, eps)
    bins = 2000
    ht = np.histogram(V1, bins)
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, axis=2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)
    return V1, A

def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=True):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)
    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img) / (1 - Mask_img/A)
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))
    return Y

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = deHaze(frame / 255.0) * 255
        out.write(processed_frame.astype('uint8'))
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video('video(1).mp4', '暗通道.mp4')
