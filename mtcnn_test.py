import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
import numpy as np




if __name__ == '__main__':

    '''
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("./IMG_9182.JPG")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #b, g, r = cv2.split(img)
    #img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print box_align
    save_name = 'r_4.jpg'
    vis_face(img_bg,bboxs,landmarks, save_name)
    '''
    s="./data_set/face_detection/WIDERFACE/WIDER_train/WIDER_train/images/40--Gymnastics/40_Gymnastics_Gymnastics_40_204.jpg 456 144 516 222"
    a=s.split(' ')
    boxes=np.array([[int(a[1]),int(a[2]),int(a[3]),int(a[4])]])
    landmarks=np.array([[[0,0],[0,0],[0,0],[0,0],[0,0]]])
    img=cv2.imread(a[0])
    #cv2.imshow("aa",img)
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vis_face(img_bg,boxes,landmarks,'r_4.jpg')

