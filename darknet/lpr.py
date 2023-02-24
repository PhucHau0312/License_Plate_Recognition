import sys, os, time
import keras
import cv2
import numpy as np
import traceback
import argparse
import darknet
from wpod_src.keras_utils 			import load_model
from os.path 					    import splitext, basename
from wpod_src.utils 				import im2single
from wpod_src.keras_utils 			import load_model, detect_lp



def resize_bbox(detections, out_size, in_size):
    
    detections = list(detections)
    for det in detections:
        det = list(det)
        det[2] = list(det[2])

        xmin, ymin, xmax, ymax = darknet.bbox2points(det[2])
        y_scale = float(out_size[0]) / in_size[0]
        x_scale = float(out_size[1]) / in_size[1]
        ymin = int(y_scale * ymin)
        ymax = int(y_scale * ymax)
        xmin = int(x_scale * xmin) if int(x_scale * xmin) > 0 else 0
        xmax = int(x_scale * xmax)

        det[2] = [xmin, ymin, xmax, ymax]

    detections = sorted(detections, key = lambda x: x[2][1])
    line = detections[0][2][3]
    begin_idx = 0 
    for i, det in enumerate(detections):
        if det[2][1] > line:
            detections[begin_idx:i] = sorted(detections[begin_idx:i], key = lambda x: x[2][0])
            begin_idx = i 
            break
    detections[begin_idx:] = sorted(detections[begin_idx:], key = lambda x: x[2][0])

    return detections


def character_detect(frame, threshold, network, class_names, class_colors):
 
    prev_time = time.time()

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), 
                                interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=threshold)
    darknet.free_image(darknet_image)
    
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    print(detections)
    darknet.print_detections(detections)

    det_time = time.time() - prev_time
    fps = int(1/(time.time() - prev_time))
    print("Detection time: {}".format(det_time))
    print("FPS: {}".format(fps))

    out_size = frame.shape[:2]
    in_size = image_resized.shape[:2]
    detections = resize_bbox(detections, out_size, in_size)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def parser():
    parser = argparse.ArgumentParser(description="License Plate Recognition")
    parser.add_argument("--input_dir", type=str, default="", help="input")
    parser.add_argument("--output_dir", default = "./output", help="output")
    parser.add_argument("--lp_threshold", default = .5, help="lp-threshold")
    parser.add_argument("--threshold", default = .5, help="threshold")

    return parser.parse_args()


def main():
    args = parser()
    wpod_net_path = './models/my-trained-model/wpod_ver4_backup.h5'
    lp_threshold = args.lp_threshold

    config_file = './cfg/yolov4-tiny-mish.cfg'
    data_file = './data/obj.data'
    weights = './backup/yolov4-tiny-mish_best.weights'
    thresh = args.threshold

    img_path = args.input_dir
    output_dir = args.output_dir

    wpod_net = load_model(wpod_net_path)
    network, class_names, class_colors = darknet.load_network(config_file, data_file, weights)

    print ('Searching for license plates using WPOD-NET')

    bname = splitext(basename(img_path))[0]
    Ivehicle = cv2.imread(img_path)
    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
    side  = int(ratio*288.)
    bound_dim = min(side + (side%(2**4)),608)
    
    Llp,LlpImgs,_,points = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
    
    pt = points[0]
    pt = np.array([pt[0,:],pt[1,:],pt[2,:],pt[3,:]], np.int32)
    pts = pt.reshape((-1,1,2))
    Ip = cv2.polylines(im2single(Ivehicle),[pts],True,(0,255,0),2)
    
    if len(LlpImgs):
        Ilp = LlpImgs[0]
        
    text = []
    if len(Ilp): 
        Ilp = np.array(Ilp*255.0, dtype=np.uint8)
        image, detections = character_detect(Ilp, thresh, network, class_names, class_colors)

        for det in detections:
            text.append(det[0])
        text = ''.join(text)
        print(text)

    cv2.putText(Ip,text,pt[0],cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2,cv2.LINE_AA)
    cv2.imwrite('%s/%s_output.jpg' % (output_dir, bname), Ip*255.0)
    cv2.waitKey(0)


if __name__ == '__main__':

    try:
        main()		 
    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)


