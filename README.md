# License_Plate_Recognition

## Summary 

License Plate Recognition System using WPOD-NET and YOLOv4 

Paper YOLO v4: https://arxiv.org/abs/2004.10934

## Prepare dataset 

Prepare your character dataset or download dataset: [license plate dataset](https://thigiacmaytinh.com/tai-nguyen-xu-ly-anh/tong-hop-data-xu-ly-anh/?fbclid=IwAR2tajA5Ku83kIrb09ovhmb_68Zmdwo9KvV_CSNBCTbuIIsiK_FUM4W4Dh8)

## Training 

Download pretrained model: [yolov4-tiny.conv.29](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.conv.29)

Run training_yolov4_for_character.ipynb to train 

## Testing

Run darknet/lpr.py --input_dir "your image path"

## References

https://github.com/AlexeyAB/darknet
