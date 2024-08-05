from paddle.vision.transforms import Compose, Resize, CenterCrop, Normalize
import paddle.nn.functional as F
import paddle.vision.models
import paddle
import paddle.text
import paddle.nn
paddle.set_default_dtype('float32')
paddle.text.datasets.UCIHousing(mode='train')
paddle.text.datasets.UCIHousing(mode='test')

# Yolov9


class YoloV9(paddle.nn.Layer):
    def __init__(self, num_classes=80):
        super(YoloV9, self).__init__()
        self.num_classes = num_classes
        self.backbone = paddle.vision.models.resnet50(pretrained=True)
        self.head = paddle.nn.Sequential(
            paddle.nn.Conv2D(2048, 256, 1),
            paddle.nn.ReLU(),
        )
