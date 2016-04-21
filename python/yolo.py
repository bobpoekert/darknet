from c_yolo import YOLONet
import os

here = os.path.abspath('%s/..' % __file__)

_net = None
def yolo_net():
    global _net
    if _net is None:
        _net = YOLONet(
                os.path.join(here, 'yolo.cfg'),
                os.path.join(here, 'yolo.weights'))
    return _net
