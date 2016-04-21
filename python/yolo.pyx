import numpy as np
cimport numpy as np
from libc.stdlib cimport free

cdef extern from "../src/layer.h":
    ctypedef struct layer:
        int side
        int n
        int sqrt
        int classes


cdef extern from "../src/box.h":
    ctypedef struct box:
        float x
        float y
        float w
        float h

    ctypedef struct dbox:
        float dx, dy, dw, dh

    void do_nms(box *boxes, float **probs, int total, int classes, float thresh)

cdef extern from "../src/network.h":
    ctypedef struct network:
        int n
        layer *layers

    float *network_predict(network net, float *input)
    void free_network(network net)
    void set_batch_network(network *net, int b)

cdef extern from "../src/yolo.c":
    void convert_yolo_detections(float *predictions, int classes, int num, int square, int size, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)

cdef extern from "../src/image.h":
    struct image:
        int h
        int w
        int c # channels
        float *data

cdef extern from "../src/parser.h":
    network parse_network_cfg(char *filename)
    void load_weights(network *net, char *filename)

cdef class YOLONet:

    cdef network net
    cdef float nms
    cdef float thresh

    def __cinit__(self, bytes cfgfile, bytes weightfile):
        self.nms = 0.4
        self.thresh = 0.2
        cdef network net = parse_network_cfg(cfgfile)
        load_weights(&net, weightfile)
        set_batch_network(&net, 1)
        self.net = net

    def __dealloc__(self):
        free_network(self.net)

    def predict(self, np.ndarray inp):
        cdef float **probs
        cdef box *boxes
        cdef layer l = self.net.layers[self.net.n-1]
        cdef np.ndarray[float, ndim=3, mode='c'] X = inp.astype(np.float32)
        cdef float *predictions = network_predict(self.net, &X[0,0,0])
        cdef int num_detections = l.side*l.side*l.n
        convert_yolo_detections(
                predictions,
                l.classes, l.n, l.sqrt, l.side, 1, 1,
                self.thresh, probs, boxes, 0)
        if self.nms > 0:
            do_nms(boxes, probs, num_detections, l.classes, self.nms)

        res = []
        cdef box *b
        for i in range(num_detections):
            b = &boxes[i]
            res.append((b.x, b.y, b.w, b.h, [probs[i][j] for j in range(l.classes)]))
        return res
