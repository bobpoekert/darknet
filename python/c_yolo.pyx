import numpy as np
cimport numpy as np
from cython cimport view

from libc.stdlib cimport free, calloc
from libc.string cimport memcpy
import os

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
        int w
        int h
        layer *layers

    float *network_predict(network net, float *input)
    void free_network(network net)
    void set_batch_network(network *net, int b)

cdef extern from "../src/yolo.c":
    void convert_yolo_detections(float *predictions, int classes, int num, int square, int size, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)

cdef extern from "../src/image.h":
    ctypedef struct image:
        int h
        int w
        int c # channels
        float *data

    image resize_image(image inp, int w, int h)
    void free_image(image inp)
    image load_image(char *filename, int w, int h, int c)
    image load_image_color(char *filename, int w, int h)

cdef extern from "../src/parser.h":
    network parse_network_cfg(char *filename)
    void load_weights(network *net, char *filename)

cdef class Image:

    cdef image img

    # this is here to keep the reference to the underlying buffer alive
    # since we don't copy the buffer into img, just point to it
    cdef np.ndarray ndarray

    def to_ndarray(self):
        cdef int width = self.img.w
        cdef int height = self.img.h
        cdef int channels = self.img.c
        cdef int size = width * height * channels * sizeof(float)
        cdef np.ndarray res
        if self.ndarray is not None:
            return self.ndarray
        else:
            res = np.empty((height, width, channels), dtype=np.float32, order='C')
            memcpy(res.data, <char *> self.img.data, size)
            return res

    @staticmethod
    def from_ndarray(arr):
        res = Image()
        res.set_ndarray(arr)
        return res

    @staticmethod
    def load(fname, width, height):
        cdef image res_img = load_image_color(fname, width, height)
        cdef Image res = Image()
        res.img = res_img
        return res

    def width(self):
        return self.img.w

    def height(self):
        return self.img.h

    def channels(self):
        return self.img.c

    cdef image get_image(self):
        return self.img

    cdef c_set_ndarray(self, np.ndarray inp):
        cdef np.ndarray floats = inp.astype(np.float32, order='C')
        self.ndarray = floats
        cdef image input_image
        input_image.w = floats.shape[1]
        input_image.h = floats.shape[0]
        input_image.c = floats.shape[2]
        input_image.data = <float *> floats.data
        self.img = input_image

    def set_ndarray(self, inp):
        self.c_set_ndarray(inp)

    cdef set_image(self, image img):
        self.img = img

    def __dealloc__(self):
        if self.ndarray is None and self.img.data:
            free_image(self.img)

    cdef Image c_resize(self, int w, int h):
        print w, h
        cdef image resized = resize_image(self.img, w, h)
        cdef Image res = Image()
        res.set_image(resized)
        return res

    def resize(self, w, h):
        return self.c_resize(w, h)

cdef class YOLONet:

    cdef network net
    cdef float nms
    cdef float thresh

    def __cinit__(self, bytes cfgfile, bytes weightfile):
        assert os.path.exists(cfgfile), 'net config file does not exist'
        assert os.path.exists(weightfile), 'weight file does not exist'
        self.nms = 0.4
        self.thresh = 0.2
        cdef network net = parse_network_cfg(cfgfile)
        load_weights(&net, weightfile)
        set_batch_network(&net, 1)
        self.net = net

    def __dealloc__(self):
        free_network(self.net)

    cdef Image prep_image(self, Image img):
        cdef int w = self.net.w
        cdef int h = self.net.h

        if img.width() == w and img.height() == h:
            return img
        else:
            return img.c_resize(w, h)

    def predict(self, thing):
        if isinstance(thing, Image):
            return self.predict_image(thing)
        elif hasattr(thing, 'dtype'):
            return self.predict_ndarray(thing)

    def predict_ndarray(self, np.ndarray arr):
        cdef Image img = Image()
        img.c_set_ndarray(arr)
        return self.predict_image(img)

    def predict_image(self, Image inp):
        cdef layer l
        cdef Image prepped_pyimage = self.prep_image(inp)
        cdef image prepped_image = prepped_pyimage.get_image()
        cdef int num_detections
        cdef float **probs
        cdef float *predictions
        cdef box *b
        cdef int classes
        try:
            l = self.net.layers[self.net.n-1]
            classes = l.classes
            num_detections = l.side*l.side*l.n

            probs = <float **> calloc(num_detections, sizeof(float *))
            for j in range(num_detections):
                probs[j] = <float *> calloc(l.classes, sizeof(float *))
            boxes = <box *> calloc(num_detections, sizeof(box))

            predictions = network_predict(self.net, prepped_image.data)
            convert_yolo_detections(
                    predictions,
                    l.classes, l.n, l.sqrt, l.side,
                    self.net.w, self.net.h,
                    self.thresh, probs, boxes, 0)
            if self.nms > 0:
                do_nms(boxes, probs, num_detections, l.classes, self.nms)

            res = []
            for i in range(num_detections):
                b = &boxes[i]
                res.append((b.x, b.y, b.w, b.h, [probs[i][j] for j in range(l.classes)]))
            return res
        finally:
            if probs:
                for i in range(num_detections):
                    free(probs[i])
                free(probs)
            if boxes:
                free(boxes)
