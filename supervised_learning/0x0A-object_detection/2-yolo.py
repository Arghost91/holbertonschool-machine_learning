#!/usr/bin/env python3
"""
Process Outputs
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Class Yolo based on 0_yolo.py
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        * model_path is the path to where a Darknet Keras model is stored
        * classes_path is the path to where the list of class names used
        for the Darknet model, listed in order of index, can be found
        * class_t is a float representing the box score threshold for the
        initial filtering step
        * nms_t is a float representing the IOU threshold for non-max
        suppression
        * anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
            * outputs is the number of outputs (predictions) made by the
            Darknet model
            * anchor_boxes is the number of anchor boxes used for each
            prediction
            * 2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(filepath=model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [names[:-1] for names in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        * outputs is a list of numpy.ndarrays containing the predictions
        from the Darknet model for a single image:
            * Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
                * grid_height & grid_width => the height and width of the
                grid used for the output
                * anchor_boxes => the number of anchor boxes used
                * 4 => (t_x, t_y, t_w, t_h)
                * 1 => box_confidence
                * classes => class probabilities for all classes
        * image_size is a numpy.ndarray containing the image’s original
        size [image_height, image_width]
        * Returns a tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        i_h, i_w = image_size
        i = 0
        for out in outputs:
            g_h, g_w, a_b, _ = out.shape
            boxes.append(out[:, :, :, :4])
            confid = (1 / (1 + np.exp(-out[:, :, :, 4:5])))
            box_confidences.append(confid)
            prob = (1 / (1 + np.exp(-out[:, :, :, 5:])))
            box_class_probs.append(prob)

            t_x = out[:, :, :, 0]
            t_y = out[:, :, :, 1]
            t_w = out[:, :, :, 2]
            t_h = out[:, :, :, 3]

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            cx = np.indices((g_h, g_h, a_b))[1]
            cy = np.indices((g_h, g_h, a_b))[0]

            bx = ((1 / (1 + np.exp(-t_x))) + cx) / g_w
            by = ((1 / (1 + np.exp(-t_y))) + cy) / g_h
            bw = (np.exp(t_w) * pw) / self.model.input.shape[1].value
            bh = (np.exp(t_h) * ph) / self.model.input.shape[2].value

            x1 = (bx - (bw / 2)) * i_w
            x2 = (bx + (bw / 2)) * i_w
            y1 = (by - (bh / 2)) * i_h
            y2 = (by + (bh / 2)) * i_h

            boxes[i][:, :, :, 0] = x1
            boxes[i][:, :, :, 1] = y1
            boxes[i][:, :, :, 2] = x2
            boxes[i][:, :, :, 3] = y2
            i += 1
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        * boxes: a list of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes,
        4) containing the processed boundary boxes for each output, respectively
        * box_confidences: a list of numpy.ndarrays of shape (grid_height, grid_width,
        anchor_boxes, 1) containing the processed box confidences for each output, respectively
        * box_class_probs: a list of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes,
        classes) containing the processed box class probabilities for each output, respectively
        * Returns a tuple of (filtered_boxes, box_classes, box_scores)
        """
        box = [j.reshape(-1, 4) for j in boxes]
        box_scores = []
        scores = []
        boxes = []
        cla = []
        for i in range(len(boxes)):
            box_scores.append(box_confidences[i] * box_class_probs[i])
            classes = np.argmax(box_scores[i], axis=3)
            class_scores = np.max(box_scores[i], axis=3)
            filt = class_scores[i] >= self.class_t
        for j, k in zip(class_scores, filt):
            scores += j[k]
        for l, m in zip(boxes, filt):
            boxes += l[m]
        for n, o in zip(classes, filt):
            cla += n[o].flatten()
        classes = np.concatenate(classes)
        class_scores = np.concatenate(class_scores)
        box = np.concatenate(box)
        return (filtered_boxes, box_classes, box_scores)
