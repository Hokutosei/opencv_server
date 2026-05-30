import cv2
import os
import ssl
import urllib.request
import logging

from config import MODELS_DIR, SSD_INPUT_SIZE, SSD_MEAN, DETECTION_CONF_THRESHOLD, DETECTION_NMS_THRESHOLD

logger = logging.getLogger(__name__)

COCO_LABELS = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Embedded deploy.prototxt (avoids network download for this config file)
DEPLOY_PROTOTXT = r'''name: "SSD_300x300"
input: "data"
input_shape { dim: 1 dim: 3 dim: 300 dim: 300 }
layer { name: "data_bn" type: "BatchNorm" bottom: "data" top: "data_bn"
  param { lr_mult: 0 decay_mult: 0 } param { lr_mult: 0 decay_mult: 0 } param { lr_mult: 0 decay_mult: 0 } }
layer { name: "data_scale" type: "Scale" bottom: "data_bn" top: "data_bn"
  param { lr_mult: 1.0 decay_mult: 0.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  scale_param { bias_term: true } }
layer { name: "conv1_1" type: "Convolution" bottom: "data_bn" top: "conv1_1"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 32 pad: 1 kernel_size: 3 stride: 2 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv1_1_relu" type: "ReLU" bottom: "conv1_1" top: "conv1_1" }
layer { name: "conv1_2" type: "Convolution" bottom: "conv1_1" top: "conv1_2"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 32 pad: 1 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv1_2_relu" type: "ReLU" bottom: "conv1_2" top: "conv1_2" }
layer { name: "conv2_1" type: "Convolution" bottom: "conv1_2" top: "conv2_1"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 64 pad: 1 kernel_size: 3 stride: 2 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv2_1_relu" type: "ReLU" bottom: "conv2_1" top: "conv2_1" }
layer { name: "conv2_2" type: "Convolution" bottom: "conv2_1" top: "conv2_2"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 64 pad: 1 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv2_2_relu" type: "ReLU" bottom: "conv2_2" top: "conv2_2" }
layer { name: "pool1" type: "Pooling" bottom: "conv2_2" top: "pool1" pooling_param { pool: MAX kernel_size: 2 stride: 2 } }
layer { name: "conv3_1" type: "Convolution" bottom: "pool1" top: "conv3_1"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 64 pad: 1 kernel_size: 1 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv3_1_relu" type: "ReLU" bottom: "conv3_1" top: "conv3_1" }
layer { name: "conv3_2" type: "Convolution" bottom: "conv3_1" top: "conv3_2"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 128 pad: 1 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv3_2_relu" type: "ReLU" bottom: "conv3_2" top: "conv3_2" }
layer { name: "conv3_3" type: "Convolution" bottom: "conv3_2" top: "conv3_3"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 128 pad: 1 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv3_3_relu" type: "ReLU" bottom: "conv3_3" top: "conv3_3" }
layer { name: "pool2" type: "Pooling" bottom: "conv3_3" top: "pool2" pooling_param { pool: MAX kernel_size: 2 stride: 2 } }
layer { name: "conv4_1" type: "Convolution" bottom: "pool2" top: "conv4_1"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 128 pad: 1 kernel_size: 1 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv4_1_relu" type: "ReLU" bottom: "conv4_1" top: "conv4_1" }
layer { name: "conv4_2" type: "Convolution" bottom: "conv4_1" top: "conv4_2"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 256 pad: 1 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv4_2_relu" type: "ReLU" bottom: "conv4_2" top: "conv4_2" }
layer { name: "conv4_3" type: "Convolution" bottom: "conv4_2" top: "conv4_3"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 256 pad: 1 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv4_3_relu" type: "ReLU" bottom: "conv4_3" top: "conv4_3" }
layer { name: "conv5_1" type: "Convolution" bottom: "conv4_3" top: "conv5_1"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 128 pad: 1 kernel_size: 1 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv5_1_relu" type: "ReLU" bottom: "conv5_1" top: "conv5_1" }
layer { name: "conv5_2" type: "Convolution" bottom: "conv5_1" top: "conv5_2"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 256 pad: 1 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv5_2_relu" type: "ReLU" bottom: "conv5_2" top: "conv5_2" }
layer { name: "conv5_3" type: "Convolution" bottom: "conv5_2" top: "conv5_3"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 256 pad: 1 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv5_3_relu" type: "ReLU" bottom: "conv5_3" top: "conv5_3" }
layer { name: "pool3" type: "Pooling" bottom: "conv5_3" top: "pool3" pooling_param { pool: MAX kernel_size: 2 stride: 2 } }
layer { name: "fc6" type: "Convolution" bottom: "pool3" top: "fc6"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 1024 pad: 3 kernel_size: 6 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "fc6_relu" type: "ReLU" bottom: "fc6" top: "fc6" }
layer { name: "fc7" type: "Convolution" bottom: "fc6" top: "fc7"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 1024 kernel_size: 1 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "fc7_relu" type: "ReLU" bottom: "fc7" top: "fc7" }
layer { name: "conv6_1" type: "Convolution" bottom: "fc7" top: "conv6_1"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 256 pad: 1 kernel_size: 1 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv6_1_relu" type: "ReLU" bottom: "conv6_1" top: "conv6_1" }
layer { name: "conv6_2" type: "Convolution" bottom: "conv6_1" top: "conv6_2"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 512 pad: 1 kernel_size: 3 stride: 2 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv6_2_relu" type: "ReLU" bottom: "conv6_2" top: "conv6_2" }
layer { name: "conv7_1" type: "Convolution" bottom: "conv6_2" top: "conv7_1"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 128 pad: 1 kernel_size: 1 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv7_1_relu" type: "ReLU" bottom: "conv7_1" top: "conv7_1" }
layer { name: "conv7_2" type: "Convolution" bottom: "conv7_1" top: "conv7_2"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 256 pad: 1 kernel_size: 3 stride: 2 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv7_2_relu" type: "ReLU" bottom: "conv7_2" top: "conv7_2" }
layer { name: "conv8_1" type: "Convolution" bottom: "conv7_2" top: "conv8_1"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 128 pad: 1 kernel_size: 1 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv8_1_relu" type: "ReLU" bottom: "conv8_1" top: "conv8_1" }
layer { name: "conv8_2" type: "Convolution" bottom: "conv8_1" top: "conv8_2"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 256 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv8_2_relu" type: "ReLU" bottom: "conv8_2" top: "conv8_2" }
layer { name: "conv9_1" type: "Convolution" bottom: "conv8_2" top: "conv9_1"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 128 pad: 1 kernel_size: 1 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv9_1_relu" type: "ReLU" bottom: "conv9_1" top: "conv9_1" }
layer { name: "conv9_2" type: "Convolution" bottom: "conv9_1" top: "conv9_2"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 256 kernel_size: 3 weight_filler { type: "msra" variance_norm: FAN_OUT } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv9_2_relu" type: "ReLU" bottom: "conv9_2" top: "conv9_2" }
layer { name: "conv4_3_norm" type: "Normalize" bottom: "conv4_3" top: "conv4_3_norm"
  norm_param { across_spatial: false scale_filler { type: "constant" value: 20 } channel_shared: false } }
layer { name: "conv4_3_norm_mbox_loc" type: "Convolution" bottom: "conv4_3_norm" top: "conv4_3_norm_mbox_loc"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 16 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv4_3_norm_mbox_loc_perm" type: "Permute" bottom: "conv4_3_norm_mbox_loc" top: "conv4_3_norm_mbox_loc_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv4_3_norm_mbox_loc_flat" type: "Flatten" bottom: "conv4_3_norm_mbox_loc_perm" top: "conv4_3_norm_mbox_loc_flat"
  flatten_param { axis: 1 } }
layer { name: "conv4_3_norm_mbox_conf" type: "Convolution" bottom: "conv4_3_norm" top: "conv4_3_norm_mbox_conf"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 8 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv4_3_norm_mbox_conf_perm" type: "Permute" bottom: "conv4_3_norm_mbox_conf" top: "conv4_3_norm_mbox_conf_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv4_3_norm_mbox_conf_flat" type: "Flatten" bottom: "conv4_3_norm_mbox_conf_perm" top: "conv4_3_norm_mbox_conf_flat"
  flatten_param { axis: 1 } }
layer { name: "conv4_3_norm_mbox_priorbox" type: "PriorBox" bottom: "conv4_3_norm" bottom: "data" top: "conv4_3_norm_mbox_priorbox"
  prior_box_param { min_size: 30.0 max_size: 60.0 aspect_ratio: 2 flip: true clip: false variance: 0.1 variance: 0.1 variance: 0.2 variance: 0.2 step: 8 offset: 0.5 } }
layer { name: "fc7_mbox_loc" type: "Convolution" bottom: "fc7" top: "fc7_mbox_loc"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 24 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "fc7_mbox_loc_perm" type: "Permute" bottom: "fc7_mbox_loc" top: "fc7_mbox_loc_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "fc7_mbox_loc_flat" type: "Flatten" bottom: "fc7_mbox_loc_perm" top: "fc7_mbox_loc_flat"
  flatten_param { axis: 1 } }
layer { name: "fc7_mbox_conf" type: "Convolution" bottom: "fc7" top: "fc7_mbox_conf"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 12 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "fc7_mbox_conf_perm" type: "Permute" bottom: "fc7_mbox_conf" top: "fc7_mbox_conf_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "fc7_mbox_conf_flat" type: "Flatten" bottom: "fc7_mbox_conf_perm" top: "fc7_mbox_conf_flat"
  flatten_param { axis: 1 } }
layer { name: "fc7_mbox_priorbox" type: "PriorBox" bottom: "fc7" bottom: "data" top: "fc7_mbox_priorbox"
  prior_box_param { min_size: 60.0 max_size: 111.0 aspect_ratio: 2 aspect_ratio: 3 flip: true clip: false variance: 0.1 variance: 0.1 variance: 0.2 variance: 0.2 step: 16 offset: 0.5 } }
layer { name: "conv6_2_mbox_loc" type: "Convolution" bottom: "conv6_2" top: "conv6_2_mbox_loc"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 24 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv6_2_mbox_loc_perm" type: "Permute" bottom: "conv6_2_mbox_loc" top: "conv6_2_mbox_loc_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv6_2_mbox_loc_flat" type: "Flatten" bottom: "conv6_2_mbox_loc_perm" top: "conv6_2_mbox_loc_flat"
  flatten_param { axis: 1 } }
layer { name: "conv6_2_mbox_conf" type: "Convolution" bottom: "conv6_2" top: "conv6_2_mbox_conf"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 12 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv6_2_mbox_conf_perm" type: "Permute" bottom: "conv6_2_mbox_conf" top: "conv6_2_mbox_conf_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv6_2_mbox_conf_flat" type: "Flatten" bottom: "conv6_2_mbox_conf_perm" top: "conv6_2_mbox_conf_flat"
  flatten_param { axis: 1 } }
layer { name: "conv6_2_mbox_priorbox" type: "PriorBox" bottom: "conv6_2" bottom: "data" top: "conv6_2_mbox_priorbox"
  prior_box_param { min_size: 111.0 max_size: 162.0 aspect_ratio: 2 aspect_ratio: 3 flip: true clip: false variance: 0.1 variance: 0.1 variance: 0.2 variance: 0.2 step: 32 offset: 0.5 } }
layer { name: "conv7_2_mbox_loc" type: "Convolution" bottom: "conv7_2" top: "conv7_2_mbox_loc"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 24 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv7_2_mbox_loc_perm" type: "Permute" bottom: "conv7_2_mbox_loc" top: "conv7_2_mbox_loc_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv7_2_mbox_loc_flat" type: "Flatten" bottom: "conv7_2_mbox_loc_perm" top: "conv7_2_mbox_loc_flat"
  flatten_param { axis: 1 } }
layer { name: "conv7_2_mbox_conf" type: "Convolution" bottom: "conv7_2" top: "conv7_2_mbox_conf"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 12 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv7_2_mbox_conf_perm" type: "Permute" bottom: "conv7_2_mbox_conf" top: "conv7_2_mbox_conf_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv7_2_mbox_conf_flat" type: "Flatten" bottom: "conv7_2_mbox_conf_perm" top: "conv7_2_mbox_conf_flat"
  flatten_param { axis: 1 } }
layer { name: "conv7_2_mbox_priorbox" type: "PriorBox" bottom: "conv7_2" bottom: "data" top: "conv7_2_mbox_priorbox"
  prior_box_param { min_size: 162.0 max_size: 213.0 aspect_ratio: 2 aspect_ratio: 3 flip: true clip: false variance: 0.1 variance: 0.1 variance: 0.2 variance: 0.2 step: 64 offset: 0.5 } }
layer { name: "conv8_2_mbox_loc" type: "Convolution" bottom: "conv8_2" top: "conv8_2_mbox_loc"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 16 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv8_2_mbox_loc_perm" type: "Permute" bottom: "conv8_2_mbox_loc" top: "conv8_2_mbox_loc_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv8_2_mbox_loc_flat" type: "Flatten" bottom: "conv8_2_mbox_loc_perm" top: "conv8_2_mbox_loc_flat"
  flatten_param { axis: 1 } }
layer { name: "conv8_2_mbox_conf" type: "Convolution" bottom: "conv8_2" top: "conv8_2_mbox_conf"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 8 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv8_2_mbox_conf_perm" type: "Permute" bottom: "conv8_2_mbox_conf" top: "conv8_2_mbox_conf_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv8_2_mbox_conf_flat" type: "Flatten" bottom: "conv8_2_mbox_conf_perm" top: "conv8_2_mbox_conf_flat"
  flatten_param { axis: 1 } }
layer { name: "conv8_2_mbox_priorbox" type: "PriorBox" bottom: "conv8_2" bottom: "data" top: "conv8_2_mbox_priorbox"
  prior_box_param { min_size: 213.0 max_size: 264.0 aspect_ratio: 2 flip: true clip: false variance: 0.1 variance: 0.1 variance: 0.2 variance: 0.2 step: 100 offset: 0.5 } }
layer { name: "conv9_2_mbox_loc" type: "Convolution" bottom: "conv9_2" top: "conv9_2_mbox_loc"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 16 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv9_2_mbox_loc_perm" type: "Permute" bottom: "conv9_2_mbox_loc" top: "conv9_2_mbox_loc_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv9_2_mbox_loc_flat" type: "Flatten" bottom: "conv9_2_mbox_loc_perm" top: "conv9_2_mbox_loc_flat"
  flatten_param { axis: 1 } }
layer { name: "conv9_2_mbox_conf" type: "Convolution" bottom: "conv9_2" top: "conv9_2_mbox_conf"
  param { lr_mult: 1.0 decay_mult: 1.0 } param { lr_mult: 2.0 decay_mult: 0.0 }
  convolution_param { num_output: 8 pad: 1 kernel_size: 3 weight_filler { type: "msra" } bias_filler { type: "constant" value: 0 } } }
layer { name: "conv9_2_mbox_conf_perm" type: "Permute" bottom: "conv9_2_mbox_conf" top: "conv9_2_mbox_conf_perm"
  permute_param { order: 0 order: 2 order: 3 order: 1 } }
layer { name: "conv9_2_mbox_conf_flat" type: "Flatten" bottom: "conv9_2_mbox_conf_perm" top: "conv9_2_mbox_conf_flat"
  flatten_param { axis: 1 } }
layer { name: "conv9_2_mbox_priorbox" type: "PriorBox" bottom: "conv9_2" bottom: "data" top: "conv9_2_mbox_priorbox"
  prior_box_param { min_size: 264.0 max_size: 315.0 aspect_ratio: 2 flip: true clip: false variance: 0.1 variance: 0.1 variance: 0.2 variance: 0.2 step: 300 offset: 0.5 } }
layer { name: "mbox_loc" type: "Concat"
  bottom: "conv4_3_norm_mbox_loc_flat" bottom: "fc7_mbox_loc_flat" bottom: "conv6_2_mbox_loc_flat"
  bottom: "conv7_2_mbox_loc_flat" bottom: "conv8_2_mbox_loc_flat" bottom: "conv9_2_mbox_loc_flat"
  top: "mbox_loc" concat_param { axis: 1 } }
layer { name: "mbox_conf" type: "Concat"
  bottom: "conv4_3_norm_mbox_conf_flat" bottom: "fc7_mbox_conf_flat" bottom: "conv6_2_mbox_conf_flat"
  bottom: "conv7_2_mbox_conf_flat" bottom: "conv8_2_mbox_conf_flat" bottom: "conv9_2_mbox_conf_flat"
  top: "mbox_conf" concat_param { axis: 1 } }
layer { name: "mbox_priorbox" type: "Concat"
  bottom: "conv4_3_norm_mbox_priorbox" bottom: "fc7_mbox_priorbox" bottom: "conv6_2_mbox_priorbox"
  bottom: "conv7_2_mbox_priorbox" bottom: "conv8_2_mbox_priorbox" bottom: "conv9_2_mbox_priorbox"
  top: "mbox_priorbox" concat_param { axis: 2 } }
layer { name: "mbox_conf_reshape" type: "Reshape" bottom: "mbox_conf" top: "mbox_conf_reshape"
  reshape_param { shape { dim: 0 dim: -1 dim: 2 } } }
layer { name: "mbox_conf_softmax" type: "Softmax" bottom: "mbox_conf_reshape" top: "mbox_conf_softmax"
  softmax_param { axis: 2 } }
layer { name: "mbox_conf_flatten" type: "Flatten" bottom: "mbox_conf_softmax" top: "mbox_conf_flatten"
  flatten_param { axis: 1 } }
layer { name: "detection_out" type: "DetectionOutput"
  bottom: "mbox_loc" bottom: "mbox_conf_flatten" bottom: "mbox_priorbox" top: "detection_out"
  include { phase: TEST }
  detection_output_param { num_classes: 2 share_location: true background_label_id: 0
    nms_param { nms_threshold: 0.45 top_k: 400 } code_type: CENTER_SIZE keep_top_k: 200 confidence_threshold: 0.01 } }
'''


def ensure_models():
    """Ensure SSD model files exist. Returns (prototxt_path, model_path) or (prototxt_path, None) if download fails."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    prototxt_path = os.path.join(MODELS_DIR, "deploy.prototxt")
    model_path = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt_path):
        logger.info("Writing embedded deploy.prototxt...")
        with open(prototxt_path, "w") as f:
            f.write(DEPLOY_PROTOTXT)
        logger.info("Wrote deploy.prototxt")

    if not os.path.exists(model_path):
        logger.info("Downloading SSD caffemodel (~10MB)...")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Python/OpenCV-Server"})
            with urllib.request.urlopen(req, context=ctx) as response:
                with open(model_path, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            logger.info("Downloaded SSD caffemodel")
        except Exception as e:
            logger.warning(f"Failed to download SSD model: {e}")
            logger.warning("Object detection disabled. Only Haar cascade face detection available.")
            if os.path.exists(model_path):
                os.remove(model_path)
            return prototxt_path, None

    return prototxt_path, model_path


class ObjectDetector:
    def __init__(self, prototxt_path: str = None, model_path: str = None):
        if prototxt_path is None or model_path is None:
            prototxt_path, model_path = ensure_models()

        if model_path is None:
            self._model = None
            logger.warning("ObjectDetector initialized in disabled mode (no model)")
            return

        self._model = cv2.dnn.DetectionModel(prototxt_path, model_path)
        self._model.setInputSize(SSD_INPUT_SIZE)
        self._model.setInputMean(SSD_MEAN)
        self._model.setInputSwapRB(False)
        self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def detect(self, frame, conf_threshold: float = None) -> list[dict]:
        """
        Detect objects in a BGR frame.
        Returns list of dicts: {class_id, label, confidence, bbox: (x, y, w, h)}
        """
        if self._model is None:
            return []

        if conf_threshold is None:
            conf_threshold = DETECTION_CONF_THRESHOLD

        class_ids, confidences, boxes = self._model.detect(
            frame,
            confThreshold=conf_threshold,
            nmsThreshold=DETECTION_NMS_THRESHOLD
        )

        results = []
        if class_ids is not None and len(class_ids) > 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
                label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else f"class_{class_id}"
                results.append({
                    "class_id": int(class_id),
                    "label": label,
                    "confidence": float(confidence),
                    "bbox": (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                })
        return results
