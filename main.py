from copyreg import dispatch_table
import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import statistics
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import math
from collections import Counter, defaultdict
from collections import deque
# import albumentations as A
from PIL import Image as im
import onnx
from onnx_tf.backend import prepare
from tensorflow import keras
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', 'data/video/test_n.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', "outputs/video_results/test1.avi", 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_float('show', 0, 'to show output')

## Loading models
print("Started Initial Configurations --------------")
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model_esrgan = hub.load(SAVED_MODEL_PATH)  ##ESRGAN MODEL
model_unet = onnx.load('model_data/unet.onnx')  # segmentation model
gender_model = keras.models.load_model("model_data/gender.h5")  # gender model path
age_model = keras.models.load_model("model_data/age.h5")  # age model path

print("Initial configuration finished --------------------")


def preprocess_image(image_batch):
    """ Loads images as batches and preprocesses to make it model ready
      Args:
        image_batch: Batches of images
  """
    shr_image = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    shr_size = (tf.convert_to_tensor(shr_image.shape[1:-1]) // 4) * 4
    shr_image = tf.image.crop_to_bounding_box(shr_image, 0, 0, shr_size[0], shr_size[1])
    shr_image = tf.cast(shr_image, tf.float32)
    return shr_image


def resize_func(img):
    img = tf.image.resize(
        img, (128, 128), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=False,
        antialias=False, name=None
    ) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img

outputs_ids_age = defaultdict(list)
outputs_ids_gd = defaultdict(list)
outputs_show = {}

def age_gender_pred(images_dict):
    global outputs_show
    file = open("outputs/labels.txt", "wb")
    tf_rep = prepare(model_unet)
    for id in images_dict:
        id_images = images_dict[id]
        # print(f"Len for id {id} = ", len(id_images))
        if len(id_images) == 4:
            paths = f"outputs/persons/person-{id}"
            if not os.path.exists(paths):
                os.makedirs(paths)
            lr_image = preprocess_image(id_images)  # Change path or image accordingly
            hr_image = model_esrgan(lr_image)
            img_batch = np.array(hr_image, dtype='float32')
            # print(img_batch[0, :, :, :].astype("uint8"))
            # print(img_batch[0, :, :, :].min(),  img_batch[0, :, :, :].max())
            # cv2.imshow('super resolved image', img_batch[0, :, :, :].astype("uint8"))
            # cv2.waitKey(1)
            encoded_images = tf.map_fn(resize_func, img_batch)
            op = tf_rep.run(np.asarray(encoded_images, dtype=np.float32))
            res = op._0
            res[res > 0.0] = 1
            res[res <= 0.0] = 0
            mean = np.mean(res, axis=0)
            mean1 = mean.copy()
            mean1 = np.moveaxis(mean1, -1, 0)
            mean1 = np.moveaxis(mean1, -1, 1)
            assert mean1.shape[0] == 128
            # plt.imshow(mean1.astype("uint8"))
            assert mean.shape[0] == 1
            ans_gender = gender_model.predict(mean)
            if ans_gender[0][0] >= 0.5:
                # print("Male")
                gd = "Male"
            else:
                # print("Female")
                gd = "Female"
            ans_age = age_model.predict(mean)
            # print('Age is: {}'.format(int(ans_age[0][0])))
            timing = str(time.time()).split(".")[0]
            cv2.imwrite(paths + f"/{timing}.jpg", np.array(id_images[0]).astype(np.uint8))
            outputs_ids_age[id].append(int(ans_age[0][0]))
            outputs_ids_gd[id].append(gd)
            images_dict[id].pop(0)
    for id in outputs_ids_age:
        mini = np.percentile(outputs_ids_age[id], q=25)
        maxi = np.percentile(outputs_ids_age[id], q=75)
        median_age = np.median(outputs_ids_age[id])
        mode_gd = statistics.mode(outputs_ids_gd[id])
        outputs_show[id] = (median_age, mode_gd)
        # print(outputs_show)
        # print(f"Person - {id}, Gender - {mode_gd}, Age Range- ({mini}-{maxi}), Age - {median_age}\n")
        file.write(f"Person - {id}, Gender - {mode_gd}, Age Range- ({mini}-{maxi}), Age - {median_age}\n")
    file.close()
    


def main(_argv):
    global outputs_show
    # Initialising the Keras model to predict Demographics of image
    # model = tf.keras.models.load_model('model_data/New_32CL_5LR_43Epoc')

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416
    video_path = FLAGS.video

    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    memory = {}
    cropped_images = defaultdict(list)

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_num += 1
        # print('Frame #: ', frame_num)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.50
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        # print(names)
        names = np.array(names)
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        disp = deepcopy(frame)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # Tracking midpoints
            midpoint = track.tlbr_midpoint(bbox)

            if track.track_id not in memory:
                cropped_images[track.track_id] = []
                memory[track.track_id] = deque(maxlen=2)

            memory[track.track_id].append(midpoint)
            previous_midpoint = memory[track.track_id][0]

            # --------------- cropping image ---------
            xmin, ymin, xmax, ymax = bbox
            cropped_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            image_data_flat = cropped_img.shape[0] * cropped_img.shape[1]
            if image_data_flat > 120 * 60:
                cropped_img = cv2.resize(cropped_img, (60, 120), cv2.INTER_AREA)
            else:
                cropped_img = cv2.resize(cropped_img, (60, 120), cv2.INTER_LINEAR)
            cropped_images[track.track_id].append(cropped_img)

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            # cv2.imshow(f"tracked - {track.track_id}", frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            cv2.rectangle(disp, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.line(disp, midpoint, previous_midpoint, (0, 255, 0), 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(
            # track.track_id)))*17, int(bbox[1])), color, -1)
            try:
                cv2.putText(disp, class_name + "-" + str(track.track_id) + " " + str(outputs_show[track.track_id]), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.5,
                            (255, 255, 255), 2)
                # print("done")
            except:
                pass
        age_gender_pred(cropped_images)  # result output
        # calculate frames per second of entire model
        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = np.asarray(disp)
        result = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
        
        if FLAGS.show:
            cv2.imshow("Output Video", result)

    # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xff == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
