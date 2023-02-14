import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video1', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video2', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output1', None, 'path to output video')
flags.DEFINE_string('output2', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_string('masking_image', None, 'path to masking image file')

def calc_distance(feat1, feat2):
    distance = np.linalg.norm(feat1 - feat2)
    return distance

def main(_argv):
    #add calculate distance
    import cv2
    import settings
    from reid_onnx_helper import ReidHelper
    
    helper = ReidHelper(settings.ReID)
    feat1 = None

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric1 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker1 = Tracker(metric1)
    tracker2 = Tracker(metric2)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path1 = FLAGS.video1
    video_path2 = FLAGS.video2

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid1 = cv2.VideoCapture(int(video_path1))
        vid2 = cv2.VideoCapture(int(video_path2))
    except:
        vid1 = cv2.VideoCapture(video_path1)
        vid2 = cv2.VideoCapture(video_path2)

    out1 = None
    out2 = None

    # get video ready to save locally if flag is set
    if FLAGS.output1:
        # by default VideoCapture returns float instead of int
        width1 = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps1 = int(vid1.get(cv2.CAP_PROP_FPS))
        codec1 = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out1 = cv2.VideoWriter(FLAGS.output1, codec1, fps1, (width1, height1))
    
    if FLAGS.output2:
        width2 = int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps2 = int(vid2.get(cv2.CAP_PROP_FPS))
        codec2 = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out2 = cv2.VideoWriter(FLAGS.output2, codec2, fps2, (width2, height2))

    frame_num1 = 0
    frame_num2 = 0

    map_list_track1 = []
    map_list_track2 = []

    # while video is running
    while True:
        return_value1, frame1 = vid1.read()
        return_value2, frame2 = vid2.read()

        if return_value1:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image1 = Image.fromarray(frame1)
            #image.save('C:/Users/USER/cp2/yolov4-deepsort/outputs/sample.jpg','JPEG')
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num1 +=1
        print('Frame-1 #: ', frame_num1)
        frame_size1 = frame1.shape[:2]

        if return_value2:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image2 = Image.fromarray(frame2)
            #image.save('C:/Users/USER/cp2/yolov4-deepsort/outputs/sample.jpg','JPEG')
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num2 +=1
        print('Frame-2 #: ', frame_num2)
        frame_size2 = frame2.shape[:2]

        if FLAGS.masking_image:
            masking_image = cv2.imread(FLAGS.masking_image)/255
            image_data1 = cv2.resize(frame1*masking_image, (input_size, input_size))
        else:
            image_data1 = cv2.resize(frame1, (input_size, input_size))

        image_data1 = image_data1 / 255.
        image_data1 = image_data1[np.newaxis, ...].astype(np.float32)

        image_data2 = cv2.resize(frame2, (input_size, input_size))
        image_data2 = image_data2 / 255.
        image_data2 = image_data2[np.newaxis, ...].astype(np.float32)

        start_time = time.time()

        batch_data1 = tf.constant(image_data1)
        pred_bbox1 = infer(batch_data1)
        for key, value in pred_bbox1.items():
            boxes = value[:, :, 0:4]
            pred_conf1 = value[:, :, 4:]
        
        boxes1, scores1, classes1, valid_detections1 = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf1, (tf.shape(pred_conf1)[0], -1, tf.shape(pred_conf1)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
            
        batch_data2 = tf.constant(image_data2)
        pred_bbox2 = infer(batch_data2)
        for key, value in pred_bbox2.items():
            boxes = value[:, :, 0:4]
            pred_conf2 = value[:, :, 4:]

        boxes2, scores2, classes2, valid_detections2 = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf2, (tf.shape(pred_conf2)[0], -1, tf.shape(pred_conf2)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects1 = valid_detections1.numpy()[0]
        bboxes1 = boxes1.numpy()[0]
        bboxes1 = bboxes1[0:int(num_objects1)]
        scores1 = scores1.numpy()[0]
        scores1 = scores1[0:int(num_objects1)]
        classes1 = classes1.numpy()[0]
        classes1 = classes1[0:int(num_objects1)]

        num_objects2 = valid_detections2.numpy()[0]
        bboxes2 = boxes2.numpy()[0]
        bboxes2 = bboxes2[0:int(num_objects2)]
        scores2 = scores2.numpy()[0]
        scores2 = scores2[0:int(num_objects2)]
        classes2 = classes2.numpy()[0]
        classes2 = classes2[0:int(num_objects2)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h1, original_w1, _ = frame1.shape
        bboxes1 = utils.format_boxes(bboxes1, original_h1, original_w1)

        original_h2, original_w2, _ = frame2.shape
        bboxes2 = utils.format_boxes(bboxes2, original_h2, original_w2)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox1 = [bboxes1, scores1, classes1, num_objects1]
        pred_bbox2 = [bboxes2, scores2, classes2, num_objects2]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names1 = []
        deleted_indx1 = []

        names2 = []
        deleted_indx2 = []

        for i in range(num_objects1):
            class_indx1 = int(classes1[i])
            class_name1 = class_names[class_indx1]
            if class_name1 not in allowed_classes:
                deleted_indx1.append(i)
            else:
                names1.append(class_name1)
        
        for i in range(num_objects2):
            class_indx2 = int(classes2[i])
            class_name2 = class_names[class_indx2]
            if class_name2 not in allowed_classes:
                deleted_indx2.append(i)
            else:
                names2.append(class_name2)

        names1 = np.array(names1)
        count1 = len(names1)

        names2 = np.array(names2)
        count2 = len(names2)
        # if FLAGS.count:
        #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        #     print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes1 = np.delete(bboxes1, deleted_indx1, axis=0)
        scores1 = np.delete(scores1, deleted_indx1, axis=0)

        bboxes2 = np.delete(bboxes2, deleted_indx2, axis=0)
        scores2 = np.delete(scores2, deleted_indx2, axis=0)

        # encode yolo detections and feed to tracker
        features1 = encoder(frame1, bboxes1)
        detections1 = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes1, scores1, names1, features1)]

        features2 = encoder(frame2, bboxes2)
        detections2 = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes2, scores2, names2, features2)]


        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs1 = np.array([d.tlwh for d in detections1])
        scores1 = np.array([d.confidence for d in detections1])
        classes1 = np.array([d.class_name for d in detections1])
        indices1 = preprocessing.non_max_suppression(boxs1, classes1, nms_max_overlap, scores1)
        detections1 = [detections1[i] for i in indices1]

        gallery_img = []
        for i in range(len(detections1)):
            xmin, ymin, w, h = detections1[i].tlwh[0], detections1[i].tlwh[1], detections1[i].tlwh[2], detections1[i].tlwh[3]
            xmax = xmin + w
            ymax = ymin + h
            # print(f'detections1===={i} : {xmin, ymin, xmax, ymax}===========')
            gallery_img.append([frame1[int(ymin):int(ymax),int(xmin):int(xmax)],i])

            # xmin, ymin, xmax, ymax = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

        boxs2 = np.array([d.tlwh for d in detections2])
        scores2 = np.array([d.confidence for d in detections2])
        classes2 = np.array([d.class_name for d in detections2])
        indices2 = preprocessing.non_max_suppression(boxs2, classes2, nms_max_overlap, scores2)
        detections2 = [detections2[i] for i in indices2]        
        
        sort_rank = []
        for i in range(len(detections2)):
            xmin, ymin, w, h = detections2[i].tlwh[0], detections2[i].tlwh[1], detections2[i].tlwh[2], detections2[i].tlwh[3]
            xmax = xmin + w
            ymax = ymin + h
            query_img = frame2[int(ymin):int(ymax),int(xmin):int(xmax)]
            if query_img.shape[0] >= 785 and query_img.shape[1] >= 1150:
                reader = easyocr.Reader(['ko', 'en'], gpu=False)
                result =  reader.readtext(query_img, detail=0)

                max_text = ''
                for i in result:
                    if len(i) >= len(max_text):
                        max_text = i
                # print(f'------------max_text: {max_text}---------------------------------------')
                # print(f'------------result: {result}---------------------------------------')
                # print(f'detections1===={i} : {query_img.shape}===========')
                # cv2.imshow("",query_img)
                # cv2.waitKey(0)
                if len(gallery_img)!=0:
                    feat1 = helper.infer(query_img)
                    for j in range(len(gallery_img)):
                        feat = helper.infer(gallery_img[j][0])
                        distance = calc_distance(feat1, feat)
                        sort_rank.append([distance, gallery_img[j][1]])
                    sort_rank.sort(key=lambda x: x[0])
                    rank_1_img_order = sort_rank[0][1]

        # Call the tracker
        tracker1.predict()
        tracker1.update(detections1)

        tracker2.predict()
        tracker2.update(detections2)
        
        # update tracks
        track1_id_list = []
        for track in tracker1.tracks:
            track1_id_list.append(track.track_id)
        
        if len(map_list_track1)!=0:
            for idx,i in enumerate(map_list_track1):
                if i[0] not in track1_id_list:
                    del(map_list_track1[idx])

        for index, track in enumerate(tracker1.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            if len(sort_rank)!=0:
                if index == rank_1_img_order:
                    if any(track.track_id in l for l in map_list_track1):
                        for i in range(len(map_list_track1)):
                            if map_list_track1[i][0] == track.track_id:
                                save_car_plate = map_list_track1[i][1]
                                if len(save_car_plate) > len(max_text):
                                    car_plate = save_car_plate
                                else:
                                    map_list_track1[i][1] = max_text
                                    car_plate = max_text
                        # print(f'------------car_plate: {car_plate}---------------------------------------')
                    else:
                        map_list_track1.append([track.track_id, max_text])
                        car_plate = max_text
                        # print(f'------------car_plate: {car_plate}---------------------------------------')
                    cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len('map-')+len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame1, "map-" + class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    cv2.putText(frame1, car_plate ,(int(bbox[0]), int(bbox[3]+20)),0, 0.75, (255,255,255),2)

                else:
                    cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame1, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                
            else:
                if any(track.track_id in l for l in map_list_track1):
                    for i in range(len(map_list_track1)):
                        if map_list_track1[i][0] == track.track_id:
                            car_plate = map_list_track1[i][1]
                    cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len('map-')+len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame1, "map-" + class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    cv2.putText(frame1, car_plate ,(int(bbox[0]), int(bbox[3]+20)),0, 0.75, (255,255,255),2)
                else:
                    cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame1, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker-1 ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        track2_id_list = []
        for track in tracker2.tracks:
            track2_id_list.append(track.track_id)
        
        if len(map_list_track2)!=0:
            for idx,i in enumerate(map_list_track2):
                if i[0] not in track2_id_list:
                    del(map_list_track2[idx])

        for index, track in enumerate(tracker2.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            query_img_2 = frame2[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
        
            cv2.rectangle(frame2, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            if len(sort_rank)!=0:
                if query_img_2.shape[0] >= 780 and query_img_2.shape[1] >= 1100:
                    if any(track.track_id in l for l in map_list_track2):
                        for i in range(len(map_list_track2)):
                            if map_list_track2[i][0] == track.track_id:
                                save_car_plate = map_list_track2[i][1]
                                if len(save_car_plate) > len(max_text):
                                    car_plate = save_car_plate
                                else:
                                    map_list_track2[i][1] = max_text
                                    car_plate = max_text
                    else:
                        map_list_track2.append([track.track_id, max_text])
                        car_plate = max_text

                    cv2.rectangle(frame2, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len('map-')+len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame2, "map-" + class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    cv2.putText(frame2, car_plate ,(int(bbox[0]), int(bbox[3]+20)),0, 0.75, (255,255,255),2)
                else:
                    cv2.rectangle(frame2, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame2, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            else:
                if any(track.track_id in l for l in map_list_track2):
                    for i in range(len(map_list_track2)):
                        if map_list_track2[i][0] == track.track_id:
                            car_plate = map_list_track2[i][1]
                    cv2.rectangle(frame2, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len('map-')+len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame2, "map-" + class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    cv2.putText(frame2, car_plate ,(int(bbox[0]), int(bbox[3]+20)),0, 0.75, (255,255,255),2)
                else:
                    cv2.rectangle(frame2, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame2, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            # cv2.imshow(f"{query_img.shape}",query_img)
            # cv2.waitKey(0)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker-2 ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result1 = np.asarray(frame1)
        result1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

        result2 = np.asarray(frame2)
        result2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
        
        # if not FLAGS.dont_show:
        #     cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output1:
            out1.write(result1)

        if FLAGS.output2:
            out2.write(result2)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
