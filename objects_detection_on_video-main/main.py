import cv2
import math
import numpy as np
from art import tprint
import sqlite3
from typing import Dict, List
from datetime import datetime


class object_point(object):
    def __init__(self, x, y, n, tracking_status, classes, appearance_time, disappearance_time, filename):
        self.coord = (x, y, n, tracking_status, classes, appearance_time, disappearance_time, filename)


list_obj: List[object_point] = []
count_car = 0
count_person = 0
count_truck = 0
count_bus = 0
# спецтехника
count_plant = 0
count_animal = 0
# пешеходные переходы
count_traffic = 0
# рекламные щиты


sqlite_connection = sqlite3.connect('bd.db')
cursor = sqlite_connection.cursor()

video = " "


def add_obj(x, y, classes_name, video):
    global count_car
    global count_person
    global count_truck
    global count_bus
    global count_plant
    global count_animal
    global count_traffic

    find_flag = 0  # Object not found, 1 - found
    num_detection_obj = 0
    min_distance = 100
    i = 0
    list_number = -1
    for obj in list_obj:
        distance = math.sqrt(math.pow((x - obj[0]), 2) + math.pow((y - obj[1]), 2))
        if distance < 20 and obj[3] >= 0 and distance < min_distance and obj[4] == classes_name:
            min_distance = distance
            list_number = i
        i = i + 1
    if list_number >= 0:
        list_obj[list_number][0] = x
        list_obj[list_number][1] = y
        list_obj[list_number][3] = 1
        find_flag = 1
        num_detection_obj = list_obj[list_number][2]
    if find_flag == 0:

        if classes_name in ["bicycle", "car", "motorbike"]:
            list_obj.append([x, y, count_car, 1, classes_name, datetime.now(), "", video])
            num_detection_obj = count_car
            count_car += 1

        if classes_name in ["person"]:
            list_obj.append([x, y, count_person, 1, classes_name, datetime.now(), "", video])
            num_detection_obj = count_person
            count_person += 1

        if classes_name in ["truck", "train"]:
            list_obj.append([x, y, count_truck, 1, classes_name, datetime.now(), "", video])
            num_detection_obj = count_truck
            count_truck += 1

        if classes_name in ["bus"]:
            list_obj.append([x, y, count_bus, 1, classes_name, datetime.now(), "", video])
            num_detection_obj = count_bus
            count_bus += 1

        if classes_name in ["potted", "plant"]:
            list_obj.append([x, y, count_plant, 1, classes_name, datetime.now(), "", video])
            num_detection_obj = count_plant
            count_plant += 1

        if classes_name in ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bar", "zebra", "giraffe"]:
            list_obj.append([x, y, count_animal, 1, classes_name, datetime.now(), "", video])
            num_detection_obj = count_animal
            count_animal += 1

        if classes_name in ["traffic light", "fire hydrant", "stop sign", "parking meter"]:
            list_obj.append([x, y, count_traffic, 1, classes_name, datetime.now(), "", video])
            num_detection_obj = count_traffic
            count_traffic += 1

    # print(obj[0])

    return num_detection_obj


def apply_yolo_object_detection(image_to_process):
    """
    Recognition and determination of the coordinates of objects on the image
    :param image_to_process: original image
    :return: image with marked objects and captions to them
    """

    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0
    car_count = 0
    person_count = 0
    truck_count = 0
    bus_count = 0
    # спецтехника
    plant_count = 0
    animal_count = 0
    # пешеходные переходы
    traffic_count = 0
    # рекламные щиты

    # Starting a search for objects in an image
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Selection
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        # For debugging, we draw objects included in the desired classes
        if classes[class_index] in classes_to_look_for:

            # print(classes[class_index])
            # print(box[1])
            num = 0
            objects_count += 1
            if classes[class_index] in ["bicycle", "car", "motorbike"]:
                car_count += 1
                num = add_obj(box[0], box[1], classes[class_index], video)

            if classes[class_index] in ["person"]:
                person_count += 1
                num = add_obj(box[0], box[1], classes[class_index], video)

            if classes[class_index] in ["truck", "train"]:
                truck_count += 1
                num = add_obj(box[0], box[1], classes[class_index], video)

            if classes[class_index] in ["bus"]:
                bus_count += 1
                num = add_obj(box[0], box[1], classes[class_index], video)

            if classes[class_index] in ["potted", "plant"]:
                plant_count += 1
                num = add_obj(box[0], box[1], classes[class_index], video)

            if classes[class_index] in ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bar", "zebra",
                                        "giraffe"]:
                animal_count += 1
                num = add_obj(box[0], box[1], classes[class_index], video)

            if classes[class_index] in ["traffic light", "fire hydrant", "stop sign", "parking meter"]:
                traffic_count += 1
                num = add_obj(box[0], box[1], classes[class_index], video)

            image_to_process = draw_object_bounding_box(image_to_process, class_index, box, num)

    for obj in list_obj:
        if obj[3] == 6:
            obj[3] = -1
            obj[6] = datetime.now()
            print(obj)

            sqlite_select_query = "insert into data(class,time1,time2,video) values('" + str(obj[4]) + "','" + str(
                obj[5]) + "','" + str(obj[6]) + "','" + str(obj[7]) + "');"
            cursor.execute(sqlite_select_query)
            sqlite_connection.commit()
            # record = cursor.fetchall()
            # print(record)

        if obj[3] == 0:
            obj[3] = 2

        if obj[3] == 1:
            obj[3] = 0

        if obj[3] > 1:
            obj[3] = obj[3] + 1

    final_image = draw_object_count(image_to_process, objects_count, person_count, car_count, truck_count, bus_count,
                                    plant_count, animal_count, traffic_count)
    return final_image


def draw_object_bounding_box(image_to_process, index, box, num):
    """
    Drawing object borders with captions
    :param image_to_process: original image
    :param index: index of object class defined with YOLO
    :param box: coordinates of the area around the object
    :return: image with marked objects
    """

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    # final_image = cv2.putText(final_image, text+" "+str(num)+" "+str(box[0])+" "+str(box[1]), start, font,
    #                          font_size, color, width, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text + " " + str(num), start, font,
                              font_size, color, width, cv2.LINE_AA)

    return final_image


def draw_object_count(image_to_process, objects_count, person_count, car_count, truck_count, bus_count, plant_count,
                      animal_count, traffic_count):
    """
    Signature of the number of found objects in the image
    :param image_to_process: original image
    :param objects_count: the number of objects of the desired class
    :return: image with labeled number of found objects
    """

    start = (10, 20)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3

    text = "Objects found: " + str(objects_count)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    start = (10, 50)
    text = "Person found: " + str(person_count)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    start = (10, 80)
    text = "Car found: " + str(car_count)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    start = (10, 110)
    text = "Truck found: " + str(truck_count)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    start = (10, 140)
    text = "Bus found: " + str(bus_count)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    start = (10, 170)
    text = "Plant found: " + str(plant_count)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    start = (10, 200)
    text = "Animal found: " + str(animal_count)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    start = (10, 230)
    text = "Traffic found: " + str(traffic_count)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    return final_image


def start_video_object_detection(video: str):
    """
    Захват и анализ видео в режиме реального времени
    """

    while True:
        try:
            # Capturing a picture from a video
            video_camera_capture = cv2.VideoCapture(video)

            while video_camera_capture.isOpened():
                ret, frame = video_camera_capture.read()
                if not ret:
                    break

                # Application of object recognition methods on a video frame from YOLO
                frame = apply_yolo_object_detection(frame)

                # Displaying the processed image on the screen with a reduced window size
                frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
                cv2.imshow("Video Capture", frame)
                cv2.waitKey(1)

            video_camera_capture.release()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':

    # Logo
    tprint("Object detection")

    # Loading YOLO scales from files and setting up the network
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg",
                                     "Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Loading from a file of object classes that YOLO can detect
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    # Determining classes that will be prioritized for search in an image
    # The names are in the file coco.names.txt

    video = "C:\\Users\\user\\Downloads\\FPV1.mp4"

    look_for = "person,bicycle,car,truck, bus,potted plant,bird, cat, dog, horse, sheep, cow,elephant, bar," \
               "zebra,giraffe,traffic light,fire hydrant, stop sign, parking meter".split(',')

    # Delete spaces
    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for

    start_video_object_detection(video)

    cursor.close()
