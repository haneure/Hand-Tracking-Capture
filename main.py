# CV2
import copy
import csv
import itertools
import json
import math
# Python Utils
import socket

import cv2
import cvzone
# Mediapipe
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from utils import CvFpsCalc

from model import LandmarkClassifier

# Global variable

prevLmList = []
lmCount = 0

# Parameters
width, height = 1280, 720

# Mode 2
lmToTest = []
lmToTestCount = 0
handToTest = []
tempDataToTest = []
lmToTestLandmarkIndex = 0
xModifier = 0
yModifier = 0
zModifier = 0
modifierMode = 'plus'

# get_distance in real life
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
# Use Polynomial Function
# Second Order Polynomial Function
# Quadratic function (1 Curve / bumps)

# get_distance in virtual space
z = [70]
zInVR = [2]

cleanImg = []
img = []

label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

data = {
    "left": {},
    "right": {}
}


def modify_z(hand, hand_sign_id):
    global data
    lmList = hand['lmList']

    # print(hand)
    # print(lmList)

    if hand['type'] == 'Left':
        tempData = []

        lmList = check_hand_sign_id(lmList, hand_sign_id)

        for lm in lmList:
            tempData.extend([lm[0], height - lm[1], lm[2]])
            # data['left'] = str.encode(str(tempData))
            data['left']['lmList'] = str(tempData)
    else:
        tempData = []

        lmList = check_hand_sign_id(lmList, hand_sign_id)

        for lm in lmList:
            tempData.extend([lm[0], height - lm[1], lm[2]])
            # data['left'] = str.encode(str(tempData))
            # data['right']['lmList'] = str(tempData)
        pass


def check_hand_sign_id(lmList, hand_sign_id):
    # if hand_sign_id == 0:
    # print("before")
    # print(lmList[7])
    # lmList[7][2] += -25
    # print(lmList[7])
    #
    # print("after")
    # print(lmList[8])
    # lmList[8][2] += -50
    # print(lmList[8])

    # elif hand_sign_id == 99:
    # print("before")
    # print(lmList[11])
    # lmList[11][2] += -30
    # print(lmList[11])
    #
    # print("after")
    # print(lmList[12])
    # lmList[12][2] += -55
    # print(lmList[12])

    return lmList


def modify_lmToTest():
    global lmToTestLandmarkIndex
    global tempDataToTest
    global handToTest
    global modifierMode
    global xModifier
    global yModifier
    global zModifier
    global cleanImg
    global img
    global data
    global label

    # 21 * 3 landmarks
    # label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # print(lmToTest[7])

    tempDataToTest = []
    tempImg = cleanImg
    number = -1

    key = cv2.waitKey(1)

    if key != -1:
        # print(key)
        img = cleanImg

        number = -1
        if 48 <= key <= 57:  # number 0 - 9
            number = key - 48
            logging_modifier_csv(number, handToTest)

        if key == 93:
            lmToTestLandmarkIndex += 1
            xModifier = label[lmToTestLandmarkIndex * 3]
            yModifier = label[lmToTestLandmarkIndex * 3 + 1]
            zModifier = label[lmToTestLandmarkIndex * 3 + 2]
        elif key == 91:
            lmToTestLandmarkIndex -= 1
            xModifier = label[lmToTestLandmarkIndex * 3]
            yModifier = label[lmToTestLandmarkIndex * 3 + 1]
            zModifier = label[lmToTestLandmarkIndex * 3 + 2]

        if key == 97:
            xModifier += 1
            lmToTest[lmToTestLandmarkIndex][0] += 1
            # label[lmToTestLandmarkIndex * 3] += 1
        if key == 115:
            yModifier += 1
            lmToTest[lmToTestLandmarkIndex][1] += 1
            # label[lmToTestLandmarkIndex * 3 + 1] += yModifier
        if key == 100:
            zModifier += 1
            lmToTest[lmToTestLandmarkIndex][2] += 1
            # label[lmToTestLandmarkIndex * 3 + 2] += zModifier

        if key == 122:
            xModifier -= 1
            lmToTest[lmToTestLandmarkIndex][0] -= 1
            # label[lmToTestLandmarkIndex * 3] -= 1
        if key == 120:
            yModifier -= 1
            lmToTest[lmToTestLandmarkIndex][1] -= 1
            # label[lmToTestLandmarkIndex * 3 + 1] -= 1
        if key == 99:
            zModifier -= 1
            lmToTest[lmToTestLandmarkIndex][2] -= 1
            # label[lmToTestLandmarkIndex * 3 + 2] -= 1

        label[lmToTestLandmarkIndex * 3] = xModifier
        label[lmToTestLandmarkIndex * 3 + 1] = yModifier
        label[lmToTestLandmarkIndex * 3 + 2] = zModifier

        # print(label)

        for lm in lmToTest:
            tempDataToTest.extend([lm[0], height - lm[1], lm[2]])
            data['right']['lmList'] = str(tempDataToTest)

    # print(tempData)

    # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    # tempImg = cv2.resize(cleanImg, (0, 0), None, 0.5, 0.5)

    if lmToTest:
        # Draw info
        cv2.putText(img, "lmList[" + str(lmToTestLandmarkIndex) + "] = " +
                    "x: " + str(lmToTest[lmToTestLandmarkIndex][0]) + " " +
                    "y: " + str(lmToTest[lmToTestLandmarkIndex][1]) + " " +
                    "z: " + str(lmToTest[lmToTestLandmarkIndex][2]),
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
                    cv2.LINE_AA)

        # Modifier info
        cv2.putText(img, "lmList[" + str(lmToTestLandmarkIndex) + "] = " +
                    "x: " + str(label[lmToTestLandmarkIndex * 3]) + " " +
                    "y: " + str(label[lmToTestLandmarkIndex * 3 + 1]) + " " +
                    "z: " + str(label[lmToTestLandmarkIndex * 3 + 2]),
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
                    cv2.LINE_AA)

    cv2.imshow("Image", img)

    # img = cleanImg

    # print("before")
    # print(lmList[7])
    # lmList[7][2] += -25
    # print(lmList[7])


def main():
    global img
    global lmToTest
    global lmToTestCount
    global cleanImg
    global data
    global xModifier
    global yModifier
    global zModifier
    global label

    isRecording = 1

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    webcamCount = 0

    # Hand Detector
    detector = HandDetector(maxHands=2, detectionCon=0.8)

    # Communication
    sockets = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 5052)

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Model load
    landmark_classifier = LandmarkClassifier()

    # Read labels
    with open('model/landmark_classifier/landmark_label.csv',
              encoding='utf-8-sig') as f:
        landmark_classifier_label = csv.reader(f)
        landmark_classifier_label = [
            row[0] for row in landmark_classifier_label
        ]

    mode = 0
    getCleanImg = 0
    # img = []

    while cap:
        fps = cvFpsCalc.get()

        key = cv2.waitKey(10)
        prevKey = key
        if key != prevKey:
            print(key)
        if key == 27:  # ESC
            break
        if mode == 2:  # Enter
            if key == 13:
                cleanImg = img
                lmToTestCount = lmToTestCount + 1
                if webcamCount == 1:
                    webcamCount -= 1
                if isRecording:
                    isRecording = 0

        number, mode = select_mode(key, mode)

        # Get The frame from the webcam
        if isRecording:
            success, img = cap.read()
        else:
            success, img = cap.read()

            while webcamCount <= 0:
                success, img = cap.read()
                # Modify lmToTest
                modify_lmToTest()

                jsonData = json.dumps(data)
                sockets.sendto(str.encode(str(jsonData)), serverAddressPort)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    webcamCount += 1
                    isRecording = 1
                    lmToTestCount -= 1
                    xModifier = 0
                    yModifier = 0
                    zModifier = 0
                    label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0]
                    break

        # Hands
        hands, img = detector.findHands(img)

        jsonData = ""

        # Landmark values - (x,y,z) * 21
        if hands:
            # Get the first hand detected
            hand1 = hands[0]
            # print(hand1)
            # print(hand1)
            # if hand1['type'] == 'Left':
            #     if mode != 2:
            #         data['right'] = {}
            # else:
            #     if mode != 2:
            #         data['left'] = {}

            if len(hands) > 1:
                hand1 = hands[0]
                hand2 = hands[1]

                # print(hand2)
                data = get_landmark(hand2, mode)

                pre_processed_landmark_list = pre_process_landmark(hand2)

                # Not needed for now
                # if hand2['type'] == 'Left':
                #     type2 = 0
                # else:
                #     type2 = 1

                # Hand gesture classification
                # pre_processed_landmark_list.insert(0, type2)
                # print(data['left']['lmList'])
                hand_sign_id = landmark_classifier([pre_processed_landmark_list])
                draw_hand_info(img, hand2, landmark_classifier_label[hand_sign_id])
                if hand2['type'] == 'Left':
                    data['left']['gesture'] = landmark_classifier_label[hand_sign_id]
                else:
                    data['right']['gesture'] = landmark_classifier_label[hand_sign_id]


                # Modify z
                modify_z(hand2, hand_sign_id)

                # Write data to the csv
                if mode == 1:
                    logging_csv(number, mode, pre_processed_landmark_list)
                elif mode == 3:
                    logging_modifier_csv(number, hand2)

            data = get_landmark(hand1, mode)

            pre_processed_landmark_list = pre_process_landmark(hand1)

            # Not needed for now
            # if hand1['type'] == 'Left':
            #     type1 = 0
            # else:
            #     type1 = 1

            # Hand gesture classification
            # pre_processed_landmark_list.insert(0, type1)
            # print(data['left']['lmList'])
            hand_sign_id = landmark_classifier([pre_processed_landmark_list])
            draw_hand_info(img, hand1, landmark_classifier_label[hand_sign_id])
            if hand1['type'] == 'Left':
                data['left']['gesture'] = landmark_classifier_label[hand_sign_id]
            else:
                data['right']['gesture'] = landmark_classifier_label[hand_sign_id]

            # Write data to the csv
            if mode == 1:
                logging_csv(number, mode, pre_processed_landmark_list)
            elif mode == 3:
                logging_modifier_csv(number, hand1)

            # Modify z
            modify_z(hand1, hand_sign_id)

            # print(str.encode(str(jsonData)))
            # print(jsonData)
        else:
            data['left'] = {}
            data['right'] = {}

        jsonData = json.dumps(data)
        # print(jsonData)
        sockets.sendto(str.encode(str(jsonData)), serverAddressPort)

        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

        # Draw other details
        draw_info(img, fps, mode, number)
        # draw_hand_info(img, dista)
        cv2.imshow("Image", img)


# Get the landmark of the hand
def get_landmark(hand, mode):
    # type = ''

    global prevLmList
    global lmCount
    global lmToTest
    global lmToTestCount
    global handToTest
    global tempDataToTest
    global data

    stabilize = 0
    lmListDifference = []

    if hand['type'] == 'Left':
        # type = 'Left'
        tempData = []

        if lmToTestCount <= 0:
            tempDataToTest = []
        if mode == 2:
            if lmToTestCount <= 0:
                lmToTest = copy.deepcopy(hand['lmList'])
                handToTest = hand

            distance = get_distance(handToTest)
            for lm in lmToTest:
                tempDataToTest.extend([lm[0], height - lm[1], lm[2]])
                data['left']['lmList'] = str(tempDataToTest)
                data['left']['type'] = hand['type']
                data['left']['distance'] = distance

        else:
            # print(hand)
            lmList = hand['lmList']

            if lmCount == 0:
                prevLmList = lmList
                lmCount += 1
            elif lmCount > 0:
                difference = abs(np.subtract(lmList, prevLmList))

                # print("lmList: ", lmList)
                # print("prevLmList: ", prevLmList)

                # Stabilize here
                # Only update the landmarks if it is different by 4
                # for i in range(len(lmList)):
                #     # print(lmList[i], difference[i])
                #     for j in range(len(lmList[i])):
                #         if difference[i][j] < 4:
                #             lmList[i][j] = prevLmList[i][j]

                # print("difference: ", difference)
                lmCount -= 1

            distance = get_distance(hand)
            # print("left distance: ", distance)

            for lm in lmList:
                tempData.extend([lm[0], height - lm[1], lm[2]])
                data['left']['lmList'] = str(tempData)
                data['left']['type'] = hand['type']
                data['left']['distance'] = distance
    else:
        tempData = []
        if lmToTestCount <= 0:
            tempDataToTest = []
        if mode == 2:
            if lmToTestCount <= 0:
                lmToTest = copy.deepcopy(hand['lmList'])
                handToTest = hand

            distance = get_distance(handToTest)
            for lm in lmToTest:
                tempDataToTest.extend([lm[0], height - lm[1], lm[2]])
                data['left']['lmList'] = str(tempDataToTest)
                data['left']['type'] = hand['type']
                data['left']['distance'] = distance
        else:
            lmList = hand['lmList']

            distance = get_distance(hand)
            # print("right distance: ", distance)

            for lm in lmList:
                tempData.extend([lm[0], height - lm[1], lm[2]])
                data['right']['lmList'] = str(tempData)
                data['right']['type'] = hand['type']
                data['right']['distance'] = distance

    return data


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list['lmList'])

    # Convert to relative coordinates
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # Convert to one dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    # print(temp_landmark_list)

    return temp_landmark_list


def draw_hand_info(     img, hand, hand_gesture):
    x, y, w, h = hand['bbox']

    cv2.putText(img, hand_gesture, (x, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1,
                cv2.LINE_AA)


def draw_info(img, fps, mode, number):
    global lmToTestLandmarkIndex
    global lmToTestCount

    cv2.putText(img, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    mode_info = ['Logging landmark point', 'Testing one landmark', 'Testing landmark no modifier']

    if mode == 1 or mode == 2 or mode == 3 :
        cv2.putText(img, "MODE: " + mode_info[mode - 1], (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(img, "Record landmark point :" + str(number), (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # if lmToTest:
        #     # print(img.shape[1])
        #     # print(img.shape[0])
        #     # print(lmToTest[0])
        #     if lmToTestCount <= 0:
        #         cv2.putText(img, "lmList[" + str(lmToTestLandmarkIndex) + "] = " +
        #                     "x: " + str(lmToTest[lmToTestLandmarkIndex][0]) + " " +
        #                     "y: " + str(lmToTest[lmToTestLandmarkIndex][1]) + " " +
        #                     "z: " + str(lmToTest[lmToTestLandmarkIndex][2]),
        #                     (15, 300),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        #                     cv2.LINE_AA)

        # print(lmList[7])
        # lmList[7][2] += -25
        # print(lmToTest)


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # number 0 - 9
        number = key - 48
    if key == 110:  # n for return to normal mode
        mode = 0
    if key == 114:  # r for record landmark
        mode = 1
    if key == 116:  # t for test 1 input with modifier (1 image/frame modifier)
        mode = 2
    if key == 121:  # y for test 1 input but doesn't need modifier
        mode = 3
    return number, mode


def logging_modifier_csv(number, handToTest):
    global lmToTest
    global label

    pre_processed_landmark = pre_process_landmark(handToTest)

    if 0 <= number <= 9:
        csv_path = 'model/landmark_classifier/landmark_modifier.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *label, *pre_processed_landmark])
    return


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/landmark_classifier/landmark.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


# Get the difference between previous landmark and current landmark
# To stabilize, less stutter
def count_difference(lmList):
    global prevLmList
    global lmCount

    if lmCount == 0:
        prevLmList = lmList
        lmCount += 1
        return []
    elif lmCount > 0:
        difference = np.subtract(lmList, prevLmList)

        # print("lmList: ", lmList)
        # print("prevLmList: ", prevLmList)

        # for i in range(len(lmList)):
        #     # print(lmList[i], difference[i])
        #     for j in range(len(lmList[i])):
        #         if difference[i][j] < 1:
        #             lmList[i][j] = prevLmList[i][j]

        # print("difference: ", lmList)
        lmCount -= 1
        return lmList


def get_distance(hand):
    global img, x, y

    # # get_distance in real life
    # x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    # y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    # coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
    # # Use Polynomial Function
    # # Second Order Polynomial Function
    # # Quadratic function (1 Curve / bumps)

    lmList = hand['lmList']
    x, y, w, h = hand['bbox']

    x1, y1, z1 = lmList[5]
    x2, y2, z2 = lmList[17]

    distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
    A, B, C = coff
    distanceCM = A * distance ** 2 + B * distance + C

    # print(distanceCM, distance)

    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
    # cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 150, y))
    cv2.putText(img, f'{int(distanceCM)} cm', (x + 150, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1,
                cv2.LINE_AA)

    return distanceCM

if __name__ == '__main__':
    main()
