import math
import cv2
import numpy as np


green = {'low_h': 45, 'high_h': 80, 'low_s': 0, 'high_s': 255, 'low_v': 0, 'high_v': 255}
white = {'low_h': 0, 'high_h': 180, 'low_s': 0, 'high_s': 52, 'low_v': 115, 'high_v': 255}

min_cartridge_area = 1500
min_cartridge_eccentricity = 0.93
max_cartridge_eccentricity = 0.98


def get_contours_img(img, color):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(img_HSV, (color['low_h'], color['low_s'], color['low_v']),
                                  (color['high_h'], color['high_s'], color['high_v']))
    cartridge_contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cartridge_contours = [contour for contour in cartridge_contours if
                          cv2.contourArea(contour) > cv2.arcLength(contour, closed=True)]
    filtered_contours = []

    for contour in cartridge_contours:
        moments = cv2.moments(contour)
        cov_matrix = np.array([[moments['nu20'], moments['nu11']], [moments['nu11'], moments['nu02']]])
        eig_vals = np.sort(np.linalg.eigvals(cov_matrix))
        eccentricity = math.sqrt(1 - eig_vals[0] / eig_vals[1])
        area = moments['m00']
        if area > min_cartridge_area and max_cartridge_eccentricity > eccentricity > min_cartridge_eccentricity:
            filtered_contours.append(contour)
    return filtered_contours


def find_angle_of_rotation(contours, image_annotations=None):
    angles = []
    centers = []
    for contour in contours:
        moments = cv2.moments(contour)
        center_of_mass = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        angle = 90 - math.degrees(0.5 * math.atan2(2 * moments['nu11'], moments['nu20'] - moments['nu02']))
        rotated_rectangle = cv2.minAreaRect(contour)
        center_point = tuple(np.int0(rotated_rectangle[0]))
        # this correction relies on the fact that the cartridge is narrower on one end than the other so the center of
        # mass should lie below the geometric center
        vector_com_to_center = np.array([center_point[0] - center_of_mass[0], -center_point[1] + center_of_mass[1]])
        vector_orientation = np.array([-math.sin(math.radians(angle)), math.cos(math.radians(angle))])
        if np.dot(vector_com_to_center, vector_orientation) < 0:
            angle = angle - 180
        angles.append(angle)
        centers.append(center_point)

        # Annotate the image to make debugging easier
        if image_annotations is not None:
            image_annotations = cv2.drawContours(image_annotations, [contour], 0, (0, 0, 0), 1)
            box = np.int0(cv2.boxPoints(rotated_rectangle))
            image_annotations = cv2.drawContours(image_annotations, [box], 0, (0, 0, 0), 1)
            # image_annotations = cv2.circle(image_annotations, center_point, 1, (0, 0, 255))
            # image_annotations = cv2.circle(image_annotations, center_of_mass, 1, (0, 255, 0))
            image_annotations = cv2.putText(image_annotations, str(round(angle, 1)) + "d", center_point,
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.7, color=(0, 255, 128),
                                            thickness=1)
    return image_annotations, angles, centers


def orientation_detection(img, color, annotated_image=None):
    contours = get_contours_img(img, color)
    annotated_image, angles, centers = find_angle_of_rotation(contours, annotated_image)
    return annotated_image, angles, centers


test_images = ['img.png', 'img_1.png', 'img_2.png', 'img_3.png', 'img_4.png', 'img_5.png', 'img_6.png']
for image_uri in test_images:
    img = cv2.imread("test_images" + '/' + image_uri)
    annotated_image, _, _ = orientation_detection(img, green, img.copy())
    annotated_image, _, _ = orientation_detection(img, white, annotated_image)
    cv2.imwrite("annotated_images/" + image_uri, annotated_image)
    cv2.imshow("orientation detection", annotated_image)
    cv2.waitKey(0)
