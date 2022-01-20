import argparse
import itertools
import math
import os
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.ndimage import rotate

'''
These are the HSV thresholds for the green cartridges and the white cartridges
'''
green = {'low_h': 20, 'high_h': 80, 'low_s': 50, 'high_s': 255, 'low_v': 0, 'high_v': 255}
white = {'low_h': 0, 'high_h': 180, 'low_s': 0, 'high_s': 135, 'low_v': 70, 'high_v': 255}

'''
We can use some coarse heuristics to filter out noise and only find the contours that correspond to the cartridges
These will need to be tuned when we implement it on a raspberry pi/ with rpi camera
'''
min_cartridge_area = 1500
min_cartridge_eccentricity = 0.93
max_cartridge_eccentricity = 0.98

canny_threshold_1 = 50
canny_threshold_2 = 50


# Pass in the RGB image to this function
# Pass in the the color of cartridge to look for (green or white)
def get_contours_img(img, color):
    # convert the image to greyscale
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # use a thresholding function to find all pixels within the appropriate color range
    frame_threshold = cv2.inRange(img_HSV, (color['low_h'], color['low_s'], color['low_v']),
                                  (color['high_h'], color['high_s'], color['high_v']))
    # use the opencv contours function to find contours of the image, pass RETR_EXTERNAL flag so we only consider
    # external contours and not the ones found inside the cartridge
    cartridge_contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    # filter the contours based on some heuristics of how the cartridge looks like
    # (size eccentricity etc)
    # I use image moments to determine the area and eccentricity of the contour
    # https://en.wikipedia.org/wiki/Image_moment
    for contour in cartridge_contours:
        moments = cv2.moments(contour)
        area = moments['m00']
        if area < min_cartridge_area: continue
        cov_matrix = np.array([[moments['nu20'], moments['nu11']], [moments['nu11'], moments['nu02']]])
        eig_vals = np.sort(np.linalg.eigvals(cov_matrix))
        eccentricity = math.sqrt(1 - eig_vals[0] / eig_vals[1])
        if max_cartridge_eccentricity > eccentricity > min_cartridge_eccentricity:
            filtered_contours.append(contour)
    mask = cv2.drawContours(np.zeros_like(img), filtered_contours, -1, (255, 255, 255), thickness=-1)
    threshold_image = np.bitwise_and(img, mask)
    cartridges = []
    for contour in filtered_contours:
        bounding_rectangle = cv2.boundingRect(contour)
        x, y, w, h = bounding_rectangle
        cartridges.append(threshold_image[y:y + h, x:x + w])
    return filtered_contours, cartridges


def find_angle_of_rotation(contours, image_annotations=None):
    angles = []
    centers = []
    for contour in contours:
        # use image moments to get relevant features of the image
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


def get_internal_features(cartridges, angles):
    all_descriptors = []
    for angle, cartridge in zip(angles, cartridges):
        cartridge = rotate(cartridge, -angle, reshape=True)
        height, width, _ = cartridge.shape
        greyscale_image = cv2.cvtColor(cartridge, cv2.COLOR_BGR2HSV)[:, :, 2]
        edges = cv2.Canny(greyscale_image, canny_threshold_1, canny_threshold_2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        morphed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morphed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.convexHull(contour) for contour in contours]
        cv2.imshow('edges', edges)
        cv2.imshow('internal_features', cv2.drawContours(cartridge, contours, -1, (255, 255, 255), 1))
        # get rid of noisy points/contours
        valid_contours = [i for i, cont in enumerate(contours) if cv2.contourArea(cont) > cv2.arcLength(cont, False)]
        # generate descriptors for each contour using the contour moments
        contour_descriptors = []
        for i in valid_contours:
            moments = cv2.moments(contours[i])
            area = moments['m00']
            x = moments['m10'] / moments['m00']
            y = moments['m01'] / moments['m00']
            cov_matrix = np.array([[moments['nu20'], moments['nu11']], [moments['nu11'], moments['nu02']]])
            eig_vals = np.sort(np.linalg.eigvals(cov_matrix))
            eccentricity = math.sqrt(1 - eig_vals[0] / eig_vals[1])
            contour_descriptors.append([i, area, x, y, eccentricity])

        if len(contour_descriptors) < 1:
            all_descriptors.append(None)
            print('error')
            continue

        # relative position to center of the sample, select the contour with the largest area
        contour_descriptors.sort(key=lambda descriptor: descriptor[1], reverse=True)
        cartridge_area = contour_descriptors[0][1]
        contour_x, contour_y = contour_descriptors[0][2], contour_descriptors[0][3]
        contour_descriptors = [
            np.array([d[1] / cartridge_area, (d[2] - contour_x) / width, (d[3] - contour_y) / height, d[4]])
            for d in contour_descriptors]
        all_descriptors.append(contour_descriptors)
    return all_descriptors


def fit_GMM(images_directory, n_components, color):
    X = []
    images = os.listdir(images_directory)
    if len(images) == 0:
        print("No Training Images, ReRun with take_images flag")
        exit()
    for image_file in os.listdir(images_directory):
        img = cv2.imread(os.path.join(images_directory, image_file))
        contours, cartridges = get_contours_img(img, color)
        _, angles, _ = find_angle_of_rotation(contours, None)
        X.extend(get_internal_features(cartridges, angles))
    gm = GaussianMixture(n_components=n_components, covariance_type='diag', n_init=5)
    X = list(itertools.chain.from_iterable(X))
    gm.fit(X)
    return gm


def detect_flip(all_descriptor, centers, face_down_model, annotated_image):
    is_face_down = []
    for center, descriptor in zip(centers, all_descriptor):
        score = face_down_model.score(descriptor)
        is_face_down.append(score > 0)
        if annotated_image is not None:
            print(score)
            if score < 2:
                annotated_image = cv2.circle(annotated_image, center, 30, (0, 255, 0), 2)
            if score > 2:
                annotated_image = cv2.circle(annotated_image, center, 30, (0, 0, 255), 2)
    return annotated_image, is_face_down


def orientation_detection(img, color, face_down_model, annotated_image=None):
    contours, cartridges = get_contours_img(img, color)
    annotated_image, angles, centers = find_angle_of_rotation(contours, annotated_image)
    all_descriptors = get_internal_features(cartridges, angles)
    flips = detect_flip(all_descriptors, centers, face_down_model, annotated_image)
    return annotated_image, angles, centers, flips


def take_training_images(name, num_training_images):
    cap = cv2.VideoCapture(0)
    i = 0
    ret = True
    while ret and i < num_training_images:
        ret, frame = cap.read()
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(30)
        if key != -1:
            cv2.imwrite(os.path.join('training_images', name, str(i) + ".png"), frame)
            i += 1
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect Cartridge Orientation')

    parser.add_argument('--take_green_down_pictures', action='store_true',
                        help='flag to train green up/down detector')
    parser.add_argument('--take_white_down_pictures', action='store_true',
                        help='flag to train white up/down detector')
    parser.add_argument('--run_detection_algo', action='store_true',
                        help='flag to run algorithm')

    args = parser.parse_args()

    if args.take_green_down_pictures:
        take_training_images('green_down', 15)
        exit()
    if args.take_white_down_pictures:
        take_training_images('white_down', 15)
        exit()
    if args.run_detection_algo:
        green_down = fit_GMM('training_images/green_down', 4, green)
        white_down = fit_GMM('training_images/white_down', 4, white)
        cap = cv2.VideoCapture(0)
        ret = True
        while ret:
            ret, frame = cap.read()
            annotated_image, _, _, _ = orientation_detection(frame, green, green_down, frame.copy())
            annotated_image, _, _, _ = orientation_detection(frame, white, white_down, annotated_image)
            cv2.imshow("orientation detection", annotated_image)
            cv2.waitKey(10)
        exit()
