import cv2
import numpy as np
from PIL import Image
import dlib


class Croper:
    def __init__(self, path_of_lm):
        # download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor(path_of_lm)

    def get_landmark(self, img_np, face_fields):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        detector = dlib.get_frontal_face_detector()  # detector face
        dets = detector(img_np, 1)
        if len(dets) == 0:
            return None

        if face_fields is None:
            d = dets[0]  # get first face
        else:
            # user crop face in frontend
            # real size
            originalHeight, originalWidth, _ = img_np.shape
            squareFace = face_fields[0]
            canvasWidth = squareFace["canvasWidth"]
            canvasHeight = squareFace["canvasHeight"]
            # Calculate the scale factor
            scaleFactorX = originalWidth / canvasWidth
            scaleFactorY = originalHeight / canvasHeight

            # Calculate the new position and size of the square face on the original image
            # Convert canvas square face coordinates to original image coordinates
            newX1 = squareFace['x'] * scaleFactorX
            newX2 = (squareFace['x'] + squareFace['width']) * scaleFactorX
            newY1 = squareFace['y'] * scaleFactorY
            newY2 = (squareFace['y'] + squareFace['height']) * scaleFactorY

            d_user_crop = dlib.rectangle(int(newX1), int(newY1), int(newX2), int(newY2))
            # print(int(newX1), int(newY1), int(newX2), int(newY2))
            # get the center point of d_new
            user_crop_center = d_user_crop.center()
            # check if user_crop_center is inside det
            for det in dets:
                if det.contains(user_crop_center):
                    print("center is inside det")
                    d = det
                    break
            else:
                d = []

        # Get the landmarks/parts for the face in box d.
        try:
            shape = self.predictor(img_np, d)
        except Exception as err:
            raise "Error..Not predict face"

        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        return lm

    @staticmethod
    def align_face(img, lm, output_size=1024):
        """
        :param filepath: str
        :return: PIL Image
        """
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]  # Addition of binocular difference and double mouth difference
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Shrink
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
        else:
            rsize = (int(np.rint(float(img.size[0]))), int(np.rint(float(img.size[1]))))

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            quad -= crop[0:2]

        # Transform.
        quad = (quad + 0.5).flatten()
        lx = max(min(quad[0], quad[2]), 0)
        ly = max(min(quad[1], quad[7]), 0)
        rx = min(max(quad[4], quad[6]), img.size[0])
        ry = min(max(quad[3], quad[5]), img.size[0])

        # Save aligned image.
        return rsize, crop, [lx, ly, rx, ry]

    def crop(self, img_np_list, still=False, xsize=512, face_fields=None):    # first frame for all video
        img_np = img_np_list[0]
        lm = self.get_landmark(img_np, face_fields)
        if lm is None:
            raise 'can not detect the landmark from source image'
        rsize, crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=xsize)
        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        for _i in range(len(img_np_list)):
            _inp = img_np_list[_i]
            _inp = cv2.resize(_inp, (rsize[0], rsize[1]))
            _inp = _inp[cly:cry, clx:crx]

            if not still:
                _inp = _inp[ly:ry, lx:rx]

            img_np_list[_i] = _inp
        return img_np_list, crop, quad
