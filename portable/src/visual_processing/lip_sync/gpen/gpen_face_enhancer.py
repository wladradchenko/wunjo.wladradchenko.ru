import os
import sys
import cv2
import numpy as np

from .face_parse.face_parsing import FaceParse
from .face_detect.retinaface_detection import RetinaFaceDetection


# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]

DEFAULT_CROP_SIZE = (96, 112)


class FaceEnhancement(object):
    def __init__(self, model_path_retina, model_path_parse, size=512, device='cuda'):
        self.facedetector = RetinaFaceDetection(model_path=model_path_retina, device=device)
        self.faceparser = FaceParse(model_path=model_path_parse, device=device)
        self.size = size
        self.threshold = 0.9

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points((self.size, self.size), inner_padding_factor, outer_padding, default_square)

    def mask_postprocess(self, mask, thres=20):
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        return mask.astype(np.float32)

    def process(self, img, ori_img, bbox=None, possion_blending=False):
        facebs, landms = self.facedetector.detect(img.copy())

        orig_faces, enhanced_faces = [], []
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(ori_img.shape, dtype=np.uint8)

        mask_sharp = None

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold: continue

            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts,  crop_size=(self.size, self.size))

            # enhance the face
            ef = of

            orig_faces.append(of)
            enhanced_faces.append(ef)

            # print(ef.shape)
            # tmp_mask = self.mask
            '''
            0: 'background' 1: 'skin'   2: 'nose'
            3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
            6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
            9: 'r_ear'  10: 'mouth' 11: 'u_lip'
            12: 'l_lip' 13: 'hair'  14: 'hat'
            15: 'ear_r' 16: 'neck_l'    17: 'neck'
            18: 'cloth'
            '''

            # no ear, no neck, no hair&hat,  only face region
            mm = [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            mask_sharp = self.faceparser.process(ef, mm)[0] / 255.
            tmp_mask = self.mask_postprocess(mask_sharp)
            tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
            mask_sharp = cv2.resize(mask_sharp, ef.shape[:2])

            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)
            mask_sharp = cv2.warpAffine(mask_sharp, tfm_inv, (width, height), flags=3)

            if min(fh, fw) < 100:  # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)

            tmp_img = cv2.warpAffine(of, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
            full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]

        if mask_sharp is not None:
            mask_sharp = cv2.GaussianBlur(mask_sharp, (0, 0), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)

            full_mask = full_mask[:, :, np.newaxis]
            mask_sharp = mask_sharp[:, :, np.newaxis]

            if possion_blending is True:
                if bbox is not None:
                    y1, y2, x1, x2 = bbox
                    mask_bbox = np.zeros_like(mask_sharp)
                    mask_bbox[y1:y2 - 5, x1:x2] = 1
                    full_img, ori_img, full_mask = [cv2.resize(x, (512, 512)) for x in (full_img, ori_img, np.float32(mask_sharp * mask_bbox))]
                else:
                    full_img, ori_img, full_mask = [cv2.resize(x, (512, 512)) for x in (full_img, ori_img, full_mask)]

                img = laplacian_pyramid_blending_with_mask(full_img, ori_img, full_mask, 6)
                img = np.clip(img, 0, 255)
                img = np.uint8(cv2.resize(img, (width, height)))

            else:
                img = cv2.convertScaleAbs(ori_img * (1 - full_mask) + full_img * full_mask)
                img = cv2.convertScaleAbs(ori_img * (1 - mask_sharp) + img * mask_sharp)

            return img, orig_faces, enhanced_faces
        return img, None, None


def laplacian_pyramid_blending_with_mask(A, B, m, num_levels = 6):
    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtract upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        gm = gm[:,:,np.newaxis]
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))


def _umeyama(src, dst, estimate_scale=True, scale=1.0):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = scale

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T, scale


def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm


def warp_and_crop_face(src_img, facial_pts, reference_pts=None, crop_size=(96, 112), align_type='smilarity'):
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3:
        raise FaceWarpException('reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2 or ref_pts_shp[0] == 3:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3: # or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2 or src_pts_shp[0] == 3:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
        tfm_inv = cv2.getAffineTransform(ref_pts[0:3], src_pts[0:3])
    elif align_type == 'cv2_rigid':
        tfm, _ = cv2.estimateAffinePartial2D(src_pts[0:3], ref_pts[0:3])
        tfm_inv, _ = cv2.estimateAffinePartial2D(ref_pts[0:3], src_pts[0:3])
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
        tfm_inv = get_affine_transform_matrix(ref_pts, src_pts)
    else:
        params, scale = _umeyama(src_pts, ref_pts)
        tfm = params[:2, :]

        params, _ = _umeyama(ref_pts, src_pts, False, scale=1.0/scale)
        tfm_inv = params[:2, :]

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]), flags=3)

    return face_img, tfm_inv


def get_reference_facial_points(output_size=None, inner_padding_factor=0.0, outer_padding=(0, 0), default_square=False):
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if (output_size and output_size[0] == tmp_crop_size[0] and output_size[1] == tmp_crop_size[1]):
        print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
        return tmp_5pts

    if (inner_padding_factor == 0 and outer_padding == (0, 0)):
        if output_size is None:
            print('No paddings to do: return default reference points')
            return tmp_5pts
        else:
            raise FaceWarpException('No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0) and output_size is None):
        output_size = tmp_crop_size * (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
        print('deduced from paddings, output_size = ', output_size)

    if not (outer_padding[0] < output_size[0]
            and outer_padding[1] < output_size[1]):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1])')

    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException('Must have (output_size - outer_padding) = some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor
    reference_5point = tmp_5pts + np.array(outer_padding)

    return reference_5point
