/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "facedetector.h"

#include <QMutex>
#include <QMutexLocker>
#include <QStandardPaths>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

namespace {

constexpr int INPUT_SIZE = 640;
// A face always has some structure. Only near-uniform boxes are dropped — a
// low-key portrait has little contrast too, and its face must survive.
constexpr double FLAT_STDDEV = 3.0;

struct FaceProposal
{
    float x0, y0, x1, y1;
    float prob;
};

QString modelPath()
{
    // AppDataLocation, and no "wunjo/" in front of it — the way every other
    // lookup in this application is written. GenericDataLocation happens to
    // work on Linux, where it lists /app/share and the installed path really is
    // share/wunjo/ai/, and cannot work on macOS for two separate reasons: it
    // names ~/Library/Application Support and never the application bundle, and
    // inside the bundle the file sits at Contents/Resources/ai/ with no wunjo
    // component at all, because DATA_INSTALL_PREFIX is empty there.
    //
    // AppDataLocation resolves to share/wunjo on Linux and to the bundle's
    // Resources on macOS, so the same relative path finds the model on both.
    return QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("ai/mnet.25_v2.simplify.onnx"));
}

cv::dnn::Net &network(bool &ok)
{
    static QMutex mutex;
    static cv::dnn::Net net;
    static bool loaded = false;
    static bool loadOk = false;
    QMutexLocker locker(&mutex);
    if (!loaded) {
        loaded = true;
        const QString path = modelPath();
        if (!path.isEmpty()) {
            try {
                net = cv::dnn::readNetFromONNX(path.toStdString());
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                loadOk = true;
            } catch (const cv::Exception &) {
                loadOk = false;
            }
        }
    }
    ok = loadOk;
    return net;
}

// RetinaFace anchors: baseSize 16, ratio 1, two scales per stride
void generateProposals(const cv::Mat &score, const cv::Mat &bbox, int featStride, const QList<float> &scales, float probThreshold,
                       QList<FaceProposal> &proposals)
{
    const int h = bbox.size[2];
    const int w = bbox.size[3];
    const int offset = w * h;
    const int numAnchors = int(scales.size());
    const float *scoreData = score.ptr<float>();
    const float *bboxData = bbox.ptr<float>();

    for (int q = 0; q < numAnchors; q++) {
        // square anchor centered in a 16px cell, scaled
        const float cxa = 16.f * 0.5f;
        const float anchorSide = 16.f * scales.at(q);
        const float ax0 = cxa - anchorSide * 0.5f;
        const float ay0 = cxa - anchorSide * 0.5f;
        const float anchorW = anchorSide;
        const float anchorH = anchorSide;

        const float *scorePlane = scoreData + (q + numAnchors) * offset;
        const float *bboxPlane = bboxData + q * 4 * offset;

        float anchorY = ay0;
        for (int i = 0; i < h; i++) {
            float anchorX = ax0;
            for (int j = 0; j < w; j++) {
                const int index = i * w + j;
                const float prob = scorePlane[index];
                if (prob >= probThreshold) {
                    const float dx = bboxPlane[index];
                    const float dy = bboxPlane[index + offset];
                    const float dw = bboxPlane[index + offset * 2];
                    const float dh = bboxPlane[index + offset * 3];

                    const float cx = anchorX + anchorW * 0.5f;
                    const float cy = anchorY + anchorH * 0.5f;
                    const float pbCx = cx + anchorW * dx;
                    const float pbCy = cy + anchorH * dy;
                    const float pbW = anchorW * std::exp(dw);
                    const float pbH = anchorH * std::exp(dh);

                    proposals.append({pbCx - pbW * 0.5f, pbCy - pbH * 0.5f, pbCx + pbW * 0.5f, pbCy + pbH * 0.5f, prob});
                }
                anchorX += float(featStride);
            }
            anchorY += float(featStride);
        }
    }
}

QList<int> nmsSortedBboxes(const QList<FaceProposal> &faces, float nmsThreshold)
{
    QList<int> picked;
    for (int i = 0; i < faces.size(); i++) {
        const FaceProposal &a = faces.at(i);
        const float areaA = (a.x1 - a.x0) * (a.y1 - a.y0);
        bool keep = true;
        for (int j : std::as_const(picked)) {
            const FaceProposal &b = faces.at(j);
            const float inter = std::max(0.f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0)) * std::max(0.f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
            const float areaB = (b.x1 - b.x0) * (b.y1 - b.y0);
            if (inter / (areaA + areaB - inter) > nmsThreshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            picked.append(i);
        }
    }
    return picked;
}

} // namespace

bool FaceDetector::isAvailable()
{
    bool ok = false;
    network(ok);
    return ok;
}

QList<QRectF> FaceDetector::detect(const QImage &image, float probThreshold, float nmsThreshold)
{
    QList<QRectF> result;
    bool ok = false;
    cv::dnn::Net &net = network(ok);
    if (!ok || image.isNull()) {
        return result;
    }

    // RGB, raw 0..255 float values (as in the JS reference). The frame is fitted
    // into the model's square input without distortion: RetinaFace is trained on
    // undistorted faces, and stretching a 16:9 frame to 640x640 both deforms the
    // boxes and invents faces out of the stretched flat areas.
    const QImage rgb = image.convertToFormat(QImage::Format_RGB888);
    cv::Mat mat(rgb.height(), rgb.width(), CV_8UC3, const_cast<uchar *>(rgb.constBits()), rgb.bytesPerLine());
    const double scale = std::min(double(INPUT_SIZE) / mat.cols, double(INPUT_SIZE) / mat.rows);
    const int fittedWidth = std::max(1, int(std::lround(mat.cols * scale)));
    const int fittedHeight = std::max(1, int(std::lround(mat.rows * scale)));
    cv::Mat fitted;
    cv::resize(mat, fitted, cv::Size(fittedWidth, fittedHeight));
    cv::Mat input(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    fitted.copyTo(input(cv::Rect(0, 0, fittedWidth, fittedHeight)));
    const cv::Mat blob = cv::dnn::blobFromImage(input, 1.0, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(), /*swapRB*/ false, /*crop*/ false, CV_32F);

    QList<FaceProposal> proposals;
    static QMutex inferenceMutex;
    {
        QMutexLocker locker(&inferenceMutex);
        try {
            net.setInput(blob);
            const std::vector<cv::String> outNames = {
                "face_rpn_cls_prob_reshape_stride32", "face_rpn_bbox_pred_stride32", "face_rpn_cls_prob_reshape_stride16",
                "face_rpn_bbox_pred_stride16",        "face_rpn_cls_prob_reshape_stride8", "face_rpn_bbox_pred_stride8"};
            std::vector<cv::Mat> outs;
            net.forward(outs, outNames);
            generateProposals(outs[0], outs[1], 32, {32.f, 16.f}, probThreshold, proposals);
            generateProposals(outs[2], outs[3], 16, {8.f, 4.f}, probThreshold, proposals);
            generateProposals(outs[4], outs[5], 8, {2.f, 1.f}, probThreshold, proposals);
        } catch (const cv::Exception &) {
            return result;
        }
    }

    std::sort(proposals.begin(), proposals.end(), [](const FaceProposal &a, const FaceProposal &b) { return a.prob > b.prob; });
    const QList<int> picked = nmsSortedBboxes(proposals, nmsThreshold);

    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_RGB2GRAY);
    for (int i : picked) {
        const FaceProposal &f = proposals.at(i);
        // back from the fitted area to the frame — anything the model placed in
        // the padding beside it is not on the image at all
        const qreal x0 = qBound(0.f, f.x0, float(fittedWidth)) / fittedWidth;
        const qreal y0 = qBound(0.f, f.y0, float(fittedHeight)) / fittedHeight;
        const qreal x1 = qBound(0.f, f.x1, float(fittedWidth)) / fittedWidth;
        const qreal y1 = qBound(0.f, f.y1, float(fittedHeight)) / fittedHeight;
        if (x1 - x0 < 0.005 || y1 - y0 < 0.005) {
            continue;
        }
        // measured on the source frame, not on the fitted input
        const cv::Rect crop(cv::Point(int(x0 * gray.cols), int(y0 * gray.rows)), cv::Point(int(x1 * gray.cols), int(y1 * gray.rows)));
        const cv::Rect box = crop & cv::Rect(0, 0, gray.cols, gray.rows);
        if (box.width > 1 && box.height > 1) {
            cv::Scalar mean;
            cv::Scalar stddev;
            cv::meanStdDev(gray(box), mean, stddev);
            if (stddev[0] < FLAT_STDDEV) {
                continue;
            }
        }
        result.append(QRectF(QPointF(x0, y0), QPointF(x1, y1)));
    }
    return result;
}
