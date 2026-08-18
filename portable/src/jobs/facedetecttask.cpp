/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "facedetecttask.h"
#include "ai/facedatastore.h"
#include "ai/facedetector.h"
#include "bin/projectclip.h"
#include "bin/projectitemmodel.h"
#include "core.h"
#include "doc/kthumb.h"
#include "doc/wunjodoc.h"

#include <mlt++/MltProducer.h>

#include <KLocalizedString>
#include <QImage>
#include <QScopedPointer>

FaceDetectTask::FaceDetectTask(const ObjectId &owner, const QString &binId, const QString &saveFolder, QObject *object, int from, int to)
    : AbstractTask(owner, AbstractTask::ANALYSECLIPJOB, object)
    , m_binId(binId)
    , m_saveFolder(saveFolder)
    , m_urgentFrom(from)
    , m_urgentTo(to)
{
    m_description = i18n("Detecting faces");
}

QList<QPair<int, int>> FaceDetectTask::passes(int duration) const
{
    if (m_urgentFrom < 0 || m_urgentTo <= m_urgentFrom) {
        return {{0, duration}};
    }
    const int from = qBound(0, m_urgentFrom, duration);
    const int to = qBound(from, m_urgentTo, duration);
    // The urgent part, then the whole clip. The second pass costs nothing where
    // the first has already been: every frame in the store is skipped.
    return {{from, to}, {0, duration}};
}

void FaceDetectTask::start(const QString &binId, QObject *object, int from, int to)
{
    const ObjectId owner(WunjoObjectType::BinClip, binId.toInt(), QUuid());
    if (pCore->taskManager.hasPendingJob(owner, AbstractTask::ANALYSECLIPJOB)) {
        return;
    }
    if (!FaceDetector::isAvailable()) {
        pCore->displayMessage(i18n("Face detection model is missing from the installation."), ErrorMessage);
        return;
    }
    const QString folder = pCore->currentDoc() ? pCore->currentDoc()->projectDataFolder() : QString();
    if (object == nullptr) {
        // report progress on the bin item
        object = pCore->projectItemModel()->getClipByBinID(binId).get();
    }
    FaceDetectTask *task = new FaceDetectTask(owner, binId, folder, object, from, to);
    pCore->taskManager.startTask(owner.itemId, task);
}

void FaceDetectTask::run()
{
    AbstractTaskDone whenFinished(m_owner.itemId, this);
    if (m_isCanceled || pCore->taskManager.isBlocked()) {
        return;
    }
    QMutexLocker lock(&m_runMutex);
    auto binClip = pCore->projectItemModel()->getClipByBinID(m_binId);
    if (!binClip) {
        return;
    }
    const int duration = binClip->getFramePlaytime();
    if (duration <= 0) {
        return;
    }
    std::unique_ptr<Mlt::Producer> producer = binClip->getThumbProducer();
    if (producer == nullptr) {
        return;
    }
    FaceDataStore &store = FaceDataStore::instance();
    // analyse around the model's input resolution — a bigger frame brings no
    // extra precision, it is fitted into 640x640 anyway. The extra height over
    // the model input pays off for a clip that only fills part of the frame.
    const int imageHeight = 540;
    const int imageWidth = qRound(imageHeight * pCore->getCurrentDar());
    // The producer renders into the project profile, so a clip of another aspect
    // comes back letter- or pillarboxed. Those bars carry no face, cost the
    // detector resolution and used to produce boxes of their own, so analysis
    // runs on the content only and the results are mapped back to frame space.
    QRectF content(0, 0, 1, 1);
    const QSize clipSize = binClip->frameSize();
    if (clipSize.width() > 0 && clipSize.height() > 0) {
        const double clipDar = double(clipSize.width()) / clipSize.height();
        const double frameDar = pCore->getCurrentDar();
        if (clipDar < frameDar - 0.001) {
            const double width = clipDar / frameDar;
            content = QRectF((1. - width) / 2, 0, width, 1);
        } else if (clipDar > frameDar + 0.001) {
            const double height = frameDar / clipDar;
            content = QRectF(0, (1. - height) / 2, 1, height);
        }
    }
    // sample on the store's analysis grid — intermediate positions are
    // interpolated on display, so decoding + inference drop by the same factor
    int analysed = 0;
    const QList<QPair<int, int>> plan = passes(duration);
    const int total = qMax(1, plan.constFirst().second - plan.constFirst().first + (plan.size() > 1 ? duration : 0));
    for (const QPair<int, int> &pass : plan) {
    for (int pos = pass.first; pos < pass.second; pos += FaceDataStore::kAnalyseStep) {
        if (m_isCanceled || pCore->taskManager.isBlocked()) {
            break;
        }
        if (!store.isEnabled(m_binId)) {
            // Switched off while this was running. Nobody is going to look at
            // what comes out, so the rest of a ten-minute file is not worth
            // decoding — it used to grind on to the end regardless.
            m_isCanceled = true;
            break;
        }
        analysed += FaceDataStore::kAnalyseStep;
        if (store.hasFrameData(m_binId, pos)) {
            continue;
        }
        producer->seek(pos);
        QScopedPointer<Mlt::Frame> frame(producer->get_frame());
        if (frame == nullptr || !frame->is_valid()) {
            continue;
        }
        frame->set("consumer.deinterlacer", "onefield");
        frame->set("consumer.top_field_first", -1);
        frame->set("consumer.rescale", "nearest");
        const QImage image = KThumb::getFrame(frame.get(), imageWidth, imageHeight, imageWidth);
        if (image.isNull()) {
            continue;
        }
        QList<QRectF> faces;
        if (content == QRectF(0, 0, 1, 1)) {
            faces = FaceDetector::detect(image);
        } else {
            const QRect crop(qRound(content.x() * image.width()), qRound(content.y() * image.height()), qRound(content.width() * image.width()),
                             qRound(content.height() * image.height()));
            const QList<QRectF> found = FaceDetector::detect(image.copy(crop));
            for (const QRectF &face : found) {
                faces.append(QRectF(content.x() + face.x() * content.width(), content.y() + face.y() * content.height(), face.width() * content.width(),
                                    face.height() * content.height()));
            }
        }
        store.addFrameData(m_binId, pos, faces);
        if (pos == pass.first) {
            // the monitor only redraws the boxes when the playhead moves, so a
            // clip standing still under it would stay bare until the user seeks
            Q_EMIT pCore->faceDataChanged(m_binId);
        }
        const int val = qBound(1, 100 * analysed / total, 99);
        if (m_progress != val) {
            m_progress = val;
            if (m_object) {
                QMetaObject::invokeMethod(m_object, "updateJobProgress");
            }
        }
    }
    // The part that was asked for is done: show it before starting on the rest.
    Q_EMIT pCore->faceDataChanged(m_binId);
    if (m_isCanceled || pCore->taskManager.isBlocked()) {
        break;
    }
    }
    if (!m_isCanceled) {
        store.markComplete(m_binId);
    }
    Q_EMIT pCore->faceDataChanged(m_binId);
    // keep whatever was analysed, even on cancel
    store.saveClip(m_binId, m_saveFolder);
}
