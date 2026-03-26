#include <QApplication>
#include <QMainWindow>
#include <QFileInfo>
#include <QScreen>
#include <QTimer>
#include <QSurfaceFormat>
#include <QMenu>
#include <QAction>
#include <QSystemTrayIcon>
#include <QIcon>
#include <QDesktopServices>
#include <QUrl>
#include <QLabel>
#include <QPixmap>
#include <QFont>
#include <QVBoxLayout>
#include <QDir>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <algorithm>
#include "engine/Renderer.hpp"
#include "common/SettingsManager.hpp"
#include "ui/SettingsWindow.hpp"
#include "ai/ChatController.hpp"
#include "ui/ChatWindow.hpp"
#include "ui/theme/ThemeApi.hpp"
#include "common/Utils.hpp"
#include "audio/OfflineVoiceService.hpp"
#include <QEvent>
#include <QWindow>
#include <QShortcut>
#include <QTranslator>
#include <QtGlobal>
#include <functional>
#include <QJsonArray>
#include <QJsonObject>
#include <QDateTime>
#include <QTime>
#include <QHash>
#include <QTextDocument>
#include <QAbstractTextDocumentLayout>
#include <QTextOption>
#include <QFontMetrics>
#include <QPalette>
#include <QStandardPaths>
#include <QFile>
#include <QTextStream>
#include <QMutex>
#include <cstdlib>
#include <cmath>

namespace {
QString startupLogFilePath()
{
    QString base = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    if (base.isEmpty())
        base = QDir::homePath();
    QDir d(base);
    d.mkpath(QStringLiteral("logs"));
    return d.filePath(QStringLiteral("logs/startup.log"));
}

void installStartupLogger()
{
    static QMutex s_mu;
    static const QString s_path = startupLogFilePath();
    qInstallMessageHandler([](QtMsgType type, const QMessageLogContext& ctx, const QString& msg){
        Q_UNUSED(ctx);
        QMutexLocker locker(&s_mu);
        QFile f(s_path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
            return;
        QTextStream ts(&f);
        const QString t = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss.zzz"));
        QString level;
        switch (type) {
        case QtDebugMsg: level = QStringLiteral("D"); break;
        case QtInfoMsg: level = QStringLiteral("I"); break;
        case QtWarningMsg: level = QStringLiteral("W"); break;
        case QtCriticalMsg: level = QStringLiteral("C"); break;
        case QtFatalMsg: level = QStringLiteral("F"); break;
        }
        ts << t << " [" << level << "] " << msg << "\n";
        ts.flush();
    });

    {
        QMutexLocker locker(&s_mu);
        QFile f(s_path);
        if (f.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
            QTextStream ts(&f);
            const QString t = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss.zzz"));
            ts << "\n" << t << " [I] " << QStringLiteral("---- startup ----") << "\n";
            ts.flush();
        }
    }
}

QString formatExceptionMessage(const std::exception& e)
{
    const QString w = QString::fromLocal8Bit(e.what());
    return w.isEmpty() ? QObject::tr("未知错误") : w;
}

void showModelLoadError(const QString& modelPath, const QString& detail)
{
    QMessageBox::critical(
        nullptr,
        QObject::tr("XiaoMo"),
        QObject::tr("模型加载失败：\n%1\n\n%2").arg(modelPath, detail));
}

QString normalizeBubbleStyleId(const QString& styleId)
{
    const QString s = styleId.trimmed().toLower();
    if (s == QStringLiteral("imessage") || s == QStringLiteral("i-message") || s == QStringLiteral("ios"))
        return QStringLiteral("iMessage");
    if (s == QStringLiteral("outline") || s == QStringLiteral("flowbite"))
        return QStringLiteral("Outline");
    if (s == QStringLiteral("cloud") || s == QStringLiteral("cloudy") || s == QStringLiteral("cloud-bubble"))
        return QStringLiteral("Cloud");
    if (s == QStringLiteral("heart") || s == QStringLiteral("love"))
        return QStringLiteral("Heart");
    if (s == QStringLiteral("comic") || s == QStringLiteral("manga") || s == QStringLiteral("burst"))
        return QStringLiteral("Comic");
    if (s == QStringLiteral("round") || s == QStringLiteral("material") || s == QStringLiteral("era") || s == QStringLiteral("default"))
        return QStringLiteral("Round");
    return QStringLiteral("Round");
}

class PetSpeechBubbleWidget final : public QWidget
{
public:
    explicit PetSpeechBubbleWidget(QWidget* parent = nullptr)
        : QWidget(parent)
    {
        setObjectName(QStringLiteral("petSpeechBubble"));
        setWindowFlag(Qt::Tool, true);
        setWindowFlag(Qt::FramelessWindowHint, true);
        setWindowFlag(Qt::NoDropShadowWindowHint, true);
        setWindowFlag(Qt::WindowStaysOnTopHint, SettingsManager::instance().windowAlwaysOnTop());
        setAttribute(Qt::WA_TranslucentBackground, true);
        setAttribute(Qt::WA_TransparentForMouseEvents, true);
        setFocusPolicy(Qt::NoFocus);

        m_label = new QLabel(this);
        m_label->setObjectName(QStringLiteral("petSpeechText"));
        m_label->setWordWrap(true);
        m_label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        m_label->setStyleSheet(QStringLiteral("QLabel#petSpeechText{color: transparent;}"));

        m_layout = new QVBoxLayout(this);
        m_layout->setSpacing(0);
        m_layout->addWidget(m_label);
        refreshLayoutMargins();
        updateTextLayoutForStyle();
    }

    QLabel* label() const { return m_label; }

    void setBubbleText(const QString& text)
    {
        if (!m_label)
            return;
        if (m_label->text() == text)
            return;
        m_label->setText(text);
        updateTextLayoutForStyle();
        updateGeometry();
        update();
    }

    void setBubbleStyle(const QString& styleId)
    {
        const QString normalized = normalizeBubbleStyleId(styleId);
        if (m_styleId == normalized)
            return;
        m_styleId = normalized;
        applyTailSizeForStyle();
        refreshLayoutMargins();
        updateTextLayoutForStyle();
        updateGeometry();
        update();
    }

    void setTailOnLeft(bool onLeft)
    {
        if (m_tailOnLeft == onLeft)
            return;
        m_tailOnLeft = onLeft;
        refreshLayoutMargins();
        updateTextLayoutForStyle();
        updateGeometry();
        update();
    }

protected:
    void paintEvent(QPaintEvent* event) override
    {
        Q_UNUSED(event);
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);

        const QString styleId = normalizeBubbleStyleId(m_styleId);
        const QRectF r = QRectF(rect()).adjusted(2.0, 2.0, -2.0, -2.0);

        qreal tailW = qreal(m_tailW);
        qreal tailH = qreal(m_tailH);
        qreal radius = 16.0;

        QColor fill(20, 20, 20, 205);
        QColor border = Qt::transparent;
        QColor text = QColor(255, 255, 255);
        qreal borderW = 1.0;
        bool drawShadow = true;
        bool drawBorder = false;
        bool wantCloud = false;
        bool wantComic = false;
        bool wantHeart = false;

        if (styleId == QStringLiteral("iMessage"))
        {
            radius = 18.0;
            fill = QColor(46, 170, 79, 235);
            border = Qt::transparent;
            text = QColor(255, 255, 255);
            drawShadow = true;
            drawBorder = false;
            borderW = 0.0;
        }
        else if (styleId == QStringLiteral("Outline"))
        {
            radius = 14.0;
            fill = QColor(255, 255, 255, 0);
            border = QColor(255, 255, 255, 220);
            text = QColor(255, 255, 255);
            drawShadow = false;
            drawBorder = true;
            borderW = 2.0;
        }
        else if (styleId == QStringLiteral("Cloud"))
        {
            radius = 22.0;
            fill = QColor(255, 255, 255, 236);
            border = QColor(0, 0, 0, 45);
            text = QColor(18, 18, 18);
            drawShadow = true;
            drawBorder = true;
            wantCloud = true;
            borderW = 1.2;
        }
        else if (styleId == QStringLiteral("Heart"))
        {
            radius = 0.0;
            fill = QColor(255, 92, 140, 225);
            border = Qt::transparent;
            text = QColor(255, 255, 255);
            drawShadow = true;
            drawBorder = false;
            wantHeart = true;
        }
        else if (styleId == QStringLiteral("Comic"))
        {
            radius = 10.0;
            fill = QColor(255, 232, 110, 238);
            border = QColor(20, 20, 20, 210);
            text = QColor(20, 20, 20);
            drawShadow = false;
            drawBorder = true;
            wantComic = true;
            borderW = 2.0;
        }

        const QRectF bodyRect = m_tailOnLeft ? r.adjusted(tailW, 0.0, 0.0, 0.0) : r.adjusted(0.0, 0.0, -tailW, 0.0);
        if (radius > 0.0)
            radius = std::min<qreal>(radius, std::max<qreal>(6.0, std::min(bodyRect.width(), bodyRect.height()) * 0.5 - 1.0));

        QPainterPath bubble;
        if (wantHeart)
        {
            const QRectF hr = bodyRect.adjusted(2.0, 2.0, -2.0, -2.0);
            double xmin = 1e9, xmax = -1e9;
            double ymin = 1e9, ymax = -1e9;
            QVector<QPointF> raw;
            raw.reserve(900);

            const int n = 720;
            for (int i = 0; i <= n; ++i)
            {
                const double t = (double(i) / double(n)) * 6.283185307179586;
                const double s = std::sin(t);
                const double c = std::cos(t);
                const double x = 16.0 * s * s * s;
                const double y = -(13.0 * c - 5.0 * std::cos(2.0 * t) - 2.0 * std::cos(3.0 * t) - std::cos(4.0 * t));
                raw.push_back(QPointF(x, y));
                xmin = std::min(xmin, x); xmax = std::max(xmax, x);
                ymin = std::min(ymin, y); ymax = std::max(ymax, y);
            }

            const double w0 = std::max(1e-6, xmax - xmin);
            const double h0 = std::max(1e-6, ymax - ymin);
            const double s0 = 0.92 * std::min(double(hr.width()) / w0, double(hr.height()) / h0);
            const double cx0 = (xmin + xmax) * 0.5;
            const double cy0 = (ymin + ymax) * 0.5;
            const double dy = double(hr.height()) * 0.02;

            bubble = QPainterPath();
            bubble.setFillRule(Qt::WindingFill);
            bool first = true;
            for (const QPointF& p : raw)
            {
                const qreal px = qreal((p.x() - cx0) * s0 + hr.center().x());
                const qreal py = qreal((p.y() - cy0) * s0 + hr.center().y() + dy);
                if (first) { bubble.moveTo(px, py); first = false; }
                else { bubble.lineTo(px, py); }
            }
            bubble.closeSubpath();
        }
        else if (wantCloud)
        {
            const QRectF cr = bodyRect.adjusted(2.0, 2.0, -2.0, -2.0);
            const qreal w = cr.width();
            const qreal h = cr.height();
            const qreal bodyR = std::max<qreal>(10.0, std::min<qreal>(h * 0.32, w * 0.18));
            const qreal bumpR = std::max<qreal>(8.0, std::min<qreal>(bodyR * 0.95, w * 0.13));

            QPainterPath cloud;
            cloud.setFillRule(Qt::WindingFill);
            const QRectF base = cr.adjusted(0.0, bumpR * 0.75, 0.0, 0.0);
            cloud.addRoundedRect(base, bodyR, bodyR);

            const qreal y = base.top() + bodyR * 0.10;
            const qreal x0 = base.left() + bumpR * 1.05;
            const qreal x1 = base.left() + w * 0.30;
            const qreal x2 = base.center().x();
            const qreal x3 = base.left() + w * 0.70;
            const qreal x4 = base.right() - bumpR * 1.05;

            cloud.addEllipse(QPointF(x0, y + bumpR * 0.15), bumpR * 0.95, bumpR * 0.90);
            cloud.addEllipse(QPointF(x1, y - bumpR * 0.10), bumpR * 1.20, bumpR * 1.10);
            cloud.addEllipse(QPointF(x2, y - bumpR * 0.22), bumpR * 1.40, bumpR * 1.25);
            cloud.addEllipse(QPointF(x3, y - bumpR * 0.06), bumpR * 1.18, bumpR * 1.08);
            cloud.addEllipse(QPointF(x4, y + bumpR * 0.16), bumpR * 0.92, bumpR * 0.88);

            bubble = cloud;
        }
        else if (wantComic)
        {
            const QPointF c = bodyRect.center();
            const qreal rx = bodyRect.width() * 0.50;
            const qreal ry = bodyRect.height() * 0.50;
            const int n = 18;
            QPolygonF poly;
            poly.reserve(n);
            for (int i = 0; i < n; ++i)
            {
                const qreal a = (qreal(i) / qreal(n)) * 6.283185307179586;
                const qreal k = (i % 2 == 0) ? 1.0 : 0.72;
                poly << QPointF(c.x() + std::cos(a) * rx * k, c.y() + std::sin(a) * ry * k);
            }
            bubble.addPolygon(poly);
        }
        else
        {
            bubble.addRoundedRect(bodyRect, radius, radius);
        }

        bubble.setFillRule(Qt::WindingFill);

        if (tailW > 0.0 && tailH > 0.0 && !wantHeart)
        {
            const qreal tailY = bodyRect.bottom() - std::max<qreal>(1.0, radius) * 0.60;

            if (styleId == QStringLiteral("iMessage"))
            {
                QPainterPath tail;
                if (m_tailOnLeft)
                {
                    const QPointF p0(bodyRect.left(), tailY);
                    const QPointF p1(r.left() + tailW * 0.20, tailY + tailH * 0.10);
                    const QPointF p2(r.left(), tailY + tailH * 0.60);
                    const QPointF p3(bodyRect.left(), tailY + tailH);

                    tail.moveTo(p0);
                    tail.cubicTo(QPointF(bodyRect.left() - tailW * 0.85, tailY + tailH * 0.05), p1, p2);
                    tail.cubicTo(QPointF(r.left() + tailW * 0.35, tailY + tailH * 0.92), QPointF(bodyRect.left() - tailW * 0.15, tailY + tailH), p3);
                    tail.closeSubpath();
                }
                else
                {
                    const QPointF p0(bodyRect.right(), tailY);
                    const QPointF p1(r.right() - tailW * 0.20, tailY + tailH * 0.10);
                    const QPointF p2(r.right(), tailY + tailH * 0.60);
                    const QPointF p3(bodyRect.right(), tailY + tailH);

                    tail.moveTo(p0);
                    tail.cubicTo(QPointF(bodyRect.right() + tailW * 0.85, tailY + tailH * 0.05), p1, p2);
                    tail.cubicTo(QPointF(r.right() - tailW * 0.35, tailY + tailH * 0.92), QPointF(bodyRect.right() + tailW * 0.15, tailY + tailH), p3);
                    tail.closeSubpath();
                }
                bubble.addPath(tail);
            }
            else if (wantCloud)
            {
                const qreal c1 = std::max<qreal>(3.0, tailW * 0.18);
                const qreal c2 = std::max<qreal>(4.0, tailW * 0.30);
                const qreal c3 = std::max<qreal>(5.0, tailW * 0.42);
                if (m_tailOnLeft)
                {
                    const qreal x = r.left() + tailW * 0.10;
                    bubble.addEllipse(QPointF(x + c1, tailY + tailH * 0.90), c1, c1);
                    bubble.addEllipse(QPointF(x + c2 + tailW * 0.10, tailY + tailH * 0.62), c2, c2);
                    bubble.addEllipse(QPointF(x + c3 + tailW * 0.22, tailY + tailH * 0.30), c3, c3);
                }
                else
                {
                    const qreal x = r.right() - tailW * 0.10;
                    bubble.addEllipse(QPointF(x - c1, tailY + tailH * 0.90), c1, c1);
                    bubble.addEllipse(QPointF(x - (c2 + tailW * 0.10), tailY + tailH * 0.62), c2, c2);
                    bubble.addEllipse(QPointF(x - (c3 + tailW * 0.22), tailY + tailH * 0.30), c3, c3);
                }
            }
            else if (wantComic)
            {
                QPolygonF tail;
                if (m_tailOnLeft)
                {
                    tail << QPointF(bodyRect.left() + radius * 0.15, tailY + tailH * 0.10)
                         << QPointF(r.left() + tailW * 0.10, tailY + tailH * 0.28)
                         << QPointF(bodyRect.left() + radius * 0.28, tailY + tailH * 0.46)
                         << QPointF(r.left() + tailW * 0.05, tailY + tailH * 0.62)
                         << QPointF(bodyRect.left() + radius * 0.22, tailY + tailH * 0.86);
                }
                else
                {
                    tail << QPointF(bodyRect.right() - radius * 0.15, tailY + tailH * 0.10)
                         << QPointF(r.right() - tailW * 0.10, tailY + tailH * 0.28)
                         << QPointF(bodyRect.right() - radius * 0.28, tailY + tailH * 0.46)
                         << QPointF(r.right() - tailW * 0.05, tailY + tailH * 0.62)
                         << QPointF(bodyRect.right() - radius * 0.22, tailY + tailH * 0.86);
                }
                bubble.addPolygon(tail);
            }
            else
            {
                QPolygonF tail;
                if (m_tailOnLeft)
                {
                    tail << QPointF(bodyRect.left(), tailY)
                         << QPointF(r.left(), tailY + tailH * 0.35)
                         << QPointF(bodyRect.left(), tailY + tailH);
                }
                else
                {
                    tail << QPointF(bodyRect.right(), tailY)
                         << QPointF(r.right(), tailY + tailH * 0.35)
                         << QPointF(bodyRect.right(), tailY + tailH);
                }
                bubble.addPolygon(tail);
            }
        }

        if (drawShadow)
        {
            QColor shadow = Qt::black;
            shadow.setAlphaF(0.18);
            painter.setPen(Qt::NoPen);
            painter.setBrush(shadow);
            painter.save();
            painter.translate(0.0, 1.5);
            painter.drawPath(bubble);
            painter.restore();
        }

        if (drawBorder)
        {
            QPen p(border, borderW);
            if (wantComic)
                p.setJoinStyle(Qt::MiterJoin);
            painter.setPen(p);
        }
        else
            painter.setPen(Qt::NoPen);
        painter.setBrush(fill);
        painter.drawPath(bubble);

        if (m_label && !m_label->text().trimmed().isEmpty())
        {
            painter.save();
            painter.setClipPath(bubble);

            const QRectF tr = m_label->geometry();
            QTextDocument doc;
            doc.setDefaultFont(m_label->font());
            doc.setPlainText(m_label->text());

            QTextOption opt;
            opt.setWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
            opt.setAlignment(m_label->alignment());
            doc.setDefaultTextOption(opt);
            doc.setTextWidth(tr.width());

            QAbstractTextDocumentLayout::PaintContext ctx;
            ctx.palette.setColor(QPalette::Text, text);
            ctx.clip = QRectF(0.0, 0.0, tr.width(), tr.height());

            painter.translate(tr.topLeft());
            doc.documentLayout()->draw(&painter, ctx);
            painter.restore();
        }
    }

private:
    void updateTextLayoutForStyle()
    {
        if (!m_label || !m_layout)
            return;

        const QString styleId = normalizeBubbleStyleId(m_styleId);
        const QString text = m_label->text();

        Qt::Alignment align = Qt::AlignLeft | Qt::AlignVCenter;
        if (styleId == QStringLiteral("Heart") || styleId == QStringLiteral("Comic"))
            align = Qt::AlignHCenter | Qt::AlignVCenter;
        m_label->setAlignment(align);
        m_layout->setAlignment(m_label, align);

        if (text.trimmed().isEmpty())
        {
            m_label->setFixedSize(1, 1);
            return;
        }

        int bubbleMinW = 90;
        int bubbleMaxW = 320;
        double targetRatio = 1.65;
        if (styleId == QStringLiteral("Heart")) { bubbleMinW = 240; bubbleMaxW = 440; targetRatio = 0.90; }
        else if (styleId == QStringLiteral("Cloud")) { bubbleMinW = 120; bubbleMaxW = 360; targetRatio = 1.85; }
        else if (styleId == QStringLiteral("Comic")) { bubbleMinW = 130; bubbleMaxW = 340; targetRatio = 1.35; }
        else if (styleId == QStringLiteral("iMessage")) { bubbleMinW = 110; bubbleMaxW = 360; targetRatio = 1.75; }
        else if (styleId == QStringLiteral("Outline")) { bubbleMinW = 110; bubbleMaxW = 360; targetRatio = 1.75; }

        const QMargins mg = m_layout->contentsMargins();
        const int minTextW = qMax(60, bubbleMinW - mg.left() - mg.right());
        const int maxTextW = qMax(minTextW, bubbleMaxW - mg.left() - mg.right());

        QTextDocument doc;
        doc.setDefaultFont(m_label->font());
        doc.setPlainText(text);
        QTextOption opt = doc.defaultTextOption();
        opt.setWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
        doc.setDefaultTextOption(opt);

        const QFontMetrics fm(m_label->font());

        auto eval = [&](int w) -> QPair<double, QSize> {
            doc.setTextWidth(w);
            doc.adjustSize();
            const int docW = qMax(1, int(std::ceil(doc.size().width())));
            const int docH = qMax(fm.height(), int(std::ceil(doc.size().height())));
            const int bubbleW = docW + mg.left() + mg.right();
            const int bubbleH = docH + mg.top() + mg.bottom();
            const double ratio = bubbleH > 0 ? (double(bubbleW) / double(bubbleH)) : 999.0;
            const double area = double(bubbleW) * double(bubbleH);
            const double areaNorm = area / double(bubbleMaxW * bubbleMaxW);
            const double score = std::abs(ratio - targetRatio) + areaNorm * 0.12;
            return {score, QSize(docW, docH)};
        };

        int bestW = minTextW;
        QSize bestSz;
        double bestScore = 1e18;
        const int step = (maxTextW - minTextW <= 80) ? 10 : 20;
        for (int w = minTextW; w <= maxTextW; w += step)
        {
            const auto r = eval(w);
            const double score = r.first;
            const QSize sz = r.second;
            if (score < bestScore - 1e-6 || (std::abs(score - bestScore) <= 1e-6 && sz.width() > bestSz.width()))
            {
                bestScore = score;
                bestW = w;
                bestSz = sz;
            }
        }
        {
            const auto r = eval(maxTextW);
            const double score = r.first;
            const QSize sz = r.second;
            if (score < bestScore - 1e-6 || (std::abs(score - bestScore) <= 1e-6 && sz.width() > bestSz.width()))
            {
                bestScore = score;
                bestW = maxTextW;
                bestSz = sz;
            }
        }

        Q_UNUSED(bestW);
        m_label->setFixedWidth(bestSz.width());
        m_label->setFixedHeight(bestSz.height());
    }

    void applyTailSizeForStyle()
    {
        const QString styleId = normalizeBubbleStyleId(m_styleId);
        int w = 10;
        int h = 12;
        if (styleId == QStringLiteral("iMessage")) { w = 12; h = 18; }
        else if (styleId == QStringLiteral("Cloud")) { w = 0; h = 0; }
        else if (styleId == QStringLiteral("Heart")) { w = 0; h = 0; }
        else if (styleId == QStringLiteral("Comic")) { w = 0; h = 0; }
        else if (styleId == QStringLiteral("Outline")) { w = 0; h = 0; }
        m_tailW = w;
        m_tailH = h;
    }

    void refreshLayoutMargins()
    {
        if (!m_layout)
            return;
        const QString styleId = normalizeBubbleStyleId(m_styleId);
        int left = 14 + (m_tailOnLeft ? m_tailW : 0);
        int right = 14 + (!m_tailOnLeft ? m_tailW : 0);
        int top = 12;
        int bottom = 12;
        if (styleId == QStringLiteral("Cloud")) { left = right = 18; top = 16; bottom = 16; }
        else if (styleId == QStringLiteral("Heart")) { left = right = 58; top = 48; bottom = 74; }
        else if (styleId == QStringLiteral("Comic")) { left = right = 20; top = 18; bottom = 18; }
        else if (styleId == QStringLiteral("Outline")) { left = right = 16; top = 14; bottom = 14; }
        m_layout->setContentsMargins(left, top, right, bottom);
    }

    QString m_styleId{QStringLiteral("Round")};
    bool m_tailOnLeft{true};
    int m_tailW{10};
    int m_tailH{12};
    QLabel* m_label{nullptr};
    QVBoxLayout* m_layout{nullptr};
};
} // namespace

static bool isLinuxWayland()
{
#if defined(Q_OS_LINUX)
    return QGuiApplication::platformName().startsWith(QStringLiteral("wayland"));
#else
    return false;
#endif
}

#if defined(Q_OS_MACOS)
static QIcon resolveMacTrayIcon(const QString& resRoot, const QIcon& fallbackIcon)
{
    const QString preferredIconPath =
        QDir(resRoot).filePath(QStringLiteral("icons/menubar-icon-dark.png"));
    const QString fallbackMenubarIconPath =
        QDir(resRoot).filePath(QStringLiteral("icons/menubar-icon.png"));

    QIcon trayIcon(preferredIconPath);
    if (trayIcon.isNull()) {
        trayIcon = QIcon(fallbackMenubarIconPath);
    }
    if (trayIcon.isNull()) {
        trayIcon = fallbackIcon;
    }

    // Mark the menu bar icon as a template/mask icon so macOS automatically
    // renders it with the correct contrast for light and dark menu bar backgrounds.
    if (!trayIcon.isNull()) {
        trayIcon.setIsMask(true);
    }

    return trayIcon;
}
#endif

static void cacheWindowGeometry(QWidget* w)
{
#if defined(Q_OS_LINUX)
    if (!w) return;
    w->setProperty("amaigirl_cached_geometry", w->geometry());
#else
    Q_UNUSED(w);
#endif
}

static void restoreWindowGeometry(QWidget* w)
{
#if defined(Q_OS_LINUX)
    if (!w) return;
    const QRect cached = w->property("amaigirl_cached_geometry").toRect();
    if (!cached.isValid()) return;
    w->setGeometry(cached);
#else
    Q_UNUSED(w);
#endif
}

// Helper to center/reset window
static void centerAndSize(QMainWindow& win) {
    QScreen* screen = QGuiApplication::primaryScreen();
    int screenH = screen ? screen->geometry().height() : 900;
    int initH = screenH * 2 / 5;
    // 初始宽先按一半，稍后由 Renderer 发出建议再调整
    int initW = initH / 2;
    QRect scr = screen ? screen->availableGeometry() : QRect(0,0,1280,800);
    int x = scr.x() + (scr.width() - initW)/2;
    int y = scr.y() + (scr.height() - initH)/2;
    win.resize(initW, initH);
    win.move(x, y);
}

// Bring a window to front once (try without flags first; fallback to temporary topmost without re-show)
static void bringToFrontOnce(QWidget* w) {
    if (!w) return;
    w->show();
    w->raise();
    w->activateWindow();
    if (auto *wh = w->windowHandle()) wh->requestActivate();

#if defined(Q_OS_LINUX)
    if (isLinuxWayland()) {
        return;
    }
#endif

    // If already active, don't touch flags to avoid flicker
    QTimer::singleShot(50, w, [w]{
        if (w->isActiveWindow()) return;
        const bool hadTop = w->windowFlags().testFlag(Qt::WindowStaysOnTopHint);
        if (!hadTop) w->setWindowFlag(Qt::WindowStaysOnTopHint, true);
        w->raise();
        w->activateWindow();
        if (auto *wh2 = w->windowHandle()) wh2->requestActivate();
        // Remove temporary flag after a short delay without calling show() again (to avoid flicker)
        QTimer::singleShot(250, w, [w, hadTop]{
            if (!hadTop) {
                w->setWindowFlag(Qt::WindowStaysOnTopHint, false);
                w->raise(); // keep it above siblings
            }
        });
    });
}

// Helper to center any top-level widget on the screen that contains its parent/itself
static void centerOnCurrentScreen(QWidget* w)
{
    if (!w) return;
    QWidget* ref = w->parentWidget() ? w->parentWidget() : w;
    QScreen* screen = QGuiApplication::screenAt(ref->geometry().center());
    if (!screen) screen = QGuiApplication::primaryScreen();
    const QRect scr = screen ? screen->availableGeometry() : QRect(0, 0, 1280, 800);
    const QSize sz = w->size();
    const int x = scr.x() + (scr.width() - sz.width()) / 2;
    const int y = scr.y() + (scr.height() - sz.height()) / 2;
    w->move(x, y);
}

// Helper to find preferred screen by SettingsManager, fallback to primary
static QScreen* resolvePreferredScreen()
{
    const QString preferred = SettingsManager::instance().preferredScreenName().trimmed();
    if (!preferred.isEmpty())
    {
        const QList<QScreen*> screens = QGuiApplication::screens();
        for (QScreen* s : screens)
        {
            if (s && s->name() == preferred)
                return s;
        }
    }
    return QGuiApplication::primaryScreen();
}

static void moveWindowToScreenCenter(QMainWindow& win, QScreen* screen)
{
    if (!screen) screen = QGuiApplication::primaryScreen();
    const QRect scr = screen ? screen->availableGeometry() : QRect(0,0,1280,800);
    const QSize sz = win.size();
    const int x = scr.x() + (scr.width() - sz.width())/2;
    const int y = scr.y() + (scr.height() - sz.height())/2;
    win.move(x, y);
}

// Create a stable-ish signature for a screen. Used only to detect changes.
static QString screenSignature(QScreen* s)
{
    if (!s) return {};
    const QRect a = s->availableGeometry();
    return QStringLiteral("%1|%2x%3@%4,%5|dpr=%6")
        .arg(s->name())
        .arg(a.width()).arg(a.height())
        .arg(a.x()).arg(a.y())
        .arg(QString::number(s->devicePixelRatio(), 'f', 2));
}

// Reset window geometry for a target screen (same intent as clicking "还原初始状态").
static void resetWindowForScreen(QMainWindow& win, QScreen* screen, Renderer* renderer)
{
    if (!screen) screen = QGuiApplication::primaryScreen();
    const QRect scr = screen ? screen->availableGeometry() : QRect(0,0,1280,800);

    const int targetH = std::max(300, scr.height() * 2 / 5);
    const int targetW = renderer ? renderer->suggestWidthForHeight(targetH) : (targetH/2);

    win.resize(targetW, targetH);
    const int x = scr.x() + (scr.width() - targetW)/2;
    const int y = scr.y() + (scr.height() - targetH)/2;
    win.move(x, y);

    SettingsManager::instance().setWindowGeometry(win.geometry());
    // Persist which display context produced this geometry
    SettingsManager::instance().setWindowGeometryScreen(screenSignature(screen));
}

// Return true if the window rect intersects any screen's available area by at least a few pixels.
static bool isRectVisibleOnAnyScreen(const QRect& r)
{
    if (!r.isValid()) return false;
    constexpr int kMinVisible = 16; // px
    const QList<QScreen*> screens = QGuiApplication::screens();
    for (QScreen* s : screens)
    {
        if (!s) continue;
        const QRect avail = s->availableGeometry();
        const QRect inter = r.intersected(avail);
        if (inter.width() >= kMinVisible && inter.height() >= kMinVisible)
            return true;
    }
    return false;
}

static bool loadAppTranslator(QApplication& app, QTranslator& translator, const QString& languageCode)
{
    app.removeTranslator(&translator);

    // Chinese is the current source language in this project.
    if (languageCode == QStringLiteral("zh_CN")) {
        return true;
    }

    const QString baseName = QStringLiteral("amaigirl_") + languageCode;
    const QString appResI18nDir = QDir(appResourceRootPath()).filePath(QStringLiteral("i18n"));

    bool loaded = translator.load(baseName, appResI18nDir);
    if (!loaded) {
        // fallback to searching Qt resource/system paths if present in future.
        loaded = translator.load(baseName);
    }
    if (loaded) {
        app.installTranslator(&translator);
    }
    return loaded;
}

int main(int argc, char *argv[]) {
#if defined(Q_OS_LINUX)
    const QByteArray qpaEnv = qgetenv("QT_QPA_PLATFORM");
    if (qpaEnv.isEmpty()) {
        qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("wayland;xcb"));
    }
#endif

#if defined(Q_OS_WIN32)
    qputenv("QT_PLUGIN_PATH", QByteArray());
    qputenv("QT_QPA_PLATFORM_PLUGIN_PATH", QByteArray());
#endif

    QSurfaceFormat fmt;
    fmt.setDepthBufferSize(0);
    fmt.setStencilBufferSize(8);
    //fmt.setSamples(4); // MSAA x4: balance quality and memory (was 8)
    QSurfaceFormat::setDefaultFormat(fmt);

    // Work around a macOS/Qt shutdown crash in QApplication::~QApplication by
    // letting the process exit without running this destructor.
    auto* appPtr = new QApplication(argc, argv);
    QApplication& app = *appPtr;
#if defined(Q_OS_LINUX)
    if (!isLinuxWayland()) {
        QMessageBox::warning(
            nullptr,
            QStringLiteral("XiaoMo"),
            QStringLiteral("XiaoMo on Linux currently supports Wayland only.\n"
                           "Current Qt platform plugin: %1\n\n"
                           "The app will continue to run with current backend.")
                .arg(QGuiApplication::platformName()));
    }
#endif
    QCoreApplication::setApplicationName(QStringLiteral("XiaoMo"));
    QApplication::setApplicationDisplayName(QStringLiteral("XiaoMo"));
    QCoreApplication::setOrganizationName(QStringLiteral("IAIAYN"));
    QApplication::setQuitOnLastWindowClosed(false);

    installStartupLogger();
    qInfo() << "XiaoMo start"
            << "appDir=" << QCoreApplication::applicationDirPath()
            << "resRoot=" << appResourceRootPath()
            << "modelsRoot=" << SettingsManager::instance().modelsRoot();

    try {
        SettingsManager::instance().load();
        SettingsManager::instance().bootstrap(QCoreApplication::applicationDirPath());
        Theme::installApplicationStyle(app, SettingsManager::instance().theme());
    } catch (const std::exception& e) {
        QMessageBox::critical(nullptr, QObject::tr("XiaoMo"), formatExceptionMessage(e));
        std::_Exit(1);
    } catch (...) {
        QMessageBox::critical(nullptr, QObject::tr("XiaoMo"), QObject::tr("启动失败：未知错误"));
        std::_Exit(1);
    }

    const QIcon appIcon(appResourcePath(QStringLiteral("icons/app-icon.png")));
    if (!appIcon.isNull()) {
        app.setWindowIcon(appIcon);
    }

    QTranslator appTranslator;
    loadAppTranslator(app, appTranslator, SettingsManager::instance().currentLanguage());

    QMainWindow win;
    win.setWindowFlag(Qt::FramelessWindowHint, true);
    win.setWindowFlag(Qt::WindowStaysOnTopHint, SettingsManager::instance().windowAlwaysOnTop());
    win.setWindowFlag(Qt::NoDropShadowWindowHint, true);
#if defined(Q_OS_WIN32)
    win.setWindowFlag(Qt::Tool, true);
#endif
    win.setAttribute(Qt::WA_TranslucentBackground, SettingsManager::instance().windowTransparentBackground());
    win.setAutoFillBackground(!SettingsManager::instance().windowTransparentBackground());

    auto* renderer = new Renderer;
    // 始终指向当前有效的 Renderer；下方所有连接通过此指针间接调用，避免重建后悬垂指针
    Renderer** currentRenderer = &renderer;
    // 在窗口高度确定后，依据模型尺寸建议宽度
    QObject::connect(renderer, &Renderer::requestFitWidthForHeight, [&win](int h, int suggestedW){
        if (h <= 0 || suggestedW <= 0) return;
        // 保持高度，调整宽度
        QSize cur = win.size();
        if (std::abs(cur.width() - suggestedW) >= 2) {
            win.resize(suggestedW, h);
        }
    });
    renderer->setEnableBlink(SettingsManager::instance().enableBlink());
    renderer->setEnableBreath(SettingsManager::instance().enableBreath());
    renderer->setEnableGaze(SettingsManager::instance().enableGaze()); // 默认已在 SettingsManager 中改为 false
    renderer->setEnablePhysics(SettingsManager::instance().enablePhysics());
    // 注入 per-model 的去水印表达式
    renderer->setWatermarkExpression(SettingsManager::instance().watermarkExpPath());
    // 初始渲染选项：贴图上限与 MSAA
    renderer->setTextureCap(SettingsManager::instance().textureMaxDim());
    renderer->setMsaaSamples(SettingsManager::instance().msaaSamples());

    win.setCentralWidget(renderer);

    // Restore geometry if present
    if (SettingsManager::instance().hasWindowGeometry()) {
        QRect g = SettingsManager::instance().windowGeometry();
        int w_ = std::max(150, g.width());
        int h_ = std::max(300, g.height());
        win.resize(w_, h_);
        win.move(g.x(), g.y()); // allow negative and partially offscreen

        // If the last saved geometry was for a different display setup, reset like "还原初始状态".
        const QString savedSig = SettingsManager::instance().windowGeometryScreen();
        const QString nowSig = screenSignature(resolvePreferredScreen());
        if (!savedSig.isEmpty() && !nowSig.isEmpty() && savedSig != nowSig)
        {
            resetWindowForScreen(win, resolvePreferredScreen(), renderer);
        }
    } else {
        // Initial size: height = 2/5 of current screen height, width = height / 2
        QScreen* screen = QGuiApplication::primaryScreen();
        int screenH = screen ? screen->geometry().height() : 900; // fallback
        int initH = screenH * 2 / 5;
        int initW = initH / 2;
        win.setMinimumSize(150, 300);
        win.resize(initW, initH);

        SettingsManager::instance().setWindowGeometry(win.geometry());
        SettingsManager::instance().setWindowGeometryScreen(screenSignature(screen));
    }

    // Only relocate the window when it would be invisible due to screen changes.
    // If the display setup hasn't changed, we keep the persisted position as-is.
    if (!isRectVisibleOnAnyScreen(win.geometry()))
    {
        resetWindowForScreen(win, resolvePreferredScreen(), renderer);
    }

    win.show();
#if defined(Q_OS_LINUX)
    cacheWindowGeometry(&win);
    if (auto* wh = win.windowHandle()) {
        wh->setFlag(Qt::WindowStaysOnTopHint, SettingsManager::instance().windowAlwaysOnTop());
    }
#endif

    // Actions used by shortcuts and context menu
    auto toggleAction = new QAction(QStringLiteral("隐藏"), &win);
    toggleAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_H)); // Ctrl+H
    auto chatAction = new QAction(QStringLiteral("聊天"), &win);
    chatAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_T)); // Ctrl+T
    auto settingsAction = new QAction(QStringLiteral("设置"), &win);
    settingsAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_S)); // Ctrl+S
    auto quitAction = new QAction(QStringLiteral("退出"), &win);

    // Tray icon
    QSystemTrayIcon* tray = nullptr;
    auto createTray = [&tray, &app, &win, toggleAction, chatAction, settingsAction, quitAction]() {
        if (tray) return;
        tray = new QSystemTrayIcon(&app);
#if defined(Q_OS_MACOS)
        const QString resRoot = appResourceRootPath();
        tray->setIcon(resolveMacTrayIcon(resRoot, app.windowIcon()));
#else
        QIcon trayIcon = app.windowIcon();
        if (trayIcon.isNull()) {
            const QString resRoot = appResourceRootPath();
            QIcon icon(QDir(resRoot).filePath(QStringLiteral("icons/app-icon.png")));
            if (!icon.isNull()) trayIcon = icon;
        }
        tray->setIcon(trayIcon);
#endif
        tray->setToolTip(QStringLiteral("XiaoMo"));
        auto* menu = new QMenu();
        menu->addAction(toggleAction);
        menu->addAction(chatAction);
        menu->addAction(settingsAction);
        menu->addSeparator();
        menu->addAction(quitAction);
        tray->setContextMenu(menu);
        tray->show();
        toggleAction->setText(win.isVisible() ? QObject::tr("隐藏") : QObject::tr("显示"));
    };
    auto destroyTray = [&]() {
        if (!tray) return;
        tray->hide();
        tray->deleteLater();
        tray = nullptr;
        toggleAction->setText(win.isVisible() ? QObject::tr("隐藏") : QObject::tr("显示"));
    };

    createTray();

    // Settings window (default hidden)
    auto settingsWnd = new SettingsWindow(&win);

    // Chat window + controller
    auto chatWnd = new ChatWindow(&win);
    chatWnd->hide();
    auto chatCtl = new ChatController(&app);
    chatCtl->setChatWindow(chatWnd);
    chatCtl->setRenderer(renderer);
    chatCtl->applyPreferredAudioOutput();

    auto voiceSvc = new OfflineVoiceService(&app);
    voiceSvc->reloadFromSettings();
    QObject::connect(settingsWnd, &SettingsWindow::offlineVoiceSettingsChanged, voiceSvc, [voiceSvc]{
        voiceSvc->reloadFromSettings();
    });

    auto bubbleWnd = new PetSpeechBubbleWidget(nullptr);
    bubbleWnd->setBubbleStyle(SettingsManager::instance().chatBubbleStyle());
    bubbleWnd->hide();

    auto bubbleHideTimer = new QTimer(bubbleWnd);
    bubbleHideTimer->setSingleShot(true);
    QObject::connect(bubbleHideTimer, &QTimer::timeout, bubbleWnd, [bubbleWnd]{ bubbleWnd->hide(); });

    auto condenseForBubble = [](QString s) -> QString {
        s.replace('\n', ' ');
        s.replace('\r', ' ');
        s = s.simplified();
        if (s.size() > 140)
            s = s.left(140).trimmed() + QStringLiteral("…");
        return s;
    };

    auto placeBubble = [&win, bubbleWnd]{
        if (!bubbleWnd->isVisible()) return;
        if (!win.isVisible()) { bubbleWnd->hide(); return; }

        bubbleWnd->adjustSize();
        const QSize sz = bubbleWnd->sizeHint().expandedTo(QSize(60, 40));
        bubbleWnd->resize(sz);

        const QRect wg = win.frameGeometry();
        QScreen* s = QGuiApplication::screenAt(wg.center());
        if (!s) s = QGuiApplication::primaryScreen();
        const QRect scr = s ? s->availableGeometry() : QRect(0,0,1280,800);

        const int margin = 12;
        int x = wg.right() + margin;
        int y = wg.top() + 24;
        if (x + bubbleWnd->width() > scr.right() - margin)
            x = wg.left() - margin - bubbleWnd->width();
        bubbleWnd->setTailOnLeft(x >= wg.right());
        x = std::clamp(x, scr.left() + margin, scr.right() - margin - bubbleWnd->width());
        y = std::clamp(y, scr.top() + margin, scr.bottom() - margin - bubbleWnd->height());
        bubbleWnd->move(x, y);
        bubbleWnd->raise();
    };

    QObject::connect(chatCtl, &ChatController::assistantBubbleTextChanged, &app, [bubbleWnd, bubbleHideTimer, placeBubble, condenseForBubble](const QString& text, bool isFinal){
        const QString t = condenseForBubble(text);
        if (t.isEmpty()) {
            bubbleHideTimer->stop();
            bubbleWnd->hide();
            return;
        }
        bubbleWnd->setBubbleStyle(SettingsManager::instance().chatBubbleStyle());
        bubbleWnd->setBubbleText(t);
        bubbleWnd->show();
        placeBubble();
        if (isFinal) bubbleHideTimer->start(6000);
        else bubbleHideTimer->stop();
    });

    bool voiceAwaitingAssistant = false;
    QObject::connect(voiceSvc, &OfflineVoiceService::wakeWordDetected, &app, [bubbleWnd, bubbleHideTimer, placeBubble]{
        bubbleHideTimer->stop();
        bubbleWnd->setBubbleStyle(SettingsManager::instance().chatBubbleStyle());
        bubbleWnd->setBubbleText(QObject::tr("嗯？"));
        bubbleWnd->show();
        placeBubble();
        bubbleHideTimer->start(1200);
    });
    QObject::connect(voiceSvc, &OfflineVoiceService::sttPartialText, &app, [bubbleWnd, bubbleHideTimer, placeBubble](const QString& text){
        const QString t = text.trimmed();
        if (t.isEmpty()) return;
        bubbleHideTimer->stop();
        bubbleWnd->setBubbleStyle(SettingsManager::instance().chatBubbleStyle());
        bubbleWnd->setBubbleText(QObject::tr("…%1").arg(t));
        bubbleWnd->show();
        placeBubble();
    });
    QObject::connect(voiceSvc, &OfflineVoiceService::sttFinalText, &app, [chatCtl, bubbleHideTimer, bubbleWnd, placeBubble, &voiceAwaitingAssistant](const QString& text){
        const QString t = text.trimmed();
        if (t.isEmpty()) return;
        voiceAwaitingAssistant = true;
        bubbleHideTimer->stop();
        bubbleWnd->setBubbleStyle(SettingsManager::instance().chatBubbleStyle());
        bubbleWnd->setBubbleText(QObject::tr("你：%1").arg(t));
        bubbleWnd->show();
        placeBubble();
        chatCtl->triggerLocalPrompt(t, QString(), QString());
    });

    QObject::connect(renderer, &Renderer::requestChangeBubbleStyle, &app, [bubbleWnd](const QString& styleId){
        SettingsManager::instance().setChatBubbleStyle(styleId);
        if (bubbleWnd) {
            bubbleWnd->setBubbleStyle(styleId);
            if (bubbleWnd->isVisible()) bubbleWnd->update();
        }
    });
    QObject::connect(chatCtl, &ChatController::assistantBubbleTextChanged, &app, [voiceSvc, &voiceAwaitingAssistant](const QString& text, bool isFinal){
        if (!isFinal) return;
        if (!voiceAwaitingAssistant) return;
        voiceAwaitingAssistant = false;
        voiceSvc->speakText(text);
    });

    QHash<QString, QDateTime> reminderLastFired;
    auto runReminderTask = [chatCtl, currentRenderer](const QJsonObject& o){
        const QString motion = o.value(QStringLiteral("motionGroup")).toString().trimmed();
        const QString expr = o.value(QStringLiteral("expressionName")).toString().trimmed();
        const QString text = o.value(QStringLiteral("text")).toString();
        const bool writeToHistory = o.value(QStringLiteral("writeToHistory")).toBool(true);
        const QString kind = o.value(QStringLiteral("kind")).toString(QStringLiteral("assistant"));
        if (kind == QStringLiteral("ask")) {
            chatCtl->triggerLocalPrompt(text, motion, expr);
        } else if (!text.trimmed().isEmpty()) {
            chatCtl->postLocalAssistantMessage(text, motion, expr, writeToHistory);
        } else if (*currentRenderer) {
            if (!expr.isEmpty()) (*currentRenderer)->setExpressionName(expr);
            if (!motion.isEmpty()) (*currentRenderer)->setMotionGroup(motion);
        }
    };

    auto* reminderTimer = new QTimer(&app);
    reminderTimer->setInterval(5000);
    QObject::connect(reminderTimer, &QTimer::timeout, &app, [&reminderLastFired, runReminderTask]{
        const QDateTime now = QDateTime::currentDateTime();
        const QDate today = now.date();
        const QJsonArray tasks = SettingsManager::instance().reminderTasks();
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (!tasks.at(i).isObject()) continue;
            const QJsonObject o = tasks.at(i).toObject();
            if (!o.value(QStringLiteral("enabled")).toBool(true)) continue;

            const QString rawId = o.value(QStringLiteral("id")).toString();
            const QString id = rawId.isEmpty() ? QStringLiteral("idx_%1").arg(i) : rawId;

            const QString mode = o.value(QStringLiteral("mode")).toString(QStringLiteral("daily"));
            if (mode == QStringLiteral("interval"))
            {
                const int intervalMinutes = std::max(1, o.value(QStringLiteral("intervalMinutes")).toInt(60));
                const QDateTime last = reminderLastFired.value(id);
                if (!last.isValid()) {
                    reminderLastFired.insert(id, now);
                    continue;
                }
                if (last.secsTo(now) < intervalMinutes * 60)
                    continue;
                reminderLastFired.insert(id, now);
                runReminderTask(o);
            }
            else
            {
                const QString timeStr = o.value(QStringLiteral("time")).toString(QStringLiteral("09:00"));
                const QTime target = QTime::fromString(timeStr, QStringLiteral("HH:mm"));
                if (!target.isValid()) continue;
                if (now.time().hour() != target.hour() || now.time().minute() != target.minute())
                    continue;

                const QDateTime last = reminderLastFired.value(id);
                if (last.isValid() && last.date() == today)
                    continue;

                reminderLastFired.insert(id, now);
                runReminderTask(o);
            }
        }
    });
    reminderTimer->start();

    class BubbleFollowFilter : public QObject {
    public:
        std::function<void()> onUpdate;
        QWidget* bubble{nullptr};
        bool eventFilter(QObject* obj, QEvent* ev) override {
            Q_UNUSED(obj);
            if (!bubble) return QObject::eventFilter(obj, ev);
            if (ev->type() == QEvent::Move || ev->type() == QEvent::Resize || ev->type() == QEvent::Show || ev->type() == QEvent::Hide) {
                if (onUpdate) onUpdate();
                if (ev->type() == QEvent::Hide) bubble->hide();
            }
            return QObject::eventFilter(obj, ev);
        }
    };
    static BubbleFollowFilter s_bubbleFollow;
    s_bubbleFollow.bubble = bubbleWnd;
    s_bubbleFollow.onUpdate = placeBubble;
    win.installEventFilter(&s_bubbleFollow);

    QObject::connect(settingsWnd, &SettingsWindow::preferredAudioOutputChanged, chatCtl, [chatCtl](const QString&){
        chatCtl->applyPreferredAudioOutput();
    });

    QObject::connect(settingsWnd, &SettingsWindow::preferredScreenChanged, &app, [&win, currentRenderer](const QString&){
        // Switching display should behave like "还原初始状态".
        resetWindowForScreen(win, resolvePreferredScreen(), *currentRenderer);
    });

    QObject::connect(settingsWnd, &SettingsWindow::windowAlwaysOnTopChanged, &app, [&win, bubbleWnd](bool on){
        const bool wasVisible = win.isVisible();
        win.setWindowFlag(Qt::WindowStaysOnTopHint, on);
        if (wasVisible) {
            win.show();
            win.raise();
            win.activateWindow();
        }
#if defined(Q_OS_LINUX)
        if (auto* wh = win.windowHandle()) {
            wh->setFlag(Qt::WindowStaysOnTopHint, on);
        }
#endif
        if (bubbleWnd) {
            bubbleWnd->setWindowFlag(Qt::WindowStaysOnTopHint, on);
            if (bubbleWnd->isVisible()) {
                bubbleWnd->show();
                bubbleWnd->raise();
            }
        }
    });

    QObject::connect(settingsWnd, &SettingsWindow::windowTransparentBackgroundChanged, &app, [&win, currentRenderer](bool on){
        win.setAttribute(Qt::WA_TranslucentBackground, on);
        win.setAutoFillBackground(!on);
        if (*currentRenderer) (*currentRenderer)->update();
        win.update();
    });

    QObject::connect(settingsWnd, &SettingsWindow::windowMousePassthroughChanged, &app, [&win, currentRenderer](bool on){
#if defined(Q_OS_MACOS) || defined(Q_OS_LINUX) || defined(Q_OS_WIN32)
        if (auto* wh = win.windowHandle()) {
            wh->setFlag(Qt::WindowTransparentForInput, on);
        }
#endif
        if (*currentRenderer) {
            (*currentRenderer)->setAttribute(Qt::WA_TransparentForMouseEvents, on);
            (*currentRenderer)->update();
        }
    });

    auto toggleChat = [chatWnd]{
        static bool s_firstShow = true;
        if (chatWnd->isVisible()) {
#if defined(Q_OS_LINUX)
            cacheWindowGeometry(chatWnd);
#endif
            chatWnd->hide();
        } else {
            if (s_firstShow) {
                // 首次显示时强制居中。之后不再改位置，避免每次 show() 被平台重新“自动摆放”导致漂移。
                centerOnCurrentScreen(chatWnd);
                s_firstShow = false;
            }
#if defined(Q_OS_LINUX)
            restoreWindowGeometry(chatWnd);
#endif
            bringToFrontOnce(chatWnd);
#if defined(Q_OS_LINUX)
            QTimer::singleShot(0, chatWnd, [chatWnd]{ restoreWindowGeometry(chatWnd); });
            cacheWindowGeometry(chatWnd);
#endif
        }
    };

    // Global shortcut for toggling chat window visibility (works even when main window is hidden)
    auto* chatShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_T), &win);
    chatShortcut->setContext(Qt::ApplicationShortcut);
    QObject::connect(chatShortcut, &QShortcut::activated, &app, [toggleChat]{ toggleChat(); });

    QObject::connect(chatAction, &QAction::triggered, &app, [toggleChat]{ toggleChat(); });

    QObject::connect(settingsWnd, &SettingsWindow::requestOpenChat, &app, [toggleChat]{ toggleChat(); });

    // --- Add shortcuts like "聊天" ---
    auto* toggleShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_H), &win);
    toggleShortcut->setContext(Qt::ApplicationShortcut);
    QObject::connect(toggleShortcut, &QShortcut::activated, &app, [winPtr=&win, toggleAction]{
        if (winPtr->isVisible()) {
#if defined(Q_OS_LINUX)
            cacheWindowGeometry(winPtr);
#endif
            winPtr->hide();
            toggleAction->setText(QStringLiteral("显示"));
        } else {
#if defined(Q_OS_LINUX)
            restoreWindowGeometry(winPtr);
#endif
            winPtr->show();
            winPtr->raise();
            winPtr->activateWindow();
#if defined(Q_OS_LINUX)
            if (auto* wh = winPtr->windowHandle()) {
                wh->setFlag(Qt::WindowStaysOnTopHint, SettingsManager::instance().windowAlwaysOnTop());
            }
            QTimer::singleShot(0, winPtr, [winPtr]{ restoreWindowGeometry(winPtr); });
            cacheWindowGeometry(winPtr);
#endif
            toggleAction->setText(QStringLiteral("隐藏"));
        }
    });

    auto* settingsShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_S), &win);
    settingsShortcut->setContext(Qt::ApplicationShortcut);
    QObject::connect(settingsShortcut, &QShortcut::activated, &app, [settingsWnd]{
        if (settingsWnd->isVisible()) {
#if defined(Q_OS_LINUX)
            cacheWindowGeometry(settingsWnd);
#endif
            settingsWnd->hide();
        } else {
#if defined(Q_OS_LINUX)
            restoreWindowGeometry(settingsWnd);
#endif
            bringToFrontOnce(settingsWnd);
#if defined(Q_OS_LINUX)
            QTimer::singleShot(0, settingsWnd, [settingsWnd]{ restoreWindowGeometry(settingsWnd); });
            cacheWindowGeometry(settingsWnd);
#endif
        }
    });

    QObject::connect(settingsWnd, &SettingsWindow::aiSettingsChanged, chatCtl, [chatCtl]{
        Q_UNUSED(chatCtl);
        // config is pulled on send; no-op placeholder.
    });

    QObject::connect(settingsWnd, &SettingsWindow::languageChanged, &app, [&app, &appTranslator](const QString& code){
        const bool loaded = loadAppTranslator(app, appTranslator, code);
        if (!loaded && code != QStringLiteral("zh_CN")) {
            QMessageBox::information(nullptr,
                                     QObject::tr("提示"),
                                     QObject::tr("未找到对应语言包，已回退为内置中文文案。"));
        }
    });

    QObject::connect(settingsWnd, &SettingsWindow::themeChanged, &app, [&app](const QString& themeId){
        Theme::applyTheme(app, themeId);
    });

    QObject::connect(settingsWnd, &SettingsWindow::requestLoadModel, [currentRenderer, chatCtl](const QString& json){
        if (*currentRenderer) {
            try {
                (*currentRenderer)->load(json);
            } catch (const std::exception& e) {
                showModelLoadError(json, formatExceptionMessage(e));
            } catch (...) {
                showModelLoadError(json, QObject::tr("未知错误"));
            }
        }

        // derive model folder + dir from current settings
        const QString folder = SettingsManager::instance().selectedModelFolder();
        const QString modelDir = QDir(SettingsManager::instance().modelsRoot()).filePath(folder);
        chatCtl->onModelChanged(folder, modelDir);
        chatCtl->setRenderer(*currentRenderer);
    });
    QObject::connect(settingsWnd, &SettingsWindow::llmStyleChanged, chatCtl, [chatCtl](const QString& s){
        chatCtl->setLlmStyle(s);
    });
    QObject::connect(settingsWnd, &SettingsWindow::llmModelSizeChanged, &app, [currentRenderer, chatCtl](const QString&){
        const QString folder = SettingsManager::instance().selectedModelFolder();
        const QString modelDir = QDir(SettingsManager::instance().modelsRoot()).filePath(folder);
        chatCtl->onModelChanged(folder, modelDir);
        chatCtl->setRenderer(*currentRenderer);
    });

    QObject::connect(settingsWnd, &SettingsWindow::toggleBlink, &win, [currentRenderer](bool on){ if (*currentRenderer) (*currentRenderer)->setEnableBlink(on); });
    QObject::connect(settingsWnd, &SettingsWindow::toggleBreath, &win, [currentRenderer](bool on){ if (*currentRenderer) (*currentRenderer)->setEnableBreath(on); });
    QObject::connect(settingsWnd, &SettingsWindow::toggleGaze, &win, [currentRenderer](bool on){ if (*currentRenderer) (*currentRenderer)->setEnableGaze(on); });
    QObject::connect(settingsWnd, &SettingsWindow::togglePhysics, &win, [currentRenderer](bool on){ if (*currentRenderer) (*currentRenderer)->setEnablePhysics(on); });
    // watermark change
    QObject::connect(settingsWnd, &SettingsWindow::watermarkChanged, &win, [currentRenderer](const QString& p){ if (*currentRenderer) (*currentRenderer)->setWatermarkExpression(p); });
    QObject::connect(settingsWnd, &SettingsWindow::textureCapChanged, &win, [currentRenderer](int d){ if (*currentRenderer) (*currentRenderer)->setTextureCap(d); });

    // Disable MSAA rebuild logic for now (user rolled back earlier); keep signal connected harmlessly.
    QObject::connect(settingsWnd, &SettingsWindow::msaaChanged, &app, [currentRenderer](int samples){ if (*currentRenderer) (*currentRenderer)->setMsaaSamples(samples); Q_UNUSED(samples); });

    QObject::connect(settingsAction, &QAction::triggered, [settingsWnd]{
        if (settingsWnd->isVisible()) {
#if defined(Q_OS_LINUX)
            cacheWindowGeometry(settingsWnd);
#endif
            settingsWnd->hide();
        } else {
#if defined(Q_OS_LINUX)
            restoreWindowGeometry(settingsWnd);
#endif
            bringToFrontOnce(settingsWnd);
#if defined(Q_OS_LINUX)
            QTimer::singleShot(0, settingsWnd, [settingsWnd]{ restoreWindowGeometry(settingsWnd); });
            cacheWindowGeometry(settingsWnd);
#endif
        }
    });

    QObject::connect(toggleAction, &QAction::triggered, [winPtr=&win, toggleAction]{
        if (winPtr->isVisible()) {
#if defined(Q_OS_LINUX)
            cacheWindowGeometry(winPtr);
#endif
            winPtr->hide();
        } else {
#if defined(Q_OS_LINUX)
            restoreWindowGeometry(winPtr);
#endif
            winPtr->show();
            winPtr->raise();
            winPtr->activateWindow();
#if defined(Q_OS_LINUX)
            if (auto* wh = winPtr->windowHandle()) {
                wh->setFlag(Qt::WindowStaysOnTopHint, SettingsManager::instance().windowAlwaysOnTop());
            }
            QTimer::singleShot(0, winPtr, [winPtr]{ restoreWindowGeometry(winPtr); });
            cacheWindowGeometry(winPtr);
#endif
        }
        toggleAction->setText(winPtr->isVisible() ? QObject::tr("隐藏") : QObject::tr("显示"));
    });

    QObject::connect(quitAction, &QAction::triggered, &app, &QCoreApplication::quit);

    // 右键模型触发的菜单信号：不再依赖任务栏
    QObject::connect(renderer, &Renderer::requestToggleMain, toggleAction, &QAction::trigger);
    QObject::connect(renderer, &Renderer::requestOpenSettings, settingsAction, &QAction::trigger);
    QObject::connect(renderer, &Renderer::requestOpenChat, chatAction, &QAction::trigger);
    QObject::connect(renderer, &Renderer::requestQuit, quitAction, &QAction::trigger);
    QObject::connect(renderer, &Renderer::requestChangeStyle, chatCtl, [chatCtl](const QString& s){
        chatCtl->setLlmStyle(s);
    });
    QObject::connect(renderer, &Renderer::requestSwitchModel, &app, [currentRenderer, chatCtl](const QString& folder){
        auto entries = SettingsManager::instance().scanModels();
        QString json;
        for (const auto& e : entries) if (e.folderName == folder) { json = e.jsonPath; break; }
        if (json.isEmpty()) return;
        SettingsManager::instance().setSelectedModelFolder(folder);
        if (*currentRenderer) {
            try {
                (*currentRenderer)->load(json);
            } catch (const std::exception& e) {
                showModelLoadError(json, formatExceptionMessage(e));
            } catch (...) {
                showModelLoadError(json, QObject::tr("未知错误"));
            }
        }
        const QString modelDir = QDir(SettingsManager::instance().modelsRoot()).filePath(folder);
        chatCtl->onModelChanged(folder, modelDir);
        chatCtl->setRenderer(*currentRenderer);
    });

    // Event filter to persist geometry on move/resize
    class WinEventFilter : public QObject {
        bool eventFilter(QObject* obj, QEvent* ev) override {
            auto* w = qobject_cast<QMainWindow*>(obj);
            if (!w) return QObject::eventFilter(obj, ev);
            if (ev->type() == QEvent::Move || ev->type() == QEvent::Resize) {
                SettingsManager::instance().setWindowGeometry(w->geometry());
                // Update geometry display signature too (used for display-change detection).
                QScreen* s = QGuiApplication::screenAt(w->geometry().center());
                if (!s) s = resolvePreferredScreen();
                SettingsManager::instance().setWindowGeometryScreen(screenSignature(s));
            }
            return QObject::eventFilter(obj, ev);
        }
    };
    static WinEventFilter s_filter;
    win.installEventFilter(&s_filter);

    // Wire reset request
    QObject::connect(settingsWnd, &SettingsWindow::requestResetWindow, [winPtr=&win, currentRenderer]{
         QScreen* screen = QGuiApplication::screenAt(winPtr->geometry().center());
         if (!screen) screen = QGuiApplication::primaryScreen();
         resetWindowForScreen(*winPtr, screen, *currentRenderer);
     });

    // If no stored geometry, center to defaults
    if (!SettingsManager::instance().hasWindowGeometry()) {
        centerAndSize(win);
        SettingsManager::instance().setWindowGeometry(win.geometry());
    }

    // Choose initial model from settings (selected folder), fallback to any scanned model
    QString initialJson;
    auto entries = SettingsManager::instance().scanModels();
    QString folder = SettingsManager::instance().selectedModelFolder();
    for (const auto& e : entries) if (e.folderName == folder) { initialJson = e.jsonPath; break; }
    if (initialJson.isEmpty() && !entries.isEmpty()) {
        // Prefer Hiyori if available
        for (const auto& e : entries) {
            if (e.folderName.compare("Hiyori", Qt::CaseInsensitive) == 0) { initialJson = e.jsonPath; break; }
        }
        if (initialJson.isEmpty()) initialJson = entries.front().jsonPath;
    }

    if (!initialJson.isEmpty()) {
        const QString modelPath = initialJson;
        QTimer::singleShot(0, [currentRenderer, modelPath, chatCtl]{
            if (*currentRenderer) {
                try {
                    (*currentRenderer)->load(modelPath);
                } catch (const std::exception& e) {
                    showModelLoadError(modelPath, formatExceptionMessage(e));
                } catch (...) {
                    showModelLoadError(modelPath, QObject::tr("未知错误"));
                }
            }
            const QString folder = SettingsManager::instance().selectedModelFolder();
            const QString modelDir = QDir(SettingsManager::instance().modelsRoot()).filePath(folder);
            chatCtl->onModelChanged(folder, modelDir);
        });
    }

    const int exitCode = app.exec();
    Q_UNUSED(appPtr);
    std::_Exit(exitCode);
}
