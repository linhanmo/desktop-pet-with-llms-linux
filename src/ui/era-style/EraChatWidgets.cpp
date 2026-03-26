#include "ui/era-style/EraChatWidgets.hpp"

#include "ui/era-style/EraStyleColor.hpp"
#include "ui/era-style/EraStyleHelper.hpp"

#include <QAbstractTextDocumentLayout>
#include <QContextMenuEvent>
#include <QCoreApplication>
#include <QEnterEvent>
#include <QGuiApplication>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QResizeEvent>
#include <QStyleHints>
#include <QTextDocument>
#include <QTextOption>
#include <QTimer>

#include <cmath>

namespace {
QString toRgba(const QColor& color)
{
    return QStringLiteral("rgba(%1, %2, %3, %4)")
        .arg(color.red())
        .arg(color.green())
        .arg(color.blue())
        .arg(QString::number(color.alphaF(), 'f', 3));
}

QString normalizeBubbleStyleId(const QString& styleId)
{
    const QString s = styleId.trimmed().toLower();
    if (s == QStringLiteral("imessage") || s == QStringLiteral("i-message") || s == QStringLiteral("ios"))
        return QStringLiteral("iMessage");
    if (s == QStringLiteral("outline") || s == QStringLiteral("flowbite"))
        return QStringLiteral("Outline");
    if (s == QStringLiteral("round"))
        return QStringLiteral("Round");
    if (s == QStringLiteral("era") || s == QStringLiteral("material") || s == QStringLiteral("default"))
        return QStringLiteral("Era");
    return QStringLiteral("Era");
}

QColor imessageUserFillTop(const EraStyleColor::ThemePalette& pal)
{
    if (EraStyleColor::isDark())
        return pal.accentHover;
    return QColor(0, 122, 255);
}

QColor imessageUserFillBottom(const EraStyleColor::ThemePalette& pal)
{
    if (EraStyleColor::isDark())
        return pal.accentPressed;
    return QColor(11, 147, 246);
}

QColor imessageAssistantFill(const EraStyleColor::ThemePalette& pal)
{
    if (EraStyleColor::isDark())
        return QColor(44, 44, 46);
    return QColor(229, 229, 234);
}

QColor outlineFill(bool isUserBubble, const EraStyleColor::ThemePalette& pal)
{
    if (isUserBubble)
    {
        QColor c = pal.accent;
        c.setAlphaF(EraStyleColor::isDark() ? 0.22 : 0.12);
        return c;
    }
    QColor c = EraStyleColor::isDark() ? pal.panelRaised : pal.panelBackground;
    c.setAlphaF(EraStyleColor::isDark() ? 0.78 : 0.92);
    return c;
}

QColor outlineBorder(bool isUserBubble, const EraStyleColor::ThemePalette& pal)
{
    if (isUserBubble)
        return pal.accent;
    return EraStyleColor::isDark() ? pal.borderPrimary : pal.borderSecondary;
}

QColor roundFill(bool isUserBubble, const EraStyleColor::ThemePalette& pal)
{
    if (isUserBubble)
        return pal.accent;
    return EraStyleColor::isDark() ? pal.panelRaised : pal.panelBackground;
}

QColor roundBorder(bool isUserBubble, const EraStyleColor::ThemePalette& pal)
{
    if (isUserBubble)
        return pal.accentPressed;
    return EraStyleColor::isDark() ? pal.borderPrimary : pal.borderSecondary;
}

bool isThemeEvent(QEvent::Type type)
{
    return type == QEvent::ApplicationPaletteChange
        || type == QEvent::PaletteChange
        || type == QEvent::ThemeChange
        || type == QEvent::StyleChange;
}

QPixmap tintedIconPixmap(const QIcon& icon, const QSize& logicalSize, const QColor& tint, qreal devicePixelRatio)
{
    if (icon.isNull())
        return {};

    const QSize deviceSize(
        qMax(1, qRound(logicalSize.width() * devicePixelRatio)),
        qMax(1, qRound(logicalSize.height() * devicePixelRatio))
    );

    QPixmap base = icon.pixmap(deviceSize);
    if (base.isNull())
        return {};

    base.setDevicePixelRatio(devicePixelRatio);
    QImage image = base.toImage().convertToFormat(QImage::Format_ARGB32_Premultiplied);

    QPainter painter(&image);
    painter.setCompositionMode(QPainter::CompositionMode_SourceIn);
    painter.fillRect(image.rect(), tint);

    return QPixmap::fromImage(image);
}
}  // namespace

EraChatComposerEdit::EraChatComposerEdit(QWidget* parent)
    : QTextEdit(parent)
{
    init();
}

int EraChatComposerEdit::preferredHeight(int minLines, int maxLines) const
{
    const QFontMetrics fm(font());
    const int lineHeight = fm.lineSpacing();
    const int minHeight = lineHeight * minLines;
    const int maxHeight = lineHeight * maxLines;
    const int docHeight = qMax(lineHeight, documentHeight());
    return qBound(minHeight, docHeight, maxHeight);
}

int EraChatComposerEdit::documentHeight() const
{
    const QFontMetrics fm(font());
    return qMax(
        fm.lineSpacing(),
        int(std::ceil(document()->documentLayout()->documentSize().height()))
    );
}

void EraChatComposerEdit::keyPressEvent(QKeyEvent* event)
{
    if ((event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
        && !(event->modifiers() & Qt::ShiftModifier))
    {
        event->accept();
        emit sendRequested();
        return;
    }

    QTextEdit::keyPressEvent(event);
}

void EraChatComposerEdit::contextMenuEvent(QContextMenuEvent* event)
{
    QMenu* menu = createStandardContextMenu();
    if (!menu)
        return;
    menu->setAttribute(Qt::WA_TranslucentBackground);
    for (QAction* a : menu->actions())
        a->setIcon(QIcon());
    menu->exec(event->globalPos());
    delete menu;
}

void EraChatComposerEdit::resizeEvent(QResizeEvent* event)
{
    QTextEdit::resizeEvent(event);
    emit metricsChanged();
}

void EraChatComposerEdit::changeEvent(QEvent* event)
{
    QTextEdit::changeEvent(event);
    if (!event)
        return;

    if (event->type() == QEvent::EnabledChange || isThemeEvent(event->type()))
    {
        QTimer::singleShot(0, this, [this] {
            refreshAppearance();
            emit metricsChanged();
        });
    }
}

void EraChatComposerEdit::init()
{
    setAttribute(Qt::WA_MacShowFocusRect, false);
    setFrameStyle(QFrame::NoFrame);
    setAcceptRichText(false);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setContentsMargins(0, 0, 0, 0);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    document()->setDocumentMargin(0.0);
    setWordWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
    setLineWrapMode(QTextEdit::WidgetWidth);
    setStyleSheet(QStringLiteral(
        "QTextEdit{background:transparent; border:none; padding:0px; margin:0px;}"
    ));
    EraStyle::installHoverScrollBars(this, true, false);

    if (auto* layout = document()->documentLayout())
    {
        connect(layout, &QAbstractTextDocumentLayout::documentSizeChanged, this, [this](const QSizeF&) {
            emit metricsChanged();
        });
    }

    if (auto* hints = QGuiApplication::styleHints())
    {
        connect(hints, &QStyleHints::colorSchemeChanged, this, [this](Qt::ColorScheme) {
            QTimer::singleShot(0, this, [this] {
                refreshAppearance();
                emit metricsChanged();
            });
        });
    }

    refreshAppearance();
}

void EraChatComposerEdit::refreshAppearance()
{
    if (QCoreApplication::closingDown() || m_updatingColors)
        return;

    m_updatingColors = true;
    const EraStyleColor::ThemePalette& pal = EraStyleColor::themePalette();

    QPalette editPalette = palette();
    editPalette.setColor(QPalette::Text, isEnabled() ? pal.textPrimary : pal.textDisabled);
    editPalette.setColor(QPalette::PlaceholderText, isEnabled() ? pal.textMuted : pal.textDisabled);
    editPalette.setColor(QPalette::Highlight, pal.selectionBackground);
    editPalette.setColor(QPalette::HighlightedText, pal.selectionText);
    setPalette(editPalette);
    viewport()->setAutoFillBackground(false);
    viewport()->update();
    m_updatingColors = false;
}

EraIconToolButton::EraIconToolButton(QWidget* parent)
    : QToolButton(parent)
{
    init();
}

void EraIconToolButton::setTone(Tone tone)
{
    if (m_tone == tone)
        return;

    m_tone = tone;
    update();
}

void EraIconToolButton::setIconLogicalSize(int size)
{
    const int clamped = qMax(8, size);
    if (m_iconLogicalSize == clamped)
        return;

    m_iconLogicalSize = clamped;
    updateGeometry();
    update();
}

QSize EraIconToolButton::sizeHint() const
{
    const int side = qMax(m_iconLogicalSize + 12, 28);
    return QSize(side, side);
}

QSize EraIconToolButton::minimumSizeHint() const
{
    return sizeHint();
}

void EraIconToolButton::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const EraStyleColor::ThemePalette& pal = EraStyleColor::themePalette();
    QColor background = Qt::transparent;
    QColor iconColor = pal.textMuted;

    if (m_tone == Tone::Accent)
    {
        if (!isEnabled())
        {
            background = pal.inputBackgroundDisabled;
            iconColor = pal.textDisabled;
        }
        else if (m_pressed)
        {
            background = pal.accentPressed;
            iconColor = pal.onAccentText;
        }
        else if (m_hovered)
        {
            background = pal.accentHover;
            iconColor = pal.onAccentText;
        }
        else
        {
            background = pal.accent;
            iconColor = pal.onAccentText;
        }
    }
    else
    {
        if (!isEnabled())
        {
            background = Qt::transparent;
            iconColor = pal.textDisabled;
        }
        else if (m_pressed)
        {
            background = pal.tabActiveBackground;
            iconColor = pal.textPrimary;
        }
        else if (m_hovered)
        {
            background = pal.hoverBackground;
            iconColor = pal.textPrimary;
        }
        else
        {
            background = Qt::transparent;
            iconColor = pal.textMuted;
        }
    }

    const QRectF buttonRect = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
    painter.setPen(Qt::NoPen);
    painter.setBrush(background);
    painter.drawRoundedRect(buttonRect, buttonRect.height() / 2.0, buttonRect.height() / 2.0);

    const int logicalIconSize = qMin(m_iconLogicalSize, qMax(8, qMin(width(), height()) - 6));
    if (!icon().isNull())
    {
        const QRect iconRect(
            (width() - logicalIconSize) / 2,
            (height() - logicalIconSize) / 2,
            logicalIconSize,
            logicalIconSize
        );

        const QPixmap pix = tintedIconPixmap(icon(), QSize(logicalIconSize, logicalIconSize), iconColor, devicePixelRatioF());
        if (!pix.isNull())
            painter.drawPixmap(iconRect, pix);
    }
    else if (!text().isEmpty())
    {
        painter.setPen(iconColor);
        painter.drawText(rect(), Qt::AlignCenter, text());
    }
}

void EraIconToolButton::enterEvent(QEnterEvent* event)
{
    m_hovered = true;
    update();
    QToolButton::enterEvent(event);
}

void EraIconToolButton::leaveEvent(QEvent* event)
{
    m_hovered = false;
    m_pressed = false;
    update();
    QToolButton::leaveEvent(event);
}

void EraIconToolButton::mousePressEvent(QMouseEvent* event)
{
    QToolButton::mousePressEvent(event);
    m_pressed = isDown();
    update();
}

void EraIconToolButton::mouseReleaseEvent(QMouseEvent* event)
{
    QToolButton::mouseReleaseEvent(event);
    m_pressed = isDown();
    update();
}

void EraIconToolButton::changeEvent(QEvent* event)
{
    QToolButton::changeEvent(event);
    if (event && (event->type() == QEvent::EnabledChange || isThemeEvent(event->type())))
        update();
}

void EraIconToolButton::init()
{
    setAttribute(Qt::WA_MacShowFocusRect, false);
    setFocusPolicy(Qt::NoFocus);
    setCursor(Qt::PointingHandCursor);
    setAutoRaise(true);
    setMouseTracking(true);
}

EraChatBubbleBox::EraChatBubbleBox(bool isUserBubble, QWidget* parent)
    : QWidget(parent)
    , m_isUserBubble(isUserBubble)
{
    setAttribute(Qt::WA_StyledBackground, false);
    setAutoFillBackground(false);
    m_bubbleStyle = QStringLiteral("Era");
}

void EraChatBubbleBox::setBubbleStyle(const QString& styleId)
{
    const QString normalized = normalizeBubbleStyleId(styleId);
    if (m_bubbleStyle == normalized)
        return;
    m_bubbleStyle = normalized;
    update();
}

void EraChatBubbleBox::setUserBubble(bool isUserBubble)
{
    if (m_isUserBubble == isUserBubble)
        return;

    m_isUserBubble = isUserBubble;
    update();
}

void EraChatBubbleBox::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const EraStyleColor::ThemePalette& pal = EraStyleColor::themePalette();
    const QString styleId = normalizeBubbleStyleId(m_bubbleStyle);

    if (styleId == QStringLiteral("iMessage"))
    {
        const QRectF r = QRectF(rect()).adjusted(2.0, 2.0, -2.0, -2.0);
        const qreal radius = 18.0;
        const qreal tailW = 12.0;
        const qreal tailH = 18.0;

        const QRectF bodyRect = m_isUserBubble ? r.adjusted(0.0, 0.0, -tailW, 0.0) : r.adjusted(tailW, 0.0, 0.0, 0.0);

        QPainterPath bubble;
        bubble.addRoundedRect(bodyRect, radius, radius);

        const qreal tailBaseY = bodyRect.bottom() - radius * 0.70;
        QPainterPath tail;
        if (m_isUserBubble)
        {
            const QPointF p0(bodyRect.right(), tailBaseY);
            const QPointF p1(r.right() - tailW * 0.20, tailBaseY + tailH * 0.10);
            const QPointF p2(r.right(), tailBaseY + tailH * 0.60);
            const QPointF p3(bodyRect.right(), tailBaseY + tailH);

            tail.moveTo(p0);
            tail.cubicTo(QPointF(bodyRect.right() + tailW * 0.85, tailBaseY + tailH * 0.05), p1, p2);
            tail.cubicTo(QPointF(r.right() - tailW * 0.35, tailBaseY + tailH * 0.92), QPointF(bodyRect.right() + tailW * 0.15, tailBaseY + tailH), p3);
            tail.closeSubpath();
        }
        else
        {
            const QPointF p0(bodyRect.left(), tailBaseY);
            const QPointF p1(r.left() + tailW * 0.20, tailBaseY + tailH * 0.10);
            const QPointF p2(r.left(), tailBaseY + tailH * 0.60);
            const QPointF p3(bodyRect.left(), tailBaseY + tailH);

            tail.moveTo(p0);
            tail.cubicTo(QPointF(bodyRect.left() - tailW * 0.85, tailBaseY + tailH * 0.05), p1, p2);
            tail.cubicTo(QPointF(r.left() + tailW * 0.35, tailBaseY + tailH * 0.92), QPointF(bodyRect.left() - tailW * 0.15, tailBaseY + tailH), p3);
            tail.closeSubpath();
        }

        bubble.addPath(tail);

        QColor shadow = Qt::black;
        shadow.setAlphaF(EraStyleColor::isDark() ? 0.22 : 0.10);
        painter.setPen(Qt::NoPen);
        painter.setBrush(shadow);
        painter.save();
        painter.translate(0.0, 1.5);
        painter.drawPath(bubble);
        painter.restore();

        if (m_isUserBubble)
        {
            QLinearGradient g(r.topLeft(), r.bottomLeft());
            g.setColorAt(0.0, imessageUserFillTop(pal));
            g.setColorAt(1.0, imessageUserFillBottom(pal));
            painter.setBrush(g);
        }
        else
        {
            painter.setBrush(imessageAssistantFill(pal));
        }
        painter.drawPath(bubble);
        return;
    }

    if (styleId == QStringLiteral("Outline"))
    {
        const QRectF r = QRectF(rect()).adjusted(2.0, 2.0, -2.0, -2.0);
        const qreal radius = 14.0;
        const qreal tailW = 10.0;
        const qreal tailH = 12.0;
        const QRectF bodyRect = m_isUserBubble ? r.adjusted(0.0, 0.0, -tailW, 0.0) : r.adjusted(tailW, 0.0, 0.0, 0.0);

        QPainterPath bubble;
        bubble.addRoundedRect(bodyRect, radius, radius);
        const qreal tailY = bodyRect.bottom() - radius * 0.58;
        QPolygonF tail;
        if (m_isUserBubble)
        {
            tail << QPointF(bodyRect.right(), tailY)
                 << QPointF(r.right(), tailY + tailH * 0.35)
                 << QPointF(bodyRect.right(), tailY + tailH);
        }
        else
        {
            tail << QPointF(bodyRect.left(), tailY)
                 << QPointF(r.left(), tailY + tailH * 0.35)
                 << QPointF(bodyRect.left(), tailY + tailH);
        }
        bubble.addPolygon(tail);

        painter.setPen(QPen(outlineBorder(m_isUserBubble, pal), 1.2));
        painter.setBrush(outlineFill(m_isUserBubble, pal));
        painter.drawPath(bubble);
        return;
    }

    if (styleId == QStringLiteral("Round"))
    {
        const QRectF r = QRectF(rect()).adjusted(2.0, 2.0, -2.0, -2.0);
        const qreal radius = 16.0;
        const qreal tailW = 10.0;
        const qreal tailH = 12.0;
        const QRectF bodyRect = m_isUserBubble ? r.adjusted(0.0, 0.0, -tailW, 0.0) : r.adjusted(tailW, 0.0, 0.0, 0.0);

        QPainterPath bubble;
        bubble.addRoundedRect(bodyRect, radius, radius);
        const qreal tailY = bodyRect.bottom() - radius * 0.60;
        QPolygonF tail;
        if (m_isUserBubble)
        {
            tail << QPointF(bodyRect.right(), tailY)
                 << QPointF(r.right(), tailY + tailH * 0.35)
                 << QPointF(bodyRect.right(), tailY + tailH);
        }
        else
        {
            tail << QPointF(bodyRect.left(), tailY)
                 << QPointF(r.left(), tailY + tailH * 0.35)
                 << QPointF(bodyRect.left(), tailY + tailH);
        }
        bubble.addPolygon(tail);

        QColor shadow = Qt::black;
        shadow.setAlphaF(EraStyleColor::isDark() ? 0.30 : 0.14);
        painter.setPen(Qt::NoPen);
        painter.setBrush(shadow);
        painter.save();
        painter.translate(0.0, 1.5);
        painter.drawPath(bubble);
        painter.restore();

        painter.setPen(QPen(roundBorder(m_isUserBubble, pal), 1.0));
        painter.setBrush(roundFill(m_isUserBubble, pal));
        painter.drawPath(bubble);
        return;
    }

    QColor background;
    QColor border;
    if (m_isUserBubble)
    {
        background = pal.accent;
        border = pal.accentPressed;
    }
    else
    {
        background = EraStyleColor::isDark() ? pal.panelRaised : pal.panelBackground;
        border = EraStyleColor::isDark() ? pal.borderPrimary : pal.borderSecondary;
    }

    painter.setPen(QPen(border, 1.0));
    painter.setBrush(background);
    painter.drawRoundedRect(rect().adjusted(1, 1, -1, -1), 14, 14);
}

void EraChatBubbleBox::changeEvent(QEvent* event)
{
    QWidget::changeEvent(event);
    if (event && isThemeEvent(event->type()))
        update();
}

EraChatBubbleTextView::EraChatBubbleTextView(bool isUserMessage, QWidget* parent)
    : QTextBrowser(parent)
    , m_isUserMessage(isUserMessage)
{
    init();
}

void EraChatBubbleTextView::setBubbleStyle(const QString& styleId)
{
    const QString normalized = normalizeBubbleStyleId(styleId);
    if (m_bubbleStyle == normalized)
        return;
    m_bubbleStyle = normalized;
    refreshAppearance();
}

void EraChatBubbleTextView::setUserMessage(bool isUserMessage)
{
    if (m_isUserMessage == isUserMessage)
        return;

    m_isUserMessage = isUserMessage;
    refreshAppearance();
}

void EraChatBubbleTextView::changeEvent(QEvent* event)
{
    QTextBrowser::changeEvent(event);
    if (!event)
        return;

    if (isThemeEvent(event->type()) || event->type() == QEvent::EnabledChange)
    {
        QTimer::singleShot(0, this, [this] { refreshAppearance(); });
    }
}

void EraChatBubbleTextView::contextMenuEvent(QContextMenuEvent* event)
{
    QMenu* menu = createStandardContextMenu();
    if (!menu)
        return;

    const EraStyleColor::ThemePalette& pal = EraStyleColor::themePalette();
    QPalette menuPalette = menu->palette();
    menuPalette.setColor(QPalette::Window, pal.popupBackground);
    menuPalette.setColor(QPalette::Base, pal.popupBackground);
    menuPalette.setColor(QPalette::Button, pal.popupBackground);
    menuPalette.setColor(QPalette::Text, pal.textPrimary);
    menuPalette.setColor(QPalette::WindowText, pal.textPrimary);
    menuPalette.setColor(QPalette::Highlight, pal.hoverBackground);
    menuPalette.setColor(QPalette::HighlightedText, pal.textPrimary);
    menu->setPalette(menuPalette);
    menu->setAutoFillBackground(true);
    menu->setAttribute(Qt::WA_StyledBackground, true);
    menu->setStyleSheet(QStringLiteral(
        "QMenu{background:%1; color:%2; border:1px solid %3; border-radius:8px; padding:4px;}"
        "QMenu::item{background:%1; color:%2; padding:6px 18px 6px 12px; border-radius:4px;}"
        "QMenu::item:selected{background:%4; color:%2;}"
        "QMenu::item:disabled{background:%1; color:%5;}"
        "QMenu::indicator{width:0px; height:0px;}"
    )
                            .arg(toRgba(pal.popupBackground))
                            .arg(toRgba(pal.textPrimary))
                            .arg(toRgba(pal.divider))
                            .arg(toRgba(pal.hoverBackground))
                            .arg(toRgba(pal.textDisabled)));
    menu->setAttribute(Qt::WA_TranslucentBackground);
    for (QAction* a : menu->actions())
        a->setIcon(QIcon());
    menu->exec(event->globalPos());
    delete menu;
}

void EraChatBubbleTextView::init()
{
    setOpenExternalLinks(true);
    setFrameShape(QFrame::NoFrame);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    setContextMenuPolicy(Qt::DefaultContextMenu);
    setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::LinksAccessibleByMouse);
    document()->setDocumentMargin(0.0);
    setContentsMargins(0, 0, 0, 0);
    setWordWrapMode(QTextOption::WordWrap);
    setLineWrapMode(QTextEdit::WidgetWidth);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    if (auto* hints = QGuiApplication::styleHints())
    {
        connect(hints, &QStyleHints::colorSchemeChanged, this, [this](Qt::ColorScheme) {
            QTimer::singleShot(0, this, [this] { refreshAppearance(); });
        });
    }

    m_bubbleStyle = QStringLiteral("Era");
    refreshAppearance();
}

void EraChatBubbleTextView::refreshAppearance()
{
    if (QCoreApplication::closingDown() || m_updatingColors)
        return;

    m_updatingColors = true;
    const EraStyleColor::ThemePalette& pal = EraStyleColor::themePalette();
    const QString styleId = normalizeBubbleStyleId(m_bubbleStyle);
    QColor textColor = m_isUserMessage ? pal.onAccentText : pal.textPrimary;
    if (styleId == QStringLiteral("iMessage"))
    {
        if (m_isUserMessage)
            textColor = QColor(255, 255, 255);
        else
            textColor = EraStyleColor::isDark() ? QColor(235, 235, 245) : QColor(18, 18, 18);
    }
    else if (styleId == QStringLiteral("Outline"))
    {
        textColor = m_isUserMessage ? pal.accent : pal.textPrimary;
    }
    const QColor disabledColor = pal.textDisabled;

    QPalette widgetPalette = palette();
    bool paletteChanged = false;
    const QColor resolvedTextColor = isEnabled() ? textColor : disabledColor;

    if (widgetPalette.color(QPalette::Text) != resolvedTextColor)
    {
        widgetPalette.setColor(QPalette::Text, resolvedTextColor);
        paletteChanged = true;
    }
    if (widgetPalette.color(QPalette::Base) != Qt::transparent)
    {
        widgetPalette.setColor(QPalette::Base, Qt::transparent);
        paletteChanged = true;
    }
    if (widgetPalette.color(QPalette::PlaceholderText) != disabledColor)
    {
        widgetPalette.setColor(QPalette::PlaceholderText, disabledColor);
        paletteChanged = true;
    }
    if (paletteChanged)
        setPalette(widgetPalette);

    viewport()->setAutoFillBackground(false);
    viewport()->update();
    m_updatingColors = false;
}

EraChatListWidget::EraChatListWidget(QWidget* parent)
    : QListWidget(parent)
{
    init();
}

void EraChatListWidget::changeEvent(QEvent* event)
{
    QListWidget::changeEvent(event);
    if (!event)
        return;

    if (isThemeEvent(event->type()) || event->type() == QEvent::EnabledChange)
    {
        QTimer::singleShot(0, this, [this] { refreshAppearance(); });
    }
}

void EraChatListWidget::init()
{
    setAttribute(Qt::WA_MacShowFocusRect, false);
    setFrameShape(QFrame::NoFrame);
    setStyleSheet(QStringLiteral(
        "QListWidget{background:transparent; border:none; outline:none; padding:0px;}"
        "QListWidget::item{background:transparent; border:none; margin:0px; padding:0px;}"
    ));

    if (viewport())
    {
        viewport()->setAutoFillBackground(false);
        viewport()->setStyleSheet(QStringLiteral("background:transparent; border:none;"));
    }

    EraStyle::installHoverScrollBars(this, true, false);

    if (auto* hints = QGuiApplication::styleHints())
    {
        connect(hints, &QStyleHints::colorSchemeChanged, this, [this](Qt::ColorScheme) {
            QTimer::singleShot(0, this, [this] { refreshAppearance(); });
        });
    }

    refreshAppearance();
}

void EraChatListWidget::refreshAppearance()
{
    if (viewport())
    {
        viewport()->setAutoFillBackground(false);
        viewport()->update();
    }

    update();
}
