#include "ui/ChatWindow.hpp"

#include "common/SettingsManager.hpp"
#include "common/Utils.hpp"

#include <QBoxLayout>
#include <QDateTime>
#include <QDir>
#include <QEvent>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QListWidget>
#include <QDesktopServices>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QTimer>
#include <QTextBrowser>
#include <QTextEdit>
#include <QTextOption>
#include <QUrl>

#include "ui/theme/ThemeApi.hpp"
#include "ui/theme/ThemeWidgets.hpp"

namespace {

constexpr int kComposerMinLines = 2;
constexpr int kComposerMaxLines = 5;
constexpr int kComposerRadius = 18;
constexpr int kComposerSendButtonSize = 34;
constexpr int kComposerSendIconSize = 18;
constexpr int kComposerFooterControlSize = 24;
constexpr int kComposerClearIconSize = 19;
constexpr qreal kComposerInputDocumentMargin = 3.0;

QPixmap circleAvatar(const QPixmap& src, int logicalSize, qreal devicePixelRatio)
{
    if (src.isNull()) return {};

    const int px = qMax(1, int(std::round(logicalSize * devicePixelRatio)));

    // Scale in device pixels to avoid blur on Retina.
    QPixmap scaled = src.scaled(px, px, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);

    QPixmap out(px, px);
    out.fill(Qt::transparent);

    QPainter p(&out);
    p.setRenderHint(QPainter::Antialiasing, true);
    QPainterPath clip;
    clip.addEllipse(0, 0, px, px);
    p.setClipPath(clip);
    p.drawPixmap(0, 0, scaled);
    p.end();

    out.setDevicePixelRatio(devicePixelRatio);
    return out;
}

QString firstPngInDir(const QString& dir)
{
    QDir d(dir);
    const auto files = d.entryInfoList(QStringList{QStringLiteral("*.png"), QStringLiteral("*.PNG")}, QDir::Files, QDir::Name);
    if (files.isEmpty()) return {};
    return files.front().absoluteFilePath();
}

bool hasVisibleCharacters(const QString& text)
{
    for (const QChar ch : text)
    {
        if (!ch.isSpace())
            return true;
    }
    return false;
}

QJsonArray loadChatMessages(const QString& modelFolder)
{
    QFile f(SettingsManager::instance().chatPathForModel(modelFolder));
    if (!f.exists()) return {};
    if (!f.open(QIODevice::ReadOnly)) return {};
    QJsonParseError e;
    const auto doc = QJsonDocument::fromJson(f.readAll(), &e);
    if (e.error != QJsonParseError::NoError) return {};
    return doc.object().value("messages").toArray();
}

void saveChatMessages(const QString& modelFolder, const QJsonArray& messages)
{
    QDir dir(SettingsManager::instance().chatsDir());
    if (!dir.exists()) { dir.mkpath("."); }

    QJsonObject o;
    o["messages"] = messages;

    QFile f(SettingsManager::instance().chatPathForModel(modelFolder));
    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        f.write(QJsonDocument(o).toJson(QJsonDocument::Indented));
    }
}

class ChatMessageWidget final : public QWidget
{
    Q_OBJECT
public:
    class TypingDotsWidget final : public QWidget
    {
    public:
        explicit TypingDotsWidget(QWidget* parent = nullptr)
            : QWidget(parent)
        {
            setFixedSize(30, 12);
            m_timer.setInterval(220);
            connect(&m_timer, &QTimer::timeout, this, [this] {
                m_phase = (m_phase + 1) % 3;
                update();
            });
        }

        void setActive(bool active)
        {
            if (active)
            {
                if (!m_timer.isActive())
                    m_timer.start();
            }
            else
            {
                if (m_timer.isActive())
                    m_timer.stop();
                m_phase = 0;
                update();
            }
        }

    protected:
        void paintEvent(QPaintEvent* event) override
        {
            Q_UNUSED(event);

            QPainter painter(this);
            painter.setRenderHint(QPainter::Antialiasing, true);

            const QColor base = palette().color(QPalette::Text);
            const qreal radius = 2.2;
            const qreal centerY = height() * 0.5;
            const qreal startX = 7.0;
            const qreal step = 8.0;

            for (int i = 0; i < 3; ++i)
            {
                QColor c = base;
                c.setAlphaF(i == m_phase ? 0.95 : 0.32);
                painter.setPen(Qt::NoPen);
                painter.setBrush(c);
                const qreal cx = startX + i * step;
                painter.drawEllipse(QPointF(cx, centerY), radius, radius);
            }
        }

    private:
        QTimer m_timer;
        int m_phase{0};
    };

    ChatMessageWidget(const QPixmap& avatar, const QString& text, bool isUser, const QString& bubbleStyleId, QWidget* parent=nullptr)
        : QWidget(parent), m_isUser(isUser), m_bubbleStyleId(bubbleStyleId)
    {
        auto lay = new QHBoxLayout(this);
        lay->setContentsMargins(10, 8, 10, 8);
        lay->setSpacing(10);

        m_avatar = new QLabel(this);
        m_avatar->setFixedSize(32, 32);
        setAvatar(avatar);

        m_bubbleBox = new ThemeWidgets::ChatBubbleBox(m_isUser, this);
        auto* bubbleLay = new QVBoxLayout(m_bubbleBox);
        bubbleLay->setContentsMargins(12, 8, 12, 8);
        bubbleLay->setSpacing(0);

        m_text = new ThemeWidgets::ChatBubbleTextView(m_isUser, m_bubbleBox);
        m_text->setText(text);
        updateBubbleTextStyle();

        m_typingDots = new TypingDotsWidget(this);
        m_typingDots->setVisible(false);

        bubbleLay->addWidget(m_text);
        m_bubbleBox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

        if (m_isUser)
        {
            lay->addStretch(1);
            lay->addWidget(m_bubbleBox, 0, Qt::AlignRight | Qt::AlignTop);
            lay->addWidget(m_avatar, 0, Qt::AlignRight | Qt::AlignTop);
        }
        else
        {
            lay->addWidget(m_avatar, 0, Qt::AlignLeft | Qt::AlignTop);
            lay->addWidget(m_bubbleBox, 0, Qt::AlignLeft | Qt::AlignTop);
            lay->addWidget(m_typingDots, 0, Qt::AlignLeft | Qt::AlignVCenter);
            lay->addStretch(1);
        }

        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);

        // Initial sizing
        syncTextSizeToContent();
    }

    void setAvatar(const QPixmap& avatar)
    {
        const qreal dpr = devicePixelRatioF();
        m_avatar->setPixmap(circleAvatar(avatar, 32, dpr));
    }

    void updateBubbleTextStyle()
    {
        if (m_bubbleBox)
            m_bubbleBox->setUserBubble(m_isUser);
        if (m_text)
            m_text->setUserMessage(m_isUser);
        if (m_bubbleBox)
            m_bubbleBox->setBubbleStyle(m_bubbleStyleId);
        if (m_text)
            m_text->setBubbleStyle(m_bubbleStyleId);
    }

    void setBubbleStyle(const QString& bubbleStyleId)
    {
        const QString s = bubbleStyleId.trimmed();
        if (m_bubbleStyleId == s)
            return;
        m_bubbleStyleId = s;
        updateBubbleTextStyle();
        syncTextSizeToContent();
        updateGeometry();
        update();
    }

    void setMaxBubbleWidth(int w)
    {
        m_maxBubbleWidth = qMax(1, w);
        syncTextSizeToContent();
        updateGeometry();
    }

    void setRowWidth(int w)
    {
        m_rowWidth = qMax(1, w);
        updateGeometry();
    }

    void appendToken(const QString& t)
    {
        if (m_waitingForResponse && hasVisibleCharacters(t))
            setWaitingForResponse(false);

        m_text->moveCursor(QTextCursor::End);
        m_text->insertPlainText(t);
        m_text->moveCursor(QTextCursor::End);
        updateBubbleTextStyle();
        syncTextSizeToContent();
        updateGeometry();
        update();
    }

    void setContent(const QString& c)
    {
        if (m_waitingForResponse && hasVisibleCharacters(c))
            setWaitingForResponse(false);

        m_text->setText(c);
        updateBubbleTextStyle();
        syncTextSizeToContent();
        updateGeometry();
        update();
    }

    void setWaitingForResponse(bool waiting)
    {
        const bool normalized = waiting && !m_isUser;
        if (m_waitingForResponse == normalized)
            return;

        m_waitingForResponse = normalized;
        if (m_typingDots) {
            m_typingDots->setActive(m_waitingForResponse);
            m_typingDots->setVisible(m_waitingForResponse);
        }
        if (m_bubbleBox)
            m_bubbleBox->setVisible(!m_waitingForResponse);
        if (m_text)
            m_text->setVisible(!m_waitingForResponse);

        updateGeometry();
        update();
    }

    QString content() const
    {
        return m_text ? m_text->toPlainText() : QString();
    }

    QSize sizeHint() const override
    {
        // Height = max(avatar, visible content) + paddings.
        const int rowTopBottom = 16;
        int contentH = 0;
        if (m_waitingForResponse && m_typingDots)
            contentH = m_typingDots->sizeHint().height();
        else if (m_bubbleBox)
            contentH = m_bubbleBox->height();

        const int h = qMax(32, contentH) + rowTopBottom;
        return { m_rowWidth, h };
    }

private:
    void syncTextSizeToContent()
    {
        if (!m_text) return;

        if (m_waitingForResponse)
        {
            m_text->setVisible(false);
            if (m_typingDots)
                m_typingDots->setVisible(true);
            if (m_bubbleBox)
                m_bubbleBox->setVisible(false);
            return;
        }

        m_text->setVisible(true);
        if (m_typingDots)
            m_typingDots->setVisible(false);
        if (m_bubbleBox)
            m_bubbleBox->setVisible(true);

        m_text->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_text->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

        QTextDocument* doc = m_text->document();
        doc->setDocumentMargin(0.0);

        const QFontMetrics fm(m_text->font());

        // Only cap MAX width (3/4 of window). Don't enforce a minimum width.
        const int maxW = qMax(1, m_maxBubbleWidth);

        // Phase 1: measure ideal width without wrapping.
        {
            QTextOption opt;
            opt.setWrapMode(QTextOption::NoWrap);
            doc->setDefaultTextOption(opt);
        }
        doc->setTextWidth(-1);
        doc->adjustSize();

        const int idealW = qMax(1, int(std::ceil(doc->idealWidth())));

        const bool empty = m_text->toPlainText().isEmpty();
        int w = empty ? fm.horizontalAdvance(QStringLiteral("…")) : idealW;
        w = qMin(w, maxW);

        // Phase 2: if clamped, enable wrapping and compute final height.
        if (w >= maxW)
        {
            QTextOption opt = doc->defaultTextOption();
            opt.setWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
            doc->setDefaultTextOption(opt);
            doc->setTextWidth(w);
        }
        else
        {
            QTextOption opt;
            opt.setWrapMode(QTextOption::NoWrap);
            doc->setDefaultTextOption(opt);
            doc->setTextWidth(-1);
        }

        doc->adjustSize();

        int textH = int(std::ceil(doc->size().height()));
        textH = qMax(textH, fm.height());

        // Final width: use the actual document width (after wrapping decisions).
        // This avoids the "tiny horizontal scrollbar" caused by rounding/viewport mismatches.
        const int docW = qMax(1, int(std::ceil(doc->size().width())));
        const int finalW = qMin(docW, maxW);

        m_text->setFixedWidth(finalW);
        m_text->setFixedHeight(textH);
        if (m_bubbleBox)
        {
            m_bubbleBox->setFixedSize(finalW + 24, textH + 16);
            m_bubbleBox->updateGeometry();
        }

        m_text->updateGeometry();
    }

    bool m_isUser{false};
    QString m_bubbleStyleId;
    QLabel* m_avatar{nullptr};
    ThemeWidgets::ChatBubbleBox* m_bubbleBox{nullptr};
    ThemeWidgets::ChatBubbleTextView* m_text{nullptr};
    TypingDotsWidget* m_typingDots{nullptr};
    int m_maxBubbleWidth{360};
    int m_rowWidth{520};
    bool m_waitingForResponse{false};
};

} // namespace

class ChatWindow::Impl {
public:
    QString modelFolder;
    QString modelDir;
    QString bubbleStyleId{QStringLiteral("Era")};

    QWidget* central{nullptr};
    ThemeWidgets::ChatListWidget* list{nullptr};
    QWidget* composerCard{nullptr};
    ThemeWidgets::ChatComposerEdit* input{nullptr};
    ThemeWidgets::IconButton* sendBtn{nullptr};
    ThemeWidgets::IconButton* clearBtn{nullptr};
    ThemeWidgets::ComboBox* styleCombo{nullptr};
    ThemeWidgets::ComboBox* modelSizeCombo{nullptr};
    QLabel* countLabel{nullptr};

    QPixmap userAvatar;
    QPixmap aiAvatar;

    ChatMessageWidget* currentAiBubble{nullptr};
    QListWidgetItem* currentAiItem{nullptr};

    QJsonArray messages; // simplified format: {role, content}
    bool currentAiBubbleIsDraft{false};

    bool relayoutQueued{false};

    int streamingAssistantIndex{-1};

    void scheduleRelayout(QObject* context)
    {
        if (relayoutQueued) return;
        relayoutQueued = true;
        QMetaObject::invokeMethod(context, [this]{
            relayoutQueued = false;
            applyBubbleWidthForAll();
        }, Qt::QueuedConnection);
    }

    int viewportRowWidth() const
    {
        const int vw = (list && list->viewport()) ? list->viewport()->width() : 520;
        return qMax(1, vw - 20);
    }

    int computeBubbleMaxWidth() const
    {
        const int vw = viewportRowWidth();
        const int safe = qMax(0, vw - 32);
        const int max = int(safe * 0.75);
        return qMax(1, max);
    }

    void applyBubbleWidthForAll()
    {
        if (!list) return;
        const int maxW = computeBubbleMaxWidth();
        for (int i = 0; i < list->count(); ++i)
        {
            auto* item = list->item(i);
            if (!item) continue;
            if (auto* w = qobject_cast<ChatMessageWidget*>(list->itemWidget(item)))
            {
                w->setMaxBubbleWidth(maxW);
                w->setRowWidth(viewportRowWidth());
                const QSize hint = w->sizeHint();
                item->setSizeHint(QSize(viewportRowWidth(), hint.height()));
            }
        }
        forceListRelayout();
    }

    void applyBubbleStyleForAll()
    {
        if (!list) return;
        for (int i = 0; i < list->count(); ++i)
        {
            auto* item = list->item(i);
            if (!item) continue;
            if (auto* w = qobject_cast<ChatMessageWidget*>(list->itemWidget(item)))
                w->setBubbleStyle(bubbleStyleId);
        }
        forceListRelayout();
    }

    void forceListRelayout()
    {
        if (!list) return;
        // Force immediate geometry recalculation so bubble sizing changes are visible right away.
        list->doItemsLayout();
        list->updateGeometry();
        list->viewport()->update();
    }

    void updateInputMetrics()
    {
        if (!input) return;

        const int targetHeight = input->preferredHeight(kComposerMinLines, kComposerMaxLines);
        if (input->height() != targetHeight)
            input->setFixedHeight(targetHeight);

        const QFontMetrics fm(input->font());
        const int maxDocumentHeight = fm.lineSpacing() * kComposerMaxLines;
        const bool overflow = input->documentHeight() > maxDocumentHeight;
        input->setVerticalScrollBarPolicy(overflow ? Qt::ScrollBarAsNeeded : Qt::ScrollBarAlwaysOff);
    }

    void updateInputCount()
    {
        if (!countLabel || !input) return;
        countLabel->setText(QString::number(input->toPlainText().size()));
    }

    void rebuildFromMessages()
    {
        list->clear();
        currentAiBubble = nullptr;
        currentAiItem = nullptr;

        const int maxW = computeBubbleMaxWidth();

        for (const auto& v : messages)
        {
            const auto o = v.toObject();
            const QString role = o.value("role").toString();
            const QString content = o.value("content").toString();
            const bool isUser = (role == "user");

            auto* item = new QListWidgetItem(list);
            item->setFlags(item->flags() & ~Qt::ItemIsSelectable);

            auto* bubble = new ChatMessageWidget(isUser ? userAvatar : aiAvatar, content, isUser, bubbleStyleId);
            bubble->setMaxBubbleWidth(maxW);
            bubble->setRowWidth(viewportRowWidth());

            const QSize hint = bubble->sizeHint();
            item->setSizeHint(QSize(viewportRowWidth(), hint.height()));
            list->addItem(item);
            list->setItemWidget(item, bubble);
        }

        list->scrollToBottom();
        forceListRelayout();
    }

    ChatMessageWidget* appendBubble(const QString& role, const QString& content, bool waiting)
    {
        if (!list) return nullptr;
        const bool isUser = (role == QStringLiteral("user"));

        auto* item = new QListWidgetItem(list);
        item->setFlags(item->flags() & ~Qt::ItemIsSelectable);

        auto* bubble = new ChatMessageWidget(isUser ? userAvatar : aiAvatar, content, isUser, bubbleStyleId);
        bubble->setWaitingForResponse(waiting);
        bubble->setMaxBubbleWidth(computeBubbleMaxWidth());
        bubble->setRowWidth(viewportRowWidth());

        const QSize hint = bubble->sizeHint();
        item->setSizeHint(QSize(viewportRowWidth(), hint.height()));
        list->addItem(item);
        list->setItemWidget(item, bubble);

        if (!isUser) {
            currentAiItem = item;
        }

        list->scrollToBottom();
        return bubble;
    }

    void pushMessage(const QString& role, const QString& content)
    {
        QJsonObject o;
        o["role"] = role;
        o["content"] = content;
        messages.append(o);
        if (messages.size() > 200) {
            const int drop = messages.size() - 200;
            for (int i = 0; i < drop && !messages.isEmpty(); ++i)
                messages.removeAt(0);
            if (list) {
                for (int i = 0; i < drop && list->count() > 0; ++i) {
                    auto* item = list->takeItem(0);
                    if (!item) continue;
                    auto* w = list->itemWidget(item);
                    if (w) {
                        list->removeItemWidget(item);
                        w->deleteLater();
                    }
                    delete item;
                }
            }
        }
        saveChatMessages(modelFolder, messages);
    }

    void beginStreamingAssistant()
    {
        // Ensure persisted history has exactly one assistant placeholder for this reply.
        streamingAssistantIndex = -1;
        if (modelFolder.isEmpty()) return;

        // If last message is already an assistant placeholder (empty), reuse it.
        if (!messages.isEmpty())
        {
            const auto last = messages.last().toObject();
            if (last.value("role").toString() == QStringLiteral("assistant")
                && last.value("content").toString().isEmpty())
            {
                streamingAssistantIndex = messages.size() - 1;
                return;
            }
        }

        QJsonObject o;
        o["role"] = QStringLiteral("assistant");
        o["content"] = QString();
        messages.append(o);
        if (messages.size() > 200) {
            const int drop = messages.size() - 200;
            for (int i = 0; i < drop && !messages.isEmpty(); ++i)
                messages.removeAt(0);
            if (list) {
                for (int i = 0; i < drop && list->count() > 0; ++i) {
                    auto* item = list->takeItem(0);
                    if (!item) continue;
                    auto* w = list->itemWidget(item);
                    if (w) {
                        list->removeItemWidget(item);
                        w->deleteLater();
                    }
                    delete item;
                }
            }
        }
        streamingAssistantIndex = messages.size() - 1;
        saveChatMessages(modelFolder, messages);
    }

    void updateStreamingAssistantContent(const QString& content)
    {
        if (streamingAssistantIndex < 0 || streamingAssistantIndex >= messages.size()) return;
        auto o = messages.at(streamingAssistantIndex).toObject();
        if (o.value("role").toString() != QStringLiteral("assistant")) return;
        o["content"] = content;
        messages[streamingAssistantIndex] = o;
        saveChatMessages(modelFolder, messages);
    }

    void finalizeStreamingAssistant(const QString& finalText)
    {
        if (streamingAssistantIndex >= 0)
        {
            updateStreamingAssistantContent(finalText);
        }
        streamingAssistantIndex = -1;
    }

    // NOTE: streaming should NOT touch persisted chat history.
    // We keep the assistant bubble purely in UI during streaming and persist only once at finish.
};

ChatWindow::ChatWindow(QWidget* parent)
    : QMainWindow(parent), d(new Impl)
{
    setWindowTitle(tr("聊天"));
#if defined(Q_OS_LINUX)
    setWindowFlag(Qt::Window, true);
    setWindowFlag(Qt::CustomizeWindowHint, false);
    setWindowFlag(Qt::WindowMaximizeButtonHint, false);
#endif
    resize(520, 640);

    d->central = new QWidget(this);
    d->central->setObjectName(QStringLiteral("chatCentral"));
    setCentralWidget(d->central);

    auto* root = new QVBoxLayout(d->central);
    root->setContentsMargins(12, 12, 12, 12);
    root->setSpacing(10);

    d->list = new ThemeWidgets::ChatListWidget(d->central);
    d->list->setSpacing(6);
    d->list->setUniformItemSizes(false);
    d->list->setSelectionMode(QAbstractItemView::NoSelection);
    d->list->setFocusPolicy(Qt::NoFocus);
    d->list->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    d->list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    root->addWidget(d->list, 1);

    d->composerCard = new QWidget(d->central);
    d->composerCard->setObjectName(QStringLiteral("chatComposerCard"));
    auto* composerLayout = new QVBoxLayout(d->composerCard);
    composerLayout->setContentsMargins(14, 14, 14, 14);
    composerLayout->setSpacing(6);

    auto* editorRow = new QHBoxLayout();
    editorRow->setContentsMargins(0, 0, 0, 0);
    editorRow->setSpacing(10);

    d->input = new ThemeWidgets::ChatComposerEdit(d->composerCard);
    d->input->document()->setDocumentMargin(kComposerInputDocumentMargin);
    d->input->setPlaceholderText(tr("输入消息... (Enter 发送 / Shift+Enter 换行)"));
    d->sendBtn = new ThemeWidgets::IconButton(d->composerCard);
    d->sendBtn->setTone(ThemeWidgets::IconButton::Tone::Accent);
    d->sendBtn->setIconLogicalSize(kComposerSendIconSize);
    d->sendBtn->setFixedSize(kComposerSendButtonSize, kComposerSendButtonSize);
    d->sendBtn->setToolTip(tr("发送"));
    d->sendBtn->setIcon(Theme::themedIcon(Theme::IconToken::ChatSend));

    editorRow->addWidget(d->input, 1);
    editorRow->addWidget(d->sendBtn, 0, Qt::AlignRight | Qt::AlignBottom);
    composerLayout->addLayout(editorRow);

    auto* footerRow = new QHBoxLayout();
    footerRow->setContentsMargins(0, 0, 0, 0);
    footerRow->setSpacing(8);

    d->clearBtn = new ThemeWidgets::IconButton(d->composerCard);
    d->clearBtn->setTone(ThemeWidgets::IconButton::Tone::Ghost);
    d->clearBtn->setIconLogicalSize(kComposerClearIconSize);
    d->clearBtn->setToolTip(tr("清除"));
    d->clearBtn->setFixedSize(kComposerFooterControlSize, kComposerFooterControlSize);
    d->clearBtn->setIcon(Theme::themedIcon(Theme::IconToken::ChatClear));

    d->styleCombo = new ThemeWidgets::ComboBox(d->composerCard);
    d->styleCombo->addItem(QStringLiteral("Original"), QStringLiteral("Original"));
    d->styleCombo->addItem(QStringLiteral("Universal"), QStringLiteral("Universal"));
    d->styleCombo->addItem(QStringLiteral("Anime"), QStringLiteral("Anime"));
    d->styleCombo->setFixedWidth(120);
    {
        const QString saved = SettingsManager::instance().llmStyle();
        const int idx = d->styleCombo->findData(saved);
        d->styleCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }

    d->modelSizeCombo = new ThemeWidgets::ComboBox(d->composerCard);
    d->modelSizeCombo->addItem(QStringLiteral("1.5B"), QStringLiteral("1.5B"));
    d->modelSizeCombo->addItem(QStringLiteral("7B"), QStringLiteral("7B"));
    d->modelSizeCombo->setFixedWidth(90);
    {
        const QString saved = SettingsManager::instance().llmModelSize();
        const int idx = d->modelSizeCombo->findData(saved);
        d->modelSizeCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }

    d->countLabel = new QLabel(QStringLiteral("0"), d->composerCard);
    d->countLabel->setObjectName(QStringLiteral("chatComposerCountLabel"));
    d->countLabel->setAlignment(Qt::AlignCenter);
    d->countLabel->setFixedSize(kComposerSendButtonSize, kComposerFooterControlSize);
    d->countLabel->setContentsMargins(0, 5, 0, 0);

    footerRow->addWidget(d->clearBtn, 0, Qt::AlignLeft | Qt::AlignVCenter);
    footerRow->addWidget(d->styleCombo, 0, Qt::AlignLeft | Qt::AlignVCenter);
    footerRow->addWidget(d->modelSizeCombo, 0, Qt::AlignLeft | Qt::AlignVCenter);
    footerRow->addStretch(1);
    footerRow->addWidget(d->countLabel, 0, Qt::AlignRight | Qt::AlignVCenter);
    composerLayout->addLayout(footerRow);

    root->addWidget(d->composerCard);

    // 默认头像：统一使用 avator-icon.png（用户头像 + 模型无 png 时的 AI 默认头像）
    const QString defaultAvatarPath = appResourcePath(QStringLiteral("icons/avator-icon.png"));
    d->userAvatar = QPixmap(defaultAvatarPath);
    if (d->userAvatar.isNull())
    {
        // 兜底：如果资源缺失，至少用 app-icon
        d->userAvatar = QPixmap(appResourcePath(QStringLiteral("icons/app-icon.png")));
    }

    auto sendNow = [this]{
        if (d->modelFolder.isEmpty()) return;
        if (!d->sendBtn->isEnabled()) return; // already busy/disabled

        const QString text = d->input->toPlainText().trimmed();
        if (text.isEmpty()) return;

        // De-dup: some platforms/widgets can emit both "clicked" and our custom sendRequested
        // for a single user action. Guard by time+content.
        static QString s_lastText;
        static qint64 s_lastMs = 0;
        const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
        if (text == s_lastText && (nowMs - s_lastMs) < 400)
            return;
        s_lastText = text;
        s_lastMs = nowMs;

        d->input->clear();

        // Optimistic disable to prevent rapid double-send before controller flips busy.
        d->sendBtn->setEnabled(false);
        d->input->setEnabled(false);

        emit requestSendMessage(d->modelFolder, text);
    };

    connect(d->sendBtn, &QToolButton::clicked, this, sendNow);
    connect(d->input, &ThemeWidgets::ChatComposerEdit::sendRequested, this, sendNow);
    connect(d->input, &ThemeWidgets::ChatComposerEdit::metricsChanged, this, [this] {
        if (!d) return;
        d->updateInputMetrics();
    });
    connect(d->input, &QTextEdit::textChanged, this, [this] {
        if (!d) return;
        d->updateInputCount();
    });

    connect(d->clearBtn, &QToolButton::clicked, this, [this]{
        if (d->modelFolder.isEmpty()) return;
        const auto ret = QMessageBox::question(this, tr("确认"), tr("确定要清除聊天记录吗？"));
        if (ret != QMessageBox::Yes) return;
        emit requestClearChat(d->modelFolder);
    });
    connect(d->styleCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        emit requestLlmStyleChanged(d->styleCombo->currentData().toString());
    });
    connect(d->modelSizeCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        emit requestLlmModelSizeChanged(d->modelSizeCombo->currentData().toString());
    });

    // Responsive relayout on resize
    connect(d->list->verticalScrollBar(), &QScrollBar::rangeChanged, this, [this]{
        d->applyBubbleWidthForAll();
    });

    // Also update widths when the viewport itself changes size (more immediate than rangeChanged)
    d->list->viewport()->installEventFilter(this);

    // Initial relayout after show; without this some bubbles get an incorrect first width.
    d->updateInputMetrics();
    d->updateInputCount();
    d->scheduleRelayout(this);
}

ChatWindow::~ChatWindow() = default;

QString ChatWindow::currentAssistantDraft() const
{
    if (!d || !d->currentAiBubble) return {};
    return d->currentAiBubble->content();
}

void ChatWindow::setCurrentModel(const QString& modelFolder, const QString& modelDir)
{
    const bool folderChanged = (d->modelFolder != modelFolder);
    d->modelFolder = modelFolder;
    d->modelDir = modelDir;

    // Always reset AI avatar when model changes, so switching to "no png" doesn't keep old one.
    d->aiAvatar = {};

    // AI 头像：取模型目录第一张 png，否则回落到默认头像（avator-icon.png）
    const QString aiPng = firstPngInDir(modelDir);
    if (!aiPng.isEmpty())
    {
        d->aiAvatar = QPixmap(aiPng);
    }

    if (d->aiAvatar.isNull())
    {
        d->aiAvatar = d->userAvatar;
    }

    if (folderChanged)
        loadFromDisk(modelFolder);

    // After rebuilding, ensure widths match the current viewport size.
    d->scheduleRelayout(this);
}

void ChatWindow::setBusy(bool busy)
{
    d->sendBtn->setEnabled(!busy);
    d->input->setEnabled(!busy);
    if (d->clearBtn) d->clearBtn->setEnabled(!busy);
    if (d->styleCombo) d->styleCombo->setEnabled(!busy);
    if (d->modelSizeCombo) d->modelSizeCombo->setEnabled(!busy);
}

void ChatWindow::setLlmStyle(const QString& style)
{
    if (!d || !d->styleCombo) return;
    const int idx = d->styleCombo->findData(style.trimmed());
    if (idx < 0) return;
    if (d->styleCombo->currentIndex() == idx) return;
    QSignalBlocker b(d->styleCombo);
    d->styleCombo->setCurrentIndex(idx);
}

void ChatWindow::setLlmModelSize(const QString& size)
{
    if (!d || !d->modelSizeCombo) return;
    const int idx = d->modelSizeCombo->findData(size.trimmed());
    if (idx < 0) return;
    if (d->modelSizeCombo->currentIndex() == idx) return;
    QSignalBlocker b(d->modelSizeCombo);
    d->modelSizeCombo->setCurrentIndex(idx);
}

void ChatWindow::appendUserMessage(const QString& text)
{
    if (d->modelFolder.isEmpty()) return;
    d->pushMessage("user", text);
    d->appendBubble(QStringLiteral("user"), text, /*waiting*/false);
    d->scheduleRelayout(this);
}

void ChatWindow::appendAiMessageStart()
{
    // Start a streaming assistant bubble.
    if (d->modelFolder.isEmpty()) return;

    if (d->currentAiBubble)
        return;

    // Persist a placeholder assistant message ONCE so streaming doesn't create multiple assistant rows.
    d->beginStreamingAssistant();

    // UI-only draft bubble; the persisted content is updated via setAiMessageContent()/appendAiToken().
    d->currentAiBubbleIsDraft = true;

    if (d->list)
    {
        auto* item = new QListWidgetItem(d->list);
        auto* w = new ChatMessageWidget(d->aiAvatar, QString(), /*isUser*/false, d->bubbleStyleId, nullptr);
        w->setWaitingForResponse(true);
        w->setMaxBubbleWidth(d->computeBubbleMaxWidth());
        w->setRowWidth(d->viewportRowWidth());
        const QSize hint = w->sizeHint();
        item->setSizeHint(QSize(d->viewportRowWidth(), hint.height()));
        d->list->addItem(item);
        d->list->setItemWidget(item, w);
        d->currentAiBubble = w;
        d->currentAiItem = item;
        d->list->scrollToBottom();
        d->scheduleRelayout(this);
    }
}

void ChatWindow::appendAiToken(const QString& token)
{
    if (!d) return;
    if (!d->currentAiBubble)
        return;

    d->currentAiBubble->appendToken(token);

    // resize list item
    if (d->list)
    {
        auto* item = d->list->item(d->list->count() - 1);
        if (item)
        {
            const QSize hint = d->currentAiBubble->sizeHint();
            item->setSizeHint(QSize(d->viewportRowWidth(), hint.height()));
        }
        d->scheduleRelayout(this);
        d->list->scrollToBottom();
    }
}

void ChatWindow::appendAiMessageFinish()
{
    if (!d || !d->currentAiBubble || d->modelFolder.isEmpty())
        return;
    finalizeAssistantMessage(d->currentAiBubble->content(), /*ensureBubbleExists*/false);
}

void ChatWindow::cancelAssistantDraft()
{
    if (!d) return;

    if (d->streamingAssistantIndex >= 0 && d->streamingAssistantIndex < d->messages.size())
    {
        d->messages.removeAt(d->streamingAssistantIndex);
        d->streamingAssistantIndex = -1;
        if (!d->modelFolder.isEmpty())
            saveChatMessages(d->modelFolder, d->messages);
    }
    else
    {
        d->streamingAssistantIndex = -1;
    }

    if (d->list && d->currentAiItem)
    {
        const int row = d->list->row(d->currentAiItem);
        if (row >= 0)
        {
            QListWidgetItem* item = d->list->takeItem(row);
            QWidget* w = item ? d->list->itemWidget(item) : nullptr;
            if (w)
            {
                d->list->removeItemWidget(item);
                w->deleteLater();
            }
            delete item;
        }
    }

    d->currentAiBubble = nullptr;
    d->currentAiItem = nullptr;
    d->currentAiBubbleIsDraft = false;
}

void ChatWindow::setAiMessageContent(const QString& content)
{
    // Set/replace the assistant draft bubble content.
    if (d->modelFolder.isEmpty()) return;

    if (!d->currentAiBubble)
    {
        appendAiMessageStart();
    }

    if (d->currentAiBubble)
    {
        d->currentAiBubble->setContent(content);
        d->currentAiBubbleIsDraft = true;

        if (d->list)
        {
            auto* item = d->list->item(d->list->count() - 1);
            if (item)
            {
                const QSize hint = d->currentAiBubble->sizeHint();
                item->setSizeHint(QSize(d->viewportRowWidth(), hint.height()));
            }

            d->scheduleRelayout(this);
            d->list->scrollToBottom();
        }
    }
}

void ChatWindow::loadFromDisk(const QString& modelFolder)
{
    if (modelFolder.isEmpty()) return;
    d->messages = loadChatMessages(modelFolder);
    d->currentAiBubble = nullptr;
    d->currentAiItem = nullptr;
    d->currentAiBubbleIsDraft = false;
    d->streamingAssistantIndex = -1;
    d->rebuildFromMessages();
}

bool ChatWindow::event(QEvent* e)
{
    if (e->type() == QEvent::LanguageChange)
    {
        setWindowTitle(tr("聊天"));
        if (d)
        {
            if (d->input) d->input->setPlaceholderText(tr("输入消息... (Enter 发送 / Shift+Enter 换行)"));
            if (d->sendBtn) d->sendBtn->setToolTip(tr("发送"));
            if (d->clearBtn) d->clearBtn->setToolTip(tr("清除"));
        }
    }

    if (e->type() == QEvent::Resize)
    {
        if (d && d->list)
        {
            d->applyBubbleWidthForAll();
        }
        if (d)
        {
            d->updateInputMetrics();
        }
    }

    if (e->type() == QEvent::ApplicationPaletteChange
        || e->type() == QEvent::PaletteChange
        || e->type() == QEvent::ThemeChange
        || e->type() == QEvent::StyleChange)
    {
        if (d)
        {
            if (d->sendBtn) d->sendBtn->setIcon(Theme::themedIcon(Theme::IconToken::ChatSend));
            if (d->clearBtn) d->clearBtn->setIcon(Theme::themedIcon(Theme::IconToken::ChatClear));
            if (d->list) d->list->viewport()->update();
            d->updateInputMetrics();
            d->scheduleRelayout(this);
        }
    }

    return QMainWindow::event(e);
}

bool ChatWindow::eventFilter(QObject* obj, QEvent* e)
{
    if (d && d->list && obj == d->list->viewport())
    {
        if (e->type() == QEvent::Resize)
        {
            // Immediate relayout on viewport resize fixes the "right side empty" width glitch.
            d->applyBubbleWidthForAll();
            return false;
        }
    }

    return QMainWindow::eventFilter(obj, e);
}

#include "ChatWindow.moc"

void ChatWindow::finalizeAssistantMessage(const QString& content)
{
    finalizeAssistantMessage(content, /*ensureBubbleExists*/true);
}

void ChatWindow::finalizeAssistantMessage(const QString& content, bool ensureBubbleExists)
{
    if (!d || d->modelFolder.isEmpty())
        return;

    // Make sure we have a visible assistant bubble representing THIS reply.
    if (ensureBubbleExists)
    {
        if (!d->currentAiBubble)
            appendAiMessageStart();
        if (d->currentAiBubble)
            d->currentAiBubble->setContent(content);
    }

    // Persist final assistant content exactly once:
    // - If we are streaming, there is already a placeholder assistant row (empty at start) which we update.
    // - If we are non-streaming, update last assistant if its content is empty, otherwise append.

    // 1) Streaming path: placeholder index is tracked.
    if (d->streamingAssistantIndex >= 0 && d->streamingAssistantIndex < d->messages.size())
    {
        d->updateStreamingAssistantContent(content);
        d->streamingAssistantIndex = -1;
    }
    else
    {
        // 2) Non-stream path: prefer to update last assistant if it exists and is empty.
        if (!d->messages.isEmpty())
        {
            const auto last = d->messages.last().toObject();
            const QString role = last.value("role").toString();
            const QString lastContent = last.value("content").toString();

            if (role == QStringLiteral("assistant") && lastContent.isEmpty())
            {
                // Replace the placeholder.
                QJsonObject o = last;
                o["content"] = content;
                d->messages[d->messages.size() - 1] = o;
                saveChatMessages(d->modelFolder, d->messages);
            }
            else if (!(role == QStringLiteral("assistant") && lastContent == content))
            {
                d->pushMessage(QStringLiteral("assistant"), content);
            }
        }
        else
        {
            d->pushMessage(QStringLiteral("assistant"), content);
        }
    }

    // Rebuild UI from persisted messages to avoid leaving a draft-only bubble around.
    d->currentAiBubble = nullptr;
    d->currentAiBubbleIsDraft = false;
    if (d->list)
    {
        auto* item = d->currentAiItem ? d->currentAiItem : d->list->item(d->list->count() - 1);
        if (item)
        {
            if (auto* w = qobject_cast<ChatMessageWidget*>(d->list->itemWidget(item)))
            {
                w->setWaitingForResponse(false);
                w->setMaxBubbleWidth(d->computeBubbleMaxWidth());
                w->setRowWidth(d->viewportRowWidth());
                const QSize hint = w->sizeHint();
                item->setSizeHint(QSize(d->viewportRowWidth(), hint.height()));
            }
        }
        d->currentAiItem = nullptr;
        d->list->scrollToBottom();
    }
    d->scheduleRelayout(this);
}
