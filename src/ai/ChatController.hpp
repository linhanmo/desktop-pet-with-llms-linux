#pragma once

// Some third-party headers define macros like 'slots' that break Qt headers.
// We MUST NOT define QT_NO_KEYWORDS here, otherwise Qt's 'signals/slots' keywords
// are disabled and moc output won't compile.

#include <QObject>
#include <QString>

class ChatWindow;
class Renderer;
class LocalLlmClient;

class ChatController : public QObject
{
    Q_OBJECT
public:
    explicit ChatController(QObject* parent = nullptr);

    void setChatWindow(ChatWindow* wnd);
    void setRenderer(Renderer* renderer);

signals:
    void assistantBubbleTextChanged(const QString& text, bool isFinal);

public slots:
    void onModelChanged(const QString& modelFolder, const QString& modelDir);
    void applyPreferredAudioOutput();
    void setLlmStyle(const QString& style);
    void setLlmModelSize(const QString& size);
    void postLocalAssistantMessage(const QString& text,
                                  const QString& motionGroup,
                                  const QString& expressionName,
                                  bool writeToHistory);
    void triggerLocalPrompt(const QString& userText,
                            const QString& motionGroup,
                            const QString& expressionName);

private slots:
    void onSendRequested(const QString& modelFolder, const QString& userText);
    void onClearRequested(const QString& modelFolder);

private:
    void saveClearedChat(const QString& modelFolder) const;
    void reservedStartTtsPlayback(const QString& audioPath);
    void reservedUpdateLipSync(const QString& audioPath);

private:
    ChatWindow* m_chatWindow{};
    Renderer* m_renderer{};
    LocalLlmClient* m_llm{};

    QString m_modelFolder{QStringLiteral("__global__")};
    QString m_modelDir;
    int m_requestSeq{0};
    int m_generationSeq{0};
    QString m_generationFolder;
    QString m_assistantDraft;
};
