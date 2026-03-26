#pragma once

#include <QObject>
#include <QString>

class QProcess;

class LocalLlmClient : public QObject
{
    Q_OBJECT
public:
    explicit LocalLlmClient(QObject* parent = nullptr);

    void setPreferredStyle(const QString& style);
    QString preferredStyle() const;

    void setModelSize(const QString& size);
    QString modelSize() const;

    void setSystemPrompt(const QString& prompt);
    QString systemPrompt() const;

    bool isRunning() const;
    void abort();
    void shutdown();

    void warmUp(int maxTokens = 128);
    void generate(const QString& prompt, int maxTokens = 128);

    void clearConversation();

    QString generateSync(const QString& prompt, int maxTokens = 128) const;

Q_SIGNALS:
    void tokenReceived(const QString& token);
    void finished(const QString& fullText);
    void failed(const QString& message);
    void aborted();

private:
    QString resolveRunnerPath() const;
    QString resolveModelPath() const;

    QString m_style;
    QString m_modelSize;
    QString m_systemPrompt;
    QProcess* m_proc{};
    bool m_abortRequested{false};
    bool m_generating{false};
    QByteArray m_turnBytes;
    QString m_lastCleanStream;
    QString m_promptFilePath;
    QString m_systemPromptFilePath;
};
