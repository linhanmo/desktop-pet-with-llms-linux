#include "ai/ChatController.hpp"

#include "common/SettingsManager.hpp"
#include "ui/ChatWindow.hpp"
#include "engine/Renderer.hpp"
#include "ai/LocalLlmClient.hpp"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QTimer>

#include <algorithm>

namespace {
QJsonArray loadMessages(const QString& modelFolder)
{
    QFile f(SettingsManager::instance().chatPathForModel(modelFolder));
    if (!f.exists()) return {};
    if (!f.open(QIODevice::ReadOnly)) return {};
    QJsonParseError e;
    const auto doc = QJsonDocument::fromJson(f.readAll(), &e);
    if (e.error != QJsonParseError::NoError || !doc.isObject()) return {};
    return doc.object().value("messages").toArray();
}

void saveMessages(const QString& modelFolder, const QJsonArray& messages)
{
    QDir dir(SettingsManager::instance().chatsDir());
    if (!dir.exists()) dir.mkpath(".");

    QJsonObject o;
    o["messages"] = messages;

    QFile f(SettingsManager::instance().chatPathForModel(modelFolder));
    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        f.write(QJsonDocument(o).toJson(QJsonDocument::Indented));
    }
}

QString buildBaseSystemPrompt()
{
    const QString name = SettingsManager::instance().characterName().trimmed();
    QString systemPrompt = SettingsManager::instance().aiSystemPrompt();
    systemPrompt.replace(QStringLiteral("$name$"), name.isEmpty() ? QStringLiteral("小墨") : name);
    return systemPrompt.trimmed();
}

QString buildPromptFromHistory(const QString& modelFolder)
{
    const QJsonArray all = loadMessages(modelFolder);
    QJsonArray filtered;
    for (const auto& v : all)
    {
        if (!v.isObject()) continue;
        const QJsonObject o = v.toObject();
        const QString role = o.value("role").toString();
        if (role != QStringLiteral("user") && role != QStringLiteral("assistant")) continue;
        const QString content = o.value("content").toString().trimmed();
        if (content.isEmpty()) continue;
        filtered.append(QJsonObject{{QStringLiteral("role"), role}, {QStringLiteral("content"), content}});
    }

    const int historyCount = qMax(0, SettingsManager::instance().chatContextMessages());
    const int total = filtered.size();
    if (total <= 0)
        return {};

    const int window = qMin(total, qMax(1, historyCount + 1));
    const int start = qMax(0, total - window);

    QString prompt;
    for (int i = start; i < total; ++i)
    {
        const QJsonObject o = filtered.at(i).toObject();
        const QString role = o.value("role").toString();
        const QString content = o.value("content").toString();
        if (role == QStringLiteral("user"))
            prompt += QStringLiteral("User: ") + content + QStringLiteral("\n");
        else
            prompt += QStringLiteral("Assistant: ") + content + QStringLiteral("\n");
    }

    prompt += QStringLiteral("Assistant:");
    return prompt;
}

struct Live2DDirectives {
    QString cleanedText;
    QString motionGroup;
    QString expressionName;
};

Live2DDirectives parseLive2DDirectives(const QString& text)
{
    Live2DDirectives out;
    out.cleanedText = text;

    const QList<QRegularExpression> motionPatterns{
        QRegularExpression(QStringLiteral(R"(\[\[\s*(?:motion|动作)\s*:\s*([^\]\n]+?)\s*\]\])")),
        QRegularExpression(QStringLiteral(R"(<\s*motion\s*=\s*([^>\n]+?)\s*>)")),
    };

    const QList<QRegularExpression> exprPatterns{
        QRegularExpression(QStringLiteral(R"(\[\[\s*(?:expression|expr|表情)\s*:\s*([^\]\n]+?)\s*\]\])")),
        QRegularExpression(QStringLiteral(R"(<\s*(?:expression|expr)\s*=\s*([^>\n]+?)\s*>)")),
    };

    auto extractLast = [&](const QList<QRegularExpression>& patterns, QString* outValue) {
        for (const auto& re : patterns)
        {
            QRegularExpressionMatchIterator it = re.globalMatch(out.cleanedText);
            while (it.hasNext())
            {
                const auto m = it.next();
                const QString v = m.captured(1).trimmed();
                if (!v.isEmpty())
                    *outValue = v;
            }
            out.cleanedText.remove(re);
        }
    };

    extractLast(motionPatterns, &out.motionGroup);
    extractLast(exprPatterns, &out.expressionName);

    out.cleanedText = out.cleanedText.trimmed();
    return out;
}
} // namespace

ChatController::ChatController(QObject* parent)
    : QObject(parent)
{
    m_llm = new LocalLlmClient(this);
    setLlmStyle(SettingsManager::instance().llmStyle());

    connect(m_llm, &LocalLlmClient::tokenReceived, this, [this](const QString& token){
        if (!m_chatWindow) return;
        if (m_generationSeq != m_requestSeq) return;
        if (m_generationFolder != m_modelFolder) return;
        m_chatWindow->appendAiToken(token);
        m_assistantDraft += token;
        emit assistantBubbleTextChanged(m_assistantDraft, false);
    });

    connect(m_llm, &LocalLlmClient::finished, this, [this](const QString& fullText){
        if (!m_chatWindow) return;
        if (m_generationSeq != m_requestSeq) return;
        if (m_generationFolder != m_modelFolder) return;

        const auto d = parseLive2DDirectives(fullText);
        if (m_renderer)
        {
            if (!d.expressionName.isEmpty())
                m_renderer->setExpressionName(d.expressionName);
            if (!d.motionGroup.isEmpty())
                m_renderer->setMotionGroup(d.motionGroup);
        }
        m_assistantDraft = d.cleanedText;
        emit assistantBubbleTextChanged(m_assistantDraft, true);
        m_chatWindow->finalizeAssistantMessage(d.cleanedText, true);
        m_chatWindow->setBusy(false);
    });

    connect(m_llm, &LocalLlmClient::aborted, this, [this]{
        if (!m_chatWindow) return;
        m_chatWindow->cancelAssistantDraft();
        m_chatWindow->setBusy(false);
        m_assistantDraft.clear();
        emit assistantBubbleTextChanged(QString(), true);
    });

    connect(m_llm, &LocalLlmClient::failed, this, [this]{
        if (!m_chatWindow) return;
        m_chatWindow->cancelAssistantDraft();
        m_chatWindow->setBusy(false);
        m_assistantDraft.clear();
        emit assistantBubbleTextChanged(QString(), true);
    });
}

void ChatController::setChatWindow(ChatWindow* wnd)
{
    if (m_chatWindow == wnd)
        return;

    if (m_chatWindow)
    {
        disconnect(m_chatWindow, nullptr, this, nullptr);
    }

    m_chatWindow = wnd;

    if (!m_chatWindow)
        return;

    connect(m_chatWindow, &ChatWindow::requestSendMessage, this, &ChatController::onSendRequested);
    connect(m_chatWindow, &ChatWindow::requestClearChat, this, &ChatController::onClearRequested);
    connect(m_chatWindow, &ChatWindow::requestLlmStyleChanged, this, &ChatController::setLlmStyle);
    connect(m_chatWindow, &ChatWindow::requestLlmModelSizeChanged, this, &ChatController::setLlmModelSize);

    if (m_modelFolder.trimmed().isEmpty())
        m_modelFolder = QStringLiteral("__global__");

    if (m_modelDir.trimmed().isEmpty())
    {
        const QString folder = SettingsManager::instance().selectedModelFolder();
        m_modelDir = QDir(SettingsManager::instance().modelsRoot()).filePath(folder);
    }

    m_chatWindow->setCurrentModel(m_modelFolder, m_modelDir);
    m_chatWindow->setLlmStyle(SettingsManager::instance().llmStyle());
    m_chatWindow->setLlmModelSize(SettingsManager::instance().llmModelSize());
}

void ChatController::setRenderer(Renderer* renderer)
{
    m_renderer = renderer;
}

void ChatController::applyPreferredAudioOutput()
{
}

void ChatController::setLlmStyle(const QString& style)
{
    SettingsManager::instance().setLlmStyle(style);
    if (m_llm) m_llm->setPreferredStyle(SettingsManager::instance().llmStyle());
    if (m_chatWindow) m_chatWindow->setLlmStyle(SettingsManager::instance().llmStyle());
    if (m_llm && m_llm->isRunning())
        m_llm->shutdown();
}

void ChatController::setLlmModelSize(const QString& size)
{
    SettingsManager::instance().setLlmModelSize(size);
    if (m_llm) m_llm->setModelSize(SettingsManager::instance().llmModelSize());
    if (m_chatWindow) m_chatWindow->setLlmModelSize(SettingsManager::instance().llmModelSize());
    if (m_llm && m_llm->isRunning())
        m_llm->shutdown();
    if (!m_modelFolder.isEmpty())
        onModelChanged(m_modelFolder, m_modelDir);
}

void ChatController::postLocalAssistantMessage(const QString& text,
                                              const QString& motionGroup,
                                              const QString& expressionName,
                                              bool writeToHistory)
{
    const QString t = text.trimmed();
    if (t.isEmpty())
        return;

    if (m_renderer)
    {
        if (!expressionName.trimmed().isEmpty())
            m_renderer->setExpressionName(expressionName.trimmed());
        if (!motionGroup.trimmed().isEmpty())
            m_renderer->setMotionGroup(motionGroup.trimmed());
    }

    m_assistantDraft = t;
    emit assistantBubbleTextChanged(t, true);

    const QString folder = !m_modelFolder.isEmpty()
                               ? m_modelFolder
                               : SettingsManager::instance().selectedModelFolder();
    if (!writeToHistory || folder.isEmpty())
        return;

    QJsonArray messages = loadMessages(folder);
    QJsonObject o;
    o["role"] = QStringLiteral("assistant");
    o["content"] = t;
    messages.append(o);
    if (messages.size() > 200)
    {
        const int drop = messages.size() - 200;
        for (int i = 0; i < drop && !messages.isEmpty(); ++i)
            messages.removeAt(0);
    }
    saveMessages(folder, messages);

    if (m_chatWindow && folder == m_modelFolder)
        m_chatWindow->loadFromDisk(folder);
}

void ChatController::triggerLocalPrompt(const QString& userText,
                                       const QString& motionGroup,
                                       const QString& expressionName)
{
    if (!m_chatWindow) return;

    const QString t = userText.trimmed();
    if (t.isEmpty())
        return;

    if (m_llm && m_llm->isRunning())
        m_llm->abort();

    if (m_renderer)
    {
        if (!expressionName.trimmed().isEmpty())
            m_renderer->setExpressionName(expressionName.trimmed());
        if (!motionGroup.trimmed().isEmpty())
            m_renderer->setMotionGroup(motionGroup.trimmed());
    }

    const QString folder = !m_modelFolder.isEmpty()
                               ? m_modelFolder
                               : SettingsManager::instance().selectedModelFolder();
    if (folder.isEmpty())
        return;

    onSendRequested(folder, t);
}

void ChatController::onSendRequested(const QString& modelFolder, const QString& userText)
{
    if (!m_chatWindow) return;
    if (modelFolder.isEmpty() || userText.trimmed().isEmpty()) return;

    m_modelFolder = modelFolder;
    m_assistantDraft.clear();
    emit assistantBubbleTextChanged(QString(), false);

    if (m_llm && !m_llm->isRunning())
        m_llm->setSystemPrompt(buildBaseSystemPrompt());

    m_chatWindow->appendUserMessage(userText);

    const QString prompt = buildPromptFromHistory(modelFolder);

    m_chatWindow->appendAiMessageStart();
    m_chatWindow->setBusy(true);

    m_generationFolder = modelFolder;
    m_generationSeq = ++m_requestSeq;
    if (m_llm)
    {
        m_llm->setPreferredStyle(SettingsManager::instance().llmStyle());
        m_llm->setModelSize(SettingsManager::instance().llmModelSize());
        m_llm->generate(prompt.isEmpty() ? userText.trimmed() : prompt, SettingsManager::instance().llmMaxTokens());
    }
}

void ChatController::onClearRequested(const QString& modelFolder)
{
    if (!m_chatWindow) return;
    if (modelFolder.isEmpty()) return;

    if (m_llm)
    {
        if (m_llm->isRunning())
            m_llm->shutdown();
    }
    m_chatWindow->cancelAssistantDraft();

    saveClearedChat(modelFolder);

    m_chatWindow->loadFromDisk(modelFolder);
    m_chatWindow->setBusy(false);

    if (m_llm)
    {
        m_llm->setSystemPrompt(buildBaseSystemPrompt());
        m_llm->setModelSize(SettingsManager::instance().llmModelSize());
        m_llm->warmUp(SettingsManager::instance().llmMaxTokens());
    }
}

void ChatController::saveClearedChat(const QString& modelFolder) const
{
    saveMessages(modelFolder, QJsonArray{});
}

void ChatController::reservedStartTtsPlayback(const QString& audioPath)
{
    Q_UNUSED(audioPath);
}

void ChatController::reservedUpdateLipSync(const QString& audioPath)
{
    Q_UNUSED(audioPath);
}



void ChatController::onModelChanged(const QString& modelFolder, const QString& modelDir)
{
    m_modelDir = modelDir;
    if (m_chatWindow)
    {
        m_chatWindow->setCurrentModel(m_modelFolder, modelDir);
    }

    if (m_llm && !modelFolder.trimmed().isEmpty())
    {
        m_llm->setPreferredStyle(SettingsManager::instance().llmStyle());
        m_llm->setModelSize(SettingsManager::instance().llmModelSize());
        m_llm->setSystemPrompt(buildBaseSystemPrompt());
        m_llm->warmUp(SettingsManager::instance().llmMaxTokens());
    }
}
