#include "ai/LocalLlmClient.hpp"
#include "common/Utils.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QTimer>
#include <QtGlobal>
#include <QThread>
#include <QRegularExpression>
#include <QUuid>

namespace {

bool isValidUtf8Bytes(const QByteArray& bytes, bool allowTrailingIncomplete)
{
    const int n = bytes.size();
    int i = 0;
    while (i < n)
    {
        const unsigned char c = static_cast<unsigned char>(bytes.at(i));
        if (c <= 0x7F)
        {
            ++i;
            continue;
        }

        int need = 0;
        if (c >= 0xC2 && c <= 0xDF) need = 1;
        else if (c >= 0xE0 && c <= 0xEF) need = 2;
        else if (c >= 0xF0 && c <= 0xF4) need = 3;
        else return false;

        if (i + need >= n)
            return allowTrailingIncomplete;

        for (int j = 1; j <= need; ++j)
        {
            const unsigned char cc = static_cast<unsigned char>(bytes.at(i + j));
            if ((cc & 0xC0) != 0x80)
                return false;
        }

        i += (need + 1);
    }
    return true;
}

QString decodeProcessText(const QByteArray& bytes, bool allowTrailingIncomplete)
{
    if (bytes.isEmpty())
        return {};

    if (!allowTrailingIncomplete)
        return QString::fromUtf8(bytes);

    int cut = bytes.size();
    while (cut > 0)
    {
        const QByteArray prefix = bytes.left(cut);
        if (isValidUtf8Bytes(prefix, false))
            return QString::fromUtf8(prefix);
        --cut;
    }
    return {};
}

QString sanitizeLlamaCliOutput(QString text, const QString& promptEcho)
{
    text.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));

    const int timingsPos = text.indexOf(QStringLiteral("llama_print_timings"));
    if (timingsPos >= 0)
        text = text.left(timingsPos);

    const int promptStatsPos = text.indexOf(QStringLiteral("[ Prompt:"));
    if (promptStatsPos >= 0)
        text = text.left(promptStatsPos);

    QStringList outLines;
    const QStringList lines = text.split('\n');
    outLines.reserve(lines.size());

    auto isAsciiArtLine = [](const QString& t) -> bool {
        if (t.isEmpty()) return false;
        int strong = 0;
        for (const QChar ch : t)
        {
            const ushort u = ch.unicode();
            if (ch.isSpace())
                continue;
            if ((u >= 0x2500 && u <= 0x259F) || ch == QChar('?'))
            {
                ++strong;
                continue;
            }
            return false;
        }
        return strong > 0;
    };

    bool skippingHeader = true;
    for (const QString& rawLine : lines)
    {
        QString line = rawLine;
        if (line.endsWith('\r'))
            line.chop(1);

        const QString t = line.trimmed();

        if (t.isEmpty() && skippingHeader)
            continue;

        const auto stripRolePrefix = [](const QString& in)->QString {
            const QString s = in.trimmed();
            const QString lower = s.toLower();
            auto cutAfter = [&](const QString& prefix)->QString {
                QString out = s.mid(prefix.size()).trimmed();
                return out;
            };
            if (lower.startsWith(QStringLiteral("assistant:")))
                return cutAfter(QStringLiteral("assistant:"));
            if (lower.startsWith(QStringLiteral("assistant :")))
                return cutAfter(QStringLiteral("assistant :"));
            return in;
        };

        if (t.compare(QStringLiteral("Exiting..."), Qt::CaseInsensitive) == 0
            || t.compare(QStringLiteral("Exiting.."), Qt::CaseInsensitive) == 0
            || t.compare(QStringLiteral("Exiting."), Qt::CaseInsensitive) == 0
            || t.compare(QStringLiteral("Exiting"), Qt::CaseInsensitive) == 0)
        {
            continue;
        }

        if (!promptEcho.isEmpty()
            && (t == (QStringLiteral("> ") + promptEcho) || t == (QStringLiteral(">") + promptEcho)))
        {
            continue;
        }

        if (t.startsWith(QStringLiteral("Loading model"), Qt::CaseInsensitive)
            || t.startsWith(QStringLiteral("build"), Qt::CaseInsensitive)
            || t.startsWith(QStringLiteral("model"), Qt::CaseInsensitive)
            || t.startsWith(QStringLiteral("modalities"), Qt::CaseInsensitive)
            || t.startsWith(QStringLiteral("using custom"), Qt::CaseInsensitive)
            || t.startsWith(QStringLiteral("available commands"), Qt::CaseInsensitive))
        {
            continue;
        }
        if (t.startsWith(QStringLiteral("/exit"))
            || t.startsWith(QStringLiteral("/regen"))
            || t.startsWith(QStringLiteral("/clear"))
            || t.startsWith(QStringLiteral("/read")))
        {
            continue;
        }

        if (skippingHeader)
        {
            if (t.startsWith(QStringLiteral("System:"), Qt::CaseInsensitive)
                || t.startsWith(QStringLiteral("User:"), Qt::CaseInsensitive)
                || t.startsWith(QStringLiteral("Assistant:"), Qt::CaseInsensitive))
            {
                skippingHeader = false;
                continue;
            }

            if (t.startsWith(QStringLiteral("Loading model"), Qt::CaseInsensitive)
                || t.startsWith(QStringLiteral("build"), Qt::CaseInsensitive)
                || t.startsWith(QStringLiteral("model"), Qt::CaseInsensitive)
                || t.startsWith(QStringLiteral("modalities"), Qt::CaseInsensitive)
                || t.startsWith(QStringLiteral("available commands"), Qt::CaseInsensitive))
            {
                continue;
            }
            if (t.startsWith(QStringLiteral("/exit"))
                || t.startsWith(QStringLiteral("/regen"))
                || t.startsWith(QStringLiteral("/clear"))
                || t.startsWith(QStringLiteral("/read")))
            {
                continue;
            }
            if (isAsciiArtLine(t)) continue;
        }

        if (t.startsWith(QStringLiteral("System:"), Qt::CaseInsensitive)
            || t.startsWith(QStringLiteral("User:"), Qt::CaseInsensitive))
        {
            continue;
        }

        if (t == QStringLiteral(">") || t == QStringLiteral("> "))
            continue;

        const QString stripped = stripRolePrefix(line);
        const QString strippedTrimmed = stripped.trimmed();
        if (strippedTrimmed == QStringLiteral("Assistant:") || strippedTrimmed.isEmpty())
        {
            skippingHeader = false;
            continue;
        }

        outLines.push_back(stripped);
        skippingHeader = false;
    }

    QString out = outLines.join('\n').trimmed();
    return out;
}

QString stripAnsiEscapeCodes(QString s)
{
    static const QRegularExpression ansi(QStringLiteral(R"(\x1B\[[0-?]*[ -/]*[@-~])"));
    s.remove(ansi);
    return s;
}

QString writeUtf8TempFile(const QString& stem, const QByteArray& bytes)
{
    const QString name = stem + QStringLiteral("_") + QUuid::createUuid().toString(QUuid::WithoutBraces) + QStringLiteral(".txt");
    const QString path = QDir(QDir::tempPath()).filePath(name);
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return {};
    f.write(bytes);
    f.close();
    return path;
}

void removeTempFile(const QString& path)
{
    if (path.isEmpty()) return;
    QFile::remove(path);
}

} // namespace

LocalLlmClient::LocalLlmClient(QObject* parent)
    : QObject(parent)
{
}

void LocalLlmClient::setPreferredStyle(const QString& style)
{
    m_style = style.trimmed();
}

QString LocalLlmClient::preferredStyle() const
{
    return m_style;
}

void LocalLlmClient::setModelSize(const QString& size)
{
    m_modelSize = size.trimmed();
}

QString LocalLlmClient::modelSize() const
{
    return m_modelSize;
}

void LocalLlmClient::setSystemPrompt(const QString& prompt)
{
    m_systemPrompt = prompt;
}

QString LocalLlmClient::systemPrompt() const
{
    return m_systemPrompt;
}

bool LocalLlmClient::isRunning() const
{
    return m_proc && m_proc->state() != QProcess::NotRunning;
}

void LocalLlmClient::abort()
{
    if (!m_proc)
        return;
    if (!m_generating)
        return;
    m_abortRequested = true;
    if (m_proc->state() != QProcess::NotRunning)
        m_proc->kill();
}

void LocalLlmClient::shutdown()
{
    if (!m_proc)
        return;
    m_abortRequested = true;
    m_proc->kill();
}

void LocalLlmClient::warmUp(int maxTokens)
{
    Q_UNUSED(maxTokens);
}

void LocalLlmClient::generate(const QString& prompt, int maxTokens)
{
    const QString trimmed = prompt.trimmed();
    const QString fallback = QStringLiteral("（离线本地）%1").arg(trimmed);
    if (trimmed.isEmpty())
    {
        emit finished(fallback);
        return;
    }

    if (m_proc && m_proc->state() != QProcess::NotRunning)
        abort();

    const QString runner = resolveRunnerPath();
    const QString model = resolveModelPath();
    const QString sys = m_systemPrompt.trimmed();
    const int n = qMax(1, maxTokens);
    if (runner.isEmpty() || model.isEmpty())
    {
        emit finished(fallback);
        return;
    }

    m_promptFilePath = writeUtf8TempFile(QStringLiteral("amaigirl_prompt"), trimmed.toUtf8());
    m_systemPromptFilePath = sys.isEmpty() ? QString() : writeUtf8TempFile(QStringLiteral("amaigirl_system"), sys.toUtf8());
    const QString promptFile = m_promptFilePath;
    const QString systemFile = m_systemPromptFilePath;

    m_abortRequested = false;
    m_generating = true;
    m_turnBytes.clear();
    m_lastCleanStream.clear();

    QStringList args;
    args << QStringLiteral("-m") << model;
    args << QStringLiteral("-n") << QString::number(n);
    args << QStringLiteral("--conversation");
    args << QStringLiteral("--single-turn");
    args << QStringLiteral("--simple-io");
    args << QStringLiteral("--no-display-prompt");
    args << QStringLiteral("--no-show-timings");
    args << QStringLiteral("--log-disable");
    if (!systemFile.isEmpty())
        args << QStringLiteral("--system-prompt-file") << systemFile;
    if (!promptFile.isEmpty())
        args << QStringLiteral("-f") << promptFile;
    else
        args << QStringLiteral("-p") << trimmed;
#if defined(Q_OS_WIN)
    args << QStringLiteral("-t") << QString::number(QThread::idealThreadCount());
#endif

    m_proc = new QProcess(this);
    QProcess* proc = m_proc;
    m_proc->setProgram(runner);
    m_proc->setArguments(args);
    m_proc->setProcessChannelMode(QProcess::MergedChannels);

    const QString promptEcho = trimmed;
    connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc, promptEcho]{
        if (proc != m_proc) return;
        const QByteArray chunk = proc->readAllStandardOutput();
        if (chunk.isEmpty()) return;

        m_turnBytes += chunk;
        const QString turnText = stripAnsiEscapeCodes(decodeProcessText(m_turnBytes, /*allowTrailingIncomplete*/true));
        const QString cleaned = sanitizeLlamaCliOutput(turnText, promptEcho);
        if (cleaned.size() > m_lastCleanStream.size())
        {
            const QString delta = cleaned.mid(m_lastCleanStream.size());
            m_lastCleanStream = cleaned;
            if (!delta.isEmpty())
                emit tokenReceived(delta);
        }
    });

    connect(proc, &QProcess::errorOccurred, this, [this, proc](QProcess::ProcessError){
        if (proc != m_proc) return;
        if (m_abortRequested) return;
        emit failed(proc->errorString());
    });

    connect(proc, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this, proc, fallback, promptFile, systemFile, promptEcho](int, QProcess::ExitStatus){
        if (proc != m_proc)
        {
            proc->deleteLater();
            removeTempFile(promptFile);
            removeTempFile(systemFile);
            return;
        }
        m_proc = nullptr;
        proc->deleteLater();

        if (m_promptFilePath == promptFile) m_promptFilePath.clear();
        if (m_systemPromptFilePath == systemFile) m_systemPromptFilePath.clear();
        removeTempFile(promptFile);
        removeTempFile(systemFile);

        const bool wasAborted = m_abortRequested;
        m_abortRequested = false;

        const QString finalTurn = stripAnsiEscapeCodes(decodeProcessText(m_turnBytes, /*allowTrailingIncomplete*/false));
        QString finalText = sanitizeLlamaCliOutput(finalTurn, promptEcho);
        m_turnBytes.clear();
        m_lastCleanStream.clear();
        m_generating = false;

        if (wasAborted)
        {
            emit aborted();
            return;
        }

        if (finalText.trimmed().isEmpty())
            finalText = fallback;
        emit finished(finalText);
    });

    m_proc->start();
}

void LocalLlmClient::clearConversation()
{
}

QString LocalLlmClient::resolveRunnerPath() const
{
    const QByteArray env = qgetenv("LLAMA_RUNNER");
    if (!env.isEmpty()) {
        const QString p = QString::fromLocal8Bit(env);
        if (QFileInfo::exists(p)) return p;
    }

#if defined(Q_OS_WIN)
    {
        const QDir d(appResourcePath(QStringLiteral("bin")));
        const QString cli = d.filePath(QStringLiteral("llama-cli.exe"));
        if (QFileInfo::exists(cli)) return cli;
        const QString legacy = d.filePath(QStringLiteral("llama.exe"));
        if (QFileInfo::exists(legacy)) return legacy;
    }
#else
    {
        const QDir d(appResourcePath(QStringLiteral("bin")));
        const QString cli = d.filePath(QStringLiteral("llama-cli"));
        if (QFileInfo::exists(cli)) return cli;
        const QString legacy = d.filePath(QStringLiteral("llama"));
        if (QFileInfo::exists(legacy)) return legacy;
    }
#endif

    return {};
}

QString LocalLlmClient::resolveModelPath() const
{
    {
        const QByteArray envModel = qgetenv("LLM_MODEL");
        if (!envModel.isEmpty()) {
            const QString p = QString::fromLocal8Bit(envModel);
            if (QFileInfo::exists(p) && QFileInfo(p).isFile()) return p;
        }
    }
    auto chooseFromDir = [&](const QDir& d)->QString {
        if (!d.exists()) return {};
        const QStringList ggufs = d.entryList(QStringList{QStringLiteral("*.gguf")}, QDir::Files);
        if (ggufs.isEmpty()) return {};
        const QString s = m_style.toLower();
        auto matchFirst = [&](const QStringList& patterns)->QString {
            for (const auto& pat : patterns) {
                for (const auto& f : ggufs) {
                    if (f.toLower().contains(pat)) {
                        return d.filePath(f);
                    }
                }
            }
            return QString{};
        };
        if (s == QStringLiteral("anime")) {
            QString hit = matchFirst({QStringLiteral("anime"), QStringLiteral("anime.q")});
            if (!hit.isEmpty()) return hit;
        } else if (s == QStringLiteral("original")) {
            QString hit = matchFirst({QStringLiteral("original"), QStringLiteral("llama")});
            if (!hit.isEmpty()) return hit;
        } else if (s == QStringLiteral("universal")) {
            QString hit = matchFirst({QStringLiteral("universal")});
            if (!hit.isEmpty()) return hit;
        }
        return d.filePath(ggufs.front());
    };

    {
        const QString llmDir = appResourcePath(QStringLiteral("llm"));
        QDir d(llmDir);
        const QString sz = m_modelSize.trimmed();
        if (sz.compare(QStringLiteral("7B"), Qt::CaseInsensitive) == 0 && d.exists(QStringLiteral("7B")))
            d = QDir(d.filePath(QStringLiteral("7B")));
        else if (d.exists(QStringLiteral("1.5B")))
            d = QDir(d.filePath(QStringLiteral("1.5B")));

        const QString hit = chooseFromDir(d);
        if (!hit.isEmpty()) return hit;
    }
    {
        const QString alt = QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("../llm/checkpoints"));
        QDir d(alt);
        const QString sz = m_modelSize.trimmed();
        if (sz.compare(QStringLiteral("7B"), Qt::CaseInsensitive) == 0 && d.exists(QStringLiteral("7B")))
            d = QDir(d.filePath(QStringLiteral("7B")));
        else if (d.exists(QStringLiteral("1.5B")))
            d = QDir(d.filePath(QStringLiteral("1.5B")));

        const QString hit = chooseFromDir(d);
        if (!hit.isEmpty()) return hit;
    }
    return {};
}

QString LocalLlmClient::generateSync(const QString& prompt, int maxTokens) const
{
    const QString runner = resolveRunnerPath();
    const QString model  = resolveModelPath();
    if (runner.isEmpty() || model.isEmpty()) {
        return QStringLiteral("（离线本地）%1").arg(prompt.trimmed());
    }

    const QString sys = m_systemPrompt.trimmed();
    const QString promptFile = writeUtf8TempFile(QStringLiteral("amaigirl_prompt_sync"), prompt.toUtf8());
    const QString systemFile = sys.isEmpty() ? QString() : writeUtf8TempFile(QStringLiteral("amaigirl_system_sync"), sys.toUtf8());

    QStringList args;
    args << QStringLiteral("-m") << model;
    args << QStringLiteral("-n") << QString::number(qMax(1, maxTokens));
    args << QStringLiteral("--conversation");
    args << QStringLiteral("--single-turn");
    args << QStringLiteral("--simple-io");
    args << QStringLiteral("--no-display-prompt");
    args << QStringLiteral("--no-show-timings");
    args << QStringLiteral("--log-disable");
    if (!systemFile.isEmpty())
        args << QStringLiteral("--system-prompt-file") << systemFile;
    if (!promptFile.isEmpty())
        args << QStringLiteral("-f") << promptFile;
    else
        args << QStringLiteral("-p") << prompt;
    // Try to keep latency low; optional flags are ignored if runner doesn't support them
#if defined(Q_OS_WIN)
    args << QStringLiteral("-t") << QString::number(QThread::idealThreadCount());
#endif

    QProcess proc;
    proc.setProgram(runner);
    proc.setArguments(args);
    proc.setProcessChannelMode(QProcess::MergedChannels);
    proc.start();
    if (!proc.waitForFinished(30000)) {
        proc.kill();
        return QStringLiteral("（离线本地）%1").arg(prompt.trimmed());
    }
    const QByteArray out = proc.readAllStandardOutput();
    removeTempFile(promptFile);
    removeTempFile(systemFile);
    QByteArray bytes = out;
    const QByteArray promptBytes = prompt.trimmed().toUtf8();
    if (!promptBytes.isEmpty() && bytes.startsWith(promptBytes))
        bytes = bytes.mid(promptBytes.size());

    const QString promptEcho = prompt.trimmed();
    QString text = sanitizeLlamaCliOutput(stripAnsiEscapeCodes(decodeProcessText(bytes, /*allowTrailingIncomplete*/false)), promptEcho);
    if (text.isEmpty()) {
        text = QStringLiteral("（离线本地）%1").arg(prompt.trimmed());
    }
    return text;
}
