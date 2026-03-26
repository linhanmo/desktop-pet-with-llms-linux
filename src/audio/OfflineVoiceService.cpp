#include "audio/OfflineVoiceService.hpp"

#include "common/SettingsManager.hpp"

#include <QProcess>
#include <QTimer>
#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>
#include <QProcessEnvironment>
#include <QMessageBox>
#include <QDesktopServices>
#include <QUrl>
#include <QPushButton>
#include <QCoreApplication>
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
#include <QMediaPlayer>
#include <QAudioOutput>
#include <QDateTime>
#endif
#include <memory>
#include <functional>

#if defined(Q_OS_WIN32)
#include <windows.h>
#include <mmdeviceapi.h>
#include <audiopolicy.h>
#include <endpointvolume.h>
#endif

static QString normalizeCandidateText(QString s)
{
    s = s.trimmed();
    if (s.isEmpty()) return {};

    static const QRegularExpression re(QStringLiteral(
        R"(^\s*(?:result|text|final|partial)\s*[:：]\s*(.+?)\s*$)"
    ), QRegularExpression::CaseInsensitiveOption);
    auto m = re.match(s);
    if (m.hasMatch())
        s = m.captured(1).trimmed();

    if (s.startsWith('"') && s.endsWith('"') && s.size() >= 2)
        s = s.mid(1, s.size() - 2).trimmed();

    return s;
}

OfflineVoiceService::OfflineVoiceService(QObject* parent)
    : QObject(parent)
{
    m_sttIdleTimer = new QTimer(this);
    m_sttIdleTimer->setSingleShot(true);
    connect(m_sttIdleTimer, &QTimer::timeout, this, [this]{
        finalizeStt();
    });
}

OfflineVoiceService::SettingsSnapshot OfflineVoiceService::readSettings() const
{
    SettingsSnapshot s;
    auto& sm = SettingsManager::instance();
    s.sttEnabled = sm.offlineSttEnabled();
    s.ttsEnabled = sm.offlineTtsEnabled();
    s.wakeEnabled = sm.wakeWordEnabled();
    s.binDir = sm.sherpaOnnxBinDir();
    s.wakeWord = sm.wakeWordText();
    s.kwsArgs = sm.sherpaKwsArgs();
    s.sttArgs = sm.sherpaSttArgs();
    s.ttsArgs = sm.sherpaTtsArgs();
    s.ttsVolumePercent = sm.ttsVolumePercent();
    return s;
}

void OfflineVoiceService::applySettingsSnapshot(const SettingsSnapshot& next)
{
    const bool changedBin = (m_settings.binDir != next.binDir);
    const bool changedWake = (m_settings.wakeEnabled != next.wakeEnabled) || (m_settings.wakeWord != next.wakeWord) || (m_settings.kwsArgs != next.kwsArgs);
    const bool changedStt = (m_settings.sttEnabled != next.sttEnabled) || (m_settings.sttArgs != next.sttArgs);
    const bool changedTts = (m_settings.ttsEnabled != next.ttsEnabled) || (m_settings.ttsArgs != next.ttsArgs);

    m_settings = next;

    if (changedBin || changedWake || changedStt)
    {
        if (m_stt)
            stopStt();
        if (m_kws)
            stopKws();
    }
    if (changedBin || changedTts)
        stopTts();
}

void OfflineVoiceService::reloadFromSettings()
{
    applySettingsSnapshot(readSettings());
    start();
}

static bool g_micHintShown = false;
static bool g_speakerHintShown = false;

static void showMicHint(const QString& detail)
{
    if (g_micHintShown) return;
    g_micHintShown = true;
#if defined(Q_OS_MACOS)
    QString tip = QObject::tr("无法使用麦克风。\n\n请在“系统设置→隐私与安全性→麦克风”中授权应用访问麦克风，并确保设备连接正常。");
#elif defined(Q_OS_WIN32)
    QString tip = QObject::tr("无法使用麦克风。\n\n请在“设置→隐私→麦克风”中开启“麦克风访问”，并开启“允许桌面应用访问你的麦克风”。\nWindows 对传统桌面程序通常没有“按应用单独授权”的开关。");
#else
    QString tip = QObject::tr("无法使用麦克风。\n\n请检查桌面环境的音频输入设备设置（PulseAudio/PipeWire），并确保设备可用。");
#endif
    const QString d = detail.trimmed();
    if (!d.isEmpty())
        tip += QStringLiteral("\n\n") + d.left(800);
#if defined(Q_OS_WIN32)
    QMessageBox box(QMessageBox::Warning, QObject::tr("麦克风不可用"), tip, QMessageBox::NoButton, nullptr);
    QAbstractButton* openBtn = box.addButton(QObject::tr("打开设置"), QMessageBox::AcceptRole);
    box.addButton(QObject::tr("关闭"), QMessageBox::RejectRole);
    box.exec();
    if (box.clickedButton() == openBtn)
        QDesktopServices::openUrl(QUrl(QStringLiteral("ms-settings:privacy-microphone")));
#else
    QMessageBox::warning(nullptr, QObject::tr("麦克风不可用"), tip);
#endif
}

static void showSpeakerHint(const QString& detail)
{
    if (g_speakerHintShown) return;
    g_speakerHintShown = true;
#if defined(Q_OS_MACOS)
    QString tip = QObject::tr("无法播放音频输出。\n\n请在“系统设置→声音”中检查输出设备与音量。");
#elif defined(Q_OS_WIN32)
    QString tip = QObject::tr("无法播放音频输出。\n\n请在“设置→系统→声音”中检查输出设备与音量。");
#else
    QString tip = QObject::tr("无法播放音频输出。\n\n请检查桌面环境的音频输出设备设置，并确保音量正常。");
#endif
    const QString d = detail.trimmed();
    if (!d.isEmpty())
        tip += QStringLiteral("\n\n") + d.left(800);
#if defined(Q_OS_WIN32)
    QMessageBox box(QMessageBox::Warning, QObject::tr("音频输出不可用"), tip, QMessageBox::NoButton, nullptr);
    QAbstractButton* openBtn = box.addButton(QObject::tr("打开设置"), QMessageBox::AcceptRole);
    box.addButton(QObject::tr("关闭"), QMessageBox::RejectRole);
    box.exec();
    if (box.clickedButton() == openBtn)
        QDesktopServices::openUrl(QUrl(QStringLiteral("ms-settings:sound")));
#else
    QMessageBox::warning(nullptr, QObject::tr("音频输出不可用"), tip);
#endif
}

#if defined(Q_OS_WIN32)
static bool setAudioSessionVolumeForPid(DWORD pid, float vol01)
{
    if (pid == 0)
        return false;

    vol01 = qBound(0.0f, vol01, 1.0f);

    const HRESULT initHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool needUninit = (initHr == S_OK || initHr == S_FALSE);

    IMMDeviceEnumerator* enumerator = nullptr;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&enumerator));
    if (FAILED(hr) || !enumerator)
    {
        if (needUninit) CoUninitialize();
        return false;
    }

    IMMDevice* device = nullptr;
    hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    enumerator->Release();
    if (FAILED(hr) || !device)
    {
        if (needUninit) CoUninitialize();
        return false;
    }

    IAudioSessionManager2* mgr = nullptr;
    hr = device->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&mgr));
    device->Release();
    if (FAILED(hr) || !mgr)
    {
        if (needUninit) CoUninitialize();
        return false;
    }

    IAudioSessionEnumerator* sessions = nullptr;
    hr = mgr->GetSessionEnumerator(&sessions);
    mgr->Release();
    if (FAILED(hr) || !sessions)
    {
        if (needUninit) CoUninitialize();
        return false;
    }

    int count = 0;
    sessions->GetCount(&count);
    bool applied = false;
    for (int i = 0; i < count; ++i)
    {
        IAudioSessionControl* ctrl = nullptr;
        if (FAILED(sessions->GetSession(i, &ctrl)) || !ctrl)
            continue;

        IAudioSessionControl2* ctrl2 = nullptr;
        HRESULT hr2 = ctrl->QueryInterface(__uuidof(IAudioSessionControl2), reinterpret_cast<void**>(&ctrl2));
        if (FAILED(hr2) || !ctrl2)
        {
            ctrl->Release();
            continue;
        }

        DWORD spid = 0;
        ctrl2->GetProcessId(&spid);
        if (spid != pid)
        {
            ctrl2->Release();
            ctrl->Release();
            continue;
        }

        ISimpleAudioVolume* sav = nullptr;
        hr2 = ctrl2->QueryInterface(__uuidof(ISimpleAudioVolume), reinterpret_cast<void**>(&sav));
        if (SUCCEEDED(hr2) && sav)
        {
            sav->SetMasterVolume(vol01, nullptr);
            sav->Release();
            applied = true;
        }

        ctrl2->Release();
        ctrl->Release();
        if (applied)
            break;
    }

    sessions->Release();
    if (needUninit) CoUninitialize();
    return applied;
}
#endif

void OfflineVoiceService::start()
{
    if (m_settings.wakeEnabled)
        startKws();
}

void OfflineVoiceService::stop()
{
    stopTts();
    stopStt();
    stopKws();
}

void OfflineVoiceService::startListeningOnce()
{
    if (!m_settings.sttEnabled)
        return;
    stopKws();
    startStt();
}

void OfflineVoiceService::cancelListening()
{
    stopStt();
    if (m_settings.wakeEnabled)
        startKws();
}

void OfflineVoiceService::speakText(const QString& text)
{
    if (!m_settings.ttsEnabled)
        return;
    const QString t = text.trimmed();
    if (t.isEmpty())
        return;
    startTts(t);
}

QString OfflineVoiceService::exePath(const QString& baseName) const
{
    const QString dir = m_settings.binDir.trimmed();
    if (dir.isEmpty()) return {};
    const QString exe = baseName.endsWith(QStringLiteral(".exe"), Qt::CaseInsensitive)
                            ? baseName
                            : (baseName + QStringLiteral(".exe"));
    return QDir(dir).filePath(exe);
}

QStringList OfflineVoiceService::splitArgs(const QString& args) const
{
    return QProcess::splitCommand(args);
}

static void applySherpaEnv(QProcess* p, const QString& binDir)
{
    if (!p) return;
    if (binDir.trimmed().isEmpty()) return;

    QDir d(binDir);
    QString libDir;
    if (d.cdUp())
        libDir = d.filePath(QStringLiteral("lib"));

    auto env = QProcessEnvironment::systemEnvironment();
    const QString oldPath = env.value(QStringLiteral("PATH"));
    QStringList parts;
    const QString appDir = QCoreApplication::applicationDirPath();
    if (!appDir.trimmed().isEmpty())
        parts << appDir;
    parts << binDir;
    if (!libDir.isEmpty() && QFileInfo::exists(libDir) && QFileInfo(libDir).isDir())
        parts << libDir;
    parts << oldPath;
    env.insert(QStringLiteral("PATH"), parts.join(QDir::listSeparator()));
    p->setProcessEnvironment(env);
}

void OfflineVoiceService::startKws()
{
    if (m_kws) return;
    const QString program = exePath(QStringLiteral("sherpa-onnx-keyword-spotter-microphone"));
    if (program.isEmpty() || !QFileInfo::exists(program))
        return;

    m_kws = new QProcess(this);
    m_kws->setProgram(program);
    m_kws->setArguments(splitArgs(m_settings.kwsArgs));
    applySherpaEnv(m_kws, m_settings.binDir);
    m_kws->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_kws, &QProcess::readyReadStandardOutput, this, &OfflineVoiceService::onKwsReadyRead);
    connect(m_kws, &QProcess::errorOccurred, this, [this](QProcess::ProcessError){
        if (!m_kws) return;
        const QString out = QString::fromLocal8Bit(m_kws->readAll());
        showMicHint(out);
    });
    connect(m_kws, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this](int exitCode, QProcess::ExitStatus){
        if (!m_kws) return;
        const QString out = QString::fromLocal8Bit(m_kws->readAll());
        if (exitCode != 0)
            showMicHint(out);
        m_kws->deleteLater();
        m_kws = nullptr;
        if (exitCode == -1073741515)
            return;
        if (m_settings.wakeEnabled)
        {
            QTimer::singleShot(1200, this, [this]{ startKws(); });
        }
    });
    m_kws->start();
}

void OfflineVoiceService::stopKws()
{
    if (!m_kws) return;
    m_kws->kill();
    m_kws->deleteLater();
    m_kws = nullptr;
}

void OfflineVoiceService::startStt()
{
    if (m_stt) return;
    const QString program = exePath(QStringLiteral("sherpa-onnx-microphone"));
    if (program.isEmpty() || !QFileInfo::exists(program))
        return;

    m_sttBest.clear();
    m_sttLastPartial.clear();

    m_stt = new QProcess(this);
    m_stt->setProgram(program);
    m_stt->setArguments(splitArgs(m_settings.sttArgs));
    applySherpaEnv(m_stt, m_settings.binDir);
    m_stt->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_stt, &QProcess::readyReadStandardOutput, this, &OfflineVoiceService::onSttReadyRead);
    connect(m_stt, &QProcess::errorOccurred, this, [this](QProcess::ProcessError){
        if (!m_stt) return;
        const QString out = QString::fromLocal8Bit(m_stt->readAll());
        showMicHint(m_stt->errorString() + QStringLiteral("\n") + out);
    });
    connect(m_stt, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this](int exitCode, QProcess::ExitStatus){
        if (!m_stt) return;
        const QString out = QString::fromLocal8Bit(m_stt->readAll());
        if (exitCode != 0)
            showMicHint(out);
        m_stt->deleteLater();
        m_stt = nullptr;
        if (m_settings.wakeEnabled)
            startKws();
    });
    m_stt->start();
}

void OfflineVoiceService::stopStt()
{
    m_sttIdleTimer->stop();
    if (!m_stt) return;
    m_stt->kill();
    m_stt->deleteLater();
    m_stt = nullptr;
}

void OfflineVoiceService::finalizeStt()
{
    if (!m_stt) return;
    const QString finalText = !m_sttBest.trimmed().isEmpty() ? m_sttBest.trimmed() : m_sttLastPartial.trimmed();
    stopStt();
    if (!finalText.isEmpty())
        emit sttFinalText(finalText);
}

void OfflineVoiceService::startTts(const QString& text)
{
    stopTts();
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    const QString programGen = exePath(QStringLiteral("sherpa-onnx-offline-tts"));
    const QString programPlay = exePath(QStringLiteral("sherpa-onnx-offline-tts-play"));
    const bool canGen = !programGen.isEmpty() && QFileInfo::exists(programGen);
    const bool canPlay = !programPlay.isEmpty() && QFileInfo::exists(programPlay);
    const QString program = canGen ? programGen : programPlay;
    if (!canGen && !canPlay)
        return;
#else
    const QString program = exePath(QStringLiteral("sherpa-onnx-offline-tts-play"));
    if (program.isEmpty() || !QFileInfo::exists(program))
        return;
#endif

    QString args = m_settings.ttsArgs;
    QString escaped = text;
    escaped.replace('"', QStringLiteral("\\\""));
    const QString quoted = QStringLiteral("\"") + escaped + QStringLiteral("\"");
    if (args.contains(QStringLiteral("{text}")))
        args.replace(QStringLiteral("{text}"), quoted);
    else
        args = args + QStringLiteral(" ") + quoted;

#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    QString wavPath;
    if (canGen)
    {
        QDir cache(SettingsManager::instance().cacheDir());
        if (!cache.exists()) cache.mkpath(QStringLiteral("."));
        wavPath = cache.filePath(QStringLiteral("tts_%1.wav").arg(QDateTime::currentMSecsSinceEpoch()));
        m_ttsWavPath = wavPath;
        if (!args.contains(QStringLiteral("--output-filename")))
            args += QStringLiteral(" --output-filename \"%1\"").arg(wavPath);
    }
#endif

    m_tts = new QProcess(this);
    m_tts->setProgram(program);
    m_tts->setArguments(splitArgs(args));
    applySherpaEnv(m_tts, m_settings.binDir);
    m_tts->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_tts, &QProcess::errorOccurred, this, [this](QProcess::ProcessError){
        if (!m_tts) return;
        const QString out = QString::fromLocal8Bit(m_tts->readAll());
        showSpeakerHint(m_tts->errorString() + QStringLiteral("\n") + out);
    });
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    const int ttsVolPercent = m_settings.ttsVolumePercent;
    connect(m_tts, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this, wavPath, ttsVolPercent](int exitCode, QProcess::ExitStatus){
        if (!m_tts) return;
        const QString out = QString::fromLocal8Bit(m_tts->readAll());
        if (exitCode != 0)
            showSpeakerHint(out);
        m_tts->deleteLater();
        m_tts = nullptr;

        if (exitCode == 0 && !wavPath.isEmpty() && QFileInfo::exists(wavPath))
        {
            if (!m_ttsAudio)
                m_ttsAudio = new QAudioOutput(this);
            m_ttsAudio->setVolume(qBound(0.0, double(ttsVolPercent) / 100.0, 1.0));

            if (!m_ttsPlayer)
            {
                m_ttsPlayer = new QMediaPlayer(this);
                m_ttsPlayer->setAudioOutput(m_ttsAudio);
            }
            else
            {
                m_ttsPlayer->stop();
            }

            m_ttsPlayer->setSource(QUrl::fromLocalFile(wavPath));
            m_ttsPlayer->play();
        }
    });
#else
    connect(m_tts, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this](int exitCode, QProcess::ExitStatus){
        if (!m_tts) return;
        const QString out = QString::fromLocal8Bit(m_tts->readAll());
        if (exitCode != 0)
            showSpeakerHint(out);
        m_tts->deleteLater();
        m_tts = nullptr;
    });
#endif
    m_tts->start();

#if defined(Q_OS_WIN32)
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    if (!canGen)
    {
#endif
        const float vol01 = qBound(0.0f, float(m_settings.ttsVolumePercent) / 100.0f, 1.0f);
        auto tries = std::make_shared<int>(0);
        auto loop = std::make_shared<std::function<void()>>();
        *loop = [this, vol01, tries, loop]{
            if (!m_tts) return;
            const qint64 pid64 = m_tts->processId();
            if (pid64 <= 0) return;
            if (setAudioSessionVolumeForPid(DWORD(pid64), vol01)) return;
            if (++*tries >= 20) return;
            QTimer::singleShot(120, this, *loop);
        };
        QTimer::singleShot(60, this, *loop);
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    }
#endif
#endif
}

void OfflineVoiceService::stopTts()
{
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    if (m_ttsPlayer)
        m_ttsPlayer->stop();
#endif
    if (!m_tts) return;
    m_tts->kill();
    m_tts->deleteLater();
    m_tts = nullptr;
}

void OfflineVoiceService::onKwsReadyRead()
{
    if (!m_kws) return;
    const QString wake = m_settings.wakeWord.trimmed();
    const QString wakeCompact = QString(wake).remove(QRegularExpression(QStringLiteral("\\s+")));
    const QByteArray data = m_kws->readAllStandardOutput();
    const QString out = QString::fromLocal8Bit(data);
    const QStringList lines = out.split(QRegularExpression(QStringLiteral("[\r\n]+")), Qt::SkipEmptyParts);
    for (QString line : lines)
    {
        line = line.trimmed();
        if (line.isEmpty()) continue;
        const QString lineCompact = QString(line).remove(QRegularExpression(QStringLiteral("\\s+")));
        if (!wakeCompact.isEmpty() && (line.contains(wake, Qt::CaseInsensitive) || lineCompact.contains(wakeCompact, Qt::CaseInsensitive)))
        {
            emit wakeWordDetected();
            startListeningOnce();
            break;
        }
    }
}

void OfflineVoiceService::onSttReadyRead()
{
    if (!m_stt) return;
    const QByteArray data = m_stt->readAllStandardOutput();
    const QString out = QString::fromLocal8Bit(data);
    const QStringList lines = out.split(QRegularExpression(QStringLiteral("[\r\n]+")), Qt::SkipEmptyParts);
    for (QString line : lines)
    {
        line = normalizeCandidateText(line);
        if (line.isEmpty()) continue;
        m_sttLastPartial = line;
        if (line.size() >= m_sttBest.size())
            m_sttBest = line;
        emit sttPartialText(line);
        m_sttIdleTimer->start(1000);
    }
}
