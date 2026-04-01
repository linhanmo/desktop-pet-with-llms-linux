#include "audio/OfflineVoiceService.hpp"

#include "common/SettingsManager.hpp"

#include <QProcess>
#include <QTimer>
#include <QDir>
#include <QFileInfo>
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
static bool g_speakerHintShown = false;

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
    if (d.contains(QStringLiteral("Please provide --matcha-vocoder"), Qt::CaseInsensitive)
        || d.contains(QStringLiteral("offline-tts-vits-model.cc:Init"), Qt::CaseInsensitive)
        || d.contains(QStringLiteral("Errors in config"), Qt::CaseInsensitive))
    {
        tip = QObject::tr("离线语音合成（TTS）失败。\n\n当前选择的模型可能不兼容或缺少必要文件（例如 Matcha 模型缺少 vocoder）。\n请在设置里更换一个可用的 TTS 模型（例如 vits-melo-tts-zh_en）。");
    }
    if (!d.isEmpty())
        tip += QStringLiteral("\n\n") + (d.size() > 1200 ? d.right(1200) : d);

    QMessageBox box(QMessageBox::Warning, QObject::tr("音频输出不可用"), tip, QMessageBox::NoButton, nullptr);
    box.setWindowModality(Qt::ApplicationModal);
    box.setWindowFlag(Qt::WindowStaysOnTopHint, true);
#if defined(Q_OS_WIN32)
    QAbstractButton* openBtn = box.addButton(QObject::tr("打开设置"), QMessageBox::AcceptRole);
    box.addButton(QObject::tr("关闭"), QMessageBox::RejectRole);
    box.exec();
    if (box.clickedButton() == openBtn)
        QDesktopServices::openUrl(QUrl(QStringLiteral("ms-settings:sound")));
#else
    box.addButton(QObject::tr("关闭"), QMessageBox::RejectRole);
    box.exec();
#endif
}

static void applySherpaEnv(QProcess* p, const QString& binDir);

#if 0
struct OfflineVoiceService::Engine {
    explicit Engine(
        QObject* parent,
        std::function<void()> onWake,
        std::function<void(const QString&)> onSttPartial,
        std::function<void(const QString&)> onMicError
    )
        : parent(parent)
        , onWake(std::move(onWake))
        , onSttPartial(std::move(onSttPartial))
        , onMicError(std::move(onMicError))
    {
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
        decodeTimer = new QTimer(parent);
        decodeTimer->setInterval(30);
        decodeTimer->setTimerType(Qt::PreciseTimer);
        QObject::connect(decodeTimer, &QTimer::timeout, parent, [this] { process(); });
#endif
    }

    ~Engine() {
        stopMicIfIdle();
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
        if (decodeTimer) {
            decodeTimer->stop();
            delete decodeTimer;
            decodeTimer = nullptr;
        }
#endif
    }

    void applySettings(const SettingsSnapshot& s)
    {
        settings = s;
    }

    void startKws()
    {
        qDebug() << "[DEBUG] Engine::startKws() start";
        stopAll();
        if (!settings.wakeEnabled) {
            qDebug() << "[DEBUG] Engine::startKws() wakeEnabled is false, returning";
            return;
        }

#if defined(Q_OS_WIN32)
        qDebug() << "[DEBUG] Engine::startKws() parsing args";
        const auto args = parseArgs(settings.kwsArgs, true);
        if (!args) {
            qDebug() << "[DEBUG] Engine::startKws() parseArgs failed or missing files";
            return;
        }

        qDebug() << "[DEBUG] Engine::startKws() creating config";
        QByteArray modelType = args->modelType.toUtf8();
        QByteArray tokens = args->tokens.toUtf8();
        QByteArray encoder = args->encoder.toUtf8();
        QByteArray decoder = args->decoder.toUtf8();
        QByteArray joiner = args->joiner.toUtf8();
        QByteArray provider = args->provider.toUtf8();
        QByteArray keywordsFile = args->keywordsFile.toUtf8();

        SherpaOnnxKeywordSpotterConfig config;
        memset(&config, 0, sizeof(config));
        config.feat_config.sample_rate = 16000;
        config.feat_config.feature_dim = 80;
        config.model_config.model_type = modelType.constData();
        config.model_config.tokens = tokens.constData();
        config.model_config.provider = provider.constData();
        config.model_config.num_threads = args->numThreads;
        config.model_config.debug = 0;
        
        config.model_config.transducer.encoder = encoder.constData();
        config.model_config.transducer.decoder = decoder.constData();
        config.model_config.transducer.joiner = joiner.constData();
        
        config.max_active_paths = 4;
        config.num_trailing_blanks = 1;
        config.keywords_score = 1.0f;
        config.keywords_threshold = 0.25f;
        config.keywords_file = keywordsFile.constData();
        config.keywords_buf_size = 0;

        qDebug() << "[DEBUG] Engine::startKws() KeywordSpotter::Create...";
        const SherpaOnnxKeywordSpotter* p = SherpaOnnxCreateKeywordSpotter(&config);
        if (!p) {
            qDebug() << "[DEBUG] Engine::startKws() failed to create KeywordSpotter";
            return;
        }
        kws.reset(p);

        qDebug() << "[DEBUG] Engine::startKws() CreateStream...";
        const SherpaOnnxOnlineStream* s = SherpaOnnxCreateKeywordStream(kws.get());
        if (!s) {
            qDebug() << "[DEBUG] Engine::startKws() failed to create stream";
            return;
        }
        kwsStream.reset(s);
        mode = Mode::Wake;
        
        qDebug() << "[DEBUG] Engine::startKws() startMicIfNeeded...";
        startMicIfNeeded();
        qDebug() << "[DEBUG] Engine::startKws() done";
#else
        if (kwsProc)
            return;
        const QString program = exePath(QStringLiteral("sherpa-onnx-keyword-spotter-microphone"));
        if (program.isEmpty() || !QFileInfo::exists(program))
            return;

        kwsProc = new QProcess(parent);
        kwsProc->setProgram(program);
        kwsProc->setArguments(QProcess::splitCommand(settings.kwsArgs));
        applySherpaEnv(kwsProc, settings.binDir);
        kwsProc->setProcessChannelMode(QProcess::MergedChannels);
        QObject::connect(kwsProc, &QProcess::readyReadStandardOutput, parent, [this]{
            if (!kwsProc) return;
            const QString wake = settings.wakeWord.trimmed();
            const QString wakeCompact = QString(wake).remove(QRegularExpression(QStringLiteral("\\s+")));
            const QByteArray data = kwsProc->readAllStandardOutput();
            const QString out = QString::fromLocal8Bit(data);
            const QStringList lines = out.split(QRegularExpression(QStringLiteral("[\r\n]+")), Qt::SkipEmptyParts);
            for (QString line : lines)
            {
                line = line.trimmed();
                if (line.isEmpty()) continue;
                const QString lineCompact = QString(line).remove(QRegularExpression(QStringLiteral("\\s+")));
                if (!wakeCompact.isEmpty() && (line.contains(wake, Qt::CaseInsensitive) || lineCompact.contains(wakeCompact, Qt::CaseInsensitive)))
                {
                    onWake();
                    break;
                }
            }
        });
        QObject::connect(kwsProc, &QProcess::errorOccurred, parent, [this](QProcess::ProcessError){
            if (!kwsProc) return;
            const QString out = QString::fromLocal8Bit(kwsProc->readAll());
            onMicError(out);
        });
        QObject::connect(kwsProc, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), parent, [this](int exitCode, QProcess::ExitStatus){
            if (!kwsProc) return;
            const QString out = QString::fromLocal8Bit(kwsProc->readAll());
            if (exitCode != 0)
                onMicError(out);
            kwsProc->deleteLater();
            kwsProc = nullptr;
        });
        kwsProc->start();
        mode = Mode::Wake;
#endif
    }

    void stopKws()
    {
        qDebug() << "[DEBUG] Engine::stopKws() called, mode =" << (int)mode;
        if (mode == Mode::Wake)
            stopAll();
    }

    void startStt()
    {
        stopAll();
        if (!settings.sttEnabled)
            return;

#if defined(Q_OS_WIN32)
        const auto args = parseArgs(settings.sttArgs, false);
        if (!args)
            return;

        QByteArray modelType = args->modelType.toUtf8();
        QByteArray tokens = args->tokens.toUtf8();
        QByteArray encoder = args->encoder.toUtf8();
        QByteArray decoder = args->decoder.toUtf8();
        QByteArray joiner = args->joiner.toUtf8();
        QByteArray provider = args->provider.toUtf8();
        QByteArray decodingMethod = args->decodingMethod.toUtf8();

        SherpaOnnxOnlineRecognizerConfig config;
        memset(&config, 0, sizeof(config));
        config.feat_config.sample_rate = 16000;
        config.feat_config.feature_dim = 80;
        config.model_config.model_type = modelType.constData();
        config.model_config.tokens = tokens.constData();
        config.model_config.provider = provider.constData();
        config.model_config.num_threads = args->numThreads;
        config.model_config.debug = 0;
        
        config.model_config.transducer.encoder = encoder.constData();
        config.model_config.transducer.decoder = decoder.constData();
        config.model_config.transducer.joiner = joiner.constData();

        config.decoding_method = decodingMethod.constData();
        config.max_active_paths = 4;
        config.enable_endpoint = 1;
        config.rule1_min_trailing_silence = 2.4f;
        config.rule2_min_trailing_silence = 1.2f;
        config.rule3_min_utterance_length = 20.0f;
        config.hotwords_buf_size = 0;
        config.hotwords_score = 0.0f;
        config.blank_penalty = 0.0f;

        const SherpaOnnxOnlineRecognizer* p = SherpaOnnxCreateOnlineRecognizer(&config);
        if (!p) return;
        stt.reset(p);

        const SherpaOnnxOnlineStream* s = SherpaOnnxCreateOnlineStream(stt.get());
        if (!s) return;
        sttStream.reset(s);
        
        mode = Mode::Stt;
        startMicIfNeeded();
#else
        if (sttProc)
            return;
        const QString program = exePath(QStringLiteral("sherpa-onnx-microphone"));
        if (program.isEmpty() || !QFileInfo::exists(program))
            return;

        sttProc = new QProcess(parent);
        sttProc->setProgram(program);
        sttProc->setArguments(QProcess::splitCommand(settings.sttArgs));
        applySherpaEnv(sttProc, settings.binDir);
        sttProc->setProcessChannelMode(QProcess::MergedChannels);
        QObject::connect(sttProc, &QProcess::readyReadStandardOutput, parent, [this]{
            if (!sttProc) return;
            const QByteArray data = sttProc->readAllStandardOutput();
            const QString out = QString::fromLocal8Bit(data);
            const QStringList lines = out.split(QRegularExpression(QStringLiteral("[\r\n]+")), Qt::SkipEmptyParts);
            for (QString line : lines)
            {
                line = normalizeCandidateText(line);
                if (!line.isEmpty())
                    onSttPartial(line);
            }
        });
        QObject::connect(sttProc, &QProcess::errorOccurred, parent, [this](QProcess::ProcessError){
            if (!sttProc) return;
            const QString out = QString::fromLocal8Bit(sttProc->readAll());
            onMicError(sttProc->errorString() + QStringLiteral("\n") + out);
        });
        QObject::connect(sttProc, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), parent, [this](int exitCode, QProcess::ExitStatus){
            if (!sttProc) return;
            const QString out = QString::fromLocal8Bit(sttProc->readAll());
            if (exitCode != 0)
                onMicError(out);
            sttProc->deleteLater();
            sttProc = nullptr;
        });
        sttProc->start();
        mode = Mode::Stt;
#endif
    }

    void stopStt()
    {
        if (mode == Mode::Stt)
            stopAll();
    }

    void stopAll()
    {
        qDebug() << "[DEBUG] Engine::stopAll() called";
        mode = Mode::Idle;
#if defined(Q_OS_WIN32)
        if (kwsStream) kwsStream.reset();
        if (kws) kws.reset();
        if (sttStream) sttStream.reset();
        if (stt) stt.reset();
        pendingSamples.clear();
        lastPartial.clear();
        stopMicIfIdle();
#else
        if (kwsProc)
        {
            kwsProc->kill();
            kwsProc->deleteLater();
            kwsProc = nullptr;
        }
        if (sttProc)
        {
            sttProc->kill();
            sttProc->deleteLater();
            sttProc = nullptr;
        }
#endif
    }

private:
    enum class Mode {
        Idle,
        Wake,
        Stt
    };

    struct ParsedArgs {
        QString tokens;
        QString encoder;
        QString decoder;
        QString joiner;
        QString keywordsFile;
        QString provider{QStringLiteral("cpu")};
        QString decodingMethod{QStringLiteral("greedy_search")};
        QString modelType{QStringLiteral("zipformer")};
        int numThreads{2};
    };

    static bool fileExists(const QString& p)
    {
        return !p.trimmed().isEmpty() && QFileInfo::exists(p) && QFileInfo(p).isFile();
    }

    static std::optional<ParsedArgs> parseArgs(const QString& argLine, bool requireKeywordsFile)
    {
        qDebug() << "[DEBUG] parseArgs() input:" << argLine;
        const QStringList argv = QProcess::splitCommand(argLine);
        if (argv.isEmpty()) {
            qDebug() << "[DEBUG] parseArgs() argv is empty";
            return std::nullopt;
        }

        auto valueOf = [&](const QString& key) -> QString {
            for (int i = 0; i + 1 < argv.size(); ++i) {
                if (argv[i] == key)
                    return argv[i + 1];
            }
            return {};
        };

        ParsedArgs out;
        out.modelType = valueOf(QStringLiteral("--model-type")).trimmed();
        if (out.modelType.isEmpty())
            out.modelType = QStringLiteral("zipformer");

        out.provider = valueOf(QStringLiteral("--provider")).trimmed();
        if (out.provider.isEmpty())
            out.provider = QStringLiteral("cpu");

        const QString threads = valueOf(QStringLiteral("--num-threads")).trimmed();
        if (!threads.isEmpty()) {
            bool ok = false;
            const int n = threads.toInt(&ok);
            if (ok && n > 0)
                out.numThreads = n;
        }

        const QString dm = valueOf(QStringLiteral("--decoding-method")).trimmed();
        if (!dm.isEmpty())
            out.decodingMethod = dm;

        const QString whisperEnc = valueOf(QStringLiteral("--whisper-encoder")).trimmed();
        const QString whisperDec = valueOf(QStringLiteral("--whisper-decoder")).trimmed();
        if (!whisperEnc.isEmpty() || !whisperDec.isEmpty()) {
            qDebug() << "[DEBUG] parseArgs() whisper encoder/decoder not supported in this block";
            return std::nullopt;
        }

        out.tokens = valueOf(QStringLiteral("--tokens")).trimmed();
        out.encoder = valueOf(QStringLiteral("--encoder")).trimmed();
        out.decoder = valueOf(QStringLiteral("--decoder")).trimmed();
        out.joiner = valueOf(QStringLiteral("--joiner")).trimmed();
        out.keywordsFile = valueOf(QStringLiteral("--keywords-file")).trimmed();

        qDebug() << "[DEBUG] parseArgs() parsed tokens:" << out.tokens << "encoder:" << out.encoder;

        if (out.modelType != QStringLiteral("zipformer") && out.modelType != QStringLiteral("zipformer2")) {
            qDebug() << "[DEBUG] parseArgs() unsupported modelType:" << out.modelType;
            return std::nullopt;
        }

        if (!fileExists(out.tokens) || !fileExists(out.encoder) || !fileExists(out.decoder) || !fileExists(out.joiner)) {
            qDebug() << "[DEBUG] parseArgs() one of the core files doesn't exist. Tokens:" << fileExists(out.tokens)
                     << "Encoder:" << fileExists(out.encoder) << "Decoder:" << fileExists(out.decoder) << "Joiner:" << fileExists(out.joiner);
            return std::nullopt;
        }

        if (requireKeywordsFile)
        {
            if (out.keywordsFile.isEmpty() || !fileExists(out.keywordsFile)) {
                qDebug() << "[DEBUG] parseArgs() requireKeywordsFile is true but missing or not exists:" << out.keywordsFile;
                return std::nullopt;
            }
        }
        else if (!out.keywordsFile.isEmpty() && !fileExists(out.keywordsFile)) {
            qDebug() << "[DEBUG] parseArgs() keywords file provided but not exists:" << out.keywordsFile;
            return std::nullopt;
        }

        qDebug() << "[DEBUG] parseArgs() successful";
        return out;
    }

    void startMicIfNeeded()
    {
#if defined(Q_OS_WIN32) && defined(AMAIGIRL_USE_QT_MULTIMEDIA)
        if (mic && micDev)
        {
            if (!decodeTimer->isActive())
                decodeTimer->start();
            return;
        }

        QAudioDevice inputDevice = QMediaDevices::defaultAudioInput();
        if (inputDevice.isNull()) {
            onMicError(QStringLiteral("No default audio input device."));
            return;
        }

        QAudioFormat fmt;
        fmt.setSampleRate(16000);
        fmt.setChannelCount(1);
        fmt.setSampleFormat(QAudioFormat::Int16);

        if (!inputDevice.isFormatSupported(fmt))
            fmt = inputDevice.preferredFormat();

        if (!fmt.isValid()) {
            onMicError(QStringLiteral("Audio format not supported."));
            return;
        }

        if (fmt.sampleFormat() != QAudioFormat::Int16 && fmt.sampleFormat() != QAudioFormat::Float && fmt.sampleFormat() != QAudioFormat::Int32) {
            onMicError(QStringLiteral("Unsupported audio sample format."));
            return;
        }

        inputFormat = fmt;
        resampleStep = double(inputFormat.sampleRate()) / 16000.0;
        resamplePos = 0.0;
        monoBuf.clear();

        mic = std::make_unique<QAudioSource>(inputDevice, fmt);
        micDev = mic->start();
        if (!micDev) {
            onMicError(QStringLiteral("Failed to start audio input."));
            mic.reset();
            return;
        }

        QObject::connect(micDev, &QIODevice::readyRead, parent, [this] { onAudioReadyRead(); });
        QObject::connect(mic.get(), &QAudioSource::stateChanged, parent, [this](QAudio::State state){
            if (state == QAudio::StoppedState && mic && mic->error() != QAudio::NoError) {
                onMicError(QStringLiteral("Audio input error."));
            }
        });

        decodeTimer->start();
#endif
    }

    void stopMicIfIdle()
    {
        qDebug() << "[DEBUG] stopMicIfIdle() called, mode =" << (int)mode;
#if defined(Q_OS_WIN32) && defined(AMAIGIRL_USE_QT_MULTIMEDIA)
        if (mode != Mode::Idle)
            return;
        if (decodeTimer)
            decodeTimer->stop();
        if (micDev) {
            micDev = nullptr;
        }
        if (mic) {
            qDebug() << "[DEBUG] stopMicIfIdle() stopping mic";
            mic->stop();
            mic.reset();
        }
#endif
    }

    void onAudioReadyRead()
    {
#if defined(Q_OS_WIN32) && defined(AMAIGIRL_USE_QT_MULTIMEDIA)
        if (!micDev)
            return;
        const QByteArray data = micDev->readAll();
        if (data.isEmpty())
            return;

        const int channels = qMax(1, inputFormat.channelCount());
        const int bytesPerSample = inputFormat.bytesPerSample();
        const int bytesPerFrame = bytesPerSample * channels;
        if (bytesPerFrame <= 0 || data.size() < bytesPerFrame)
            return;

        const int frames = data.size() / bytesPerFrame;
        monoBuf.reserve(monoBuf.size() + frames);

        const char* raw = data.constData();
        for (int i = 0; i < frames; ++i)
        {
            double acc = 0.0;
            for (int ch = 0; ch < channels; ++ch)
            {
                const char* s = raw + (i * bytesPerFrame + ch * bytesPerSample);
                float v = 0.0f;
                if (inputFormat.sampleFormat() == QAudioFormat::Int16)
                {
                    int16_t x = 0;
                    memcpy(&x, s, sizeof(int16_t));
                    v = float(x) / 32768.0f;
                }
                else if (inputFormat.sampleFormat() == QAudioFormat::Int32)
                {
                    int32_t x = 0;
                    memcpy(&x, s, sizeof(int32_t));
                    v = float(double(x) / 2147483648.0);
                }
                else if (inputFormat.sampleFormat() == QAudioFormat::Float)
                {
                    float x = 0.0f;
                    memcpy(&x, s, sizeof(float));
                    v = x;
                }
                acc += v;
            }
            monoBuf.push_back(float(acc / double(channels)));
        }

        while (resamplePos + 1.0 < double(monoBuf.size()))
        {
            const int i = int(resamplePos);
            const double frac = resamplePos - double(i);
            const float a = monoBuf[size_t(i)];
            const float b = monoBuf[size_t(i + 1)];
            const float y = float((1.0 - frac) * double(a) + frac * double(b));
            pendingSamples.push_back(y);
            resamplePos += resampleStep;
        }

        const int drop = qMax(0, int(resamplePos) - 1);
        if (drop > 0)
        {
            monoBuf.erase(monoBuf.begin(), monoBuf.begin() + drop);
            resamplePos -= double(drop);
        }
#endif
    }

    void process()
    {
#if defined(Q_OS_WIN32) && defined(AMAIGIRL_USE_QT_MULTIMEDIA)
        if (mode == Mode::Idle)
            return;
        if (pendingSamples.isEmpty())
            return;

        int consume = qMin(1600, pendingSamples.size());
        if (consume <= 0)
            return;
        scratch.assign(pendingSamples.constBegin(), pendingSamples.constBegin() + consume);
        pendingSamples.erase(pendingSamples.begin(), pendingSamples.begin() + consume);

        const float* samples = scratch.data();
        const int n = int(scratch.size());

        if (mode == Mode::Wake && kws && kwsStream)
        {
            SherpaOnnxOnlineStreamAcceptWaveform(kwsStream.get(), 16000, samples, n);
            while (SherpaOnnxIsKeywordStreamReady(kws.get(), kwsStream.get()))
                SherpaOnnxDecodeKeywordStream(kws.get(), kwsStream.get());
            
            const SherpaOnnxKeywordResult* r = SherpaOnnxGetKeywordResult(kws.get(), kwsStream.get());
            if (r) {
                if (r->keyword && r->keyword[0] != '\0')
                    onWake();
                SherpaOnnxDestroyKeywordResult(r);
            }
            return;
        }

        if (mode == Mode::Stt && stt && sttStream)
        {
            SherpaOnnxOnlineStreamAcceptWaveform(sttStream.get(), 16000, samples, n);
            while (SherpaOnnxIsOnlineStreamReady(stt.get(), sttStream.get()))
                SherpaOnnxDecodeOnlineStream(stt.get(), sttStream.get());
            
            const SherpaOnnxOnlineRecognizerResult* r = SherpaOnnxGetOnlineStreamResult(stt.get(), sttStream.get());
            if (r) {
                const QString text = normalizeCandidateText(QString::fromUtf8(r->text));
                if (!text.isEmpty() && text != lastPartial)
                {
                    lastPartial = text;
                    onSttPartial(text);
                }
                SherpaOnnxDestroyOnlineRecognizerResult(r);
            }
        }
#endif
    }

private:
    QString exePath(const QString& baseName) const
    {
        const QString dir = settings.binDir.trimmed();
        if (dir.isEmpty()) return {};
#if defined(Q_OS_WIN32)
        const QString exe = baseName.endsWith(QStringLiteral(".exe"), Qt::CaseInsensitive)
                                ? baseName
                                : (baseName + QStringLiteral(".exe"));
        return QDir(dir).filePath(exe);
#else
        return QDir(dir).filePath(baseName);
#endif
    }

    QObject* parent{nullptr};
    SettingsSnapshot settings;
    Mode mode{Mode::Idle};

    std::function<void()> onWake;
    std::function<void(const QString&)> onSttPartial;
    std::function<void(const QString&)> onMicError;

#if defined(Q_OS_WIN32)
    struct KwsDeleter { void operator()(const SherpaOnnxKeywordSpotter* p) const { if(p) { qDebug() << "[DEBUG] KwsDeleter called"; SherpaOnnxDestroyKeywordSpotter(p); } } };
    struct SttDeleter { void operator()(const SherpaOnnxOnlineRecognizer* p) const { if(p) { qDebug() << "[DEBUG] SttDeleter called"; SherpaOnnxDestroyOnlineRecognizer(p); } } };
    struct StreamDeleter { void operator()(const SherpaOnnxOnlineStream* p) const { if(p) { qDebug() << "[DEBUG] StreamDeleter called"; SherpaOnnxDestroyOnlineStream(p); } } };
    std::unique_ptr<const SherpaOnnxKeywordSpotter, KwsDeleter> kws;
    std::unique_ptr<const SherpaOnnxOnlineStream, StreamDeleter> kwsStream;
    std::unique_ptr<const SherpaOnnxOnlineRecognizer, SttDeleter> stt;
    std::unique_ptr<const SherpaOnnxOnlineStream, StreamDeleter> sttStream;

#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    std::unique_ptr<QAudioSource> mic;
    QIODevice* micDev{nullptr};
    QTimer* decodeTimer{nullptr};
    QAudioFormat inputFormat;
    double resampleStep{1.0};
    double resamplePos{0.0};
    std::vector<float> monoBuf;
    QVector<float> pendingSamples;
    std::vector<float> scratch;
#endif
#else
    QProcess* kwsProc{nullptr};
    QProcess* sttProc{nullptr};
#endif

    QString lastPartial;
};
#endif

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
}

void OfflineVoiceService::stop()
{
    stopTts();
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
#if defined(Q_OS_WIN32)
    const QString exe = baseName.endsWith(QStringLiteral(".exe"), Qt::CaseInsensitive)
                            ? baseName
                            : (baseName + QStringLiteral(".exe"));
    return QDir(dir).filePath(exe);
#else
    return QDir(dir).filePath(baseName);
#endif
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

#if defined(Q_OS_LINUX)
    if (!libDir.isEmpty() && QFileInfo::exists(libDir) && QFileInfo(libDir).isDir())
    {
        const QString oldLd = env.value(QStringLiteral("LD_LIBRARY_PATH"));
        QStringList ldParts;
        ldParts << libDir;
        if (!oldLd.trimmed().isEmpty())
            ldParts << oldLd;
        env.insert(QStringLiteral("LD_LIBRARY_PATH"), ldParts.join(QDir::listSeparator()));
    }
#elif defined(Q_OS_MACOS)
    if (!libDir.isEmpty() && QFileInfo::exists(libDir) && QFileInfo(libDir).isDir())
    {
        const QString oldDyld = env.value(QStringLiteral("DYLD_LIBRARY_PATH"));
        QStringList dyldParts;
        dyldParts << libDir;
        if (!oldDyld.trimmed().isEmpty())
            dyldParts << oldDyld;
        env.insert(QStringLiteral("DYLD_LIBRARY_PATH"), dyldParts.join(QDir::listSeparator()));
    }
#endif
    p->setProcessEnvironment(env);
}

void OfflineVoiceService::startTts(const QString& text)
{
    stopTts();
    QString args = m_settings.ttsArgs;
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    const QString programGen = exePath(QStringLiteral("sherpa-onnx-offline-tts"));
    const QString programPlay = exePath(QStringLiteral("sherpa-onnx-offline-tts-play"));
    const QString programPlayAlsa = exePath(QStringLiteral("sherpa-onnx-offline-tts-play-alsa"));
    const bool canGen = !programGen.isEmpty() && QFileInfo::exists(programGen);
    const bool canPlay = !programPlay.isEmpty() && QFileInfo::exists(programPlay);
    const bool canPlayAlsa = !programPlayAlsa.isEmpty() && QFileInfo::exists(programPlayAlsa);
#if defined(Q_OS_LINUX)
    const QString program = canPlayAlsa ? programPlayAlsa : (canPlay ? programPlay : programGen);
    if (!canGen && !canPlayAlsa && !canPlay)
        return;
#else
    const QString program = canPlay ? programPlay : programGen;
    if (!canGen && !canPlay)
        return;
#endif
#else
    const QString program = exePath(QStringLiteral("sherpa-onnx-offline-tts-play"));
    if (program.isEmpty() || !QFileInfo::exists(program))
        return;
#endif

#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    QString wavPath;
    const bool useWavGen = (program == programGen);
    if (useWavGen)
    {
        QDir cache(SettingsManager::instance().cacheDir());
        if (!cache.exists()) cache.mkpath(QStringLiteral("."));
        wavPath = cache.filePath(QStringLiteral("tts_%1.wav").arg(QDateTime::currentMSecsSinceEpoch()));
        m_ttsWavPath = wavPath;
        if (!args.contains(QStringLiteral("--output-filename")))
        {
            const QString outArg = QStringLiteral("--output-filename=\"%1\"").arg(wavPath);
            if (args.contains(QStringLiteral("{text}")))
                args.replace(QStringLiteral("{text}"), outArg + QStringLiteral(" {text}"));
            else
                args = outArg + QStringLiteral(" ") + args;
        }
    }
#endif

    QString escaped = text;
    escaped.replace('"', QStringLiteral("\\\""));
    const QString quoted = QStringLiteral("\"") + escaped + QStringLiteral("\"");
    if (args.contains(QStringLiteral("{text}")))
        args.replace(QStringLiteral("{text}"), quoted);
    else
        args = args + QStringLiteral(" ") + quoted;

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
    if (program == programPlay)
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


OfflineVoiceService::OfflineVoiceService(QObject* parent)
    : QObject(parent)
{
    connect(qApp, &QCoreApplication::aboutToQuit, this, [this]{
        stop();
    });
}

OfflineVoiceService::~OfflineVoiceService() = default;

OfflineVoiceService::SettingsSnapshot OfflineVoiceService::readSettings() const
{
    SettingsSnapshot s;
    auto& sm = SettingsManager::instance();
    s.ttsEnabled = sm.offlineTtsEnabled();
    s.binDir = sm.sherpaOnnxBinDir();
    s.ttsArgs = sm.sherpaTtsArgs();
    s.ttsVolumePercent = sm.ttsVolumePercent();
    return s;
}

void OfflineVoiceService::applySettingsSnapshot(const SettingsSnapshot& next)
{
    const bool changedBin = (m_settings.binDir != next.binDir);
    const bool changedTts = (m_settings.ttsEnabled != next.ttsEnabled) || (m_settings.ttsArgs != next.ttsArgs) || (m_settings.ttsVolumePercent != next.ttsVolumePercent);

    m_settings = next;

    if (changedTts)
        g_speakerHintShown = false;

    if (changedBin || changedTts)
        stopTts();
}

void OfflineVoiceService::reloadFromSettings()
{
    qDebug() << "[DEBUG] OfflineVoiceService::reloadFromSettings() start";
    applySettingsSnapshot(readSettings());
    qDebug() << "[DEBUG] OfflineVoiceService::reloadFromSettings() middle";
    start();
    qDebug() << "[DEBUG] OfflineVoiceService::reloadFromSettings() end";
}
