#include "common/SettingsManager.hpp"
#include "common/Utils.hpp"
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFileInfo>
#include <QStandardPaths>
#include <QLocale>
#include <QDirIterator>
#include <QDateTime>
#include <QRegularExpression>
#include <QHash>
#include <QtGlobal>
#include <QCoreApplication>

static bool copyDirectoryRecursively(const QString& srcPath, const QString& dstPath)
{
    QDir src(srcPath);
    if (!src.exists()) return false;

    QDir dst(dstPath);
    if (!dst.exists() && !dst.mkpath(".")) return false;

    QDirIterator it(srcPath, QDir::NoDotAndDotDot | QDir::AllEntries, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        it.next();
        const QFileInfo fi = it.fileInfo();
        const QString rel = src.relativeFilePath(fi.absoluteFilePath());
        const QString target = dst.filePath(rel);

        if (fi.isDir()) {
            if (!QDir().mkpath(target)) return false;
            continue;
        }

        QDir().mkpath(QFileInfo(target).absolutePath());
        QFile::remove(target);
        if (!QFile::copy(fi.absoluteFilePath(), target)) return false;
    }
    return true;
}

static QString normalizeLanguageCode(const QString& code)
{
    const QString c = code.trimmed();
    if (c.startsWith("zh", Qt::CaseInsensitive)) return QStringLiteral("zh_CN");
    if (c.startsWith("en", Qt::CaseInsensitive)) return QStringLiteral("en_US");
    return QStringLiteral("en_US");
}

static QString normalizeThemeId(const QString& themeId)
{
    QString id = themeId.trimmed().toLower();
    if (id.isEmpty() || id == QStringLiteral("system"))
        return QStringLiteral("era");
    return id;
}

static QString systemLanguageCode()
{
    const QString n = QLocale::system().name();
    return normalizeLanguageCode(n);
}

static QString writableLocationOrFallback(QStandardPaths::StandardLocation location, const QString& fallback)
{
    const QString path = QStandardPaths::writableLocation(location);
    return path.isEmpty() ? fallback : path;
}

static QString appConfigBaseDir()
{
#if defined(Q_OS_WIN32)
    return writableLocationOrFallback(
        QStandardPaths::AppConfigLocation,
        QDir(QDir::homePath()).filePath(QStringLiteral("AppData/Roaming/IAIAYN/XiaoMo"))
    );
#else
    return QDir(QDir::homePath()).filePath(QStringLiteral(".XiaoMo"));
#endif
}

static QString appLocalDataBaseDir()
{
#if defined(Q_OS_WIN32)
    return writableLocationOrFallback(
        QStandardPaths::AppLocalDataLocation,
        QDir(QDir::homePath()).filePath(QStringLiteral("AppData/Local/IAIAYN/XiaoMo"))
    );
#else
    return QDir(QDir::homePath()).filePath(QStringLiteral(".XiaoMo"));
#endif
}

static QString voiceDepsRootDir()
{
    const QString appRoot = QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("voice_deps"));
    if (QFileInfo::exists(appRoot) && QFileInfo(appRoot).isDir())
        return appRoot;
    const QString bundled = appResourcePath(QStringLiteral("voice_deps"));
    if (QFileInfo::exists(bundled) && QFileInfo(bundled).isDir())
        return bundled;
    return appRoot;
}

static QString sherpaDefaultBinDir()
{
    const QString root = voiceDepsRootDir();
    const QString sherpaRoot = QDir(root).filePath(QStringLiteral("sherpa-onnx-v1.12.10-win-x64-shared"));
    const QString sherpaBin = QDir(sherpaRoot).filePath(QStringLiteral("bin"));
    if (QFileInfo::exists(sherpaBin) && QFileInfo(sherpaBin).isDir())
        return sherpaBin;
    return {};
}

static QString sherpaAnyBinDir()
{
    const QString root = voiceDepsRootDir();
    const QDir rootDir(root);
    if (!rootDir.exists()) return {};
    const QStringList subs = rootDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    for (const QString& name : subs)
    {
        if (!name.startsWith(QStringLiteral("sherpa-onnx-")))
            continue;
        const QString bin = QDir(rootDir.filePath(name)).filePath(QStringLiteral("bin"));
        if (QFileInfo::exists(bin) && QFileInfo(bin).isDir())
            return bin;
    }
    return {};
}

static QStringList listModelDirs(const QString& baseDir)
{
    QDir d(baseDir);
    if (!d.exists()) return {};
    QStringList out = d.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    out.removeAll(QStringLiteral("README"));
    return out;
}

static QString pickModelId(const QString& baseDir, const QString& preferred, const QStringList& fallbacks)
{
    const QString p = preferred.trimmed();
    if (!p.isEmpty() && QFileInfo::exists(QDir(baseDir).filePath(p)) && QFileInfo(QDir(baseDir).filePath(p)).isDir())
        return p;
    for (const QString& f : fallbacks)
    {
        if (f.trimmed().isEmpty()) continue;
        const QString one = QDir(baseDir).filePath(f);
        if (QFileInfo::exists(one) && QFileInfo(one).isDir())
            return f;
    }
    const QStringList dirs = listModelDirs(baseDir);
    return dirs.isEmpty() ? QString() : dirs.front();
}

static QString firstMatchFile(const QDir& d, const QStringList& namesOrGlobs)
{
    for (const QString& q : namesOrGlobs)
    {
        if (q.contains(QLatin1Char('*')))
        {
            const QStringList hits = d.entryList(QStringList() << q, QDir::Files, QDir::Name);
            if (!hits.isEmpty())
                return d.filePath(hits.front());
            continue;
        }
        const QString p = d.filePath(q);
        if (QFileInfo::exists(p) && QFileInfo(p).isFile())
            return p;
    }
    return {};
}

SettingsManager& SettingsManager::instance()
{
    static SettingsManager s;
    return s;
}

QString SettingsManager::configDir() const
{
    return QDir(appConfigBaseDir()).filePath(QStringLiteral("Configs"));
}

QString SettingsManager::configPath() const
{
    return QDir(configDir()).filePath("config.json");
}

QString SettingsManager::defaultModelsRoot() const
{
#if defined(Q_OS_WIN32)
    const QString documentsDir = writableLocationOrFallback(QStandardPaths::DocumentsLocation, QDir::homePath());
    return QDir(documentsDir).filePath(QStringLiteral("XiaoMo/Models"));
#else
    return QDir(QDir::homePath()).filePath(QStringLiteral(".XiaoMo/Models"));
#endif
}

QString SettingsManager::modelsRoot() const
{
    return m_modelsRoot.isEmpty() ? defaultModelsRoot() : m_modelsRoot;
}

void SettingsManager::setModelsRoot(const QString& p)
{
    m_modelsRoot = p;
    save();
}

void SettingsManager::resetModelsRootToDefault(const QString& appDir)
{
    m_modelsRoot.clear();
    save();
    bootstrap(appDir);
}

QVector<ModelEntry> SettingsManager::scanModels() const
{
    QVector<ModelEntry> out;
    QDir root(modelsRoot());
    if (!root.exists()) return out;

    for (const QFileInfo& fi : root.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot))
    {
        QDir sub(fi.absoluteFilePath());
        const auto pickFirstExisting = [&](const QStringList& candidates) -> QString {
            for (const QString& name : candidates)
            {
                const QString p = sub.filePath(name);
                if (QFileInfo::exists(p) && QFileInfo(p).isFile())
                    return p;
            }
            return {};
        };

        QString modelJson;
        {
            QStringList jsons = sub.entryList(QStringList{QStringLiteral("*.model3.json")}, QDir::Files);
            if (!jsons.isEmpty())
                modelJson = sub.filePath(jsons.front());
        }
        if (modelJson.isEmpty())
        {
            QStringList jsons = sub.entryList(QStringList{QStringLiteral("*.model.json")}, QDir::Files);
            if (!jsons.isEmpty())
                modelJson = sub.filePath(jsons.front());
        }
        if (modelJson.isEmpty())
        {
            modelJson = pickFirstExisting(QStringList{
                QStringLiteral("model3.json"),
                QStringLiteral("model.json"),
                QStringLiteral("index.json")
            });
        }

        if (!modelJson.isEmpty())
            out.push_back(ModelEntry{ fi.fileName(), modelJson });
    }
    return out;
}

QString SettingsManager::selectedModelFolder() const
{
    return m_selectedFolder;
}

void SettingsManager::setSelectedModelFolder(const QString& name)
{
    m_selectedFolder = name;
    save();
}

QString SettingsManager::theme() const
{
    return m_theme;
}

void SettingsManager::setTheme(const QString& themeId)
{
    m_theme = normalizeThemeId(themeId);
    save();
}

QString SettingsManager::currentLanguage() const
{
    if (m_currentLanguage.isEmpty()) return systemLanguageCode();
    return normalizeLanguageCode(m_currentLanguage);
}

void SettingsManager::setCurrentLanguage(const QString& code)
{
    m_currentLanguage = normalizeLanguageCode(code);
    save();
}

bool SettingsManager::hasWindowGeometry() const
{
    return m_winW > 0 && m_winH > 0;
}

QRect SettingsManager::windowGeometry() const
{
    return QRect(m_winX, m_winY, m_winW, m_winH);
}

QString SettingsManager::windowGeometryScreen() const
{
    return m_winScreen;
}

void SettingsManager::setWindowGeometryScreen(const QString& sig)
{
    m_winScreen = sig;
    save();
}

void SettingsManager::setWindowGeometry(const QRect& r)
{
    m_winX = r.x();
    m_winY = r.y();
    m_winW = r.width();
    m_winH = r.height();
    save();
}

QString SettingsManager::cacheDir() const
{
    const QString dir = QDir(appLocalDataBaseDir()).filePath(QStringLiteral(".Cache"));
    QDir().mkpath(dir);
    return dir;
}

// TTS settings removed in offline mode

QString SettingsManager::preferredScreenName() const { return m_preferredScreenName; }
void SettingsManager::setPreferredScreenName(const QString& name)
{
    m_preferredScreenName = name.trimmed();
    save();
}

bool SettingsManager::windowAlwaysOnTop() const { return m_windowAlwaysOnTop; }
void SettingsManager::setWindowAlwaysOnTop(bool v)
{
    if (m_windowAlwaysOnTop == v) return;
    m_windowAlwaysOnTop = v;
    save();
}

bool SettingsManager::windowTransparentBackground() const { return m_windowTransparentBackground; }
void SettingsManager::setWindowTransparentBackground(bool v)
{
    if (m_windowTransparentBackground == v) return;
    m_windowTransparentBackground = v;
    save();
}

bool SettingsManager::windowMousePassthrough() const { return m_windowMousePassthrough; }
void SettingsManager::setWindowMousePassthrough(bool v)
{
    if (m_windowMousePassthrough == v) return;
    m_windowMousePassthrough = v;
    save();
}

QJsonArray SettingsManager::reminderTasks() const
{
    return m_reminderTasks;
}

void SettingsManager::setReminderTasks(const QJsonArray& tasks)
{
    m_reminderTasks = tasks;
    save();
}

QString SettingsManager::llmStyle() const
{
    return m_llmStyle.isEmpty() ? QStringLiteral("Original") : m_llmStyle;
}

void SettingsManager::setLlmStyle(const QString& style)
{
    const QString s = style.trimmed();
    if (s.compare(m_llmStyle, Qt::CaseInsensitive) == 0)
        return;
    m_llmStyle = s.isEmpty() ? QStringLiteral("Original") : s;
    save();
}

QString SettingsManager::llmModelSize() const
{
    const QString s = m_llmModelSize.trimmed();
    if (s.compare(QStringLiteral("7B"), Qt::CaseInsensitive) == 0)
        return QStringLiteral("7B");
    return QStringLiteral("1.5B");
}

void SettingsManager::setLlmModelSize(const QString& size)
{
    const QString s = size.trimmed();
    const QString normalized = (s.compare(QStringLiteral("7B"), Qt::CaseInsensitive) == 0) ? QStringLiteral("7B") : QStringLiteral("1.5B");
    if (m_llmModelSize == normalized)
        return;
    m_llmModelSize = normalized;
    save();
}

static QString normalizeChatBubbleStyleId(const QString& styleId)
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

QString SettingsManager::chatBubbleStyle() const
{
    return normalizeChatBubbleStyleId(m_chatBubbleStyle);
}

void SettingsManager::setChatBubbleStyle(const QString& styleId)
{
    const QString normalized = normalizeChatBubbleStyleId(styleId);
    if (m_chatBubbleStyle == normalized)
        return;
    m_chatBubbleStyle = normalized;
    save();
}

bool SettingsManager::offlineTtsEnabled() const { return m_offlineTtsEnabled; }
void SettingsManager::setOfflineTtsEnabled(bool v)
{
    if (m_offlineTtsEnabled == v) return;
    m_offlineTtsEnabled = v;
    save();
}

QString SettingsManager::sherpaOnnxBinDir() const
{
    const QString def = sherpaDefaultBinDir();
    if (!def.isEmpty()) return def;
    const QString any = sherpaAnyBinDir();
    if (!any.isEmpty()) return any;
    const QString d = m_sherpaOnnxBinDir.trimmed();
    if (!d.isEmpty() && QFileInfo::exists(d) && QFileInfo(d).isDir())
        return d;
    return {};
}
void SettingsManager::setSherpaOnnxBinDir(const QString& dir)
{
    const QString d = dir.trimmed();
    if (m_sherpaOnnxBinDir == d) return;
    m_sherpaOnnxBinDir = d;
    save();
}

QString SettingsManager::sherpaTtsModel() const { return m_sherpaTtsModel.trimmed(); }
void SettingsManager::setSherpaTtsModel(const QString& modelId)
{
    const QString v = modelId.trimmed();
    if (m_sherpaTtsModel == v) return;
    m_sherpaTtsModel = v;
    if (!v.isEmpty())
        m_sherpaTtsArgs.clear();
    save();
}

QString SettingsManager::sherpaTtsArgs() const
{
    if (m_sherpaTtsModel.trimmed().isEmpty() && !m_sherpaTtsArgs.trimmed().isEmpty())
        return m_sherpaTtsArgs;

    const QString ttsBase = QDir(voiceDepsRootDir()).filePath(QStringLiteral("models"));
    const QString modelId = pickModelId(ttsBase, m_sherpaTtsModel, {});
    if (modelId.isEmpty()) return {};
    const QDir d(QDir(ttsBase).filePath(modelId));

    const QString tokens = firstMatchFile(d, {QStringLiteral("tokens.txt")});
    QString model = firstMatchFile(d, {QStringLiteral("model.onnx"), QStringLiteral("*.onnx")});
    if (tokens.isEmpty() || model.isEmpty())
        return {};

    const QString voices = firstMatchFile(d, {QStringLiteral("voices.bin")});
    if (!voices.isEmpty())
    {
        QStringList lexicons;
        const QString lexUs = firstMatchFile(d, {QStringLiteral("lexicon-us-en.txt")});
        const QString lexGb = firstMatchFile(d, {QStringLiteral("lexicon-gb-en.txt")});
        const QString lexZh = firstMatchFile(d, {QStringLiteral("lexicon-zh.txt")});
        const QString lexOne = firstMatchFile(d, {QStringLiteral("lexicon.txt")});
        if (!lexUs.isEmpty()) lexicons.push_back(lexUs);
        if (!lexGb.isEmpty()) lexicons.push_back(lexGb);
        if (!lexZh.isEmpty()) lexicons.push_back(lexZh);
        if (lexicons.isEmpty() && !lexOne.isEmpty()) lexicons.push_back(lexOne);

        QString args = QStringLiteral("--kokoro-model=\"%1\" --kokoro-voices=\"%2\" --kokoro-tokens=\"%3\"")
                           .arg(model, voices, tokens);

        const QString espeakDataDir = d.filePath(QStringLiteral("espeak-ng-data"));
        if (QFileInfo::exists(espeakDataDir) && QFileInfo(espeakDataDir).isDir())
            args += QStringLiteral(" --kokoro-data-dir=\"%1\"").arg(espeakDataDir);

        const QString dictDir = d.filePath(QStringLiteral("dict"));
        if (QFileInfo::exists(dictDir) && QFileInfo(dictDir).isDir())
            args += QStringLiteral(" --kokoro-dict-dir=\"%1\"").arg(dictDir);

        if (!lexicons.isEmpty())
            args += QStringLiteral(" --kokoro-lexicon=\"%1\"").arg(lexicons.join(QStringLiteral(",")));

        args += QStringLiteral(" --provider=cpu --num-threads=2 {text}");
        return args;
    }

    const QString espeakDataDir = d.filePath(QStringLiteral("espeak-ng-data"));
    if (QFileInfo::exists(espeakDataDir) && QFileInfo(espeakDataDir).isDir())
    {
        const int sid = m_sherpaTtsSid < 0 ? 0 : m_sherpaTtsSid;
        return QStringLiteral("--vits-model=\"%1\" --vits-tokens=\"%2\" --vits-data-dir=\"%3\" --sid=%4 --provider=cpu --num-threads=2 {text}")
            .arg(model, tokens, espeakDataDir, QString::number(sid));
    }

    const QString lexicon = d.filePath(QStringLiteral("lexicon.txt"));
    const QString dictDir = d.filePath(QStringLiteral("dict"));
    const bool hasDict = QFileInfo::exists(dictDir) && QFileInfo(dictDir).isDir();
    const int sid = m_sherpaTtsSid < 0 ? 0 : m_sherpaTtsSid;
    return QFileInfo::exists(lexicon)
               ? (hasDict
                      ? QStringLiteral("--vits-model=\"%1\" --vits-tokens=\"%2\" --vits-lexicon=\"%3\" --vits-dict-dir=\"%4\" --sid=%5 --provider=cpu --num-threads=2 {text}").arg(model, tokens, lexicon, dictDir, QString::number(sid))
                      : QStringLiteral("--vits-model=\"%1\" --vits-tokens=\"%2\" --vits-lexicon=\"%3\" --sid=%4 --provider=cpu --num-threads=2 {text}").arg(model, tokens, lexicon, QString::number(sid)))
               : (hasDict
                      ? QStringLiteral("--vits-model=\"%1\" --vits-tokens=\"%2\" --vits-dict-dir=\"%3\" --sid=%4 --provider=cpu --num-threads=2 {text}").arg(model, tokens, dictDir, QString::number(sid))
                      : QStringLiteral("--vits-model=\"%1\" --vits-tokens=\"%2\" --sid=%3 --provider=cpu --num-threads=2 {text}").arg(model, tokens, QString::number(sid)));
}
void SettingsManager::setSherpaTtsArgs(const QString& args)
{
    const QString a = args.trimmed();
    if (m_sherpaTtsArgs == a) return;
    m_sherpaTtsArgs = a;
    save();
}

int SettingsManager::sherpaTtsSid() const
{
    return m_sherpaTtsSid < 0 ? 0 : m_sherpaTtsSid;
}
void SettingsManager::setSherpaTtsSid(int sid)
{
    if (sid < 0) sid = 0;
    if (m_sherpaTtsSid == sid) return;
    m_sherpaTtsSid = sid;
    save();
}

int SettingsManager::ttsVolumePercent() const
{
    int v = m_ttsVolumePercent;
    if (v < 0) v = 0;
    if (v > 100) v = 100;
    return v;
}
void SettingsManager::setTtsVolumePercent(int v)
{
    if (v < 0) v = 0;
    if (v > 100) v = 100;
    if (m_ttsVolumePercent == v) return;
    m_ttsVolumePercent = v;
    save();
}

// Preferred audio output removed in offline mode

void SettingsManager::load()
{
    QDir dir(configDir());
    if (!dir.exists()) dir.mkpath(".");

    QFile f(configPath());
    if (!f.exists()) { save(); return; }
    if (!f.open(QIODevice::ReadOnly)) return;

    QJsonParseError err;
    auto doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (err.error != QJsonParseError::NoError)
    {
        f.close();
        const QString ts = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
        const QString badPath = configPath() + QStringLiteral(".bad_") + ts;
        QFile::rename(configPath(), badPath);
        save();
        return;
    }

    auto root = doc.object();
    bool needsSave = false;
    m_modelsRoot     = root.value("modelsRoot").toString(modelsRoot());
    m_selectedFolder = root.value("selectedModelFolder").toString();
    if (root.contains("theme"))
    {
        const QString rawTheme = root.value("theme").toString();
        m_theme = normalizeThemeId(rawTheme);
        if (m_theme != rawTheme)
            needsSave = true;
    }
    else
    {
        m_theme = normalizeThemeId(root.value("themeMode").toString(QStringLiteral("era")));
        if (root.contains("themeMode"))
            needsSave = true;
    }
    if (root.contains("currentLanguage"))
        m_currentLanguage = normalizeLanguageCode(root.value("currentLanguage").toString());
    else
        m_currentLanguage = systemLanguageCode();
    m_winX           = root.value("winX").toInt(-1);
    m_winY           = root.value("winY").toInt(-1);
    m_winW           = root.value("winW").toInt(0);
    m_winH           = root.value("winH").toInt(0);
    m_winScreen      = root.value("winScreen").toString();
    m_textureMaxDim  = root.value("textureMaxDim").toInt(2048);
    m_msaaSamples    = root.value("msaaSamples").toInt(4);

    // Advanced runtime selection
    m_preferredScreenName = root.value("preferredScreen").toString();
    // preferredAudioOutput removed
    m_windowAlwaysOnTop = root.value("windowAlwaysOnTop").toBool(true);
    m_windowTransparentBackground = root.value("windowTransparentBackground").toBool(true);
    m_windowMousePassthrough = root.value("windowMousePassthrough").toBool(false);

    m_reminderTasks = root.value("reminderTasks").toArray();

    // AI settings
    m_characterName = root.value("characterName").toString(QStringLiteral("小墨")).trimmed();
    if (m_characterName.isEmpty())
        m_characterName = QStringLiteral("小墨");
    m_chatContextMessages = root.value("chatContextMessages").toInt(16);
    if (m_chatContextMessages < 0) m_chatContextMessages = 0;
    if (m_chatContextMessages > 200) m_chatContextMessages = 200;
    m_llmMaxTokens = root.value("llmMaxTokens").toInt(256);
    if (m_llmMaxTokens < 1) m_llmMaxTokens = 1;
    if (m_llmMaxTokens > 4096) m_llmMaxTokens = 4096;
    m_aiSystemPrompt = root.value("aiSystemPrompt").toString();
    m_llmStyle = root.value("llmStyle").toString(QStringLiteral("Original"));
    m_llmModelSize = root.value("llmModelSize").toString(QStringLiteral("1.5B"));
    m_chatBubbleStyle = root.value("chatBubbleStyle").toString(QStringLiteral("Era"));

    m_offlineTtsEnabled = root.value("offlineTtsEnabled").toBool(false);
    m_sherpaOnnxBinDir = root.value("sherpaOnnxBinDir").toString().trimmed();
    m_sherpaTtsModel = root.value("sherpaTtsModel").toString().trimmed();
    m_sherpaTtsArgs = root.value("sherpaTtsArgs").toString();
    m_sherpaTtsSid = root.value("sherpaTtsSid").toInt(0);
    m_ttsVolumePercent = root.value("ttsVolumePercent").toInt(80);

    // TTS
    // TTS removed

    if (needsSave)
        save();
}

void SettingsManager::save() const
{
    QDir dir(configDir());
    if (!dir.exists()) dir.mkpath(".");

    QJsonObject root;
    root["modelsRoot"]          = m_modelsRoot.isEmpty() ? defaultModelsRoot() : m_modelsRoot;
    root["selectedModelFolder"] = m_selectedFolder;
    root["theme"]               = normalizeThemeId(m_theme);
    root["currentLanguage"]     = currentLanguage();
    root["winX"]                = m_winX;
    root["winY"]                = m_winY;
    root["winW"]                = m_winW;
    root["winH"]                = m_winH;
    root["winScreen"]           = m_winScreen;
    root["textureMaxDim"]       = m_textureMaxDim;
    root["msaaSamples"]         = m_msaaSamples;

    // Advanced runtime selection
    root["preferredScreen"]      = m_preferredScreenName;
    // preferredAudioOutput removed
    root["windowAlwaysOnTop"] = m_windowAlwaysOnTop;
    root["windowTransparentBackground"] = m_windowTransparentBackground;
    root["windowMousePassthrough"] = m_windowMousePassthrough;

    root["reminderTasks"] = m_reminderTasks;

    // AI settings
    root["characterName"] = m_characterName;
    root["chatContextMessages"] = m_chatContextMessages;
    root["llmMaxTokens"] = m_llmMaxTokens;
    root["aiSystemPrompt"] = m_aiSystemPrompt;
    root["llmStyle"] = llmStyle();
    root["llmModelSize"] = llmModelSize();
    root["chatBubbleStyle"] = chatBubbleStyle();

    root["offlineTtsEnabled"] = m_offlineTtsEnabled;
    root["sherpaOnnxBinDir"] = m_sherpaOnnxBinDir;
    root["sherpaTtsModel"] = m_sherpaTtsModel.trimmed();
    root["sherpaTtsArgs"] = m_sherpaTtsArgs;
    root["sherpaTtsSid"] = sherpaTtsSid();
    root["ttsVolumePercent"] = ttsVolumePercent();

    // TTS
    // TTS removed

    QFile f(configPath());
    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    }
}

void SettingsManager::bootstrap(const QString& appDir)
{
    QDir dir(configDir());
    if (!dir.exists()) dir.mkpath(".");

    const QString bundledModels = appResourcePath(QStringLiteral("models"));
    const QString legacyModels = QDir(appDir).filePath(QStringLiteral("res/models"));
    const bool hasBundledModels = QFileInfo::exists(bundledModels) && QFileInfo(bundledModels).isDir();

    bool changed = false;
    const QString currentRoot = modelsRoot();
    const bool currentRootExists = QFileInfo::exists(currentRoot) && QFileInfo(currentRoot).isDir();
    if (!currentRootExists)
    {
        if (hasBundledModels)
        {
            m_modelsRoot = bundledModels;
            changed = true;
        }
        else if (QFileInfo::exists(legacyModels) && QFileInfo(legacyModels).isDir())
        {
            m_modelsRoot = legacyModels;
            changed = true;
        }
    }

    QDir models(modelsRoot());
    if (!models.exists()) {
        if (!models.mkpath(".")) {
            // fallback to bundled models if we can't create the directory
            if (hasBundledModels) {
                m_modelsRoot = bundledModels;
                models = QDir(modelsRoot());
                changed = true;
            }
        }
    }

    // Chats dir
    {
        QDir chats(chatsDir());
        if (!chats.exists()) chats.mkpath(".");
    }

    if (hasBundledModels)
    {
        const auto entries = scanModels();
        if (entries.isEmpty() && models.absolutePath() != bundledModels)
        {
            m_modelsRoot = bundledModels;
            models = QDir(modelsRoot());
            changed = true;
        }
    }

    if (models.entryList(QDir::Dirs | QDir::NoDotAndDotDot).isEmpty() && models.absolutePath() != bundledModels) {
        QStringList candidateSources;
        candidateSources << bundledModels;
#if defined(Q_OS_MACOS)
        candidateSources << QDir(appDir).filePath(QStringLiteral("../Resources/models"));
    #elif defined(Q_OS_LINUX)
        candidateSources << QDir(appDir).filePath(QStringLiteral("../share/XiaoMo/res/models"));
#endif
        candidateSources << legacyModels;

        bool copied = false;
        for (const QString& src : candidateSources) {
            if (QFileInfo::exists(src) && QFileInfo(src).isDir()) {
                copied = copyDirectoryRecursively(src, models.absolutePath());
                if (copied) break;
            }
        }

        if (copied)
            changed = true;
    }

    if (m_selectedFolder.isEmpty())
    {
        const auto entries = scanModels();
        if (!entries.isEmpty())
        {
            m_selectedFolder = entries.front().folderName;
            changed = true;
        }
    }

    if (!m_selectedFolder.isEmpty())
    {
        ensureModelConfigExists(m_selectedFolder);
    }

    const QString voiceDepsRoot = voiceDepsRootDir();
    if (QFileInfo::exists(voiceDepsRoot) && QFileInfo(voiceDepsRoot).isDir())
    {
        if (!m_sherpaOnnxBinDir.trimmed().isEmpty() && (!QFileInfo::exists(m_sherpaOnnxBinDir) || !QFileInfo(m_sherpaOnnxBinDir).isDir()))
        {
            m_sherpaOnnxBinDir.clear();
            changed = true;
        }

        const QString ttsBase = QDir(voiceDepsRoot).filePath(QStringLiteral("models"));

        const QString ttsPicked = pickModelId(ttsBase, m_sherpaTtsModel, {});
        if (m_sherpaTtsModel.trimmed().isEmpty() && !ttsPicked.isEmpty())
        {
            m_sherpaTtsModel = ttsPicked;
            changed = true;
        }
    }

    if (changed)
        save();
}

QString SettingsManager::modelConfigPath(const QString& modelFolder) const
{
    if (modelFolder.isEmpty()) return {};
    return QDir(configDir()).filePath(modelFolder + ".config.json");
}

QString SettingsManager::chatsDir() const
{
    return QDir(appLocalDataBaseDir()).filePath(QStringLiteral("Chats"));
}

QString SettingsManager::chatPathForModel(const QString& modelFolder) const
{
    if (modelFolder.isEmpty()) return {};
    return QDir(chatsDir()).filePath(modelFolder + ".chat.json");
}

void SettingsManager::ensureModelConfigExists(const QString& folder) const
{
    if (folder.isEmpty()) return;

    QFile f(modelConfigPath(folder));
    if (f.exists()) return;

    QJsonObject o;
    o["enableBlink"]   = true;
    o["enableBreath"]  = true;
    o["enableGaze"]    = false;
    o["enablePhysics"] = true;
    o["poseAB"]        = 0;

    saveModelConfigObject(folder, o);
}

QJsonObject SettingsManager::loadModelConfigObject(const QString& folder) const
{
    QJsonObject o;
    if (folder.isEmpty()) return o;

    QFile f(modelConfigPath(folder));
    if (!f.exists()) return o;
    if (!f.open(QIODevice::ReadOnly)) return o;

    QJsonParseError err;
    auto doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (err.error != QJsonParseError::NoError) return o;

    return doc.object();
}

void SettingsManager::saveModelConfigObject(const QString& folder, const QJsonObject& obj) const
{
    if (folder.isEmpty()) return;

    QDir d(configDir());
    if (!d.exists()) d.mkpath(".");

    QFile f(modelConfigPath(folder));
    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        f.write(QJsonDocument(obj).toJson(QJsonDocument::Indented));
    }
}

bool SettingsManager::enableBlink() const
{
    return loadModelConfigObject(selectedModelFolder()).value("enableBlink").toBool(true);
}

bool SettingsManager::enableBreath() const
{
    return loadModelConfigObject(selectedModelFolder()).value("enableBreath").toBool(true);
}

bool SettingsManager::enableGaze() const
{
    return loadModelConfigObject(selectedModelFolder()).value("enableGaze").toBool(false);
}

bool SettingsManager::enablePhysics() const
{
    return loadModelConfigObject(selectedModelFolder()).value("enablePhysics").toBool(true);
}

void SettingsManager::setEnableBlink(bool v)
{
    auto f = selectedModelFolder();
    auto o = loadModelConfigObject(f);
    o["enableBlink"] = v;
    saveModelConfigObject(f, o);
}

void SettingsManager::setEnableBreath(bool v)
{
    auto f = selectedModelFolder();
    auto o = loadModelConfigObject(f);
    o["enableBreath"] = v;
    saveModelConfigObject(f, o);
}

void SettingsManager::setEnableGaze(bool v)
{
    auto f = selectedModelFolder();
    auto o = loadModelConfigObject(f);
    o["enableGaze"] = v;
    saveModelConfigObject(f, o);
}

void SettingsManager::setEnablePhysics(bool v)
{
    auto f = selectedModelFolder();
    auto o = loadModelConfigObject(f);
    o["enablePhysics"] = v;
    saveModelConfigObject(f, o);
}

QString SettingsManager::watermarkExpPath() const
{
    const QString folder = selectedModelFolder();
    auto o = loadModelConfigObject(folder);
    const QString p = o.value("watermarkExpPath").toString("");
    if (p.isEmpty()) return {};
    if (QFileInfo::exists(p) && QFileInfo(p).isFile()) return p;

    const QString base = QFileInfo(p).fileName();
    if (base.isEmpty()) return p;

    static const QHash<QString, QString> map = {
        { QStringLiteral("叉腰.exp3.json"), QStringLiteral("hands_on_hips.exp3.json") },
        { QStringLiteral("抬手.exp3.json"), QStringLiteral("raise_hand.exp3.json") },
        { QStringLiteral("消除高光.exp3.json"), QStringLiteral("remove_highlights.exp3.json") },
        { QStringLiteral("爱心眼.exp3.json"), QStringLiteral("heart_eyes.exp3.json") },
        { QStringLiteral("瞪眼.exp3.json"), QStringLiteral("wide_eyes.exp3.json") },
        { QStringLiteral("羞涩.exp3.json"), QStringLiteral("shy.exp3.json") },
        { QStringLiteral("舌头.exp3.json"), QStringLiteral("tongue_out.exp3.json") },

        { QStringLiteral("乳沟.exp3.json"), QStringLiteral("cleavage.exp3.json") },
        { QStringLiteral("光手臂.exp3.json"), QStringLiteral("glow_arms.exp3.json") },
        { QStringLiteral("光环.exp3.json"), QStringLiteral("halo.exp3.json") },
        { QStringLiteral("光腿.exp3.json"), QStringLiteral("glow_legs.exp3.json") },
        { QStringLiteral("内衣.exp3.json"), QStringLiteral("underwear.exp3.json") },
        { QStringLiteral("双马尾.exp3.json"), QStringLiteral("twin_tails.exp3.json") },
        { QStringLiteral("吐舌.exp3.json"), QStringLiteral("tongue_out.exp3.json") },
        { QStringLiteral("哭眼.exp3.json"), QStringLiteral("cry_eyes.exp3.json") },
        { QStringLiteral("外衣.exp3.json"), QStringLiteral("outer_clothes.exp3.json") },
        { QStringLiteral("宠物.exp3.json"), QStringLiteral("pet.exp3.json") },
        { QStringLiteral("尾巴上摆.exp3.json"), QStringLiteral("tail_swing.exp3.json") },
        { QStringLiteral("手持武器-外衣关闭.exp3.json"), QStringLiteral("weapon_hold_outer_off.exp3.json") },
        { QStringLiteral("星星眼.exp3.json"), QStringLiteral("star_eyes.exp3.json") },
        { QStringLiteral("枷锁.exp3.json"), QStringLiteral("shackles.exp3.json") },
        { QStringLiteral("白丝.exp3.json"), QStringLiteral("white_stockings.exp3.json") },
        { QStringLiteral("白眼.exp3.json"), QStringLiteral("white_eyes.exp3.json") },
        { QStringLiteral("眼心.exp3.json"), QStringLiteral("heart_pupils.exp3.json") },
        { QStringLiteral("眼镜.exp3.json"), QStringLiteral("glasses.exp3.json") },
        { QStringLiteral("精灵耳朵.exp3.json"), QStringLiteral("elf_ears.exp3.json") },
        { QStringLiteral("背后武器.exp3.json"), QStringLiteral("back_weapon.exp3.json") },
        { QStringLiteral("脸红.exp3.json"), QStringLiteral("blush.exp3.json") },
        { QStringLiteral("裸体.exp3.json"), QStringLiteral("nude.exp3.json") },
        { QStringLiteral("话筒.exp3.json"), QStringLiteral("microphone.exp3.json") },
        { QStringLiteral("豆豆眼.exp3.json"), QStringLiteral("dot_eyes.exp3.json") },
        { QStringLiteral("黑丝.exp3.json"), QStringLiteral("black_stockings.exp3.json") },
        { QStringLiteral("黑脸.exp3.json"), QStringLiteral("dark_face.exp3.json") },

        { QStringLiteral("体型.exp3.json"), QStringLiteral("body_shape.exp3.json") },
        { QStringLiteral("哭泣.exp3.json"), QStringLiteral("cry.exp3.json") },
        { QStringLiteral("尾巴上摆动.exp3.json"), QStringLiteral("tail_swing.exp3.json") },
        { QStringLiteral("手柄.exp3.json"), QStringLiteral("gamepad.exp3.json") },
        { QStringLiteral("晕眼.exp3.json"), QStringLiteral("dizzy_eyes.exp3.json") },
        { QStringLiteral("毛毯.exp3.json"), QStringLiteral("blanket.exp3.json") },
        { QStringLiteral("白丝袜.exp3.json"), QStringLiteral("white_stockings.exp3.json") },
        { QStringLiteral("眯眼.exp3.json"), QStringLiteral("squint_eyes.exp3.json") },
        { QStringLiteral("粉色双马尾.exp3.json"), QStringLiteral("pink_twin_tails.exp3.json") },
        { QStringLiteral("紧绷眼.exp3.json"), QStringLiteral("tense_eyes.exp3.json") },
        { QStringLiteral("胸部遮挡.exp3.json"), QStringLiteral("chest_cover.exp3.json") },
        { QStringLiteral("脸黑.exp3.json"), QStringLiteral("dark_face.exp3.json") },
        { QStringLiteral("鞋子.exp3.json"), QStringLiteral("shoes.exp3.json") },
        { QStringLiteral("黑丝袜.exp3.json"), QStringLiteral("black_stockings.exp3.json") }
    };

    auto it = map.find(base);
    if (it == map.end()) return p;

    const QString candidate = QDir(QFileInfo(p).absolutePath()).filePath(it.value());
    if (!QFileInfo::exists(candidate) || !QFileInfo(candidate).isFile())
        return p;

    o["watermarkExpPath"] = candidate;
    saveModelConfigObject(folder, o);
    return candidate;
}

void SettingsManager::setWatermarkExpPath(const QString& absPath)
{
    auto f = selectedModelFolder();
    auto o = loadModelConfigObject(f);
    if (absPath.isEmpty()) o.remove("watermarkExpPath");
    else o["watermarkExpPath"] = absPath;
    saveModelConfigObject(f, o);
}

QString SettingsManager::selectedMotionGroup() const
{
    return loadModelConfigObject(selectedModelFolder()).value("selectedMotionGroup").toString("");
}

void SettingsManager::setSelectedMotionGroup(const QString& group)
{
    auto f = selectedModelFolder();
    auto o = loadModelConfigObject(f);
    if (group.isEmpty()) o.remove("selectedMotionGroup");
    else o["selectedMotionGroup"] = group;
    saveModelConfigObject(f, o);
}

QString SettingsManager::selectedExpressionName() const
{
    return loadModelConfigObject(selectedModelFolder()).value("selectedExpressionName").toString("");
}

void SettingsManager::setSelectedExpressionName(const QString& name)
{
    auto f = selectedModelFolder();
    auto o = loadModelConfigObject(f);
    if (name.isEmpty()) o.remove("selectedExpressionName");
    else o["selectedExpressionName"] = name;
    saveModelConfigObject(f, o);
}

int SettingsManager::textureMaxDim() const
{
    return m_textureMaxDim;
}

void SettingsManager::setTextureMaxDim(int dim)
{
    if (dim != 1024 && dim != 2048 && dim != 3072 && dim != 4096) dim = 2048;
    m_textureMaxDim = dim;
    save();
}

int SettingsManager::msaaSamples() const
{
    return m_msaaSamples;
}

void SettingsManager::setMsaaSamples(int samples)
{
    if (samples != 2 && samples != 4 && samples != 8) samples = 4;
    m_msaaSamples = samples;
    save();
}

int SettingsManager::poseAB() const
{
    auto o = loadModelConfigObject(selectedModelFolder());
    int v = o.value("poseAB").toInt(0);
    if (v != 0 && v != 1) v = 0;
    return v;
}

void SettingsManager::setPoseAB(int index)
{
    if (index != 0 && index != 1) index = 0;
    auto f = selectedModelFolder();
    auto o = loadModelConfigObject(f);
    o["poseAB"] = index;
    saveModelConfigObject(f, o);
}

QString SettingsManager::characterName() const
{
    return m_characterName;
}

void SettingsManager::setCharacterName(const QString& name)
{
    const QString n = name.trimmed();
    m_characterName = n.isEmpty() ? QStringLiteral("小墨") : n;
    save();
}

int SettingsManager::chatContextMessages() const
{
    return m_chatContextMessages;
}

void SettingsManager::setChatContextMessages(int count)
{
    if (count < 0) count = 0;
    if (count > 200) count = 200;
    m_chatContextMessages = count;
    save();
}

int SettingsManager::llmMaxTokens() const
{
    return m_llmMaxTokens;
}

void SettingsManager::setLlmMaxTokens(int maxTokens)
{
    if (maxTokens < 1) maxTokens = 1;
    if (maxTokens > 4096) maxTokens = 4096;
    m_llmMaxTokens = maxTokens;
    save();
}

QString SettingsManager::aiSystemPrompt() const { return m_aiSystemPrompt; }
void SettingsManager::setAiSystemPrompt(const QString& prompt) { m_aiSystemPrompt = prompt; save(); }
// aiStream removed in offline mode
