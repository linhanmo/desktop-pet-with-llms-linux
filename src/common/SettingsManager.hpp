#pragma once
#include <QString>
#include <QVector>
#include <QJsonArray>
#include <QJsonObject>
#include <QRect>

struct ModelEntry {
    QString folderName;   // e.g. "Hiyori"
    QString jsonPath;     // absolute path to *.model3.json
};

class SettingsManager {
public:
    static SettingsManager& instance();

    QString configDir() const;
    QString configPath() const;
    QString modelsRoot() const;
    void setModelsRoot(const QString& p);
    QString defaultModelsRoot() const;
    void resetModelsRootToDefault(const QString& appDir);

    QVector<ModelEntry> scanModels() const;
    QString selectedModelFolder() const;
    void setSelectedModelFolder(const QString& name);

    QString theme() const;
    void setTheme(const QString& themeId);

    // UI language code (e.g. zh_CN / en_US)
    QString currentLanguage() const;
    void setCurrentLanguage(const QString& code);

    bool hasWindowGeometry() const;
    QRect windowGeometry() const;
    void setWindowGeometry(const QRect& r);

    // The screen signature used when window geometry was last saved.
    // Used to detect display changes and reset window geometry when needed.
    QString windowGeometryScreen() const;
    void setWindowGeometryScreen(const QString& sig);

    void load();
    void save() const;
    void bootstrap(const QString& appDir);

    QString modelConfigPath(const QString& modelFolder) const;

    bool enableBlink() const;
    bool enableBreath() const;
    bool enableGaze() const;
    bool enablePhysics() const;

    void setEnableBlink(bool v);
    void setEnableBreath(bool v);
    void setEnableGaze(bool v);
    void setEnablePhysics(bool v);

    QString watermarkExpPath() const;
    void setWatermarkExpPath(const QString& absPath);

    // ---- Motion / Expression selection (per model config) ----
    QString selectedMotionGroup() const;
    void setSelectedMotionGroup(const QString& group);
    QString selectedExpressionName() const;
    void setSelectedExpressionName(const QString& name);

    int textureMaxDim() const;
    void setTextureMaxDim(int dim);
    int msaaSamples() const;
    void setMsaaSamples(int samples);

    int poseAB() const;
    void setPoseAB(int index);

    void ensureModelConfigExists(const QString& folder) const;

    // ---- AI settings (stored in global config.json) ----
    QString characterName() const;
    void setCharacterName(const QString& name);
    int chatContextMessages() const;
    void setChatContextMessages(int count);
    int llmMaxTokens() const;
    void setLlmMaxTokens(int maxTokens);
    QString aiSystemPrompt() const;
    void setAiSystemPrompt(const QString& prompt);
    QString llmStyle() const;
    void setLlmStyle(const QString& style);
    QString llmModelSize() const;
    void setLlmModelSize(const QString& size);
    QString chatBubbleStyle() const;
    void setChatBubbleStyle(const QString& styleId);

    // ---- Offline voice (Sherpa-onnx, stored in global config.json) ----
    bool offlineTtsEnabled() const;
    void setOfflineTtsEnabled(bool v);
    QString sherpaOnnxBinDir() const;
    void setSherpaOnnxBinDir(const QString& dir);
    QString sherpaTtsModel() const;
    void setSherpaTtsModel(const QString& modelId);
    QString sherpaTtsArgs() const;
    void setSherpaTtsArgs(const QString& args);
    int sherpaTtsSid() const;
    void setSherpaTtsSid(int sid);
    int ttsVolumePercent() const;
    void setTtsVolumePercent(int v);

    // Cache dir for temp files (e.g. TTS audio)
    QString cacheDir() const;

    // Chat persistence (per model, stored under the platform-specific app data directory)
    QString chatsDir() const;
    QString chatPathForModel(const QString& modelFolder) const; // xxxx.chat.json

    // ---- Advanced runtime selection (stored in global config.json) ----
    // Empty means: follow system default.
    QString preferredScreenName() const;
    void setPreferredScreenName(const QString& name);

    // ---- Window behavior (stored in global config.json) ----
    bool windowAlwaysOnTop() const;
    void setWindowAlwaysOnTop(bool v);
    bool windowTransparentBackground() const;
    void setWindowTransparentBackground(bool v);
    bool windowMousePassthrough() const;
    void setWindowMousePassthrough(bool v);

    // ---- Local scheduled reminders (stored in global config.json) ----
    QJsonArray reminderTasks() const;
    void setReminderTasks(const QJsonArray& tasks);

    // Store QAudioDevice::id() as Base64 for JSON.
    // Empty means: follow system default.
    // Removed in offline mode

private:
    SettingsManager() = default;

    QString m_modelsRoot;
    QString m_selectedFolder;
    QString m_theme{"era"};
    QString m_currentLanguage;
    int m_winX{-1}, m_winY{-1}, m_winW{0}, m_winH{0};
    QString m_winScreen; // screen signature used when saving window geometry
    int m_textureMaxDim{2048};
    int m_msaaSamples{4};

    // Advanced
    QString m_preferredScreenName;
    bool m_windowAlwaysOnTop{true};
    bool m_windowTransparentBackground{true};
    bool m_windowMousePassthrough{false};

    // AI
    QString m_characterName{QStringLiteral("小墨")};
    int m_chatContextMessages{16};
    int m_llmMaxTokens{256};
    QString m_aiSystemPrompt{"你是桌面宠物$name$，请用简短、友好的中文回复用户。"};
    QString m_llmStyle{QStringLiteral("Original")};
    QString m_llmModelSize{QStringLiteral("1.5B")};
    QString m_chatBubbleStyle{QStringLiteral("Era")};

    // Offline voice
    bool m_offlineTtsEnabled{false};
    QString m_sherpaOnnxBinDir;
    QString m_sherpaTtsModel;
    QString m_sherpaTtsArgs;
    int m_sherpaTtsSid{0};
    int m_ttsVolumePercent{80};

    QJsonArray m_reminderTasks;

    QJsonObject loadModelConfigObject(const QString& folder) const;
    void saveModelConfigObject(const QString& folder, const QJsonObject& obj) const;
};
