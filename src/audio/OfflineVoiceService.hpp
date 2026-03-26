#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <memory>

class QProcess;
class QTimer;
#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
class QMediaPlayer;
class QAudioOutput;
#endif

class OfflineVoiceService : public QObject
{
    Q_OBJECT
public:
    explicit OfflineVoiceService(QObject* parent = nullptr);
    ~OfflineVoiceService() override;

    void reloadFromSettings();
    void start();
    void stop();

    void speakText(const QString& text);

private:
    struct SettingsSnapshot {
        bool ttsEnabled{false};
        QString binDir;
        QString ttsArgs;
        int ttsVolumePercent{80};
    };

    void applySettingsSnapshot(const SettingsSnapshot& next);
    SettingsSnapshot readSettings() const;

    QString exePath(const QString& baseName) const;
    QStringList splitArgs(const QString& args) const;

    void startTts(const QString& text);
    void stopTts();

private:
    SettingsSnapshot m_settings;

    QProcess* m_tts{nullptr};

#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    QMediaPlayer* m_ttsPlayer{nullptr};
    QAudioOutput* m_ttsAudio{nullptr};
    QString m_ttsWavPath;
#endif
};
