#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

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

    void reloadFromSettings();
    void start();
    void stop();

    void startListeningOnce();
    void cancelListening();

    void speakText(const QString& text);

signals:
    void wakeWordDetected();
    void sttPartialText(const QString& text);
    void sttFinalText(const QString& text);

private:
    struct SettingsSnapshot {
        bool sttEnabled{false};
        bool ttsEnabled{false};
        bool wakeEnabled{false};
        QString binDir;
        QString wakeWord;
        QString kwsArgs;
        QString sttArgs;
        QString ttsArgs;
        int ttsVolumePercent{80};
    };

    void applySettingsSnapshot(const SettingsSnapshot& next);
    SettingsSnapshot readSettings() const;

    QString exePath(const QString& baseName) const;
    QStringList splitArgs(const QString& args) const;

    void startKws();
    void stopKws();
    void startStt();
    void stopStt();
    void finalizeStt();
    void startTts(const QString& text);
    void stopTts();

    void onKwsReadyRead();
    void onSttReadyRead();

private:
    SettingsSnapshot m_settings;

    QProcess* m_kws{nullptr};
    QProcess* m_stt{nullptr};
    QProcess* m_tts{nullptr};
    QTimer* m_sttIdleTimer{nullptr};

    QString m_sttBest;
    QString m_sttLastPartial;

#if defined(AMAIGIRL_USE_QT_MULTIMEDIA)
    QMediaPlayer* m_ttsPlayer{nullptr};
    QAudioOutput* m_ttsAudio{nullptr};
    QString m_ttsWavPath;
#endif
};
