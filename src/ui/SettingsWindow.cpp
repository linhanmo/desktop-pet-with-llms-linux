#include "ui/SettingsWindow.hpp"
#include "common/SettingsManager.hpp"
#include "common/Utils.hpp"
#include <QStandardPaths>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>
#include <QApplication>
#include <QCoreApplication>
#include <QPalette>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QGridLayout>
#include "ui/theme/ThemeApi.hpp"
#include "ui/theme/ThemeWidgets.hpp"
#include <QHBoxLayout>
#include <QFormLayout>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QDebug>
#include <QIcon>
#include <QFileDialog>
#include <QMessageBox>
#include <QDesktopServices>
#include <QUrl>
#include <QStyleFactory>
#include <QStyle>
#include <QFileSystemWatcher>
#include <QTimer>
#include <QCheckBox>
#include <QDialog>
#include <QTimeEdit>
#include <QSpinBox>
#include <QToolButton>
#include <QListWidget>
#include <QUuid>
#include <QScreen>
#include <QGuiApplication>
#include <QCursor>
#include <QFont>
#include <QLineEdit>
#include <algorithm>
#include <QPlainTextEdit>
#include <QIntValidator>
#include <QSlider>
#include <QLocale>
#include <QStyleHints>
#include <QtGlobal>

// Helper copy (recursive)
static bool copyRecursively(const QString& srcPath, const QString& dstPath) {
    QDir src(srcPath);
    if (!src.exists()) return false;
    QDir dst(dstPath);
    if (!dst.exists()) { if (!dst.mkpath(".")) return false; }
    for (const QFileInfo& info : src.entryInfoList(QDir::NoDotAndDotDot | QDir::Dirs | QDir::Files | QDir::Hidden)) {
        QString rel = src.relativeFilePath(info.absoluteFilePath());
        QString to = dst.filePath(rel);
        if (info.isDir()) {
            if (!copyRecursively(info.absoluteFilePath(), to)) return false;
        } else {
            QDir toDir = QFileInfo(to).dir(); if (!toDir.exists()) toDir.mkpath(".");
            if (!QFile::copy(info.absoluteFilePath(), to)) {
                QFile::remove(to); // try overwrite
                if (!QFile::copy(info.absoluteFilePath(), to)) return false;
            }
        }
    }
    return true;
}

namespace {
constexpr int kFormHorizontalSpacing = 12;
constexpr int kFormVerticalSpacing = 10;
constexpr int kFormLeftPadding = 20;
constexpr int kFormRightPadding = 36;
constexpr int kFormTopPadding = 12;
constexpr int kFormBottomPadding = 16;
constexpr int kInlineRowSpacing = 8;
constexpr int kSidebarFooterSpacing = 10;
constexpr int kSidebarThemeIconSize = 22;
constexpr int kSidebarFooterTopPadding = 14;
constexpr int kSidebarFooterBottomPadding = 12;
constexpr int kSidebarFooterRightInset = 14;

bool currentColorThemeIsDark(const QWidget* context)
{
    if (const QStyleHints* hints = QGuiApplication::styleHints())
    {
        const Qt::ColorScheme scheme = hints->colorScheme();
        if (scheme == Qt::ColorScheme::Dark)
            return true;
        if (scheme == Qt::ColorScheme::Light)
            return false;
    }

    const QPalette palette = context ? context->palette() : QApplication::palette();
    const qreal avgLightness = (palette.color(QPalette::Window).lightnessF()
        + palette.color(QPalette::Base).lightnessF()) * 0.5;
    return avgLightness < 0.5;
}
} // namespace

static void applyLeftAlignedFormLayout(QFormLayout* form)
{
    if (!form) return;
    // Keep all tabs visually consistent and reserve left/right breathing room.
    form->setContentsMargins(kFormLeftPadding, kFormTopPadding, kFormRightPadding, kFormBottomPadding);
    form->setHorizontalSpacing(kFormHorizontalSpacing);
    form->setVerticalSpacing(kFormVerticalSpacing);
    form->setFormAlignment(Qt::AlignLeft | Qt::AlignTop);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setRowWrapPolicy(QFormLayout::DontWrapRows);
}

class SettingsWindow::Impl {
public:
    QWidget* sidebar{nullptr};
    ThemeWidgets::TabBar*     tabs{nullptr};
    QWidget* sidebarFooter{nullptr};
    QLabel* themeSchemeIconLabel{nullptr};
    QStackedWidget* tabStack{nullptr};
    QWidget* basic{nullptr};
    QFormLayout* basicForm{nullptr};
    QWidget* pathRowWidget{nullptr};
    QWidget* topRowWidget{nullptr};
    ThemeWidgets::ComboBox* modelCombo{nullptr};
    QLabel* pathLabel{nullptr};
    ThemeWidgets::Button* chooseBtn{nullptr};
    ThemeWidgets::Button* openBtn{nullptr};
    ThemeWidgets::Button* resetDirBtn{nullptr};
    ThemeWidgets::ComboBox* themeCombo{nullptr};
    ThemeWidgets::ComboBox* languageCombo{nullptr};
    ThemeWidgets::Button* resetBtn{nullptr};
    ThemeWidgets::Switch* chkAlwaysOnTop{nullptr};
    ThemeWidgets::Switch* chkTransparentBg{nullptr};
    ThemeWidgets::Switch* chkMousePassthrough{nullptr};
    ThemeWidgets::ComboBox* petBubbleCombo{nullptr};
    QLabel* tipAlwaysOnTop{nullptr};
    QLabel* tipTransparentBg{nullptr};
    QLabel* tipMousePassthrough{nullptr};

    QWidget* modelTab{nullptr};
    QLabel* modelNameTitle{nullptr};
    QLabel* watermarkTitle{nullptr};
    QLabel* curModelName{nullptr};
    ThemeWidgets::Button* openModelDirBtn{nullptr};
    ThemeWidgets::Switch* chkBlink{nullptr};
    ThemeWidgets::Switch* chkBreath{nullptr};
    ThemeWidgets::Switch* chkGaze{nullptr};
    ThemeWidgets::Switch* chkPhysics{nullptr};
    QLabel* tipBreath{nullptr};
    QLabel* tipBlink{nullptr};
    QLabel* tipGaze{nullptr};
    QLabel* tipPhysics{nullptr};

    QLabel* wmFileLabel{nullptr};
    ThemeWidgets::Button* wmChooseBtn{nullptr};
    ThemeWidgets::Button* wmClearBtn{nullptr};

    // AI tab
    QWidget* aiTab{nullptr};
    QFormLayout* aiForm{nullptr};
    ThemeWidgets::ComboBox* llmModelSizeCombo{nullptr};
    ThemeWidgets::ComboBox* llmStyleCombo{nullptr};
    ThemeWidgets::LineEdit* characterName{nullptr};
    ThemeWidgets::LineEdit* chatContextMessages{nullptr};
    ThemeWidgets::LineEdit* llmMaxTokens{nullptr};
    QSlider* chatContextSlider{nullptr};
    QSlider* llmMaxTokensSlider{nullptr};
    QLabel* chatContextHint{nullptr};
    QLabel* llmMaxTokensHint{nullptr};
    ThemeWidgets::LineEdit* aiBaseUrl{nullptr};
    ThemeWidgets::LineEdit* aiKey{nullptr};
    ThemeWidgets::LineEdit* aiModel{nullptr};
    ThemeWidgets::PlainTextEdit* aiSystemPrompt{nullptr};
    ThemeWidgets::Switch* aiStream{nullptr};
    QWidget* characterNameRow{nullptr};
    QWidget* chatContextMessagesRow{nullptr};
    QWidget* llmMaxTokensRow{nullptr};
    QWidget* aiBaseUrlRow{nullptr};
    QWidget* aiKeyRow{nullptr};
    QWidget* aiModelRow{nullptr};
    QWidget* aiSystemPromptRow{nullptr};
    QWidget* ttsBaseUrlRow{nullptr};
    QWidget* ttsKeyRow{nullptr};
    QWidget* ttsModelRow{nullptr};
    QWidget* ttsVoiceRow{nullptr};
    QLabel* tipAiBaseUrl{nullptr};
    QLabel* tipAiKey{nullptr};
    QLabel* tipAiModel{nullptr};
    QLabel* tipAiSystemPrompt{nullptr};
    QLabel* tipCharacterName{nullptr};
    QLabel* tipChatContextMessages{nullptr};
    QLabel* tipLlmMaxTokens{nullptr};
    QLabel* tipAiStream{nullptr};
    QLabel* tipTtsBaseUrl{nullptr};
    QLabel* tipTtsKey{nullptr};
    QLabel* tipTtsModel{nullptr};
    QLabel* tipTtsVoice{nullptr};

    // TTS
    ThemeWidgets::LineEdit* ttsBaseUrl{nullptr};
    ThemeWidgets::LineEdit* ttsKey{nullptr};
    ThemeWidgets::LineEdit* ttsModel{nullptr};
    ThemeWidgets::LineEdit* ttsVoice{nullptr};

    // Offline voice
    ThemeWidgets::Switch* offlineTtsEnabled{nullptr};
    ThemeWidgets::LineEdit* sherpaOnnxBinDir{nullptr};
    ThemeWidgets::Button* sherpaChooseBtn{nullptr};
    ThemeWidgets::ComboBox* sherpaTtsModelCombo{nullptr};
    QLabel* sherpaTtsModelHint{nullptr};
    ThemeWidgets::PlainTextEdit* sherpaTtsArgs{nullptr};
    QWidget* sherpaBinRow{nullptr};
    QSpinBox* ttsSidSpin{nullptr};
    QLabel* ttsSidHint{nullptr};
    QLabel* ttsSidDesc{nullptr};
    QSlider* ttsVolumeSlider{nullptr};
    ThemeWidgets::LineEdit* ttsVolumeEdit{nullptr};

    QWidget* advancedTab{nullptr};
    QFormLayout* advancedForm{nullptr};
    QWidget* cleanupRowWidget{nullptr};
    ThemeWidgets::Button* clearCacheBtn{nullptr};
    ThemeWidgets::Button* clearChatsBtn{nullptr};
    ThemeWidgets::ComboBox* texCapCombo{nullptr};
    ThemeWidgets::ComboBox* msaaCombo{nullptr};

    // Advanced: new combos
    ThemeWidgets::ComboBox* screenCombo{nullptr};
    ThemeWidgets::ComboBox* audioOutCombo{nullptr};

    // Local reminders
    QListWidget* reminderList{nullptr};
    ThemeWidgets::Button* reminderAddBtn{nullptr};
    ThemeWidgets::Button* reminderEditBtn{nullptr};
    ThemeWidgets::Button* reminderRemoveBtn{nullptr};

    QFileSystemWatcher* fsw{nullptr};
    QTimer* debounce{nullptr};
    QWidget* central{nullptr};
};

SettingsWindow::SettingsWindow(QWidget *parent) : QMainWindow(parent), d(new Impl) {
    setWindowTitle(tr("设置"));
    setWindowFlag(Qt::Window, true);
    setWindowFlag(Qt::CustomizeWindowHint, false);
    setWindowFlag(Qt::WindowMaximizeButtonHint, false);
    resize(520, 420);

    QScreen* s = nullptr;
    if (parentWidget() && parentWidget()->screen()) s = parentWidget()->screen();
    if (!s) s = QGuiApplication::screenAt(QCursor::pos());
    if (!s) s = QGuiApplication::primaryScreen();
    if (s) {
        QRect scr = s->availableGeometry();
        int w = width(); int h = height();
        move(scr.x() + (scr.width() - w)/2, scr.y() + (scr.height() - h)/2);
    }

    d->central = new QWidget(this);
    setCentralWidget(d->central);

    auto rootLay = new QHBoxLayout(d->central);
    rootLay->setContentsMargins(12,12,12,12);
    rootLay->setSpacing(0);

    d->sidebar = new QWidget(d->central);
    d->sidebar->setObjectName(QStringLiteral("settingsSidebar"));
    auto sideLay = new QGridLayout(d->sidebar);
    sideLay->setContentsMargins(0, 0, 0, 0);
    sideLay->setHorizontalSpacing(0);
    sideLay->setVerticalSpacing(0);

    d->tabs = new ThemeWidgets::TabBar(d->sidebar);
    {
        QFont tabFont = d->tabs->font();
        if (tabFont.pointSizeF() > 0.0)
            tabFont.setPointSizeF(tabFont.pointSizeF() + 2.0);
        else
            tabFont.setPointSize(14);
        d->tabs->setFont(tabFont);
    }
    d->tabs->setOrientation(ThemeWidgets::TabBar::Orientation::Vertical);

    d->sidebarFooter = new QWidget(d->sidebar);
    d->sidebarFooter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
    auto footerLay = new QVBoxLayout(d->sidebarFooter);
    footerLay->setContentsMargins(0, kSidebarFooterTopPadding, kSidebarFooterRightInset, kSidebarFooterBottomPadding);
    footerLay->setSpacing(kSidebarFooterSpacing);

    d->themeSchemeIconLabel = new QLabel(d->sidebarFooter);
    d->themeSchemeIconLabel->setObjectName(QStringLiteral("settingsThemeSchemeIcon"));
    d->themeSchemeIconLabel->setAlignment(Qt::AlignCenter);
    d->themeSchemeIconLabel->setFixedSize(kSidebarThemeIconSize, kSidebarThemeIconSize);

    footerLay->addWidget(d->themeSchemeIconLabel, 0, Qt::AlignHCenter);

    sideLay->addWidget(d->tabs, 0, 0);
    sideLay->addWidget(d->sidebarFooter, 0, 0, Qt::AlignBottom);

    d->tabStack = new QStackedWidget(d->central);
    rootLay->addWidget(d->sidebar, 0);
    rootLay->addWidget(d->tabStack, 1);
    connect(d->tabs, &ThemeWidgets::TabBar::currentChanged, this, [this](int index){
        d->tabStack->setCurrentIndex(index);
        if (auto* fw = QApplication::focusWidget()) fw->clearFocus();
        d->tabs->setFocus(Qt::OtherFocusReason);
    });

    refreshSidebarThemeIndicator();

    // Basic tab
    d->basic = new QWidget(d->tabStack);
    auto form = new QFormLayout(d->basic);
    d->basicForm = form;
    applyLeftAlignedFormLayout(form);

    // Model path row
    auto pathRow = new QWidget(d->basic);
    d->pathRowWidget = pathRow;
    auto hl = new QHBoxLayout(pathRow); hl->setContentsMargins(0,0,0,0);
    hl->setSpacing(kInlineRowSpacing);
    d->pathLabel = new QLabel(SettingsManager::instance().modelsRoot(), pathRow);
    d->chooseBtn = new ThemeWidgets::Button(tr("选择路径"), pathRow);
    d->chooseBtn->setTone(ThemeWidgets::Button::Tone::Link);
    d->openBtn   = new ThemeWidgets::Button(tr("打开路径"), pathRow);
    d->openBtn->setTone(ThemeWidgets::Button::Tone::Link);
    d->resetDirBtn = new ThemeWidgets::Button(tr("恢复默认"), pathRow);
    d->resetDirBtn->setTone(ThemeWidgets::Button::Tone::Neutral);
    hl->addWidget(d->pathLabel, 1);
    hl->addWidget(d->chooseBtn);
    hl->addWidget(d->openBtn);
    hl->addWidget(d->resetDirBtn);
    form->addRow(tr("模型路径："), pathRow);

    // Current model row
    d->modelCombo = new ThemeWidgets::ComboBox(d->basic);
    d->resetBtn = new ThemeWidgets::Button(tr("还原初始状态"), d->basic);
    d->resetBtn->setTone(ThemeWidgets::Button::Tone::Danger);
    auto topRow = new QWidget(d->basic);
    d->topRowWidget = topRow;
    auto topLay = new QHBoxLayout(topRow); topLay->setContentsMargins(0,0,0,0);
    topLay->setSpacing(kInlineRowSpacing);
    topLay->addWidget(d->modelCombo, 1);
    topLay->addWidget(d->resetBtn);
    form->addRow(tr("当前模型："), topRow);

    // Theme
    d->themeCombo = new ThemeWidgets::ComboBox(d->basic);
    const QStringList availableThemes = Theme::availableThemeIds();
    for (const QString& id : availableThemes)
    {
        if (id == QStringLiteral("era"))
            d->themeCombo->addItem(tr("Era"), id);
        else
            d->themeCombo->addItem(id, id);
    }
    if (d->themeCombo->count() == 0)
        d->themeCombo->addItem(tr("Era"), QStringLiteral("era"));

    const QString configuredTheme = Theme::normalizeThemeId(SettingsManager::instance().theme());
    int themeIndex = d->themeCombo->findData(configuredTheme);
    if (themeIndex < 0)
    {
        const QString fallbackLabel = configuredTheme == QStringLiteral("era") ? tr("Era") : configuredTheme;
        d->themeCombo->addItem(fallbackLabel, configuredTheme);
        themeIndex = d->themeCombo->findData(configuredTheme);
    }
    d->themeCombo->setCurrentIndex(themeIndex >= 0 ? themeIndex : 0);
    form->addRow(tr("当前主题："), d->themeCombo);

    d->languageCombo = new ThemeWidgets::ComboBox(d->basic);
    d->languageCombo->addItem(tr("跟随系统"), QStringLiteral("system"));
    d->languageCombo->addItem(tr("简体中文"), QStringLiteral("zh_CN"));
    d->languageCombo->addItem(QStringLiteral("English"), QStringLiteral("en_US"));
    {
        const QString saved = SettingsManager::instance().currentLanguage();
        const QString sysCode = QLocale::system().name().startsWith("zh", Qt::CaseInsensitive)
                                    ? QStringLiteral("zh_CN")
                                    : QStringLiteral("en_US");
        if (saved == sysCode) d->languageCombo->setCurrentIndex(0);
        else {
            const int idx = d->languageCombo->findData(saved);
            d->languageCombo->setCurrentIndex(idx >= 0 ? idx : 0);
        }
    }
    form->addRow(tr("当前语言："), d->languageCombo);

    {
        d->chkAlwaysOnTop = new ThemeWidgets::Switch(tr("全局置顶"), d->basic);
        d->chkTransparentBg = new ThemeWidgets::Switch(tr("透明背景"), d->basic);
        d->chkMousePassthrough = new ThemeWidgets::Switch(tr("鼠标穿透窗口"), d->basic);

        d->chkAlwaysOnTop->setChecked(SettingsManager::instance().windowAlwaysOnTop());
        d->chkTransparentBg->setChecked(SettingsManager::instance().windowTransparentBackground());
        d->chkMousePassthrough->setChecked(SettingsManager::instance().windowMousePassthrough());

        auto mkInlineTipRowBasic = [this](ThemeWidgets::Switch* chk, const QString& tip, QLabel** outTip){
            QWidget* row = new QWidget(d->basic);
            auto lay = new QHBoxLayout(row);
            lay->setContentsMargins(0,0,0,0);
            lay->setSpacing(kInlineRowSpacing);
            chk->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
            lay->addWidget(chk);
            lay->addWidget(new QLabel(" ", row));
            auto tipLbl = new QLabel(QString::fromUtf8("ⓘ"), row);
            tipLbl->setToolTip(tip);
            tipLbl->setCursor(Qt::WhatsThisCursor);
            if (outTip) *outTip = tipLbl;
            lay->addWidget(tipLbl);
            lay->addStretch(1);
            return row;
        };

        form->addRow(mkInlineTipRowBasic(
            d->chkAlwaysOnTop,
            tr("开启后桌宠窗口将始终置顶显示（全局置顶）。"),
            &d->tipAlwaysOnTop
        ));
        form->addRow(mkInlineTipRowBasic(
            d->chkTransparentBg,
            tr("开启后窗口背景为透明，仅显示角色本体；关闭后窗口为不透明背景。"),
            &d->tipTransparentBg
        ));
        form->addRow(mkInlineTipRowBasic(
            d->chkMousePassthrough,
            tr("开启后窗口将完全不接收鼠标输入（可点击穿透到桌面/其他窗口）。\n\n提示：开启后需要 Alt+Tab 切回本程序，再在设置里关闭。"),
            &d->tipMousePassthrough
        ));
    }

    d->petBubbleCombo = new ThemeWidgets::ComboBox(d->basic);
    d->petBubbleCombo->addItem(tr("圆角"), QStringLiteral("Round"));
    d->petBubbleCombo->addItem(tr("描边"), QStringLiteral("Outline"));
    d->petBubbleCombo->addItem(QStringLiteral("iMessage"), QStringLiteral("iMessage"));
    d->petBubbleCombo->addItem(tr("云朵"), QStringLiteral("Cloud"));
    d->petBubbleCombo->addItem(tr("爱心"), QStringLiteral("Heart"));
    d->petBubbleCombo->addItem(tr("漫画"), QStringLiteral("Comic"));
    d->petBubbleCombo->setToolTip(tr("Live2D模型旁边显示的 LLM 输出气泡样式"));
    {
        const QString saved = SettingsManager::instance().chatBubbleStyle();
        const int idx = d->petBubbleCombo->findData(saved);
        d->petBubbleCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }
    form->addRow(tr("输出气泡："), d->petBubbleCombo);

    d->tabStack->addWidget(d->basic);
    d->tabs->addTab(
        tr("基本设置"),
        Theme::themedIcon(Theme::IconToken::SettingsBasic)
    );

    // Model settings tab
    d->modelTab = new QWidget(d->tabStack);
    auto modelForm = new QFormLayout(d->modelTab);
    applyLeftAlignedFormLayout(modelForm);

    auto modelNameRow = new QWidget(d->modelTab);
    auto row1 = new QHBoxLayout(modelNameRow);
    row1->setSpacing(kInlineRowSpacing);
    row1->setContentsMargins(0,0,0,0);
    d->curModelName = new QLabel("", d->modelTab);
    d->openModelDirBtn = new ThemeWidgets::Button(tr("打开当前模型路径"), d->modelTab);
    d->openModelDirBtn->setTone(ThemeWidgets::Button::Tone::Link);
    d->modelNameTitle = new QLabel(tr("模型名称："), d->modelTab);
    row1->addWidget(d->curModelName, 1);
    row1->addWidget(d->openModelDirBtn);
    modelForm->addRow(d->modelNameTitle, modelNameRow);

    auto wmRowWidget = new QWidget(d->modelTab);
    auto wmRow = new QHBoxLayout(wmRowWidget);
    wmRow->setSpacing(kInlineRowSpacing);
    wmRow->setContentsMargins(0,0,0,0);
    d->watermarkTitle = new QLabel(tr("去除水印："), d->modelTab);
    d->wmFileLabel = new QLabel(tr("无"), d->modelTab);
    d->wmFileLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    d->wmChooseBtn = new ThemeWidgets::Button(tr("选择文件"), d->modelTab);
    d->wmChooseBtn->setTone(ThemeWidgets::Button::Tone::Link);
    d->wmClearBtn  = new ThemeWidgets::Button(tr("取消所选"), d->modelTab);
    d->wmClearBtn->setTone(ThemeWidgets::Button::Tone::Neutral);
    wmRow->addWidget(d->wmFileLabel, 1);
    wmRow->addWidget(d->wmChooseBtn);
    wmRow->addWidget(d->wmClearBtn);
    modelForm->addRow(d->watermarkTitle, wmRowWidget);

    d->chkBreath  = new ThemeWidgets::Switch(tr("自动呼吸"), d->modelTab);
    d->chkBlink   = new ThemeWidgets::Switch(tr("自动眨眼"), d->modelTab);
    d->chkGaze    = new ThemeWidgets::Switch(tr("视线跟踪"), d->modelTab);
    d->chkPhysics = new ThemeWidgets::Switch(tr("物理模拟"), d->modelTab);

    d->chkBreath->setChecked(SettingsManager::instance().enableBreath());
    d->chkBlink->setChecked(SettingsManager::instance().enableBlink());
    d->chkGaze->setChecked(SettingsManager::instance().enableGaze());
    d->chkPhysics->setChecked(SettingsManager::instance().enablePhysics());

    auto mkInlineTipRow = [this](ThemeWidgets::Switch* chk, const QString& tip, QLabel** outTip){
        QWidget* row = new QWidget(d->modelTab);
        auto lay = new QHBoxLayout(row);
        lay->setContentsMargins(0,0,0,0);
        lay->setSpacing(kInlineRowSpacing);
        chk->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
        lay->addWidget(chk);
        lay->addWidget(new QLabel(" ", row));
        auto tipLbl = new QLabel(QString::fromUtf8("ⓘ"), row);
        tipLbl->setToolTip(tip);
        tipLbl->setCursor(Qt::WhatsThisCursor);
        if (outTip) *outTip = tipLbl;
        lay->addWidget(tipLbl);
        lay->addStretch(1);
        return row;
    };

    modelForm->addRow(mkInlineTipRow(d->chkBreath,  tr("让角色在静止时也会轻微起伏（身体角度/呼吸参数）。开启视线跟踪时，自动呼吸不再影响头部；关闭后恢复。关闭本项后参数会复位。"), &d->tipBreath));
    modelForm->addRow(mkInlineTipRow(d->chkBlink,   tr("自动眨眼并保留更自然的间隔与瞬目效果。关闭后眼部相关参数复位。"), &d->tipBlink));
    modelForm->addRow(mkInlineTipRow(d->chkGaze,    tr("眼球、头部与身体随鼠标方向轻微转动，远距离时幅度会衰减。默认关闭；开启后将屏蔽自动呼吸对头部的影响。关闭后恢复眼球微动策略。"), &d->tipGaze));
    modelForm->addRow(mkInlineTipRow(d->chkPhysics, tr("根据模型 physics3.json 的配置驱动物理（如头发/衣物摆动）。关闭后复位受影响参数。"), &d->tipPhysics));
    modelForm->addItem(new QSpacerItem(0,0,QSizePolicy::Minimum,QSizePolicy::Expanding));
    d->tabStack->addWidget(d->modelTab);
    d->tabs->addTab(
        tr("模型设置"),
        Theme::themedIcon(Theme::IconToken::SettingsModel)
    );

    // ---- AI tab ----
    d->aiTab = new QWidget(d->tabStack);
    {
        auto form2 = new QFormLayout(d->aiTab);
        d->aiForm = form2;
        applyLeftAlignedFormLayout(form2);

        d->characterName = new ThemeWidgets::LineEdit(SettingsManager::instance().characterName(), d->aiTab);
        d->characterName->setPlaceholderText(QStringLiteral("小墨"));

        d->chatContextMessages = new ThemeWidgets::LineEdit(QString::number(SettingsManager::instance().chatContextMessages()), d->aiTab);
        d->chatContextMessages->setPlaceholderText(QStringLiteral("16"));
        d->chatContextMessages->setValidator(new QIntValidator(0, 200, d->chatContextMessages));

        d->llmMaxTokens = new ThemeWidgets::LineEdit(QString::number(SettingsManager::instance().llmMaxTokens()), d->aiTab);
        d->llmMaxTokens->setPlaceholderText(QStringLiteral("256"));
        d->llmMaxTokens->setValidator(new QIntValidator(1, 4096, d->llmMaxTokens));

        d->aiSystemPrompt = new ThemeWidgets::PlainTextEdit(SettingsManager::instance().aiSystemPrompt(), d->aiTab);
        d->aiSystemPrompt->setPlaceholderText(tr("支持变量：$name$（角色名称）"));

        // AI/TTS tooltip helper (same symbol style as 模型设置)
        auto mkInlineInfo = [](const QString& tip, QWidget* parent, QLabel** outTip){
            QWidget* w = new QWidget(parent);
            auto hl = new QHBoxLayout(w);
            hl->setContentsMargins(0,0,0,0);
            hl->setSpacing(kInlineRowSpacing);
            hl->addStretch(1);
            hl->addWidget(new QLabel(" ", w));
            auto tipLbl = new QLabel(QString::fromUtf8("ⓘ"), w);
            tipLbl->setToolTip(tip);
            tipLbl->setCursor(Qt::WhatsThisCursor);
            if (outTip) *outTip = tipLbl;
            hl->addWidget(tipLbl);
            return w;
        };

        auto mkRowWithInfo = [&](const QString& label, QWidget* field, const QString& tip, QWidget** outRow, QLabel** outTip){
            QWidget* row = new QWidget(d->aiTab);
            auto hl = new QHBoxLayout(row);
            hl->setContentsMargins(0,0,0,0);
            hl->setSpacing(kInlineRowSpacing);
            hl->addWidget(field, 1);
            hl->addWidget(mkInlineInfo(tip, row, outTip));
            form2->addRow(label, row);
            if (outRow) *outRow = row;
        };

        mkRowWithInfo(
            tr("角色名称："),
            d->characterName,
            tr(
                "对话时 AI 的角色名。\n\n"
                "会在发送前替换 system prompt（对话人设）里的 $name$。\n\n"
                "示例：角色名称 = 小墨，那么对话人设中的“你是 $name$”会变成“你是 小墨”。"
            ),
            &d->characterNameRow,
            &d->tipCharacterName
        );

        mkRowWithInfo(
            tr("上下文条数："),
            [&]{
                auto* field = new QWidget(d->aiTab);
                auto* vl = new QVBoxLayout(field);
                vl->setContentsMargins(0,0,0,0);
                vl->setSpacing(4);

                auto* top = new QWidget(field);
                auto* hl = new QHBoxLayout(top);
                hl->setContentsMargins(0,0,0,0);
                hl->setSpacing(kInlineRowSpacing);

                d->chatContextSlider = new QSlider(Qt::Horizontal, top);
                d->chatContextSlider->setRange(0, 200);
                d->chatContextSlider->setValue(SettingsManager::instance().chatContextMessages());
                d->chatContextSlider->setSingleStep(1);
                d->chatContextSlider->setPageStep(4);

                d->chatContextMessages->setFixedWidth(90);

                hl->addWidget(d->chatContextSlider, 1);
                hl->addWidget(d->chatContextMessages, 0);

                d->chatContextHint = new QLabel(tr("范围：0～200；默认：16；推荐：1.5B=12，7B=24"), field);
                d->chatContextHint->setWordWrap(true);
                {
                    auto f = d->chatContextHint->font();
                    f.setPointSize(qMax(8, f.pointSize() - 1));
                    d->chatContextHint->setFont(f);
                }
                {
                    auto pal = d->chatContextHint->palette();
                    pal.setColor(QPalette::WindowText, pal.color(QPalette::PlaceholderText));
                    d->chatContextHint->setPalette(pal);
                }

                vl->addWidget(top);
                vl->addWidget(d->chatContextHint);
                return field;
            }(),
            tr(
                "发送给本地 LLM 的历史消息条数（只取最近的 user/assistant 消息）。\n\n"
                "数值越大：越“记得住”之前的对话，但速度更慢、也更容易输出变长。\n"
                "数值越小：响应更快，但更容易忘记上下文。\n\n"
                "范围：0～200（设为 0 表示不带历史）。\n\n"
                "推荐：\n"
                "  • 1.5B：12\n"
                "  • 7B：24\n\n"
                "默认：16"
            ),
            &d->chatContextMessagesRow,
            &d->tipChatContextMessages
        );

        mkRowWithInfo(
            tr("maxTokens："),
            [&]{
                auto* field = new QWidget(d->aiTab);
                auto* vl = new QVBoxLayout(field);
                vl->setContentsMargins(0,0,0,0);
                vl->setSpacing(4);

                auto* top = new QWidget(field);
                auto* hl = new QHBoxLayout(top);
                hl->setContentsMargins(0,0,0,0);
                hl->setSpacing(kInlineRowSpacing);

                d->llmMaxTokensSlider = new QSlider(Qt::Horizontal, top);
                d->llmMaxTokensSlider->setRange(1, 4096);
                d->llmMaxTokensSlider->setValue(SettingsManager::instance().llmMaxTokens());
                d->llmMaxTokensSlider->setSingleStep(8);
                d->llmMaxTokensSlider->setPageStep(64);

                d->llmMaxTokens->setFixedWidth(90);

                hl->addWidget(d->llmMaxTokensSlider, 1);
                hl->addWidget(d->llmMaxTokens, 0);

                d->llmMaxTokensHint = new QLabel(tr("范围：1～4096；默认：256；推荐：1.5B=192，7B=384"), field);
                d->llmMaxTokensHint->setWordWrap(true);
                {
                    auto f = d->llmMaxTokensHint->font();
                    f.setPointSize(qMax(8, f.pointSize() - 1));
                    d->llmMaxTokensHint->setFont(f);
                }
                {
                    auto pal = d->llmMaxTokensHint->palette();
                    pal.setColor(QPalette::WindowText, pal.color(QPalette::PlaceholderText));
                    d->llmMaxTokensHint->setPalette(pal);
                }

                vl->addWidget(top);
                vl->addWidget(d->llmMaxTokensHint);
                return field;
            }(),
            tr(
                "本地 LLM 单次最多生成的 token 数（上限越大，回复可能越长，但耗时更久）。\n\n"
                "范围：1～4096。\n\n"
                "推荐：\n"
                "  • 1.5B：192\n"
                "  • 7B：384\n\n"
                "默认：256"
            ),
            &d->llmMaxTokensRow,
            &d->tipLlmMaxTokens
        );

        // 对话人设（system prompt） + tooltip
        {
            QWidget* row = new QWidget(d->aiTab);
            auto hl = new QHBoxLayout(row);
            hl->setContentsMargins(0,0,0,0);
            hl->setSpacing(kInlineRowSpacing);
            hl->addWidget(d->aiSystemPrompt, 1);
            hl->addWidget(mkInlineInfo(
                tr(
                    "System Prompt（系统提示词 / 人设），用于规定 AI 的角色、语气、规则与边界。\n\n"
                    "对应 OpenAI 参数：messages[0].role = \"system\" 的 content。\n\n"
                    "支持变量：\n"
                    "  • $name$：会在发送前替换成“角色名称”。\n\n"
                    "示例：\n"
                    "你是 $name$，是一只桌宠。回答要简短、温柔，并尽量使用口语。"
                ), row, &d->tipAiSystemPrompt));
            d->aiSystemPromptRow = row;
            form2->addRow(tr("对话人设："), row);
        }

        d->llmModelSizeCombo = new ThemeWidgets::ComboBox(d->aiTab);
        d->llmModelSizeCombo->addItem(tr("1.5B（更快）"), QStringLiteral("1.5B"));
        d->llmModelSizeCombo->addItem(tr("7B（更聪明）"), QStringLiteral("7B"));
        {
            const QString saved = SettingsManager::instance().llmModelSize();
            const int idx = d->llmModelSizeCombo->findData(saved);
            d->llmModelSizeCombo->setCurrentIndex(idx >= 0 ? idx : 0);
        }
        form2->addRow(tr("LLM规模："), d->llmModelSizeCombo);

        // LLM 风格切换（Original / Universal / Anime）
        d->llmStyleCombo = new ThemeWidgets::ComboBox(d->aiTab);
        d->llmStyleCombo->addItem(QStringLiteral("Original"), QStringLiteral("Original"));
        d->llmStyleCombo->addItem(QStringLiteral("Universal"), QStringLiteral("Universal"));
        d->llmStyleCombo->addItem(QStringLiteral("Anime"), QStringLiteral("Anime"));
        {
            const QString saved = SettingsManager::instance().llmStyle();
            const int idx = d->llmStyleCombo->findData(saved);
            d->llmStyleCombo->setCurrentIndex(idx >= 0 ? idx : 0);
        }
        form2->addRow(tr("LLM风格："), d->llmStyleCombo);

        d->offlineTtsEnabled = new ThemeWidgets::Switch(tr("启用离线语音合成（TTS，文字转语音）"), d->aiTab);
        d->offlineTtsEnabled->setChecked(SettingsManager::instance().offlineTtsEnabled());
        form2->addRow(d->offlineTtsEnabled);

        auto styleModelHint = [](QLabel* lab){
            if (!lab) return;
            lab->setWordWrap(true);
            auto f = lab->font();
            f.setPointSize(qMax(8, f.pointSize() - 1));
            lab->setFont(f);
            auto pal = lab->palette();
            pal.setColor(QPalette::WindowText, pal.color(QPalette::PlaceholderText));
            lab->setPalette(pal);
        };
        auto* ttsDesc = new QLabel(tr("TTS 会把文字转成语音播放。开启后，会自动朗读 AI 的最终回复，适合免看屏幕或提升沉浸感。"), d->aiTab);
        styleModelHint(ttsDesc);
        form2->addRow(QString(), ttsDesc);

        auto listVoiceModelDirs = []() -> QStringList {
            QStringList out;
            const QString appBase = QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("voice_deps/models"));
            const QString resBase = QDir(appResourcePath(QStringLiteral("voice_deps"))).filePath(QStringLiteral("models"));
            const QStringList bases{appBase, resBase};
            for (const QString& base : bases)
            {
                QDir d(base);
                if (!d.exists()) continue;
                const QStringList one = d.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
                for (const QString& name : one)
                {
                    if (!out.contains(name))
                        out.push_back(name);
                }
            }
            out.removeAll(QStringLiteral("tts"));
            out.sort();
            return out;
        };

        auto findModelDir = [](const QString& modelId) -> QString {
            const QString id = modelId.trimmed();
            if (id.isEmpty())
                return {};
            const QString appBase = QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("voice_deps/models"));
            const QString resBase = QDir(appResourcePath(QStringLiteral("voice_deps"))).filePath(QStringLiteral("models"));
            const QStringList bases{appBase, resBase};
            for (const QString& base : bases)
            {
                const QString p = QDir(base).filePath(id);
                if (QFileInfo::exists(p) && QFileInfo(p).isDir())
                    return p;
            }
            return {};
        };

        d->sherpaTtsModelCombo = new ThemeWidgets::ComboBox(d->aiTab);
        d->sherpaTtsModelCombo->addItem(tr("自动（推荐）"), QString());
        for (const QString& name : listVoiceModelDirs())
            d->sherpaTtsModelCombo->addItem(name, name);
        {
            const QString saved = SettingsManager::instance().sherpaTtsModel();
            const int idx = d->sherpaTtsModelCombo->findData(saved);
            d->sherpaTtsModelCombo->setCurrentIndex(idx >= 0 ? idx : 0);
        }
        form2->addRow(tr("TTS模型："), d->sherpaTtsModelCombo);
        d->sherpaTtsModelHint = new QLabel(d->aiTab);
        styleModelHint(d->sherpaTtsModelHint);
        form2->addRow(QString(), d->sherpaTtsModelHint);

        QWidget* sidRow = new QWidget(d->aiTab);
        auto sidHl = new QHBoxLayout(sidRow);
        sidHl->setContentsMargins(0,0,0,0);
        sidHl->setSpacing(0);

        d->ttsSidSpin = new QSpinBox(sidRow);
        d->ttsSidSpin->setButtonSymbols(QAbstractSpinBox::NoButtons);
        d->ttsSidSpin->setRange(0, 99999);
        d->ttsSidSpin->setSingleStep(1);
        d->ttsSidSpin->setAccelerated(true);
        d->ttsSidSpin->setKeyboardTracking(false);
        d->ttsSidSpin->setValue(SettingsManager::instance().sherpaTtsSid());
        d->ttsSidSpin->setFixedWidth(120);

        QWidget* sidBtns = new QWidget(sidRow);
        sidBtns->setFixedWidth(24);
        auto sidVl = new QVBoxLayout(sidBtns);
        sidVl->setContentsMargins(0,0,0,0);
        sidVl->setSpacing(0);

        auto sidUp = new QToolButton(sidBtns);
        sidUp->setText(QStringLiteral("▲"));
        sidUp->setFocusPolicy(Qt::NoFocus);
        sidUp->setAutoRepeat(true);
        sidUp->setAutoRepeatDelay(350);
        sidUp->setAutoRepeatInterval(60);
        sidUp->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

        auto sidDown = new QToolButton(sidBtns);
        sidDown->setText(QStringLiteral("▼"));
        sidDown->setFocusPolicy(Qt::NoFocus);
        sidDown->setAutoRepeat(true);
        sidDown->setAutoRepeatDelay(350);
        sidDown->setAutoRepeatInterval(60);
        sidDown->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

        sidVl->addWidget(sidUp);
        sidVl->addWidget(sidDown);

        sidHl->addWidget(d->ttsSidSpin);
        sidHl->addWidget(sidBtns);

        form2->addRow(tr("声线（sid）："), sidRow);

        connect(sidUp, &QToolButton::clicked, this, [this]{ if (d->ttsSidSpin) d->ttsSidSpin->stepUp(); });
        connect(sidDown, &QToolButton::clicked, this, [this]{ if (d->ttsSidSpin) d->ttsSidSpin->stepDown(); });
        d->ttsSidHint = new QLabel(d->aiTab);
        styleModelHint(d->ttsSidHint);
        form2->addRow(QString(), d->ttsSidHint);
        d->ttsSidDesc = new QLabel(d->aiTab);
        styleModelHint(d->ttsSidDesc);
        form2->addRow(QString(), d->ttsSidDesc);

        {
            QWidget* row = new QWidget(d->aiTab);
            auto hl = new QHBoxLayout(row);
            hl->setContentsMargins(0,0,0,0);
            hl->setSpacing(kInlineRowSpacing);
            d->ttsVolumeSlider = new QSlider(Qt::Horizontal, row);
            d->ttsVolumeSlider->setRange(0, 100);
            d->ttsVolumeSlider->setSingleStep(1);
            d->ttsVolumeSlider->setPageStep(5);
            d->ttsVolumeSlider->setValue(SettingsManager::instance().ttsVolumePercent());
            d->ttsVolumeEdit = new ThemeWidgets::LineEdit(QString::number(SettingsManager::instance().ttsVolumePercent()), row);
            d->ttsVolumeEdit->setFixedWidth(90);
            hl->addWidget(d->ttsVolumeSlider, 1);
            hl->addWidget(d->ttsVolumeEdit, 0);
            form2->addRow(tr("TTS音量："), row);
        }

        form2->addItem(new QSpacerItem(0,0,QSizePolicy::Minimum,QSizePolicy::Expanding));

        connect(d->characterName, &QLineEdit::textChanged, this, [this](const QString& v){ SettingsManager::instance().setCharacterName(v); });
        connect(d->chatContextSlider, &QSlider::valueChanged, this, [this](int v){
            QSignalBlocker b(d->chatContextMessages);
            d->chatContextMessages->setText(QString::number(v));
        });
        connect(d->chatContextSlider, &QSlider::sliderReleased, this, [this]{
            SettingsManager::instance().setChatContextMessages(d->chatContextSlider->value());
        });
        connect(d->chatContextMessages, &QLineEdit::editingFinished, this, [this]{
            bool ok = false;
            const int v = d->chatContextMessages->text().trimmed().toInt(&ok);
            if (ok) SettingsManager::instance().setChatContextMessages(v);
            {
                QSignalBlocker b(d->chatContextMessages);
                d->chatContextMessages->setText(QString::number(SettingsManager::instance().chatContextMessages()));
            }
            if (d->chatContextSlider) {
                QSignalBlocker b2(d->chatContextSlider);
                d->chatContextSlider->setValue(SettingsManager::instance().chatContextMessages());
            }
        });

        connect(d->llmMaxTokensSlider, &QSlider::valueChanged, this, [this](int v){
            QSignalBlocker b(d->llmMaxTokens);
            d->llmMaxTokens->setText(QString::number(v));
        });
        connect(d->llmMaxTokensSlider, &QSlider::sliderReleased, this, [this]{
            SettingsManager::instance().setLlmMaxTokens(d->llmMaxTokensSlider->value());
        });
        connect(d->llmMaxTokens, &QLineEdit::editingFinished, this, [this]{
            bool ok = false;
            const int v = d->llmMaxTokens->text().trimmed().toInt(&ok);
            if (ok) SettingsManager::instance().setLlmMaxTokens(v);
            {
                QSignalBlocker b(d->llmMaxTokens);
                d->llmMaxTokens->setText(QString::number(SettingsManager::instance().llmMaxTokens()));
            }
            if (d->llmMaxTokensSlider) {
                QSignalBlocker b2(d->llmMaxTokensSlider);
                d->llmMaxTokensSlider->setValue(SettingsManager::instance().llmMaxTokens());
            }
        });
        connect(d->aiSystemPrompt, &QPlainTextEdit::textChanged, this, [this]{ SettingsManager::instance().setAiSystemPrompt(d->aiSystemPrompt->toPlainText()); });

        auto updateSherpaModelHints = [this, findModelDir]{
            struct SidInfo
            {
                bool sidEffective{false};
                bool multiSpeaker{false};
                int minSid{0};
                int maxSid{0};
                QHash<int, QString> sidDesc;
                QString sidLabel;
            };

            auto loadSidInfo = [this, findModelDir](const QString& id) -> SidInfo {
                SidInfo info;
                const QString modelId = id.trimmed();
                if (modelId.isEmpty())
                    return info;

                const QString modelDir = findModelDir(modelId);
                const QString l = modelId.toLower();

                if (!modelDir.isEmpty())
                {
                    const QDir d(modelDir);
                    const QStringList jsons = d.entryList(QStringList() << QStringLiteral("*.onnx.json"), QDir::Files, QDir::Name);
                    if (!jsons.isEmpty())
                    {
                        QFile f(d.filePath(jsons.front()));
                        if (f.open(QIODevice::ReadOnly))
                        {
                            const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
                            const QJsonObject root = doc.object();
                            const int num = root.value(QStringLiteral("num_speakers")).toInt(0);
                            info.sidEffective = num > 0;
                            info.multiSpeaker = num > 1;
                            info.minSid = 0;
                            info.maxSid = num > 0 ? (num - 1) : 0;
                            if (root.value(QStringLiteral("speaker_id_map")).isObject())
                            {
                                const QJsonObject m = root.value(QStringLiteral("speaker_id_map")).toObject();
                                for (auto it = m.begin(); it != m.end(); ++it)
                                {
                                    const int sid = it.value().toInt(-1);
                                    if (sid >= 0)
                                        info.sidDesc.insert(sid, it.key());
                                }
                            }
                            if (info.multiSpeaker)
                                info.sidLabel = QObject::tr("多说话人，sid %1-%2").arg(info.minSid).arg(info.maxSid);
                            else
                                info.sidLabel = QObject::tr("单说话人");
                            return info;
                        }
                    }

                    if (QFileInfo::exists(d.filePath(QStringLiteral("speakers.txt"))))
                    {
                        QFile f(d.filePath(QStringLiteral("speakers.txt")));
                        if (f.open(QIODevice::ReadOnly))
                        {
                            const QStringList lines = QString::fromUtf8(f.readAll()).split(QLatin1Char('\n'), Qt::SkipEmptyParts);
                            info.sidEffective = true;
                            info.multiSpeaker = lines.size() > 1;
                            info.minSid = 0;
                            info.maxSid = lines.isEmpty() ? 0 : (lines.size() - 1);
                            for (int i = 0; i < lines.size(); ++i)
                            {
                                const QString one = lines.at(i).trimmed();
                                if (!one.isEmpty())
                                    info.sidDesc.insert(i, one);
                            }
                            info.sidLabel = info.multiSpeaker ? QObject::tr("多说话人，sid %1-%2").arg(info.minSid).arg(info.maxSid) : QObject::tr("单说话人");
                            return info;
                        }
                    }

                    if (QFileInfo::exists(d.filePath(QStringLiteral("G_multisperaker_latest.json"))))
                    {
                        QFile f(d.filePath(QStringLiteral("G_multisperaker_latest.json")));
                        if (f.open(QIODevice::ReadOnly))
                        {
                            const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
                            const QJsonObject speakers = doc.object().value(QStringLiteral("speakers")).toObject();
                            int maxSid = -1;
                            for (auto it = speakers.begin(); it != speakers.end(); ++it)
                            {
                                const int sid = it.value().toInt(-1);
                                if (sid >= 0)
                                {
                                    info.sidDesc.insert(sid, it.key());
                                    if (sid > maxSid) maxSid = sid;
                                }
                            }
                            if (maxSid >= 0)
                            {
                                info.sidEffective = true;
                                info.multiSpeaker = true;
                                info.minSid = 0;
                                info.maxSid = maxSid;
                                info.sidLabel = QObject::tr("多说话人，sid %1-%2").arg(info.minSid).arg(info.maxSid);
                                return info;
                            }
                        }
                    }
                }

                if (l.contains(QStringLiteral("vits-zh-hf-fanchen-c")))
                {
                    info.sidEffective = true;
                    info.multiSpeaker = true;
                    info.minSid = 0;
                    info.maxSid = 186;
                    info.sidLabel = QObject::tr("多说话人，sid %1-%2").arg(info.minSid).arg(info.maxSid);
                    return info;
                }
                if (l.contains(QStringLiteral("vits-zh-hf-theresa")) || l.contains(QStringLiteral("vits-zh-hf-eula")))
                {
                    info.sidEffective = true;
                    info.multiSpeaker = true;
                    info.minSid = 0;
                    info.maxSid = 803;
                    info.sidLabel = QObject::tr("多说话人，sid %1-%2").arg(info.minSid).arg(info.maxSid);
                    return info;
                }

                if (l.contains(QStringLiteral("kokoro-multi-lang")))
                {
                    info.sidEffective = false;
                    info.multiSpeaker = false;
                    info.minSid = 0;
                    info.maxSid = 0;
                    info.sidLabel = QObject::tr("多声线，不使用 sid");
                    return info;
                }

                info.sidEffective = false;
                info.multiSpeaker = false;
                info.minSid = 0;
                info.maxSid = 0;
                info.sidLabel = QObject::tr("单说话人");
                return info;
            };

            auto mtBase = [](const QString& id) -> QString {
                const QString s = id.trimmed();
                if (s.isEmpty())
                    return QObject::tr("自动：在已安装模型中选择一个。");
                const QString l = s.toLower();
                if (l.contains(QStringLiteral("kokoro-multi-lang")))
                    return QObject::tr("Kokoro 多语言模型：中英等多语言，内置 voices.bin（当前未接入声线选择）。");
                if (l.contains(QStringLiteral("melo-tts-zh_en")) || l.contains(QStringLiteral("zh_en")))
                    return QObject::tr("中英通用：支持中英文混合输入（词表外英文可能无法发音）。");
                if (l.contains(QStringLiteral("sherpa-onnx-vits-zh-ll")) || (l.contains(QStringLiteral("vits-zh-ll")) && l.contains(QStringLiteral("sherpa"))))
                    return QObject::tr("中文：预置多种说话风格（多说话人/多风格，sid=0-4）。");
                if (l.contains(QStringLiteral("vits-zh-hf-fanchen-c")))
                    return QObject::tr("中文多说话人：风格丰富（187 人）。");
                if (l.contains(QStringLiteral("vits-zh-hf-theresa")))
                    return QObject::tr("中文多说话人：口语感强（804 人）。");
                if (l.contains(QStringLiteral("vits-zh-hf-eula")))
                    return QObject::tr("中文多说话人：偏正式/宣传风（804 人）。");
                if (l.contains(QStringLiteral("vits-icefall-zh-aishell3")) || l.contains(QStringLiteral("aishell3")))
                    return QObject::tr("中文多说话人：AIShell3 数据集（174 人）。");
                if (l.contains(QStringLiteral("vits-zh-hf-fanchen-wnj")))
                    return QObject::tr("中文男声：单说话人男声模型。");
                if (l.contains(QStringLiteral("zh-baker")))
                    return QObject::tr("中文女声（Baker 1万句数据集）。");
                if (l.contains(QStringLiteral("vits-ljs")) || l.contains(QStringLiteral("ljspeech")) || l.contains(QStringLiteral("en_us-ljspeech")))
                    return QObject::tr("英文女声：经典 LJSpeech。");
                if (l.contains(QStringLiteral("vctk")))
                    return QObject::tr("英文多说话人：VCTK（109 人）。");
                if (l.contains(QStringLiteral("piper")))
                {
                    if (l.contains(QStringLiteral("lessac")))
                        return QObject::tr("Piper 美式英文男声（en_US，Lessac，清晰正式）。");
                    if (l.contains(QStringLiteral("cori")))
                        return QObject::tr("Piper 英式英文女声（en_GB，Cori）。");
                    if (l.contains(QStringLiteral("amy")))
                        return QObject::tr("Piper 英文女声（en_US，Amy，偏温和风格）。");
                    if (l.contains(QStringLiteral("alan")))
                        return QObject::tr("Piper 英式英文男声（en_GB，Alan，稳重）。");
                    if (l.contains(QStringLiteral("southern_english_female")))
                        return QObject::tr("Piper 英式英文女声（en_GB，Southern English）。");
                    if (l.contains(QStringLiteral("southern_english_male")))
                        return QObject::tr("Piper 英式英文男声（en_GB，Southern English）。");
                    if (l.contains(QStringLiteral("libritts_r")))
                        return QObject::tr("Piper 美式英文多说话人（en_US，LibriTTS-R，904 人）。");
                    if (l.contains(QStringLiteral("glados")))
                        return QObject::tr("Piper 角色音色：GLaDOS 风格（科幻/机械感）。");
                    return QObject::tr("Piper VITS：体积小/速度快，依赖 espeak-ng-data。");
                }
                if (l.contains(QStringLiteral("matcha")))
                    return QObject::tr("Matcha：当前版本未集成（需要额外适配）。");
                return QObject::tr("离线语音合成模型（TTS）。");
            };
            const QString modelId = d->sherpaTtsModelCombo ? d->sherpaTtsModelCombo->currentData().toString() : QString();
            SidInfo sidInfo = loadSidInfo(modelId);

            if (d->ttsSidSpin)
            {
                QSignalBlocker b(d->ttsSidSpin);
                d->ttsSidSpin->setRange(sidInfo.minSid, sidInfo.maxSid);
                const int v = qBound(sidInfo.minSid, SettingsManager::instance().sherpaTtsSid(), sidInfo.maxSid);
                d->ttsSidSpin->setValue(v);
            }

            auto sidTip = [&sidInfo](const QString& id) -> QString {
                const QString s = id.trimmed();
                if (s.isEmpty())
                    return QObject::tr("sid：用于多说话人/多风格模型；单说话人模型会忽略。");
                if (!sidInfo.sidEffective)
                    return QObject::tr("sid：此模型不使用 sid。");
                if (!sidInfo.multiSpeaker)
                    return QObject::tr("sid：单说话人（忽略 sid）。");
                return QObject::tr("sid：%1-%2").arg(sidInfo.minSid).arg(sidInfo.maxSid);
            };
            auto sidDesc = [&sidInfo](const QString& id, int sid) -> QString {
                if (sid < 0) sid = 0;
                const QString s = id.trimmed();
                if (s.isEmpty())
                    return QObject::tr("当前 sid=%1：说话人 %1").arg(sid);
                const QString l = s.toLower();
                if (l.contains(QStringLiteral("matcha")))
                    return QObject::tr("当前 sid=%1：Matcha（未集成，sid 无效）").arg(sid);
                if (!sidInfo.sidEffective)
                    return QObject::tr("当前 sid=%1：此模型不使用 sid").arg(sid);
                if (!sidInfo.multiSpeaker)
                    return QObject::tr("当前 sid=%1：单说话人（忽略 sid）").arg(sid);
                const QString one = sidInfo.sidDesc.contains(sid) ? sidInfo.sidDesc.value(sid) : QObject::tr("说话人 %1").arg(sid);
                return QObject::tr("当前 sid=%1：%2").arg(sid).arg(one);
            };
            if (d->sherpaTtsModelHint && d->sherpaTtsModelCombo)
            {
                const QString base = mtBase(modelId);
                const QString suffix = sidInfo.sidLabel.isEmpty() ? QString() : (QObject::tr("（%1）").arg(sidInfo.sidLabel));
                d->sherpaTtsModelHint->setText(base + suffix);
            }
            if (d->ttsSidHint && d->sherpaTtsModelCombo)
                d->ttsSidHint->setText(sidTip(modelId));
            if (d->ttsSidDesc && d->sherpaTtsModelCombo)
            {
                const int sid = d->ttsSidSpin ? d->ttsSidSpin->value() : SettingsManager::instance().sherpaTtsSid();
                d->ttsSidDesc->setText(sidDesc(modelId, sid));
            }
        };
        updateSherpaModelHints();

        auto emitVoiceChanged = [this]{
            emit offlineVoiceSettingsChanged();
        };
        connect(d->offlineTtsEnabled, &ThemeWidgets::Switch::toggled, this, [this, emitVoiceChanged](bool on){
            SettingsManager::instance().setOfflineTtsEnabled(on);
            emitVoiceChanged();
        });
        connect(d->sherpaTtsModelCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, emitVoiceChanged, updateSherpaModelHints](int){
            const QString v = d->sherpaTtsModelCombo ? d->sherpaTtsModelCombo->currentData().toString() : QString();
            SettingsManager::instance().setSherpaTtsModel(v);
            updateSherpaModelHints();
            emitVoiceChanged();
        });
        connect(d->ttsSidSpin, qOverload<int>(&QSpinBox::valueChanged), this, [this, emitVoiceChanged, updateSherpaModelHints](int sid){
            SettingsManager::instance().setSherpaTtsSid(sid);
            updateSherpaModelHints();
            emitVoiceChanged();
        });
        connect(d->ttsVolumeSlider, &QSlider::valueChanged, this, [this](int v){
            QSignalBlocker b(d->ttsVolumeEdit);
            d->ttsVolumeEdit->setText(QString::number(v));
        });
        connect(d->ttsVolumeSlider, &QSlider::sliderReleased, this, [this, emitVoiceChanged]{
            SettingsManager::instance().setTtsVolumePercent(d->ttsVolumeSlider->value());
            emitVoiceChanged();
        });
        connect(d->ttsVolumeEdit, &QLineEdit::editingFinished, this, [this, emitVoiceChanged]{
            bool ok = false;
            int v = d->ttsVolumeEdit->text().trimmed().toInt(&ok);
            if (!ok) v = SettingsManager::instance().ttsVolumePercent();
            v = qBound(0, v, 100);
            SettingsManager::instance().setTtsVolumePercent(v);
            {
                QSignalBlocker b(d->ttsVolumeEdit);
                d->ttsVolumeEdit->setText(QString::number(SettingsManager::instance().ttsVolumePercent()));
            }
            if (d->ttsVolumeSlider) {
                QSignalBlocker b2(d->ttsVolumeSlider);
                d->ttsVolumeSlider->setValue(SettingsManager::instance().ttsVolumePercent());
            }
            emitVoiceChanged();
        });
    }
    d->tabStack->addWidget(d->aiTab);
    d->tabs->addTab(
        tr("AI设置"),
        Theme::themedIcon(Theme::IconToken::SettingsAi)
    );

    // Advanced tab
    d->advancedTab = new QWidget(d->tabStack);
    auto advLay = new QFormLayout(d->advancedTab);
    d->advancedForm = advLay;
    applyLeftAlignedFormLayout(advLay);
    d->texCapCombo = new ThemeWidgets::ComboBox(d->advancedTab); d->texCapCombo->addItems({"4096","3072","2048","1024"});
    {
        int cur = SettingsManager::instance().textureMaxDim();
        int idx = d->texCapCombo->findText(QString::number(cur)); if (idx<0) idx = 2; d->texCapCombo->setCurrentIndex(idx);
    }
    advLay->addRow(tr("贴图上限："), d->texCapCombo);
    d->msaaCombo = new ThemeWidgets::ComboBox(d->advancedTab); d->msaaCombo->addItems({"2x","4x","8x"});
    {
        int cur = SettingsManager::instance().msaaSamples();
        int idx = (cur==2?0:(cur==8?2:1)); d->msaaCombo->setCurrentIndex(idx);
    }
    advLay->addRow(tr("MSAA："), d->msaaCombo);

    // ---- Model display screen ----
    d->screenCombo = new ThemeWidgets::ComboBox(d->advancedTab);
    d->screenCombo->addItem(tr("系统默认（默认）"), QString());
    {
        const QString preferred = SettingsManager::instance().preferredScreenName();
        const QList<QScreen*> screens = QGuiApplication::screens();
        for (QScreen* s : screens)
        {
            const QString name = s ? s->name() : QString();
            if (name.isEmpty()) continue;
            // If no preferred, mark the current primary as default.
            const bool isDefault = (QGuiApplication::primaryScreen() == s);
            const QString label = isDefault ? (name + tr("（默认）")) : name;
            d->screenCombo->addItem(label, name);
        }
        // select
        if (!preferred.isEmpty())
        {
            const int idx = d->screenCombo->findData(preferred);
            if (idx >= 0) d->screenCombo->setCurrentIndex(idx);
        }
        else
        {
            d->screenCombo->setCurrentIndex(0);
        }
    }
    advLay->addRow(tr("模型显示："), d->screenCombo);

    // （离线模式）移除音频输出设备选择

    {
        auto reminderCard = new QWidget(d->advancedTab);
        auto reminderLay = new QVBoxLayout(reminderCard);
        reminderLay->setContentsMargins(0, 0, 0, 0);
        reminderLay->setSpacing(kInlineRowSpacing);

        d->reminderList = new QListWidget(reminderCard);
        d->reminderList->setSelectionMode(QAbstractItemView::SingleSelection);
        d->reminderList->setMinimumHeight(140);
        reminderLay->addWidget(d->reminderList, 1);

        auto btnRow = new QWidget(reminderCard);
        auto btnLay = new QHBoxLayout(btnRow);
        btnLay->setContentsMargins(0, 0, 0, 0);
        btnLay->setSpacing(kInlineRowSpacing);
        d->reminderAddBtn = new ThemeWidgets::Button(tr("添加"), btnRow);
        d->reminderAddBtn->setTone(ThemeWidgets::Button::Tone::Link);
        d->reminderEditBtn = new ThemeWidgets::Button(tr("编辑"), btnRow);
        d->reminderEditBtn->setTone(ThemeWidgets::Button::Tone::Neutral);
        d->reminderRemoveBtn = new ThemeWidgets::Button(tr("删除"), btnRow);
        d->reminderRemoveBtn->setTone(ThemeWidgets::Button::Tone::Danger);
        btnLay->addWidget(d->reminderAddBtn);
        btnLay->addWidget(d->reminderEditBtn);
        btnLay->addWidget(d->reminderRemoveBtn);
        btnLay->addStretch(1);
        reminderLay->addWidget(btnRow);

        advLay->addRow(tr("定时任务："), reminderCard);

        auto formatReminderSummary = [](const QJsonObject& o) -> QString {
            const bool enabled = o.value(QStringLiteral("enabled")).toBool(true);
            const QString mode = o.value(QStringLiteral("mode")).toString(QStringLiteral("daily"));
            const QString kind = o.value(QStringLiteral("kind")).toString(QStringLiteral("assistant"));
            QString when;
            if (mode == QStringLiteral("interval")) {
                const int m = o.value(QStringLiteral("intervalMinutes")).toInt(60);
                when = QObject::tr("每 %1 分钟").arg(std::max(1, m));
            } else {
                const QString t = o.value(QStringLiteral("time")).toString(QStringLiteral("09:00"));
                when = QObject::tr("每天 %1").arg(t);
            }
            const QString text = o.value(QStringLiteral("text")).toString().trimmed();
            const QString motion = o.value(QStringLiteral("motionGroup")).toString().trimmed();
            const QString expr = o.value(QStringLiteral("expressionName")).toString().trimmed();
            QStringList parts;
            parts << when;
            if (!text.isEmpty()) parts << (kind == QStringLiteral("ask") ? (QObject::tr("对话：") + text) : text);
            if (!motion.isEmpty()) parts << QObject::tr("动作：%1").arg(motion);
            if (!expr.isEmpty()) parts << QObject::tr("表情：%1").arg(expr);
            QString s = parts.join(QStringLiteral(" / "));
            if (!enabled) s = QObject::tr("（停用）") + s;
            return s;
        };

        auto refreshReminderList = [this, formatReminderSummary]{
            if (!d || !d->reminderList) return;
            d->reminderList->clear();
            const QJsonArray arr = SettingsManager::instance().reminderTasks();
            for (const auto& v : arr)
            {
                if (!v.isObject()) continue;
                const QJsonObject o = v.toObject();
                const QString id = o.value(QStringLiteral("id")).toString();
                auto* item = new QListWidgetItem(formatReminderSummary(o));
                item->setData(Qt::UserRole, id);
                d->reminderList->addItem(item);
            }
            if (d->reminderEditBtn) d->reminderEditBtn->setEnabled(d->reminderList->currentRow() >= 0);
            if (d->reminderRemoveBtn) d->reminderRemoveBtn->setEnabled(d->reminderList->currentRow() >= 0);
        };

        auto getSelectedReminderId = [this]() -> QString {
            if (!d || !d->reminderList) return {};
            auto* item = d->reminderList->currentItem();
            if (!item) return {};
            return item->data(Qt::UserRole).toString();
        };

        auto findReminderById = [](const QJsonArray& arr, const QString& id, int* outIndex, QJsonObject* outObj) -> bool {
            for (int i = 0; i < arr.size(); ++i)
            {
                if (!arr.at(i).isObject()) continue;
                const QJsonObject o = arr.at(i).toObject();
                if (o.value(QStringLiteral("id")).toString() == id)
                {
                    if (outIndex) *outIndex = i;
                    if (outObj) *outObj = o;
                    return true;
                }
            }
            return false;
        };

        auto editReminderDialog = [this](QJsonObject init, bool* okOut) -> QJsonObject {
            QDialog dlg(this);
            dlg.setWindowTitle(tr("定时任务"));
            dlg.setModal(true);

            auto* form = new QFormLayout(&dlg);
            applyLeftAlignedFormLayout(form);

            auto* chkEnabled = new QCheckBox(tr("启用"), &dlg);
            chkEnabled->setChecked(init.value(QStringLiteral("enabled")).toBool(true));

            auto* kindCombo = new QComboBox(&dlg);
            kindCombo->addItem(tr("直接显示文案"), QStringLiteral("assistant"));
            kindCombo->addItem(tr("触发对话（让 AI 回复）"), QStringLiteral("ask"));
            {
                const QString kind = init.value(QStringLiteral("kind")).toString(QStringLiteral("assistant"));
                const int idx = kindCombo->findData(kind);
                if (idx >= 0) kindCombo->setCurrentIndex(idx);
            }

            auto* modeCombo = new QComboBox(&dlg);
            modeCombo->addItem(tr("每天固定时间"), QStringLiteral("daily"));
            modeCombo->addItem(tr("按间隔重复"), QStringLiteral("interval"));
            {
                const QString mode = init.value(QStringLiteral("mode")).toString(QStringLiteral("daily"));
                const int idx = modeCombo->findData(mode);
                if (idx >= 0) modeCombo->setCurrentIndex(idx);
            }

            auto* timeEdit = new QTimeEdit(&dlg);
            timeEdit->setDisplayFormat(QStringLiteral("HH:mm"));
            {
                const QString t = init.value(QStringLiteral("time")).toString(QStringLiteral("09:00"));
                const QTime tt = QTime::fromString(t, QStringLiteral("HH:mm"));
                timeEdit->setTime(tt.isValid() ? tt : QTime(9, 0));
            }

            auto* intervalSpin = new QSpinBox(&dlg);
            intervalSpin->setRange(1, 24 * 60);
            intervalSpin->setValue(std::max(1, init.value(QStringLiteral("intervalMinutes")).toInt(60)));

            auto* msgEdit = new ThemeWidgets::PlainTextEdit(&dlg);
            msgEdit->setPlainText(init.value(QStringLiteral("text")).toString());
            msgEdit->setFixedHeight(72);
            auto* msgLabel = new QLabel(tr("内容："), &dlg);

            auto* motionEdit = new ThemeWidgets::LineEdit(&dlg);
            motionEdit->setText(init.value(QStringLiteral("motionGroup")).toString());
            auto* exprEdit = new ThemeWidgets::LineEdit(&dlg);
            exprEdit->setText(init.value(QStringLiteral("expressionName")).toString());

            auto* chkWriteHistory = new QCheckBox(tr("写入对话历史"), &dlg);
            chkWriteHistory->setChecked(init.value(QStringLiteral("writeToHistory")).toBool(true));

            form->addRow(tr("状态："), chkEnabled);
            form->addRow(tr("任务类型："), kindCombo);
            form->addRow(tr("触发方式："), modeCombo);
            form->addRow(tr("时间："), timeEdit);
            form->addRow(tr("间隔（分钟）："), intervalSpin);
            form->addRow(msgLabel, msgEdit);
            form->addRow(tr("动作组："), motionEdit);
            form->addRow(tr("表情名："), exprEdit);
            form->addRow(QString(), chkWriteHistory);

            auto* btnBox = new QWidget(&dlg);
            auto* btnLay2 = new QHBoxLayout(btnBox);
            btnLay2->setContentsMargins(0, 0, 0, 0);
            btnLay2->setSpacing(kInlineRowSpacing);
            auto* okBtn = new ThemeWidgets::Button(tr("确定"), btnBox);
            okBtn->setTone(ThemeWidgets::Button::Tone::Link);
            auto* cancelBtn = new ThemeWidgets::Button(tr("取消"), btnBox);
            cancelBtn->setTone(ThemeWidgets::Button::Tone::Neutral);
            btnLay2->addStretch(1);
            btnLay2->addWidget(okBtn);
            btnLay2->addWidget(cancelBtn);
            form->addRow(QString(), btnBox);

            auto applyModeUi = [modeCombo, timeEdit, intervalSpin]{
                const QString mode = modeCombo->currentData().toString();
                const bool isInterval = (mode == QStringLiteral("interval"));
                timeEdit->setEnabled(!isInterval);
                intervalSpin->setEnabled(isInterval);
            };
            QObject::connect(modeCombo, qOverload<int>(&QComboBox::currentIndexChanged), &dlg, [applyModeUi](int){ applyModeUi(); });
            applyModeUi();

            auto applyKindUi = [kindCombo, msgLabel, chkWriteHistory]{
                const QString kind = kindCombo->currentData().toString();
                const bool isAsk = (kind == QStringLiteral("ask"));
                msgLabel->setText(isAsk ? QObject::tr("提示词：") : QObject::tr("内容："));
                chkWriteHistory->setEnabled(!isAsk);
                if (isAsk) chkWriteHistory->setChecked(true);
            };
            QObject::connect(kindCombo, qOverload<int>(&QComboBox::currentIndexChanged), &dlg, [applyKindUi](int){ applyKindUi(); });
            applyKindUi();

            QObject::connect(okBtn, &QPushButton::clicked, &dlg, &QDialog::accept);
            QObject::connect(cancelBtn, &QPushButton::clicked, &dlg, &QDialog::reject);

            const int ret = dlg.exec();
            if (okOut) *okOut = (ret == QDialog::Accepted);
            if (ret != QDialog::Accepted) return init;

            QJsonObject out = init;
            out["enabled"] = chkEnabled->isChecked();
            out["kind"] = kindCombo->currentData().toString();
            out["mode"] = modeCombo->currentData().toString();
            out["time"] = timeEdit->time().toString(QStringLiteral("HH:mm"));
            out["intervalMinutes"] = intervalSpin->value();
            out["text"] = msgEdit->toPlainText();
            out["motionGroup"] = motionEdit->text().trimmed();
            out["expressionName"] = exprEdit->text().trimmed();
            out["writeToHistory"] = chkWriteHistory->isChecked();
            if (out.value(QStringLiteral("id")).toString().trimmed().isEmpty())
                out["id"] = QUuid::createUuid().toString(QUuid::WithoutBraces);
            return out;
        };

        refreshReminderList();

        connect(d->reminderList, &QListWidget::currentRowChanged, this, [this](int){
            if (!d) return;
            const bool has = d->reminderList && d->reminderList->currentRow() >= 0;
            if (d->reminderEditBtn) d->reminderEditBtn->setEnabled(has);
            if (d->reminderRemoveBtn) d->reminderRemoveBtn->setEnabled(has);
        });

        auto doAdd = [this, refreshReminderList, editReminderDialog]{
            bool ok = false;
            QJsonObject init;
            init["enabled"] = true;
            init["kind"] = QStringLiteral("assistant");
            init["mode"] = QStringLiteral("daily");
            init["time"] = QStringLiteral("09:00");
            init["intervalMinutes"] = 60;
            init["writeToHistory"] = true;
            QJsonObject o = editReminderDialog(init, &ok);
            if (!ok) return;
            QJsonArray arr = SettingsManager::instance().reminderTasks();
            arr.append(o);
            SettingsManager::instance().setReminderTasks(arr);
            refreshReminderList();
        };

        auto doEdit = [this, refreshReminderList, editReminderDialog, findReminderById, getSelectedReminderId]{
            const QString id = getSelectedReminderId();
            if (id.isEmpty()) return;
            QJsonArray arr = SettingsManager::instance().reminderTasks();
            int idx = -1;
            QJsonObject cur;
            if (!findReminderById(arr, id, &idx, &cur)) return;
            bool ok = false;
            QJsonObject next = editReminderDialog(cur, &ok);
            if (!ok) return;
            arr[idx] = next;
            SettingsManager::instance().setReminderTasks(arr);
            refreshReminderList();
        };

        auto confirmNow = [this](const QString& title, const QString& text) -> bool {
            const auto ret = QMessageBox::question(this, title, text, QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            return ret == QMessageBox::Yes;
        };

        auto doRemove = [this, refreshReminderList, findReminderById, getSelectedReminderId, confirmNow]{
            const QString id = getSelectedReminderId();
            if (id.isEmpty()) return;
            if (!confirmNow(tr("确认删除"), tr("将删除所选定时任务。确定继续吗？"))) return;
            QJsonArray arr = SettingsManager::instance().reminderTasks();
            int idx = -1;
            if (!findReminderById(arr, id, &idx, nullptr)) return;
            arr.removeAt(idx);
            SettingsManager::instance().setReminderTasks(arr);
            refreshReminderList();
        };

        connect(d->reminderAddBtn, &QPushButton::clicked, this, doAdd);
        connect(d->reminderEditBtn, &QPushButton::clicked, this, doEdit);
        connect(d->reminderRemoveBtn, &QPushButton::clicked, this, doRemove);
        connect(d->reminderList, &QListWidget::itemDoubleClicked, this, [doEdit](QListWidgetItem*){ doEdit(); });

        if (d->reminderEditBtn) d->reminderEditBtn->setEnabled(false);
        if (d->reminderRemoveBtn) d->reminderRemoveBtn->setEnabled(false);
    }

    // 清理按钮
    auto cleanupRow = new QWidget(d->advancedTab);
    d->cleanupRowWidget = cleanupRow;
    auto cleanupLay = new QHBoxLayout(cleanupRow);
    cleanupLay->setContentsMargins(0, 0, 0, 0);
    cleanupLay->setSpacing(kInlineRowSpacing);
    auto btnClearCache = new ThemeWidgets::Button(tr("清除缓存"), cleanupRow);
    btnClearCache->setTone(ThemeWidgets::Button::Tone::Warning);
    auto btnClearChats = new ThemeWidgets::Button(tr("清除所有对话历史"), cleanupRow);
    btnClearChats->setTone(ThemeWidgets::Button::Tone::Danger);
    d->clearCacheBtn = btnClearCache;
    d->clearChatsBtn = btnClearChats;
    cleanupLay->addWidget(d->clearCacheBtn);
    cleanupLay->addWidget(d->clearChatsBtn);
    cleanupLay->addStretch(1);
    advLay->addRow(tr("清理："), cleanupRow);

    advLay->addItem(new QSpacerItem(0,0,QSizePolicy::Minimum,QSizePolicy::Expanding));
    d->tabStack->addWidget(d->advancedTab);
    d->tabs->addTab(
        tr("高级设置"),
        Theme::themedIcon(Theme::IconToken::SettingsAdvanced)
    );

    auto confirm = [this](const QString& title, const QString& text) -> bool {
        auto ret = QMessageBox::question(this, title, text, QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        return ret == QMessageBox::Yes;
    };

    connect(btnClearCache, &QPushButton::clicked, this, [this, confirm]{
        if (!confirm(tr("确认清除缓存"), tr("将删除缓存目录（.Cache）下的所有文件。确定继续吗？"))) return;
        const QString cacheDir = SettingsManager::instance().cacheDir();
        QDir dir(cacheDir);
        if (!dir.exists()) dir.mkpath(".");
        const QFileInfoList files = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
        for (const auto& fi : files) {
            QFile::remove(fi.absoluteFilePath());
        }
        const QFileInfoList subDirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
        for (const auto& di : subDirs) {
            QDir(di.absoluteFilePath()).removeRecursively();
        }
        QMessageBox::information(this, tr("完成"), tr("缓存已清除。"));
    });

    connect(btnClearChats, &QPushButton::clicked, this, [this, confirm]{
        if (!confirm(tr("确认清除所有对话历史"), tr("将清空当前聊天窗口的记录，并删除 Chats 目录下的所有会话文件。确定继续吗？"))) return;
        // ensure chats dir exists then remove all files
        const QString chatsDir = SettingsManager::instance().chatsDir();
        QDir dir(chatsDir);
        if (!dir.exists()) dir.mkpath(".");
        const QFileInfoList files = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
        for (const auto& fi : files) {
            QFile::remove(fi.absoluteFilePath());
        }
        const QFileInfoList subDirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
        for (const auto& di : subDirs) {
            QDir(di.absoluteFilePath()).removeRecursively();
        }
        // 清空当前聊天窗口显示（只影响当前窗口，不触发模型切换逻辑）
        emit requestClearAllChats();
        QMessageBox::information(this, tr("完成"), tr("所有对话历史已清除。"));
    });

    // Watcher for dynamic model folder changes
    d->fsw = new QFileSystemWatcher(this);
    d->debounce = new QTimer(this); d->debounce->setSingleShot(true); d->debounce->setInterval(250);
    auto scheduleRefresh = [this]{ d->debounce->start(); };
    connect(d->debounce, &QTimer::timeout, this, [this]{
        const QString cur = d->modelCombo->currentData().toString();
        refreshModelList();
        const int idx = d->modelCombo->findData(cur); if (idx >= 0) d->modelCombo->setCurrentIndex(idx);
        QString wm = SettingsManager::instance().watermarkExpPath();
        d->wmFileLabel->setText(wm.isEmpty()? tr("无") : QFileInfo(wm).fileName());
        emit watermarkChanged(wm);
    });
    connect(d->fsw, &QFileSystemWatcher::directoryChanged, this, [scheduleRefresh](const QString&){ scheduleRefresh(); });

    refreshModelList();

    // Init state from SettingsManager
    {
        auto& sm = SettingsManager::instance();
        QString folder = sm.selectedModelFolder(); if (!folder.isEmpty()) sm.ensureModelConfigExists(folder);
        d->chkBreath->setChecked(sm.enableBreath());
        d->chkBlink->setChecked(sm.enableBlink());
        d->chkGaze->setChecked(sm.enableGaze());
        d->chkPhysics->setChecked(sm.enablePhysics());
        QString wm = sm.watermarkExpPath(); d->wmFileLabel->setText(wm.isEmpty()? tr("无") : QFileInfo(wm).fileName());
    }

    // Connections
    connect(d->modelCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        const QString name = d->modelCombo->currentData().toString();
        if (name.isEmpty()) return;
        auto& sm = SettingsManager::instance();
        if (sm.selectedModelFolder() == name) return;
        sm.setSelectedModelFolder(name);
        sm.ensureModelConfigExists(name);
        for (const auto& e : sm.scanModels())
        {
            if (e.folderName == name)
            {
                qDebug() << "[ModelFlow][SettingsWindow] emit requestLoadModel" << "name=" << name << "json=" << e.jsonPath;
                emit requestLoadModel(e.jsonPath);
                break;
            }
        }
        d->chkBreath->setChecked(SettingsManager::instance().enableBreath());
        d->chkBlink->setChecked(SettingsManager::instance().enableBlink());
        d->chkGaze->setChecked(SettingsManager::instance().enableGaze());
        d->chkPhysics->setChecked(SettingsManager::instance().enablePhysics());

        const QString wm = SettingsManager::instance().watermarkExpPath();
        d->wmFileLabel->setText(wm.isEmpty() ? tr("无") : QFileInfo(wm).fileName());
        emit watermarkChanged(wm);

    });

    auto chooseExistingDirectory = [this](const QString& title, const QString& startDir) -> QString {
        QFileDialog dlg(this, title, startDir);
        dlg.setFileMode(QFileDialog::Directory);
        dlg.setOption(QFileDialog::ShowDirsOnly, true);
#if defined(Q_OS_LINUX)
        dlg.setOption(QFileDialog::DontUseNativeDialog, true);
#endif
        if (dlg.exec() != QDialog::Accepted) return {};
        const QStringList files = dlg.selectedFiles();
        return files.isEmpty() ? QString() : files.first();
    };

    auto chooseOpenFile = [this](const QString& title, const QString& startDir, const QString& filter) -> QString {
        QFileDialog dlg(this, title, startDir, filter);
        dlg.setFileMode(QFileDialog::ExistingFile);
#if defined(Q_OS_LINUX)
        dlg.setOption(QFileDialog::DontUseNativeDialog, true);
#endif
        if (dlg.exec() != QDialog::Accepted) return {};
        const QStringList files = dlg.selectedFiles();
        return files.isEmpty() ? QString() : files.first();
    };

    connect(d->chooseBtn, &QPushButton::clicked, this, [this, chooseExistingDirectory]{
        QString dir = chooseExistingDirectory(tr("选择模型根目录"), SettingsManager::instance().modelsRoot());
        if (dir.isEmpty()) return; SettingsManager::instance().setModelsRoot(dir); d->pathLabel->setText(dir); refreshModelList();
    });
    connect(d->openBtn, &QPushButton::clicked, this, [this]{ QDesktopServices::openUrl(QUrl::fromLocalFile(SettingsManager::instance().modelsRoot())); });
    connect(d->resetDirBtn, &QPushButton::clicked, this, [this]{ auto& sm = SettingsManager::instance(); sm.resetModelsRootToDefault(QCoreApplication::applicationDirPath()); d->pathLabel->setText(sm.modelsRoot()); refreshModelList(); });

    connect(d->resetBtn, &QPushButton::clicked, this, [this]{ emit requestResetWindow(); });

    auto refreshCurModelName = [this]{
        QString name = SettingsManager::instance().selectedModelFolder();
        if (name.isEmpty()) {
            auto entries = SettingsManager::instance().scanModels();
            int idx = -1; for (int i=0;i<entries.size();++i) if (entries[i].folderName.compare("Hiyori", Qt::CaseInsensitive)==0){ idx=i; break; }
            if (idx >= 0) name = entries[idx].folderName; else if (!entries.isEmpty()) name = entries.front().folderName;
        }
        d->curModelName->setText(name.isEmpty()? tr("(无)") : name);
        QString wm = SettingsManager::instance().watermarkExpPath(); d->wmFileLabel->setText(wm.isEmpty()? tr("无") : QFileInfo(wm).fileName());
    };
    refreshCurModelName();
    connect(d->openModelDirBtn, &QPushButton::clicked, this, [this]{ QString folder = SettingsManager::instance().selectedModelFolder(); QString root = SettingsManager::instance().modelsRoot(); if (folder.isEmpty()) { auto entries = SettingsManager::instance().scanModels(); if (!entries.isEmpty()) folder = entries.front().folderName; } if (!folder.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(QDir(root).filePath(folder))); });
    connect(d->modelCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, refreshCurModelName](int){
        refreshCurModelName();
    });

    connect(d->chkBlink, &ThemeWidgets::Switch::toggled, this, [this](bool on){ SettingsManager::instance().setEnableBlink(on); emit toggleBlink(on); });
    connect(d->chkBreath, &ThemeWidgets::Switch::toggled, this, [this](bool on){ SettingsManager::instance().setEnableBreath(on); emit toggleBreath(on); });
    connect(d->chkGaze, &ThemeWidgets::Switch::toggled, this, [this](bool on){ SettingsManager::instance().setEnableGaze(on); emit toggleGaze(on); });
    connect(d->chkPhysics, &ThemeWidgets::Switch::toggled, this, [this](bool on){ SettingsManager::instance().setEnablePhysics(on); emit togglePhysics(on); });
    connect(d->llmModelSizeCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        const QString s = d->llmModelSizeCombo->currentData().toString();
        SettingsManager::instance().setLlmModelSize(s);
        emit llmModelSizeChanged(s);
    });
    connect(d->llmStyleCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        const QString s = d->llmStyleCombo->currentData().toString();
        SettingsManager::instance().setLlmStyle(s);
        emit llmStyleChanged(s);
    });

    connect(d->wmChooseBtn, &QPushButton::clicked, this, [this, chooseOpenFile]{ QString folder = SettingsManager::instance().selectedModelFolder(); if (folder.isEmpty()) return; QString root = SettingsManager::instance().modelsRoot(); QString modelDir = QDir(root).filePath(folder); QString path = chooseOpenFile(tr("选择水印表达式文件"), modelDir, "Expression (*.exp3.json)"); if (path.isEmpty()) return; SettingsManager::instance().setWatermarkExpPath(path); d->wmFileLabel->setText(QFileInfo(path).fileName()); emit watermarkChanged(path); });
    connect(d->wmClearBtn, &QPushButton::clicked, this, [this]{ SettingsManager::instance().setWatermarkExpPath(""); d->wmFileLabel->setText(tr("无")); emit watermarkChanged(""); });

    connect(d->modelCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        auto& sm = SettingsManager::instance();
        const int curCap = sm.textureMaxDim();
        const int idxCap = d->texCapCombo->findText(QString::number(curCap));
        if (idxCap >= 0) d->texCapCombo->setCurrentIndex(idxCap);
        const int curMsaa = sm.msaaSamples();
        const int idxMsaa = (curMsaa == 2 ? 0 : (curMsaa == 8 ? 2 : 1));
        d->msaaCombo->setCurrentIndex(idxMsaa);
    });

    connect(d->texCapCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){ int dim = d->texCapCombo->currentText().toInt(); SettingsManager::instance().setTextureMaxDim(dim); emit textureCapChanged(dim); });
    connect(d->msaaCombo,  qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){ QString t = d->msaaCombo->currentText(); int samples = t.startsWith("2")?2:(t.startsWith("8")?8:4); SettingsManager::instance().setMsaaSamples(samples); emit msaaChanged(samples); });

    // Advanced: screen / audio output hot update
    connect(d->screenCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        const QString name = d->screenCombo->currentData().toString();
        SettingsManager::instance().setPreferredScreenName(name);
        emit preferredScreenChanged(name);
    });
    // （离线模式）移除音频输出设备选择事件

    connect(d->themeCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        const QString themeId = Theme::normalizeThemeId(d->themeCombo->currentData().toString());
        SettingsManager::instance().setTheme(themeId);
        emit themeChanged(themeId);
    });

    connect(d->languageCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        const QString sel = d->languageCombo->currentData().toString();
        QString code;
        if (sel == QStringLiteral("system")) {
            code = QLocale::system().name().startsWith("zh", Qt::CaseInsensitive)
                       ? QStringLiteral("zh_CN")
                       : QStringLiteral("en_US");
        } else {
            code = sel;
        }
        SettingsManager::instance().setCurrentLanguage(code);
        emit languageChanged(code);
    });

    connect(d->petBubbleCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){
        SettingsManager::instance().setChatBubbleStyle(d->petBubbleCombo->currentData().toString());
    });

    connect(d->chkAlwaysOnTop, &ThemeWidgets::Switch::toggled, this, [this](bool on){
        SettingsManager::instance().setWindowAlwaysOnTop(on);
        emit windowAlwaysOnTopChanged(on);
    });
    connect(d->chkTransparentBg, &ThemeWidgets::Switch::toggled, this, [this](bool on){
        SettingsManager::instance().setWindowTransparentBackground(on);
        emit windowTransparentBackgroundChanged(on);
    });
    connect(d->chkMousePassthrough, &ThemeWidgets::Switch::toggled, this, [this](bool on){
        SettingsManager::instance().setWindowMousePassthrough(on);
        emit windowMousePassthroughChanged(on);
    });

    // （离线模式）移除联网 AI/TTS 相关设置项连接，仅保留人设文本变更

    // 主题完全交给系统与 Qt 平台插件处理，不做任何手动 apply
}

SettingsWindow::~SettingsWindow() = default;

void SettingsWindow::refreshSidebarThemeIndicator()
{
    if (!d || !d->themeSchemeIconLabel)
        return;

    const Theme::IconToken token = currentColorThemeIsDark(this)
        ? Theme::IconToken::ThemeDarkIndicator
        : Theme::IconToken::ThemeLightIndicator;

    const QIcon icon = Theme::themedIcon(token);
    d->themeSchemeIconLabel->setPixmap(icon.pixmap(QSize(kSidebarThemeIconSize, kSidebarThemeIconSize)));
}

bool SettingsWindow::event(QEvent* e)
{
    if (e->type() == QEvent::Show || e->type() == QEvent::WindowActivate)
    {
        if (d && d->petBubbleCombo)
        {
            const QString saved = SettingsManager::instance().chatBubbleStyle();
            const int idx = d->petBubbleCombo->findData(saved);
            if (idx >= 0 && d->petBubbleCombo->currentIndex() != idx)
            {
                QSignalBlocker b(d->petBubbleCombo);
                d->petBubbleCombo->setCurrentIndex(idx);
            }
        }
    }

    if (e->type() == QEvent::LanguageChange)
    {
        setWindowTitle(tr("设置"));
        if (d)
        {
            if (d->chooseBtn) d->chooseBtn->setText(tr("选择路径"));
            if (d->openBtn) d->openBtn->setText(tr("打开路径"));
            if (d->resetDirBtn) d->resetDirBtn->setText(tr("恢复默认"));
            if (d->resetBtn) d->resetBtn->setText(tr("还原初始状态"));
            if (d->openModelDirBtn) d->openModelDirBtn->setText(tr("打开当前模型路径"));
            if (d->modelNameTitle) d->modelNameTitle->setText(tr("模型名称："));
            if (d->watermarkTitle) d->watermarkTitle->setText(tr("去除水印："));
            if (d->wmChooseBtn) d->wmChooseBtn->setText(tr("选择文件"));
            if (d->wmClearBtn) d->wmClearBtn->setText(tr("取消所选"));
            if (d->chkBreath) d->chkBreath->setText(tr("自动呼吸"));
            if (d->chkBlink) d->chkBlink->setText(tr("自动眨眼"));
            if (d->chkGaze) d->chkGaze->setText(tr("视线跟踪"));
            if (d->chkPhysics) d->chkPhysics->setText(tr("物理模拟"));
            if (d->chkAlwaysOnTop) d->chkAlwaysOnTop->setText(tr("全局置顶"));
            if (d->chkTransparentBg) d->chkTransparentBg->setText(tr("透明背景"));
            if (d->chkMousePassthrough) d->chkMousePassthrough->setText(tr("鼠标穿透窗口"));
            if (d->aiSystemPrompt) d->aiSystemPrompt->setPlaceholderText(tr("支持变量：$name$（角色名称）"));
            if (d->aiStream) d->aiStream->setText(tr("是否流式输出"));
            if (d->clearCacheBtn) d->clearCacheBtn->setText(tr("清除缓存"));
            if (d->clearChatsBtn) d->clearChatsBtn->setText(tr("清除所有对话历史"));

            if (d->themeCombo) {
                const int idxEra = d->themeCombo->findData(QStringLiteral("era"));
                if (idxEra >= 0) d->themeCombo->setItemText(idxEra, tr("Era"));
            }
            if (d->languageCombo) {
                const int idxSystem = d->languageCombo->findData(QStringLiteral("system"));
                if (idxSystem >= 0) d->languageCombo->setItemText(idxSystem, tr("跟随系统"));
                const int idxZh = d->languageCombo->findData(QStringLiteral("zh_CN"));
                if (idxZh >= 0) d->languageCombo->setItemText(idxZh, tr("简体中文"));
            }

            if (d->basicForm) {
                if (auto* label = qobject_cast<QLabel*>(d->basicForm->labelForField(d->pathRowWidget))) {
                    label->setText(tr("模型路径："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->basicForm->labelForField(d->topRowWidget))) {
                    label->setText(tr("当前模型："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->basicForm->labelForField(d->themeCombo))) {
                    label->setText(tr("当前主题："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->basicForm->labelForField(d->languageCombo))) {
                    label->setText(tr("当前语言："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->basicForm->labelForField(d->petBubbleCombo))) {
                    label->setText(tr("输出气泡："));
                }
            }

            if (d->aiForm) {
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->characterNameRow))) {
                    label->setText(tr("角色名称："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->chatContextMessagesRow))) {
                    label->setText(tr("上下文条数："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->llmMaxTokensRow))) {
                    label->setText(tr("maxTokens："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->aiBaseUrlRow))) {
                    label->setText(tr("对话API："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->aiKeyRow))) {
                    label->setText(tr("对话KEY："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->aiModelRow))) {
                    label->setText(tr("对话模型："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->aiSystemPromptRow))) {
                    label->setText(tr("对话人设："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->ttsBaseUrlRow))) {
                    label->setText(tr("语音API："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->ttsKeyRow))) {
                    label->setText(tr("语音KEY："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->ttsModelRow))) {
                    label->setText(tr("语音模型："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->aiForm->labelForField(d->ttsVoiceRow))) {
                    label->setText(tr("语音音色："));
                }
            }

            if (d->advancedForm) {
                if (auto* label = qobject_cast<QLabel*>(d->advancedForm->labelForField(d->texCapCombo))) {
                    label->setText(tr("贴图上限："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->advancedForm->labelForField(d->msaaCombo))) {
                    label->setText(tr("MSAA："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->advancedForm->labelForField(d->screenCombo))) {
                    label->setText(tr("模型显示："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->advancedForm->labelForField(d->audioOutCombo))) {
                    label->setText(tr("音频输出："));
                }
                if (auto* label = qobject_cast<QLabel*>(d->advancedForm->labelForField(d->cleanupRowWidget))) {
                    label->setText(tr("清理："));
                }
            }

            if (d->tabs) {
                d->tabs->setTabText(0, tr("基本设置"));
                if (d->tabs->count() > 1) d->tabs->setTabText(1, tr("模型设置"));
                if (d->tabs->count() > 2) d->tabs->setTabText(2, tr("AI设置"));
                if (d->tabs->count() > 3) d->tabs->setTabText(3, tr("高级设置"));
            }

            if (d->wmFileLabel && d->wmFileLabel->text() == QStringLiteral("无")) {
                d->wmFileLabel->setText(tr("无"));
            }
            if (d->curModelName && d->curModelName->text() == QStringLiteral("(无)")) {
                d->curModelName->setText(tr("(无)"));
            }

            // Tooltips: re-apply translated content at runtime.
            if (d->tipBreath) d->tipBreath->setToolTip(tr("让角色在静止时也会轻微起伏（身体角度/呼吸参数）。开启视线跟踪时，自动呼吸不再影响头部；关闭后恢复。关闭本项后参数会复位。"));
            if (d->tipBlink) d->tipBlink->setToolTip(tr("自动眨眼并保留更自然的间隔与瞬目效果。关闭后眼部相关参数复位。"));
            if (d->tipGaze) d->tipGaze->setToolTip(tr("眼球、头部与身体随鼠标方向轻微转动，远距离时幅度会衰减。默认关闭；开启后将屏蔽自动呼吸对头部的影响。关闭后恢复眼球微动策略。"));
            if (d->tipPhysics) d->tipPhysics->setToolTip(tr("根据模型 physics3.json 的配置驱动物理（如头发/衣物摆动）。关闭后复位受影响参数。"));
            if (d->tipAlwaysOnTop) d->tipAlwaysOnTop->setToolTip(tr("开启后桌宠窗口将始终置顶显示（全局置顶）。"));
            if (d->tipTransparentBg) d->tipTransparentBg->setToolTip(tr("开启后窗口背景为透明，仅显示角色本体；关闭后窗口为不透明背景。"));
            if (d->tipMousePassthrough) d->tipMousePassthrough->setToolTip(tr("开启后窗口将完全不接收鼠标输入（可点击穿透到桌面/其他窗口）。\n\n提示：开启后需要 Alt+Tab 切回本程序，再在设置里关闭。"));

            if (d->tipAiBaseUrl) d->tipAiBaseUrl->setToolTip(tr("填写 OpenAI 兼容接口的 Base URL。\n例： https://example.com/v1 （只填到 /v1 即可，程序会自动补全为 /v1/chat/completions）\n\n对应 OpenAI 参数：请求地址（endpoint）。\n用于：Chat Completions（/chat/completions）。"));
            if (d->tipAiKey) d->tipAiKey->setToolTip(tr("API Key（可留空）。\n通常是 Bearer Token：Authorization: Bearer <api_key>。\n\n对应 OpenAI 参数：请求头 Authorization。\n留空时将不发送 Authorization 头（适用于本地/内网免鉴权服务）。"));
            if (d->tipAiModel) d->tipAiModel->setToolTip(tr("要使用的对话模型名称。\n例：gpt-4o-mini、qwen-plus、deepseek-chat 等（取决于你的服务端支持）。\n\n对应 OpenAI 参数：model。"));
            if (d->tipCharacterName) d->tipCharacterName->setToolTip(tr("对话时 AI 的角色名。\n\n会在发送前替换 system prompt（对话人设）里的 $name$。\n\n示例：角色名称 = 小墨，那么对话人设中的“你是 $name$”会变成“你是 小墨”。"));
            if (d->tipChatContextMessages) d->tipChatContextMessages->setToolTip(tr("发送给本地 LLM 的历史消息条数（只取最近的 user/assistant 消息）。\n\n范围：0～200（设为 0 表示不带历史）。\n\n推荐：\n  • 1.5B：12\n  • 7B：24\n\n默认：16"));
            if (d->tipLlmMaxTokens) d->tipLlmMaxTokens->setToolTip(tr("本地 LLM 单次最多生成的 token 数（上限越大，回复可能越长，但耗时更久）。\n\n范围：1～4096。\n\n推荐：\n  • 1.5B：192\n  • 7B：384\n\n默认：256"));
            if (d->tipAiSystemPrompt) d->tipAiSystemPrompt->setToolTip(tr("System Prompt（系统提示词 / 人设），用于规定 AI 的角色、语气、规则与边界。\n\n对应 OpenAI 参数：messages[0].role = \"system\" 的 content。\n\n支持变量：\n  • $name$：会在发送前替换成“角色名称”。\n\n示例：\n你是 $name$，是一只桌宠。回答要简短、温柔，并尽量使用口语。"));
            if (d->tipAiStream) d->tipAiStream->setToolTip(tr("开启后，AI 回复会“边生成边显示”（更像实时打字）。\n关闭后，会等完整回复生成后一次性显示。\n\n对应 OpenAI 参数：stream（true/false）。\n提示：启用语音时，仍建议开启流式输出以获得更自然的对话节奏。"));
            if (d->tipTtsBaseUrl) d->tipTtsBaseUrl->setToolTip(tr("填写 OpenAI 兼容 TTS 接口的 Base URL。\n例： https://example.com/v1 （只填到 /v1 即可，程序会自动补全为 /v1/audio/speech）\n\n对应 OpenAI 参数：请求地址（endpoint）。\n用于：Text-to-Speech（/audio/speech）。\n\n留空表示禁用语音：只显示文字，不请求语音。"));
            if (d->tipTtsKey) d->tipTtsKey->setToolTip(tr("TTS 的 API Key（可留空）。\n对应 OpenAI 参数：请求头 Authorization。\n留空时将不发送 Authorization 头。"));
            if (d->tipTtsModel) d->tipTtsModel->setToolTip(tr("TTS 模型名称。\n例：tts-1、gpt-4o-mini-tts 等（取决于服务端支持）。\n\n对应 OpenAI 参数：model。"));
            if (d->tipTtsVoice) d->tipTtsVoice->setToolTip(tr("语音音色（voice）。\n不同服务端的可用值不同，常见如：alloy、verse、aria 等。\n\n对应 OpenAI 参数：voice。"));

            if (d->chatContextHint) d->chatContextHint->setText(tr("范围：0～200；默认：16；推荐：1.5B=12，7B=24"));
            if (d->llmMaxTokensHint) d->llmMaxTokensHint->setText(tr("范围：1～4096；默认：256；推荐：1.5B=192，7B=384"));

            // Re-localize default labels in dynamic combos.
            const QString defaultSuffixZh = QStringLiteral("（默认）");
            const QString defaultSuffixEn = QStringLiteral(" (Default)");
            const QString suffix = tr("（默认）");

            auto relocalizeComboDefaults = [&](QComboBox* combo){
                if (!combo) return;
                for (int i = 0; i < combo->count(); ++i) {
                    QString text = combo->itemText(i);
                    if (i == 0) {
                        combo->setItemText(i, tr("系统默认（默认）"));
                        continue;
                    }
                    bool hasDefault = false;
                    QString base = text;
                    if (base.endsWith(defaultSuffixZh)) {
                        base.chop(defaultSuffixZh.size());
                        hasDefault = true;
                    } else if (base.endsWith(defaultSuffixEn)) {
                        base.chop(defaultSuffixEn.size());
                        hasDefault = true;
                    }
                    if (hasDefault) combo->setItemText(i, base + suffix);
                }
            };
            relocalizeComboDefaults(d->screenCombo);
            relocalizeComboDefaults(d->audioOutCombo);

            // Recalculate layout after retranslation and shrink back if content got narrower.
            auto shrinkWindowForLanguage = [this]{
                if (!d || !d->central) return;
                if (isMaximized() || isFullScreen()) return;

                if (auto* lay = d->central->layout()) {
                    lay->invalidate();
                    lay->activate();
                }

                const int currentWidth = width();
                const int minBaseWidth = 520;
                int targetWidth = qMax(minBaseWidth, minimumSizeHint().width());

                // Fallback for zh locales where delayed size hint updates can miss the shrink.
                const QString langCode = SettingsManager::instance().currentLanguage();
                if (langCode.startsWith(QStringLiteral("zh"), Qt::CaseInsensitive)
                    && targetWidth >= currentWidth
                    && currentWidth > minBaseWidth)
                {
                    targetWidth = minBaseWidth;
                }

                if (currentWidth > targetWidth) {
                    resize(targetWidth, height());
                }
            };

            QTimer::singleShot(0, this, shrinkWindowForLanguage);
            QTimer::singleShot(30, this, shrinkWindowForLanguage);
        }
    }

    if (e->type() == QEvent::ApplicationPaletteChange
        || e->type() == QEvent::PaletteChange
        || e->type() == QEvent::ThemeChange
        || e->type() == QEvent::StyleChange)
    {
        if (d && d->tabs)
        {
            d->tabs->setTabIcon(0, Theme::themedIcon(Theme::IconToken::SettingsBasic));
            if (d->tabs->count() > 1) d->tabs->setTabIcon(1, Theme::themedIcon(Theme::IconToken::SettingsModel));
            if (d->tabs->count() > 2) d->tabs->setTabIcon(2, Theme::themedIcon(Theme::IconToken::SettingsAi));
            if (d->tabs->count() > 3) d->tabs->setTabIcon(3, Theme::themedIcon(Theme::IconToken::SettingsAdvanced));
        }

        refreshSidebarThemeIndicator();
    }

    return QMainWindow::event(e);
}

static void addWatchDirIfExists(QFileSystemWatcher* fsw, const QString& path) {
    QDir d(path); if (d.exists()) fsw->addPath(path);
}

void SettingsWindow::refreshModelList() {
    auto models = SettingsManager::instance().scanModels();
    const QSignalBlocker blocker(d->modelCombo);
    d->modelCombo->clear();
    QString sel = SettingsManager::instance().selectedModelFolder();
    int cur = -1; int i=0;
    for (const auto& e : models) {
        d->modelCombo->addItem(e.folderName, e.folderName);
        d->modelCombo->setItemData(i, e.jsonPath, Qt::UserRole + 1);
        if (e.folderName == sel) cur = i;
        ++i;
    }
    if (cur < 0 && !models.isEmpty()) {
        for (int j=0;j<models.size();++j) if (models[j].folderName.compare("Hiyori", Qt::CaseInsensitive)==0) { cur = j; break; }
        if (cur < 0) cur = 0; SettingsManager::instance().setSelectedModelFolder(models[cur].folderName);
    }
    if (cur >= 0) d->modelCombo->setCurrentIndex(cur);
    if (d->fsw) {
        QString root = SettingsManager::instance().modelsRoot();
        QStringList watched = d->fsw->directories();
        for (const QString& w : watched) d->fsw->removePath(w);
        addWatchDirIfExists(d->fsw, root);
    }
}
