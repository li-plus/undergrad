#include "settingsdlg.h"
#include "ui_settingsdlg.h"

#include <QPushButton>
#include "config/gameconfig.h"
#include "mainwindow.h"


SettingsDlg::SettingsDlg(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SettingsDlg)
{
    ui->setupUi(this);
    ui->radioButtonHintOff->setChecked(!gameConfig.getShowHint());
    ui->radioButtonHintOn->setChecked(gameConfig.getShowHint());

    ui->radioButtonAudioOff->setChecked(!gameConfig.getAudioOn());
    ui->radioButtonAudioOn->setChecked(gameConfig.getAudioOn());

    ui->radioButtonBlack->setChecked(gameConfig.getMySide() == BLACK);
    ui->radioButtonRed->setChecked(gameConfig.getMySide() == RED);

    ui->radioButtonClient->setChecked(gameConfig.getAppType() == CLIENT);
    ui->radioButtonServer->setChecked(gameConfig.getAppType() == SERVER);

    ui->spinBoxStepSec->setValue(gameConfig.getStepSec());

    connect(ui->buttonBox->button(QDialogButtonBox::Ok), &QPushButton::clicked, this, &SettingsDlg::updateConfig);
    this->setWindowModality(Qt::ApplicationModal);
}

void SettingsDlg::updateConfig()
{
    gameConfig.setShowHint(ui->radioButtonHintOn->isChecked());
    gameConfig.setAppType(ui->radioButtonClient->isChecked() ? CLIENT : SERVER);
    gameConfig.setGameSide(ui->radioButtonRed->isChecked() ? RED : BLACK);
    gameConfig.setStepSec(ui->spinBoxStepSec->value());
    gameConfig.setAudioOn(ui->radioButtonAudioOn->isChecked());
}

SettingsDlg::~SettingsDlg()
{
    delete ui;
}
