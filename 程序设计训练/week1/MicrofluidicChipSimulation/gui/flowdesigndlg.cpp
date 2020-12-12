#include "flowdesigndlg.h"
#include "ui_flowdesigndlg.h"
#include "definition.h"
#include <QPushButton>
#include <QMessageBox>


int FlowDesignDlg::flow1 = -1;
int FlowDesignDlg::flow2 = -1;
int FlowDesignDlg::flow3 = -1;

FlowDesignDlg::FlowDesignDlg(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FlowDesignDlg)
{
    ui->setupUi(this);

    connect(ui->btnStart, &QPushButton::clicked, this, &FlowDesignDlg::onOkClicked);

    connect(ui->spinBoxFlow1, QOverload<int>::of(&QSpinBox::valueChanged), this, &FlowDesignDlg::checkValue);
    connect(ui->spinBoxFlow2, QOverload<int>::of(&QSpinBox::valueChanged), this, &FlowDesignDlg::checkValue);
    connect(ui->spinBoxFlow3, QOverload<int>::of(&QSpinBox::valueChanged), this, &FlowDesignDlg::checkValue);


    if(flow1 > 0 && flow2 > 0 && flow3 > 0)
    {
        ui->spinBoxFlow1->setValue(flow1);
        ui->spinBoxFlow2->setValue(flow2);
        ui->spinBoxFlow3->setValue(flow3);
    }
    this->setWindowTitle(tr("Targeted Flows Input Dialog"));

    this->setWindowModality(Qt::ApplicationModal);
}
void FlowDesignDlg::checkValue()
{
    if(ui->spinBoxFlow1->value() + ui->spinBoxFlow2->value() + ui->spinBoxFlow3->value() != 400)
    {
        ui->btnStart->setEnabled(false);
    }
    else
    {
        ui->btnStart->setEnabled(true);
    }
}

void FlowDesignDlg::onOkClicked()
{
    std::vector<double> targetFlow(OUTPUT_NUM);
    targetFlow[0] = flow1 = ui->spinBoxFlow1->value();
    targetFlow[1] = flow2 = ui->spinBoxFlow2->value();
    targetFlow[2] = flow3 = ui->spinBoxFlow3->value();

    ui->progressBar->setMaximum(0);
    ui->btnStart->setEnabled(false);
    ui->spinBoxFlow1->setEnabled(false);
    ui->spinBoxFlow2->setEnabled(false);
    ui->spinBoxFlow3->setEnabled(false);
    emit flowDesign(targetFlow);
}
void FlowDesignDlg::closeEvent(QCloseEvent *)
{
    emit stopDesign();
}

FlowDesignDlg::~FlowDesignDlg()
{
    delete ui;
}
