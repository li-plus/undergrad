#include "pipeioconfigdlg.h"
#include "ui_pipeioconfigdlg.h"

#include <QPushButton>
#include <QMessageBox>



PipeIOConfigDlg::PipeIOConfigDlg(bool isNew, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PipeIOConfigDlg)
{
    this->isNew = isNew;
    ui->setupUi(this);
    connect(ui->spinBoxIn1, QOverload<int>::of(&QSpinBox::valueChanged), this, &PipeIOConfigDlg::checkValue);
    connect(ui->spinBoxIn2, QOverload<int>::of(&QSpinBox::valueChanged), this, &PipeIOConfigDlg::checkValue);
    connect(ui->spinBoxOut1, QOverload<int>::of(&QSpinBox::valueChanged), this, &PipeIOConfigDlg::checkValue);
    connect(ui->spinBoxOut2, QOverload<int>::of(&QSpinBox::valueChanged), this, &PipeIOConfigDlg::checkValue);
    connect(ui->spinBoxOut3, QOverload<int>::of(&QSpinBox::valueChanged), this, &PipeIOConfigDlg::checkValue);
    connect(ui->spinBoxSideLength, QOverload<int>::of(&QSpinBox::valueChanged), this, &PipeIOConfigDlg::checkValue);
    connect(ui->buttonBox->button(QDialogButtonBox::Ok), &QPushButton::clicked, this, &PipeIOConfigDlg::onOkClicked);
    this->setWindowModality(Qt::ApplicationModal);
    setWindowTitle(tr("Input & Output Pipes Configuration"));
}

void PipeIOConfigDlg::setIO(int sideLength, const std::vector<int> &inputCol, const std::vector<int> &outputCol)
{
    ui->spinBoxIn1->setValue(inputCol[0]);
    ui->spinBoxIn2->setValue(inputCol[1]);
    ui->spinBoxOut1->setValue(outputCol[0]);
    ui->spinBoxOut2->setValue(outputCol[1]);
    ui->spinBoxOut3->setValue(outputCol[2]);
    ui->spinBoxSideLength->setValue(sideLength);
}

void PipeIOConfigDlg::checkValue()
{
    int in1 = ui->spinBoxIn1->value();
    int in2 = ui->spinBoxIn2->value();
    int out1 = ui->spinBoxOut1->value();
    int out2 = ui->spinBoxOut2->value();
    int out3 = ui->spinBoxOut3->value();
    int sideLength = ui->spinBoxSideLength->value();

    if(in1 >= in2 || out1 >= out2 || out1 >= out3 || out2 >= out3)
    {
        ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
        return ;
    }
    if(in1 >= sideLength || in2 >= sideLength || out1 >= sideLength || out2 >= sideLength || out3 >= sideLength)
    {
        ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
        return ;
    }
    ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);

}
void PipeIOConfigDlg::onOkClicked()
{
    int in1 = ui->spinBoxIn1->value();
    int in2 = ui->spinBoxIn2->value();
    int out1 = ui->spinBoxOut1->value();
    int out2 = ui->spinBoxOut2->value();
    int out3 = ui->spinBoxOut3->value();
    int sideLength = ui->spinBoxSideLength->value();
    emit PipeIOChanged(sideLength, std::vector<int>({in1, in2}), std::vector<int>({out1, out2, out3}), isNew);
}

PipeIOConfigDlg::~PipeIOConfigDlg()
{
    delete ui;
}
