#ifndef FLOWDESIGNDLG_H
#define FLOWDESIGNDLG_H

#include <QDialog>
#include <QProgressDialog>

namespace Ui {
class FlowDesignDlg;
}

class FlowDesignDlg : public QDialog
{
    Q_OBJECT

public:
    explicit FlowDesignDlg(QWidget *parent = nullptr);
    ~FlowDesignDlg();

signals:
    void flowDesign(std::vector<double> targetFlow);
    void stopDesign();
public slots:
    void onOkClicked();
    void checkValue();


protected:
    void closeEvent(QCloseEvent *) override;

private:
    Ui::FlowDesignDlg *ui;
    static int flow1, flow2, flow3;
    QProgressDialog *progressDlg;

};

#endif // FLOWDESIGNDLG_H
