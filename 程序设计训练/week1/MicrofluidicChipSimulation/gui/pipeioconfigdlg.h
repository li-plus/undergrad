#ifndef PIPEIOCONFIGDLG_H
#define PIPEIOCONFIGDLG_H

#include <QDialog>

namespace Ui {
class PipeIOConfigDlg;
}

class PipeIOConfigDlg : public QDialog
{
    Q_OBJECT

public:
    explicit PipeIOConfigDlg(bool isNew, QWidget *parent = nullptr);
    ~PipeIOConfigDlg();
    void checkValue();
    void setIO(int sideLength, const std::vector<int> &inputCol, const std::vector<int> &outputCol);
    void onOkClicked();

signals:
    void PipeIOChanged(int sideLength, std::vector<int> input, std::vector<int> output, bool isNew);

private:
    bool isNew;
    Ui::PipeIOConfigDlg *ui;
};

#endif // PIPEIOCONFIGDLG_H
