#include "table/table.h"

#include <assert.h>

#include <algorithm>

#include "macros.h"
#include "minios/os.h"
#include "page/record_page.h"
#include "record/fixed_record.h"

namespace thdb {

PageID NextPageID(PageID nCur) {
  LinkedPage *pPage = new LinkedPage(nCur);
  PageID nNext = pPage->GetNextID();
  delete pPage;
  return nNext;
}

Table::Table(PageID nTableID) {
  pTable = new TablePage(nTableID);

  _nHeadID = pTable->GetHeadID();
  _nTailID = pTable->GetTailID();
  _nNotFull = _nHeadID;
  NextNotFull();
}

Table::~Table() { delete pTable; }

Record *Table::GetRecord(PageID nPageID, SlotID nSlotID) {
  // return nullptr;  // 开始实验时删除此行
  // LAB1 BEGIN
  // TODO: 获得一条记录
  // TIPS: 利用RecordPage::GetRecord获取无格式记录数据
  // TIPS: 利用TablePage::GetFieldSize, GetTyepVec,
  // GetSizeVec三个函数可以构建空的FixedRecord对象 TIPS:
  // 利用Record::Load导入数据 ALERT: 需要注意析构所有不会返回的内容
  // LAB1 END
  RecordPage recPage(nPageID);
  auto src = recPage.GetRecord(nSlotID);
  auto rec = new FixedRecord(pTable->GetFieldSize(), pTable->GetTypeVec(),
                             pTable->GetSizeVec());
  rec->Load(src);
  delete[] src;
  return rec;
}

PageSlotID Table::InsertRecord(Record *pRecord) {
  // return std::pair<PageID, SlotID>();  // 开始实验时删除此行
  // LAB1 BEGIN
  // TODO: 插入一条记录
  // TIPS: 利用_nNotFull来获取有空间的页面
  // TIPS: 利用Record::Store获得序列化数据
  // TIPS: 利用RecordPage::InsertRecord插入数据
  // TIPS: 注意页满时更新_nNotFull
  // LAB1 END
  uint8_t src[PAGE_SIZE];
  pRecord->Store(src);
  PageID nPageID = _nNotFull;
  SlotID nSlotID;
  bool isFull;
  {
    RecordPage recPage(nPageID);
    nSlotID = recPage.InsertRecord(src);
    isFull = recPage.Full();
  }
  if (isFull) {
    NextNotFull();
  }
  return {nPageID, nSlotID};
}

void Table::DeleteRecord(PageID nPageID, SlotID nSlotID) {
  // LAB1 BEGIN
  // TIPS: 利用RecordPage::DeleteRecord插入数据
  // TIPS: 注意更新_nNotFull来保证较高的页面空间利用效率
  // LAB1 END
  RecordPage recPage(nPageID);
  bool orgFull = recPage.Full();
  recPage.DeleteRecord(nSlotID);
  if (orgFull) {
    _nNotFull = std::min(_nNotFull, nPageID);
  }
}

void Table::UpdateRecord(PageID nPageID, SlotID nSlotID,
                         const std::vector<Transform> &iTrans) {
  // LAB1 BEGIN
  // TIPS: 仿照InsertRecord从无格式数据导入原始记录
  // TIPS: 构建Record对象，利用Record::SetField更新Record对象
  // TIPS: Trasform::GetPos表示更新位置，GetField表示更新后的字段
  // TIPS: 将新的记录序列化
  // TIPS: 利用RecordPage::UpdateRecord更新一条数据
  // LAB1 END
  RecordPage recPage(nPageID);
  auto recBuf = recPage.GetRecord(nSlotID);
  FixedRecord rec(pTable->GetFieldSize(), pTable->GetTypeVec(),
                  pTable->GetSizeVec());
  rec.Load(recBuf);
  for (auto &trans : iTrans) {
    rec.SetField(trans.GetPos(), trans.GetField());
  }
  rec.Store(recBuf);
  recPage.UpdateRecord(nSlotID, recBuf);
  delete[] recBuf;
}

std::vector<PageSlotID> Table::SearchRecord(Condition *pCond) {
  //  return std::vector<PageSlotID>();  // 开始实验时删除此行
  // LAB1 BEGIN
  // TODO: 对记录的条件检索
  // TIPS: 仿照InsertRecord从无格式数据导入原始记录
  // TIPS: 依次导入各条记录进行条件判断
  // TIPS: Condition的抽象方法Match可以判断Record是否满足检索条件
  // TIPS: 返回所有符合条件的结果的pair<PageID,SlotID>
  // LAB1 END
  std::vector<PageSlotID> results;
  PageID pid = _nHeadID;
  while (pid != NULL_PAGE) {
    RecordPage recPage(pid);
    for (SlotID sid = 0; sid < recPage.GetCap(); sid++) {
      if (recPage.HasRecord(sid)) {
        auto src = recPage.GetRecord(sid);
        FixedRecord rec(pTable->GetFieldSize(), pTable->GetTypeVec(),
                        pTable->GetSizeVec());
        rec.Load(src);
        delete[] src;
        if (pCond == nullptr || pCond->Match(rec)) {
          results.emplace_back(pid, sid);
        }
      }
    }
    pid = recPage.GetNextID();
  }
  return results;
}

void Table::SearchRecord(std::vector<PageSlotID> &iPairs, Condition *pCond) {
  if (!pCond) return;
  auto it = iPairs.begin();
  while (it != iPairs.end()) {
    Record *pRecord = GetRecord(it->first, it->second);
    if (!pCond->Match(*pRecord)) {
      it = iPairs.erase(it);
    } else
      ++it;
    delete pRecord;
  }
}

void Table::Clear() {
  PageID nBegin = _nHeadID;
  while (nBegin != NULL_PAGE) {
    PageID nTemp = nBegin;
    nBegin = NextPageID(nBegin);
    MiniOS::GetOS()->DeletePage(nTemp);
  }
}

void Table::NextNotFull() {
  // LAB1 BEGIN
  // TODO: 实现一个快速查找非满记录页面的算法
  // ALERT: ！！！一定要注意！！！
  // 不要同时建立两个指向相同磁盘位置的且可变对象，否则会出现一致性问题
  // ALERT: 可以适当增加传入参数，本接口不会被外部函数调用，例如额外传入Page
  // *指针
  // TIPS:
  // 充分利用链表性质，注意全满时需要在结尾_pTable->GetTailID对应结点后插入新的结点，并更新_pTable的TailID
  // TIPS: 只需要保证均摊复杂度较低即可
  // LAB1 END

  PageID pageID = _nNotFull;
  while (pageID != NULL_PAGE) {
    RecordPage page(pageID);
    if (!page.Full()) {
      _nNotFull = pageID;
      return;
    }
    pageID = page.GetNextID();
  }
  RecordPage tailPage(_nTailID);
  RecordPage newPage(pTable->GetTotalSize(), true);
  tailPage.PushBack(&newPage);
  _nNotFull = _nTailID = newPage.GetPageID();
}

FieldID Table::GetPos(const String &sCol) const { return pTable->GetPos(sCol); }

FieldType Table::GetType(const String &sCol) const {
  return pTable->GetType(sCol);
}

Size Table::GetSize(const String &sCol) const { return pTable->GetSize(sCol); }

Record *Table::EmptyRecord() const {
  FixedRecord *pRecord = new FixedRecord(
      pTable->GetFieldSize(), pTable->GetTypeVec(), pTable->GetSizeVec());
  return pRecord;
}

bool CmpByFieldID(const std::pair<String, FieldID> &a,
                  const std::pair<String, FieldID> &b) {
  return a.second < b.second;
}

std::vector<String> Table::GetColumnNames() const {
  std::vector<String> iVec{};
  std::vector<std::pair<String, FieldID>> iPairVec(pTable->_iColMap.begin(),
                                                   pTable->_iColMap.end());
  std::sort(iPairVec.begin(), iPairVec.end(), CmpByFieldID);
  for (const auto &it : iPairVec) iVec.push_back(it.first);
  return iVec;
}

}  // namespace thdb
