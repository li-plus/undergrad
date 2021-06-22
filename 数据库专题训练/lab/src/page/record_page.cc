#include "record_page.h"

#include <assert.h>

#include <cstring>

#include "exception/exceptions.h"
#include "macros.h"

namespace thdb {

const PageOffset FIXED_SIZE_OFFSET = 12;
const PageOffset BITMAP_OFFSET = 0;
const PageOffset BITMAP_SIZE = 128;

RecordPage::RecordPage(PageOffset nFixed, bool) : LinkedPage() {
  _nFixed = nFixed;
  _pUsed = new Bitmap((DATA_SIZE - BITMAP_SIZE) / nFixed);
  _nCap = (DATA_SIZE - BITMAP_SIZE) / _nFixed;
  SetHeader((uint8_t *)&_nFixed, 2, FIXED_SIZE_OFFSET);
}

RecordPage::RecordPage(PageID nPageID) : LinkedPage(nPageID) {
  GetHeader((uint8_t *)&_nFixed, 2, FIXED_SIZE_OFFSET);
  _pUsed = new Bitmap((DATA_SIZE - BITMAP_SIZE) / _nFixed);
  _nCap = (DATA_SIZE - BITMAP_SIZE) / _nFixed;
  LoadBitmap();
}

RecordPage::~RecordPage() { StoreBitmap(); }

void RecordPage::LoadBitmap() {
  uint8_t pTemp[BITMAP_SIZE];
  memset(pTemp, 0, BITMAP_SIZE);
  GetData(pTemp, BITMAP_SIZE, BITMAP_OFFSET);
  _pUsed->Load(pTemp);
}

void RecordPage::StoreBitmap() {
  uint8_t pTemp[BITMAP_SIZE];
  memset(pTemp, 0, BITMAP_SIZE);
  _pUsed->Store(pTemp);
  SetData(pTemp, BITMAP_SIZE, BITMAP_OFFSET);
}

Size RecordPage::GetCap() const { return _nCap; }

Size RecordPage::GetUsed() const { return _pUsed->GetUsed(); }

bool RecordPage::Full() const { return _pUsed->Full(); }

void RecordPage::Clear() {
  for (SlotID i = 0; i < _nCap; ++i)
    if (HasRecord(i)) DeleteRecord(i);
}

SlotID RecordPage::InsertRecord(const uint8_t *src) {
  // return -1;  // 开始实验时删除此行
  // LAB1 BEGIN
  // TODO: 寻找空的槽位，插入数据
  // TIPS: 合理抛出异常的方式可以帮助DEBUG工作
  // TIPS: 利用_pUsed位图判断槽位是否使用，插入后需要更新_pUsed
  // TIPS: 使用SetData实现写数据
  // LAB1 END
  for (Size nSlotID = 0; nSlotID < _nCap; nSlotID++) {
    if (!HasRecord(nSlotID)) {
      SetData(src, _nFixed, BITMAP_OFFSET + BITMAP_SIZE + nSlotID * _nFixed);
      _pUsed->Set(nSlotID);
      return nSlotID;
    }
  }
  assert(false);
  return _nCap;
}

uint8_t *RecordPage::GetRecord(SlotID nSlotID) {
  // return nullptr;  // 开始实验时删除此行
  // LAB1 BEGIN
  // TODO: 获得nSlotID槽位置的数据
  // TIPS: 使用GetData实现读数据
  // TIPS: 注意需要使用new分配_nFixed大小的空间
  // LAB1 END
  if (!HasRecord(nSlotID)) {
    throw RecordPageException(nSlotID);
  }
  auto rec = new uint8_t[_nFixed];
  GetData(rec, _nFixed, BITMAP_OFFSET + BITMAP_SIZE + nSlotID * _nFixed);
  return rec;
}

bool RecordPage::HasRecord(SlotID nSlotID) { return _pUsed->Get(nSlotID); }

void RecordPage::DeleteRecord(SlotID nSlotID) {
  // LAB1 BEGIN
  // TODO: 删除一条记录
  // TIPS: 需要设置_pUsed
  // LAB1 END
  if (!HasRecord(nSlotID)) {
    throw RecordPageException(nSlotID);
  }
  _pUsed->Unset(nSlotID);
}

void RecordPage::UpdateRecord(SlotID nSlotID, const uint8_t *src) {
  if (!_pUsed->Get(nSlotID)) throw RecordPageException(nSlotID);
  SetData(src, _nFixed, BITMAP_OFFSET + BITMAP_SIZE + nSlotID * _nFixed);
}

}  // namespace thdb
