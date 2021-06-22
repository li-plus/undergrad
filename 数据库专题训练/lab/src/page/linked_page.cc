#include "page/linked_page.h"

#include "macros.h"
#include "minios/os.h"

namespace thdb {

const PageOffset NEXT_PAGE_OFFSET = 4;
const PageOffset PREV_PAGE_OFFSET = 8;

LinkedPage::LinkedPage() : Page() {
  this->_bModified = true;
  this->_nNextID = NULL_PAGE;
  this->_nPrevID = NULL_PAGE;
}

LinkedPage::LinkedPage(PageID nPageID) : Page(nPageID) {
  this->_bModified = false;
  GetHeader((uint8_t *)&_nNextID, 4, NEXT_PAGE_OFFSET);
  GetHeader((uint8_t *)&_nPrevID, 4, PREV_PAGE_OFFSET);
}

LinkedPage::~LinkedPage() {
  if (_bModified) {
    // Dirty Page Condition
    SetHeader((uint8_t *)&_nNextID, 4, NEXT_PAGE_OFFSET);
    SetHeader((uint8_t *)&_nPrevID, 4, PREV_PAGE_OFFSET);
  }
}

uint32_t LinkedPage::GetNextID() const { return _nNextID; }

uint32_t LinkedPage::GetPrevID() const { return _nPrevID; }

void LinkedPage::SetNextID(PageID nNextID) {
  this->_nNextID = nNextID;
  this->_bModified = true;
}

void LinkedPage::SetPrevID(PageID nPrevID) {
  this->_nPrevID = nPrevID;
  this->_bModified = true;
}

bool LinkedPage::PushBack(LinkedPage *pPage) {
  if (!pPage) return false;
  // LAB1 BEGIN
  // TODO: 链表结点后增加一个新的链表页面结点
  // ALERT: ！！！一定要注意！！！
  // 不要同时建立两个指向相同磁盘位置的且可变对象，否则会出现一致性问题
  // ALERT:
  // 页面对象完成修改后一定要及时析构，析构会自动调用写回。以方便其他函数重新基于这一页面建立页面对象
  // TIPS: 需要判断当前页面是否存在后续页面
  // TIPS：正确设置当前页面和pPage的PrevID以及NextID
  // LAB1 END
  pPage->SetNextID(_nNextID);
  pPage->SetPrevID(_nPageID);
  if (_nNextID != NULL_PAGE) {
    LinkedPage nextPage(_nNextID);
    nextPage.SetPrevID(pPage->_nPageID);
  }
  SetNextID(pPage->_nPageID);
  return true;
}

PageID LinkedPage::PopBack() {
  // return -1;  // 开始实验时删除此行
  // LAB1 BEGIN
  // TODO: 删除下一个链表结点，返回删除结点的PageID
  // TIPS: 需要判断当前页面是否存在后续页面
  // TIPS:
  // 正确设置当前页面的NextID，如果后一个结点不是空页，需要讨论是否存在后两页的PrevID问题
  // TIPS: 可以使用MiniOS::DeletePage在最后释放被删除的页面
  // LAB1 END
  if (_nNextID == NULL_PAGE) { return NULL_PAGE; }
  PageID popID = _nNextID;
  LinkedPage nextPage(_nNextID);
  if (nextPage._nNextID != NULL_PAGE) {
    LinkedPage nextNextPage(nextPage._nNextID);
    nextNextPage.SetPrevID(_nPageID);
  }
  SetNextID(nextPage._nNextID);

  auto os = MiniOS::GetOS();
  os->DeletePage(popID);

  return popID;
}

}  // namespace thdb
