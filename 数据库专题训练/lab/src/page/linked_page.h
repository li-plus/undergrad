#ifndef THDB_LINKED_PAGE_H_
#define THDB_LINKED_PAGE_H_

#include "defines.h"
#include "page/page.h"

namespace thdb {

/**
 * @brief 链表结点页面。
 * 
 */
class LinkedPage : public Page {
 public:
  LinkedPage();
  LinkedPage(PageID nPageID);
  virtual ~LinkedPage();

  /**
   * @brief 向当前页面后添加一个新的页面
   * 
   * @param _pPage 需要添加的页面
   * @return true 添加成功
   * @return false 添加失败
   */
  bool PushBack(LinkedPage* _pPage);
  /**
   * @brief 删除当前页面后的第一个页面 
   * 
   * @return PageID 被删除页面的页面编号
   */
  PageID PopBack();

  PageID GetNextID() const;
  PageID GetPrevID() const;

  void SetNextID(PageID nNextID);
  void SetPrevID(PageID nPrevID);

 protected:
  PageID _nNextID;
  PageID _nPrevID;

 private:
  bool _bModified;
};

}  // namespace thdb

#endif  // META_PAGE_OBJECT_H_
