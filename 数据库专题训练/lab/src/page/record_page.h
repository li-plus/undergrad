#ifndef THDB_RECORD_PAGE_H_
#define THDB_RECORD_PAGE_H_

#include "page/linked_page.h"
#include "utils/bitmap.h"

namespace thdb {

/**
 * @brief 定长记录页面。
 *
 */
class RecordPage : public LinkedPage {
 public:
  /**
   * @brief 构建一个新的定长记录页面
   * @param nFixed 定长记录长度
   */
  RecordPage(PageOffset nFixed, bool);
  /**
   * @brief 从MiniOS中重新导入一个定长记录页面
   * @param nPageID 页面编号
   */
  RecordPage(PageID nPageID);
  ~RecordPage();

  /**
   * @brief 插入一条定长记录
   *
   * @param src 记录定长格式化后的内容
   * @return SlotID 插入位置的槽编号
   */
  SlotID InsertRecord(const uint8_t *src);
  /**
   * @brief 获取指定位置的记录的内容
   *
   * @param nSlotID 槽编号
   * @return uint8_t* 记录定长格式化的内容
   */
  uint8_t *GetRecord(SlotID nSlotID);
  /**
   * @brief 判断某一个槽是否存在记录
   *
   * @param nSlotID 槽编号
   * @return true 存在记录
   * @return false 不存在记录
   */
  bool HasRecord(SlotID nSlotID);
  /**
   * @brief 删除指定位置的记录
   *
   * @param nSlotID 槽编号
   */
  void DeleteRecord(SlotID nSlotID);
  /**
   * @brief 原地更新一条记录的内容
   *
   * @param nSlotID 槽编号
   * @param src 新的定长格式化内容
   */
  void UpdateRecord(SlotID nSlotID, const uint8_t *src);

  Size GetCap() const;
  Size GetUsed() const;
  bool Full() const;
  void Clear();

 private:
  void StoreBitmap();
  void LoadBitmap();

  /**
   * @brief 表示支持的定长记录长度
   */
  PageOffset _nFixed;
  /**
   * @brief 表示页面能容纳的记录数量
   */
  Size _nCap;
  /**
   * @brief 表示槽占用状况的位图
   */
  Bitmap *_pUsed;
};

}  // namespace thdb

#endif