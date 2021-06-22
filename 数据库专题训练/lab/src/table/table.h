#ifndef THDB_TABLE_H_
#define THDB_TABLE_H_

#include "condition/condition.h"
#include "defines.h"
#include "page/table_page.h"
#include "record/record.h"
#include "record/transform.h"
#include "table/schema.h"

namespace thdb {

class Table {
 public:
  Table(PageID nTableID);
  ~Table();

  /**
   * @brief 获取一个指定位置的记录
   *
   * @param nPageID 页编号
   * @param nSlotID 槽编号
   * @return Record* 对应记录
   */
  Record *GetRecord(PageID nPageID, SlotID nSlotID);
  /**
   * @brief 插入一条数据
   *
   * @param pRecord 待插入数据
   * @return PageSlotID 插入的位置
   */
  PageSlotID InsertRecord(Record *pRecord);
  /**
   * @brief 删除一条数据
   *
   * @param nPageID 页编号
   * @param nSlotID 槽编号
   */
  void DeleteRecord(PageID nPageID, SlotID nSlotID);
  /**
   * @brief 更新一条数据
   *
   * @param nPageID 页编号
   * @param nSlotID 槽编号
   * @param iTrans 更新变化方式
   */
  void UpdateRecord(PageID nPageID, SlotID nSlotID,
                    const std::vector<Transform> &iTrans);
  /**
   * @brief 条件检索
   *
   * @param pCond 检索条件
   * @return std::vector<PageSlotID> 符合条件记录的位置
   */
  std::vector<PageSlotID> SearchRecord(Condition *pCond);

  void SearchRecord(std::vector<PageSlotID> &iPairs, Condition *pCond);
  /**
   * @brief 清空页面所有存储记录
   *
   */
  void Clear();

  FieldID GetPos(const String &sCol) const;
  FieldType GetType(const String &sCol) const;
  Size GetSize(const String &sCol) const;
  /**
   * @brief 生成一个未填充数据的空记录体
   *
   * @return Record* 生成的空记录体
   */
  Record *EmptyRecord() const;

  std::vector<String> GetColumnNames() const;

 private:
  TablePage *pTable;
  PageID _nHeadID;
  PageID _nTailID;
  /**
   * @brief 表示一个非满页编号，可用于构建一个时空高效的记录插入算法。
   */
  PageID _nNotFull;

  /**
   * @brief 查找一个可用于插入新记录的页面，不存在时自动添加一个新的页面
   *
   */
  void NextNotFull();
};

}  // namespace thdb

#endif
