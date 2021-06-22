#ifndef THDB_INDEX_H_
#define THDB_INDEX_H_

#include "defines.h"
#include "field/fields.h"

namespace thdb {

class IndexPage;

class Index {
 public:
  /**
   * @brief 构建一个特定类型的索引
   * @param iType 字段类型
   */
  Index(FieldType iType);
  /**
   * @brief 从一个页面编号构建索引
   * @param nRootID
   */
  Index(PageID nRootID);

  ~Index();

  /**
   * @brief 插入一条Key Value Pair
   * @param pKey 插入的Key
   * @param iPair 插入的Value
   * @return true 插入成功
   * @return false 插入失败
   */
  bool Insert(Field *pKey, const PageSlotID &iPair);
  /**
   * @brief 删除某个Key下所有的Key Value Pair
   * @param pKey 删除的Key
   * @return Size 删除的键值数量
   */
  Size Delete(Field *pKey);
  /**
   * @brief 删除某个Key Value Pair
   * @param pKey 删除的Key
   * @param iPair 删除的Value
   * @return true 删除成功
   * @return false 删除失败
   */
  bool Delete(Field *pKey, const PageSlotID &iPair);
  /**
   * @brief 更新某个Key Value Pair到新的Value
   * @param pKey 更新的Key
   * @param iOld 原始的Value
   * @param iNew 要更新成的新Value
   * @return true 更新成功
   * @return false 更新失败
   */
  bool Update(Field *pKey, const PageSlotID &iOld, const PageSlotID &iNew);
  /**
   * @brief 使用索引进行范围查找，左闭右开区间[pLow, pHigh)
   *
   * @param pLow
   * @param pHigh
   * @return std::vector<PageSlotID> 所有符合范围条件的Value数组
   */
  std::vector<PageSlotID> Range(Field *pLow, Field *pHigh);

  /**
   * @brief 清空索引占用的所有空间
   */
  void Clear();

  /**
   * @brief 获得根结点对应的页面编号
   * @return PageID
   */
  PageID GetRootID() const;

  void Print() const;

 private:
  IndexPage *_indexPage;
};

}  // namespace thdb

#endif
