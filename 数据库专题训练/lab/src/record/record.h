#ifndef THDB_RECORD_H_
#define THDB_RECORD_H_

#include "defines.h"
#include "field/field.h"

namespace thdb {
class Record {
 public:
  Record();
  Record(Size nFieldSize);
  virtual ~Record();

  Field *GetField(FieldID nPos) const;
  void SetField(FieldID nPos, Field *pField);
  /**
   * @brief 获得记录中字段数量
   * @return Size 记录中字段的数量
   */
  Size GetSize() const;

  virtual Size Load(const uint8_t *src) = 0;
  virtual Size Store(uint8_t *dst) const = 0;
  virtual void Build(const std::vector<String> &iRawVec) = 0;
  /**
   * @brief 记录的深拷贝
   */
  virtual Record *Copy() const = 0;

  /**
   * @brief 截取Record的部分字段
   */
  virtual void Sub(const std::vector<Size> &iPos);
  /**
   * @brief 向Record后添加部分字段
   */
  virtual void Add(Record *pRecord);
  /**
   * @brief 删除Record中一个字段
   */
  virtual void Remove(FieldID nPos);

  void Clear();
  String ToString();

 protected:
  std::vector<Field *> _iFields;
};

}  // namespace thdb

#endif
