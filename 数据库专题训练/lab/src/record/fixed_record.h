#ifndef THDB_FIXED_RECORD_H_
#define THDB_FIXED_RECORD_H_

#include "defines.h"
#include "field/field.h"
#include "record/record.h"

namespace thdb {

class FixedRecord : public Record {
 public:
  FixedRecord(Size nFieldSize, const std::vector<FieldType> &iTypeVec,
              const std::vector<Size> &iSizeVec);
  ~FixedRecord() = default;

  /**
   * @brief 记录反序列化
   *
   * @param src 反序列化源数据
   * @return Size 反序列化使用的数据长度
   */
  Size Load(const uint8_t *src) override;
  /**
   * @brief 记录序列化
   *
   * @param dst 序列化结果存储位置
   * @return Size 序列化使用的数据长度
   */
  Size Store(uint8_t *dst) const override;
  /**
   * @brief 从String数据构建记录
   *
   * @param iRawVec Insert语句中的String数组
   */
  void Build(const std::vector<String> &iRawVec) override;

  Record *Copy() const override;

  void Sub(const std::vector<Size> &iPos) override;
  void Add(Record *pRecord) override;
  void Remove(FieldID nPos) override;

 private:
  /**
   * @brief 各个字段的类型
   */
  std::vector<FieldType> _iTypeVec;
  /**
   * @brief 各个字段分配的空间长度
   */
  std::vector<Size> _iSizeVec;
};

}  // namespace thdb

#endif
