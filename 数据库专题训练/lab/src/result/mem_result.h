#ifndef THDB_MEM_RESULT_H_
#define THDB_MEM_RESULT_H_

#include "defines.h"
#include "result/result.h"

namespace thdb {

class MemResult : public Result {
 public:
  MemResult(const std::vector<String> &iHeader);
  ~MemResult();

  void PushBack(Record *pRecord) override;
  Record *GetRecord(Size nPos) const override;
  Size GetSize() const override;
  String ToString() const override;
  std::vector<String> ToVector() const override;
  void Display() const override;

 private:
  std::vector<Record *> _iData;
};

}  // namespace thdb

#endif
