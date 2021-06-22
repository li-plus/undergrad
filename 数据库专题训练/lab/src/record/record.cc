#include "record/record.h"

#include <assert.h>

#include <cstring>

#include "exception/exceptions.h"

namespace thdb {

Record::Record() { _iFields = std::vector<Field *>(); }

Record::Record(Size nFieldSize) {
  _iFields = std::vector<Field *>();
  for (Size i = 0; i < nFieldSize; ++i) _iFields.push_back(nullptr);
}

Record::~Record() { Clear(); }

Field *Record::GetField(FieldID nPos) const { return _iFields[nPos]; }

void Record::SetField(FieldID nPos, Field *pField) {
  if (_iFields[nPos]) delete _iFields[nPos];
  _iFields[nPos] = pField;
}

Size Record::GetSize() const { return _iFields.size(); }

void Record::Clear() {
  for (const auto &pField : _iFields)
    if (pField) delete pField;
  for (int i = 0; i < _iFields.size(); ++i) _iFields[i] = nullptr;
}

void Record::Sub(const std::vector<Size> &iPos) {
  bool bInSub[GetSize()];
  memset(bInSub, 0, GetSize() * sizeof(bool));
  for (const auto nPos : iPos) bInSub[nPos] = 1;
  auto itField = _iFields.begin();
  for (Size i = 0; i < GetSize(); ++i) {
    if (!bInSub[i]) {
      Field *pField = *itField;
      if (pField) delete pField;
      itField = _iFields.erase(itField);
    } else {
      ++itField;
    }
  }
}

void Record::Add(Record *pRecord) {
  for (Size i = 0; i < pRecord->GetSize(); ++i) {
    _iFields.push_back(pRecord->GetField(i)->Copy());
  }
}

void Record::Remove(FieldID nPos) {
  assert(nPos < GetSize());
  auto it = _iFields.begin() + nPos;
  if (*it) delete (*it);
  _iFields.erase(it);
}

String Record::ToString() {
  String result;
  for (const auto &pField : _iFields) {
    if (pField) {
      result += pField->ToString() + " ";
    } else {
      throw Exception();
    }
  }
  return result;
}

}  // namespace thdb
