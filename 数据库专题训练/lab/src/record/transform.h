#ifndef THDB_TRANSFORM_H_
#define THDB_TRANSFORM_H_

#include "defines.h"
#include "field/fields.h"

namespace thdb {

class Transform {
 public:
  Transform(FieldID nFieldID, FieldType iType, const String &sRaw);
  ~Transform() = default;

  Field *GetField() const;
  FieldID GetPos() const;

 private:
  FieldID _nFieldID;
  FieldType _iType;
  String _sRaw;
};

}  // namespace thdb

#endif