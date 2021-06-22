#include "field/field.h"

namespace thdb {

bool Less(Field *pA, Field *pB, FieldType iType);
bool Equal(Field *pA, Field *pB, FieldType iType);
bool Greater(Field *pA, Field *pB, FieldType iType);
}  // namespace thdb