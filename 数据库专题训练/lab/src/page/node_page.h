#ifndef THDB_NODE_PAGE_H_
#define THDB_NODE_PAGE_H_

#include "field/fields.h"
#include "page/page.h"

namespace thdb {

class IndexPage;

class NodePage : public Page {
 public:
  // header offsets
  const PageOffset SIZE_OFFSET = 8;
  const PageOffset LEAF_OFFSET = 12;

  // header
  Size size;
  bool isLeaf;

  // entries
  std::vector<Field*> keys;
  std::vector<PageSlotID> children;

  NodePage(IndexPage* indexPage_, bool isLeaf_,
           const std::vector<Field*>& fields_ = {},
           const std::vector<PageSlotID>& children_ = {});

  NodePage(IndexPage* indexPage_, PageID nPageID);

  ~NodePage() override;

  void ClearChildren();

  bool Insert(Field* pKey, const PageSlotID& iPair, NodePage* parent);

  bool Delete(Field* pKey, const PageSlotID& iPair, NodePage* parent);

  bool Update(Field* pKey, const PageSlotID& iOld, const PageSlotID& iNew);

  Size LowerBound(Field* field);

  Size UpperBound(Field* field);

  std::vector<PageSlotID> Range(Field* pLow, Field* pHigh);

  void Print(int offset) const;

 private:
  void MaintainKey(const NodePage* child, Size rank);

  std::tuple<std::vector<Field*>, std::vector<PageSlotID>> PopHalf();

  Size GetChildRank(PageID child);

  void Load();

  void Store();

 private:
  IndexPage* _indexPage;
};

}  // namespace thdb

#endif