#include "node_page.h"

#include <iostream>
#include <memory>

#include "exception/index_exception.h"
#include "field/fields.h"
#include "index_page.h"
#include "macros.h"
#include "minios/os.h"

namespace thdb {

NodePage::NodePage(IndexPage* indexPage_, bool isLeaf_,
                   const std::vector<Field*>& keys_,
                   const std::vector<PageSlotID>& children_) {
  _indexPage = indexPage_;
  size = keys.size();
  isLeaf = isLeaf_;
  assert(keys.size() == children.size());
  for (auto key : keys_) {
    keys.emplace_back(key->Copy());
  }
  children = children_;
}

NodePage::NodePage(IndexPage* indexPage_, PageID nPageID) : Page(nPageID) {
  _indexPage = indexPage_;
  Load();
}

NodePage::~NodePage() {
  Store();
  for (auto key : keys) {
    delete key;
  }
}

void NodePage::ClearChildren() {
  // clear all children's pages
  if (!isLeaf) {
    for (auto& child : children) {
      {
        NodePage childNode(_indexPage, child.first);
        childNode.ClearChildren();
      }
      MiniOS::GetOS()->DeletePage(child.first);
    }
  }
}

bool NodePage::Insert(Field* pKey, const PageSlotID& iPair, NodePage* parent) {
  if (isLeaf) {
    auto pos = UpperBound(pKey);
    keys.insert(keys.begin() + pos, pKey->Copy());
    children.insert(children.begin() + pos, iPair);
  } else {
    // find which child to insert
    auto rank = UpperBound(pKey);
    if (rank >= children.size()) {
      rank = children.size() - 1;
    }
    NodePage child(_indexPage, children[rank].first);
    child.Insert(pKey, iPair, this);
  }
  // maintain last key
  if (!isLeaf) {
    NodePage lastChild(_indexPage, children.back().first);
    if (!Equal(keys.back(), lastChild.keys.back(), _indexPage->keyType)) {
      delete keys.back();
      keys.back() = lastChild.keys.back()->Copy();
    }
  }
  // solve overflow
  std::shared_ptr<NodePage> newRoot;
  if (keys.size() > _indexPage->capacity) {
    if (!parent) {
      // assign new root page
      newRoot = std::make_shared<NodePage>(_indexPage, false);
      parent = newRoot.get();
      parent->keys.emplace_back(keys.back()->Copy());
      parent->children.emplace_back(_nPageID, NULL_SLOT);
      _indexPage->rootPage = parent->_nPageID;
    }
    // create brother node
    std::vector<Field*> halfKeys;
    std::vector<PageSlotID> halfChildren;
    std::tie(halfKeys, halfChildren) = PopHalf();
    NodePage bro(_indexPage, isLeaf, halfKeys, halfChildren);

    auto splitKey = keys.back();
    Size rank = parent->GetChildRank(_nPageID);
    parent->keys.insert(parent->keys.begin() + rank, splitKey->Copy());
    parent->children.insert(parent->children.begin() + rank + 1,
                            std::make_pair(bro._nPageID, NULL_SLOT));
  }
  return true;
}

bool NodePage::Delete(Field* pKey, const PageSlotID& iPair, NodePage* parent) {
  Size rankLo = LowerBound(pKey);
  Size rankHi = UpperBound(pKey);
  bool isDeleted = false;
  if (isLeaf) {
    for (Size i = rankLo; i < rankHi; i++) {
      if (children[i] == iPair) {
        keys.erase(keys.begin() + i);
        children.erase(children.begin() + i);
        isDeleted = true;
        break;
      }
    }
  } else {
    for (Size i = rankLo; i < std::min(rankHi + 1, (Size)children.size());
         i++) {
      NodePage child(_indexPage, children[i].first);
      if (child.Delete(pKey, iPair, this)) {
        isDeleted = true;
        break;
      }
    }
  }
  // maintain parent's key
  Size rank = NULL_SLOT;
  if (parent) {
    rank = parent->GetChildRank(_nPageID);
    parent->MaintainKey(this, rank);
  }
  // solve underflow
  Size underflowThresh = (_indexPage->capacity + 1) / 2;
  if (keys.size() < underflowThresh) {
    if (!parent) {
      // root node: underflow is allowed
      if (!isLeaf && keys.size() <= 1) {
        // If root node is not leaf and it is empty, delete the root
        PageID newRootId = children[0].first;
        _indexPage->rootPage = newRootId;
        // TODO: delete current page
      }
      return isDeleted;
    }
    if (0 < rank) {
      // current node has left brother, load it
      NodePage bro(_indexPage, parent->children[rank - 1].first);
      if (bro.children.size() > underflowThresh) {
        // If left brother is rich, borrow one node from it
        keys.insert(keys.begin(), bro.keys.back());
        children.insert(children.begin(), bro.children.back());
        bro.keys.pop_back();
        bro.children.pop_back();
        // Maintain parent's key as the node's max key
        parent->MaintainKey(&bro, rank - 1);
        // underflow is solved
        return isDeleted;
      }
    }
    if (rank + 1 < parent->children.size()) {
      // current node has right brother, load it
      NodePage bro(_indexPage, parent->children[rank + 1].first);
      if (bro.children.size() > underflowThresh) {
        // If right brother is rich, borrow one node from it
        keys.push_back(bro.keys.front());
        children.push_back(bro.children.front());
        bro.keys.erase(bro.keys.begin());
        bro.children.erase(bro.children.begin());
        // Maintain parent's key as the node's max key
        parent->MaintainKey(this, rank);
        // underflow is solved
        return isDeleted;
      }
    }
    // neither brothers is rich, need to merge
    if (0 < rank) {
      // merge with left brother, transfer all children of current node to left
      // brother
      NodePage bro(_indexPage, parent->children[rank - 1].first);
      bro.keys.insert(bro.keys.end(), keys.begin(), keys.end());
      bro.children.insert(bro.children.end(), children.begin(), children.end());
      keys.clear();
      children.clear();
      parent->keys.erase(parent->keys.begin() + rank);
      parent->children.erase(parent->children.begin() + rank);
      // maintain parent's key
      parent->MaintainKey(&bro, rank - 1);
      // TODO: free current page
    } else {
      assert(rank + 1 < parent->children.size());
      // merge with right brother, transfer all children of right brother to
      // current node
      NodePage bro(_indexPage, parent->children[rank + 1].first);
      // Transfer all right brother's valid rid to current node
      keys.insert(keys.end(), bro.keys.begin(), bro.keys.end());
      children.insert(children.end(), bro.children.begin(), bro.children.end());
      bro.keys.clear();
      bro.children.clear();
      parent->keys.erase(parent->keys.begin() + rank);
      parent->children.erase(parent->children.begin() + rank + 1);
      // maintain parent's key
      parent->MaintainKey(this, rank);
      // TODO: free current page
    }
  }
  return isDeleted;
}

bool NodePage::Update(Field* pKey, const PageSlotID& iOld,
                      const PageSlotID& iNew) {
  Size rankLow = pKey ? LowerBound(pKey) : 0;
  Size rankHigh = pKey ? UpperBound(pKey) : children.size();

  if (isLeaf) {
    for (Size i = rankLow; i < rankHigh; i++) {
      if (Equal(keys[i], pKey, _indexPage->keyType) && children[i] == iOld) {
        children[i] = iNew;
        return true;
      }
    }
  } else {
    for (Size i = rankLow; i < std::min(rankHigh + 1, (Size)children.size());
         i++) {
      NodePage child(_indexPage, children[i].first);
      if (child.Update(pKey, iOld, iNew)) {
        return true;
      }
    }
  }
  return false;
}

Size NodePage::LowerBound(Field* field) {
  if (!field) {
  }
  Size nBegin = 0, nEnd = keys.size();
  while (nBegin < nEnd) {
    Size nMid = (nBegin + nEnd) / 2;
    if (!Less(keys[nMid], field, _indexPage->keyType)) {
      nEnd = nMid;
    } else {
      nBegin = nMid + 1;
    }
  }
  return nBegin;
}

Size NodePage::UpperBound(Field* field) {
  Size nBegin = 0, nEnd = keys.size();
  while (nBegin < nEnd) {
    Size nMid = (nBegin + nEnd) / 2;
    if (Greater(keys[nMid], field, _indexPage->keyType)) {
      nEnd = nMid;
    } else {
      nBegin = nMid + 1;
    }
  }
  return nBegin;
}

std::vector<PageSlotID> NodePage::Range(Field* pLow, Field* pHigh) {
  std::vector<PageSlotID> ans;
  Size rankLow = pLow ? LowerBound(pLow) : 0;
  Size rankHigh = pHigh ? LowerBound(pHigh) : children.size();

  if (isLeaf) {
    ans.assign(children.begin() + rankLow, children.begin() + rankHigh);
  } else {
    for (Size i = rankLow; i < std::min(rankHigh + 1, (Size)children.size());
         i++) {
      NodePage child(_indexPage, children[i].first);
      auto subAns = child.Range(pLow, pHigh);
      ans.insert(ans.end(), subAns.begin(), subAns.end());
    }
  }
  return ans;
}

void NodePage::MaintainKey(const NodePage* child, Size rank) {
  delete keys[rank];
  keys[rank] = child->keys.back()->Copy();
}

std::tuple<std::vector<Field*>, std::vector<PageSlotID>> NodePage::PopHalf() {
  std::vector<Field*> halfKeys;
  std::vector<PageSlotID> halfChildren;
  Size mid = keys.size() / 2;
  halfKeys.insert(halfKeys.end(), keys.begin() + mid, keys.end());
  halfChildren.insert(halfChildren.end(), children.begin() + mid,
                      children.end());
  keys.erase(keys.begin() + mid, keys.end());
  children.erase(children.begin() + mid, children.end());
  return std::make_tuple(halfKeys, halfChildren);
}

Size NodePage::GetChildRank(PageID child) {
  for (Size i = 0; i < children.size(); i++) {
    if (children[i].first == child) {
      return i;
    }
  }
  throw IndexException();
}

void NodePage::Print(int offset) const {
  for (Size i = keys.size() - 1; i != (Size)-1; i--) {
    std::cout << std::string(offset, ' ') << keys[i]->ToString() << '\n';
    if (!isLeaf) {
      NodePage child(_indexPage, children[i].first);
      child.Print(offset + 2);
    }
  }
}

void NodePage::Load() {
  // load headers
  GetHeader((uint8_t*)&size, sizeof(Size), SIZE_OFFSET);
  GetHeader((uint8_t*)&isLeaf, sizeof(bool), LEAF_OFFSET);

  // load entries
  uint8_t keyBuf[8];
  PageOffset offset = 0;
  for (Size i = 0; i < size; i++) {
    GetData(keyBuf, _indexPage->keyLen, offset);
    offset += _indexPage->keyLen;

    Field* key;
    if (_indexPage->keyType == FieldType::INT_TYPE) {
      key = new IntField;
      key->SetData(keyBuf, sizeof(int));
    } else if (_indexPage->keyType == FieldType::FLOAT_TYPE) {
      key = new FloatField;
      key->SetData(keyBuf, sizeof(double));
    } else {
      throw IndexException();
    }
    keys.push_back(key);

    PageSlotID rid;
    GetData((uint8_t*)&rid, sizeof(PageSlotID), offset);
    offset += sizeof(PageSlotID);
    children.push_back(rid);
  }
}

void NodePage::Store() {
  // store headers
  size = keys.size();

  SetHeader((uint8_t*)&size, sizeof(Size), SIZE_OFFSET);
  SetHeader((uint8_t*)&isLeaf, sizeof(bool), LEAF_OFFSET);

  // store entries
  uint8_t buf[8];
  PageOffset offset = 0;
  for (Size i = 0; i < size; i++) {
    auto key = keys[i];
    key->GetData(buf, _indexPage->keyLen);
    SetData(buf, _indexPage->keyLen, offset);
    offset += _indexPage->keyLen;

    auto& child = children[i];
    SetData((uint8_t*)&child, sizeof(PageSlotID), offset);
    offset += sizeof(PageSlotID);
  }
}

}  // namespace thdb