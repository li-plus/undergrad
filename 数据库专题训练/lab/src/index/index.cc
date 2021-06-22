#include "index/index.h"

#include <cfloat>
#include <memory>

#include "exception/index_exception.h"
#include "macros.h"
#include "minios/os.h"
#include "page/index_page.h"
#include "page/node_page.h"

namespace thdb {

Index::Index(FieldType iType) {
  // TODO: 建立一个新的根结点，注意需要基于类型判断根结点的属性
  // TODO: 根结点需要设为中间结点
  // TODO: 注意记录RootID
  Size keyLen;
  if (iType == FieldType::INT_TYPE) {
    keyLen = sizeof(int);
  } else if (iType == FieldType::FLOAT_TYPE) {
    keyLen = sizeof(double);
  } else {
    // string & none type is not supported
    throw IndexException();
  }
  _indexPage = new IndexPage(NULL_PAGE, iType, keyLen);
  NodePage rootPage(_indexPage, true);
  _indexPage->rootPage = rootPage.GetPageID();
}

Index::Index(PageID nPageID) {
  // TODO: 记录RootID即可
  _indexPage = new IndexPage(nPageID);
}

Index::~Index() {
  // TODO: 如果不添加额外的指针，理论上不用额外写回内容
  delete _indexPage;
}

void Index::Clear() {
  // TODO: 利用RootID获得根结点
  // TODO: 利用根结点的Clear函数清除全部索引占用页面
  {
    NodePage rootPage(_indexPage, _indexPage->rootPage);
    rootPage.ClearChildren();
  }
  MiniOS::GetOS()->DeletePage(_indexPage->rootPage);
}

PageID Index::GetRootID() const { return _indexPage->GetPageID(); }

void Index::Print() const {
  NodePage rootPage(_indexPage, _indexPage->rootPage);
  rootPage.Print(0);
}

bool Index::Insert(Field *pKey, const PageSlotID &iPair) {
  // TODO: 利用RootID获得根结点
  // TODO: 利用根结点的Insert执行插入
  // TODO: 根结点满时，需要进行分裂操作，同时更新RootID
  NodePage rootPage(_indexPage, _indexPage->rootPage);
  return rootPage.Insert(pKey, iPair, nullptr);
}

Size Index::Delete(Field *pKey) {
  // ALERT:
  // 结点合并实现难度较高，没有测例，不要求实现，感兴趣的同学可自行实现并设计测例
  // TODO: 利用RootID获得根结点
  // TODO: 利用根结点的Delete执行删除
  std::shared_ptr<Field> pHigh;
  if (auto intLow = dynamic_cast<IntField *>(pKey)) {
    pHigh = std::make_shared<IntField>(intLow->GetIntData() + 1);
  } else if (auto floatLow = dynamic_cast<FloatField *>(pKey)) {
    pHigh =
        std::make_shared<FloatField>(floatLow->GetFloatData() + FLT_EPSILON);
  } else {
    throw IndexException();
  }
  Size cnt = 0;
  auto rids = Range(pKey, pHigh.get());
  for (auto &rid : rids) {
    if (Delete(pKey, rid)) {
      cnt++;
    }
  }
  return cnt;
}

bool Index::Delete(Field *pKey, const PageSlotID &iPair) {
  // ALERT:
  // 结点合并实现难度较高，没有测例，不要求实现，感兴趣的同学可自行实现并设计测例
  // TODO: 利用RootID获得根结点
  // TODO: 利用根结点的Delete执行删除
  NodePage rootPage(_indexPage, _indexPage->rootPage);
  return rootPage.Delete(pKey, iPair, nullptr);
}

bool Index::Update(Field *pKey, const PageSlotID &iOld,
                   const PageSlotID &iNew) {
  // TODO: 利用RootID获得根结点
  // TODO: 利用根结点的Update执行删除
  NodePage rootPage(_indexPage, _indexPage->rootPage);
  return rootPage.Update(pKey, iOld, iNew);
}

std::vector<PageSlotID> Index::Range(Field *pLow, Field *pHigh) {
  // TODO: 利用RootID获得根结点
  // TODO: 利用根结点的Range执行范围查找
  NodePage rootPage(_indexPage, _indexPage->rootPage);
  return rootPage.Range(pLow, pHigh);
}

}  // namespace thdb
