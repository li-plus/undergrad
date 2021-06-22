#ifndef THDB_PAGE_H_
#define THDB_PAGE_H_

#include "defines.h"

namespace thdb {
/**
 * @brief 最为基本的格式化页面对象，实现了基本的页面内容读写操作。
 * 对于页面划分为头和数据两部分，页面合计占用4096字节，其中头部分占用64字节，在macros.h中进行了定义。
 */
class Page {
 public:
  Page();
  Page(PageID nPageID);
  virtual ~Page();

  PageID GetPageID() const;
  void SetPageID(PageID nPageID);

  /**
   * @brief 读出无格式页面头部分的内容。
   *
   * @param dst 读出内容存放地址
   * @param nSize 读出内容长度
   * @param nOffset 读出内容在头部分起始位置
   */
  void GetHeader(uint8_t *dst, PageOffset nSize, PageOffset nOffset) const;
  /**
   * @brief 写入无格式页面头部分的内容。
   *
   * @param src 写入内容存放地址
   * @param nSize 写入内容长度
   * @param nOffset 读出内容在头部分起始位置
   */
  void SetHeader(const uint8_t *src, PageOffset nSize, PageOffset nOffset);

  /**
   * @brief 读出无格式页面数据部分的内容。
   *
   * @param dst 读出内容存放地址
   * @param nSize 读出内容长度
   * @param nOffset 读出内容在数据部分起始位置
   */
  void GetData(uint8_t *dst, PageOffset nSize, PageOffset nOffset) const;
  /**
   * @brief 写入无格式页面数据部分的内容。
   *
   * @param src 写入内容存放地址
   * @param nSize 写入内容长度
   * @param nOffset 读出内容在数据部分起始位置
   */
  void SetData(const uint8_t *src, PageOffset nSize, PageOffset nOffset);

 protected:
  PageID _nPageID;

 private:
  bool _bModified;
};

}  // namespace thdb

#endif