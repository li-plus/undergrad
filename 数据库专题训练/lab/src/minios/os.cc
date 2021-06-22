#include "minios/os.h"

#include <cstring>
#include <fstream>

#include "exception/exceptions.h"
#include "macros.h"
#include "minios/raw_page.h"
#include "settings.h"

namespace thdb {

MiniOS *MiniOS::os = nullptr;

MiniOS *MiniOS::GetOS() {
  if (os == nullptr) os = new MiniOS();
  return os;
}

void MiniOS::WriteBack() {
  if (os != nullptr) {
    delete os;
    os = nullptr;
  }
}

MiniOS::MiniOS() {
  _pMemory = new RawPage *[MEM_PAGES];
  _pUsed = new Bitmap(MEM_PAGES);
  memset(_pMemory, 0, MEM_PAGES);
  _nClock = 0;
  LoadBitmap();
  LoadPages();
}

MiniOS::~MiniOS() {
  StoreBitmap();
  StorePages();
  for (size_t i = 0; i < MEM_PAGES; ++i)
    if (_pMemory[i]) delete _pMemory[i];
  delete[] _pMemory;
  delete _pUsed;
}

PageID MiniOS::NewPage() {
  Size tmp = _nClock;
  do {
    if (!_pUsed->Get(_nClock)) {
      _pMemory[_nClock] = new RawPage();
      _pUsed->Set(_nClock);
      return _nClock;
    } else {
      _nClock += 1;
      _nClock %= MEM_PAGES;
    }
  } while (_nClock != tmp);
  throw NewPageException();
}

void MiniOS::DeletePage(PageID pid) {
  if (!_pUsed->Get(pid)) throw PageNotInitException(pid);
  delete _pMemory[pid];
  _pMemory[pid] = 0;
  _pUsed->Unset(pid);
}

void MiniOS::ReadPage(PageID pid, uint8_t *dst, PageOffset nSize,
                      PageOffset nOffset) {
  if (!_pUsed->Get(pid)) throw PageNotInitException(pid);
  _pMemory[pid]->Read(dst, nSize, nOffset);
}

void MiniOS::WritePage(PageID pid, const uint8_t *src, PageOffset nSize,
                       PageOffset nOffset) {
  if (!_pUsed->Get(pid)) throw PageNotInitException(pid);
  _pMemory[pid]->Write(src, nSize, nOffset);
}

void MiniOS::LoadBitmap() {
  std::ifstream fin("THDB_BITMAP", std::ios::binary);
  if (!fin) return;
  uint8_t pTemp[MEM_PAGES / 8];
  fin.read((char *)pTemp, MEM_PAGES / 8);
  fin.close();
  _pUsed->Load(pTemp);
}

void MiniOS::LoadPages() {
  std::ifstream fin("THDB_PAGE", std::ios::binary);
  if (!fin) return;
  uint8_t pTemp[PAGE_SIZE];
  for (uint32_t i = 0; i < MEM_PAGES; ++i) {
    if (_pUsed->Get(i)) {
      fin.read((char *)pTemp, PAGE_SIZE);
      _pMemory[i] = new RawPage();
      _pMemory[i]->Write(pTemp, PAGE_SIZE);
    }
  }
  fin.close();
}

void MiniOS::StoreBitmap() {
  std::ofstream fout("THDB_BITMAP", std::ios::binary);
  if (!fout) return;
  uint8_t pTemp[MEM_PAGES / 8];
  _pUsed->Store(pTemp);
  fout.write((char *)pTemp, MEM_PAGES / 8);
  fout.close();
}

void MiniOS::StorePages() {
  std::ofstream fout("THDB_PAGE", std::ios::binary);
  if (!fout) return;
  uint8_t pTemp[PAGE_SIZE];
  for (uint32_t i = 0; i < MEM_PAGES; ++i) {
    if (_pUsed->Get(i)) {
      _pMemory[i]->Read(pTemp, PAGE_SIZE);
      fout.write((char *)pTemp, PAGE_SIZE);
    }
  }
  fout.close();
}

Size MiniOS::GetUsedSize() const {
  if (!_pUsed) throw OsException();
  return _pUsed->GetSize();
}

}  // namespace thdb
