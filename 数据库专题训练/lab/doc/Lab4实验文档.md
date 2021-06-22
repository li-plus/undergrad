# Lab4 实验文档

## Lab4 实验说明

本次实验对实验框架加入事务支持，并要求同学们实现简单的 MVCC 多版本并发控制机制，有效避免脏读和不可重复读现象，实验前可先复习一下数据库不同隔离级别相关概念，以及脏读、幻读、不可重复读等现象的定义。

实验前先合并 lab4 新增的实验代码，若合并出现冲突需手动解决，合并后确保不影响前 3 次实验测试的正确性。

## 重要备注

本实验测例不考虑**索引**，lab2 实现差异不会对本实验产生影响。

## 框架主要变更

- 新增类：TransactionManager, Transaction
- 接口修改：
  - Instance 类的 InsertRecord, DeleteRecord, UpdateRecord, SearchRecord, GetRecord 接口增加一个传入参数 Transaction *txn，表示进行当前操作的事务，同时为了不影响前 3 次实验，为该参数添加默认值 nullptr。
  - Instance 类的 CreateTable 函数增加传入参数 useTxn，默认值为 false，表示创建 table 时是否需要考虑事务。

## Lab4 需要实现的接口

- manager
  - TransactionManager：事务管理器
    - Begin：事务开始
    - Commit：事务提交
    - Abort：事务终止
- transaction
  - Transaction：事务，自行添加需要的函数和变量

同时，还需要对 Instance 类的 InsertRecord, DeleteRecord, UpdateRecord, SearchRecord, GetRecord 接口进行修改，加入对事务的支持。

## 测试说明

由于目前实验框架没有加入对多线程的支持，因此测试采取单线程模拟多事务的方式进行，具体可参考测试仓库代码及注释。

**至少需通过 Insert 相关测试**，包括 CommitInsert1, CommitInsert2, AbortInsert1, AbortInsert2 四个测试函数。此外，我们提供了 Update 和 Delete 接口的测试函数，默认为 DISABLED 状态，感兴趣的同学可自行测试。

## 参考实现思路

MVCC 有多种实现机制，不同数据库对 MVCC 的实现也不尽相同，下面给出一种可以通过 Insert 测试的参考实现思路，该实现不是最优的，感兴趣的同学可以尝试其他实现方案。

首先我们需要记录插入的每一条 record 属于哪个事务，在 Instance 类的 CreateTable 函数中，若参数 useTxn 为 true，可以在创建表时增加一列，用于记录创建该 record 的事务 id。对于 Insert 函数，若参数 txn 不是空指针，则需要为记录添加一个 Field，表示当前事务 id。而 GetRecord 函数在返回记录时，也需要将事务 id 列删除。

下一步，在通过 Search 函数查找记录时，需要根据当前事务 id， record 中的事务 id，以及当前事务开始时的活跃事务来判断该记录是否可见。要实现该功能，可以在 TransactionManager 中维护一个当前活跃事务列表（即 Begin 之后还没有 Commit 或 Abort 的事务列表），调用 Begin 新建事务时，将该活跃事务列表存入新建的 Transaction 对象中。

最后，为实现 Abort 功能，可以在 Transaction 中维护一个 WriteRecord 队列，记录该事务进行过的修改，事务 Abort 后，通过该队列将事务所作的修改恢复。

## Lab4 报告说明

在本次实验报告中说明你的 MVCC 实现方案，遇到的难点和关键点。

（选做）感兴趣的同学可以调研一下开源数据库（如 MySQL, PostgreSQL 等）的 MVCC 实现机制。

## Lab4 截止时间

2021年5月23日（第13周周日）晚23:59分前。

届时网络学堂会布置作业，在作业中提交报告和实验4最终版本 CI 测试的 job id。
