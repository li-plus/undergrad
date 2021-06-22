# Lab1 实验文档

## Lab1 实验说明

本次实验主要有两个任务：

1. 阅读代码，对于整体实验框架有一个结构性的理解
2. 设计底层记录页面组织，完成记录管理的各项基本功能

## Lab1 相关模块说明

各个模块的说明在**实验框架.md**中已经有比较详细的说明。
本次实验中，主要需要关注于**record**，**page**，**table**这3个模块。

1. 从record模块入手，先完成记录的序列化和反序列化
2. 对于page模块，设计记录页面的整体组织形式
3. 在table模块中，结合record模块和page模块，完成各项有关记录操作的接口

## Lab1 需要实现的接口

- record
  - FixedRecord: 定长记录
    - Load 记录反序列化
    - Store 记录序列化
  - VariableRecord: 用于变长记录的序列化，需要自己设计
- page
  - LinkedPage: 链表页面
    - PushBack 插入页面
    - PopBack 移除页面
  - RecordPage: 定长记录存储页面
    - GetRecord 页面内获取记录
    - InsertRecord 页面内插入记录
    - DeleteRecord 页面内删除记录
    - UpdateRecord 页面内更新记录
  - ToastPage: 用于变长记录存储的页面，需要自己设计
- table
  - Table
    - GetRecord 表内获取记录
    - InsertRecord 表内插入记录
    - DeleteRecord 表内删除记录
    - UpdateRecord 表内更新记录
    - SearchRecord 记录条件检索
    - NextNotFull 更新非满页面的页面编号，必要时插入新的页面

所有需要实现的部分在上述各个函数的 **// Lab1 Begin** 到 **// Lab 1 End** 之间，建议按照模块说明中的上面所列的文件顺序依次实现。其中各个模块中具有 **// TIPS:** 标记来提示大家一些实现思路，以及需要重点关注的 **// ALERT:** 标记表示非常重要的提示。

## Lab1 报告说明

请在报告中描述你所做的工作，如代码实现思路，难点和关键点等。如果你发现了实验框架的任何问题，也欢迎在报告中提出，对于有价值的问题我们会给予一定的加分奖励，并在之后的框架中进行改进。

## Lab1 截止时间

2021年3月28日（第五周周日）晚23:59分前。
届时网络学堂会布置作业，在作业中提交报告和实验1最终版本 CI 测试的 job id。

job id 的获取方法：在仓库主页的 clone 按钮左下方会有一个绿色的√或红色的×，点击它会进入 pipeline 页面，再点击下方的 test_oj 即可进入 job 页面，此时页面上方会显示 job #\<id\>。或者直接访问 https://git.tsinghua.edu.cn/dbtrain-2021/dbtrain-lab-(number)/-/jobs/ 查看自己的所有 job，其中 (number) 改为自己的学号。
