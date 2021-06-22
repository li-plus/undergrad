# Lab2 实验文档

# Lab2 实验说明

本次实验有2个任务：
1. 合并Lab2新增的实验框架代码，保证合并后不影响Lab1原始测试
2. 设计索引的底层存储结构，实现基于索引进行高速查找的算法。索引要求至少支持整型(Integer)和浮点型数据(Double)两种基本的字段类型。只需要支持单字段索引。
补充：对于字符串型数据(String)一般不使用常规索引结构，有兴趣的同学可以阅读相关资料自行探索，本次实验不做要求。

# Lab2 实验合并

针对于不同基础的同学，本次实验有两个实现选择：
1. lab2-clean: 只提供了顶层索引操作接口的分支
2. lab2-simple: 在分支1的基础上，提供了一种比较简单的树结构结点页面设计，实现的算法比较简单，性能一般，但是有比较完整的实现注释。**注意，这个算法实现的不是平衡树，仅保证一般测例情况下性能接近B树/B+树**

能力较强的同学可以尝试直接从lab2-clean的基础上进行本次实验。

实验合并的方式直接仿照本次实验错误更新的方法合并对应分支并解决合并冲突即可。需要注意，合并完成后，建议清除原始数据库并重新运行Lab1实验测试。应该保证Lab1测试仍能正确运行的情况下开始本次实验。

```bash
cd build            
cmake ..
make
./bin/thdb_clear    # 清除原有数据库
```

# Lab2 更新内容和相关模块

1. field模块添加了field指针的复制函数Copy，用于指针的深拷贝。同时添加了Equal, Less, Greater三个函数，用于同类型指针大小比较
2. manager模块添加了index manager，用于管理index的创建、删除等操作，与table manager实现类似
3. system中instance对于查询的执行过程添加了对于index的支持，默认在字段存在索引情况下优先通过索引进行查询，**不要对此模块代码进行修改，否则将视为作弊行为**
4. parser中添加了对于index相关语法的支持，包括创建和删除和显示index的相关语法
5. 添加了index模块，需要同学们实现
6. page模块中需要添加新的页面，lab2-simple中添加了node page用于表示树结点的页面

新增支持的语法：
1. ALTER TABLE *TABLE NAME* ADD INDEX (*COLUMN NAMES*); # 在*TABLE NAME*表的各个*COLUMN NAMES*上依次建立索引
2. ALTER TABLE *TABLE NAME* DROP INDEX (*COLUMN NAMES*); # 在*TABLE NAME*表的各个*COLUMN NAMES*上依次删除索引
3. SHOW INDEXES; # 显示数据库上所有的索引，输出所在表名，列名，类型，长度等信息

实现思路：
1. 合并分支，保持Lab1测试仍能通过
2. 实现用于支持索引的相关数据类型的存储页面，例如中间结点页面、叶结点页面等，自行设计相关测试
3. 完成index最上层接口的实现

# Lab2 需要实现的接口

必要的顶层接口：
- index
  - Index 索引，需要利用内部变量或者继承区分不同类型索引
    - 构建和析构函数
    - Insert 插入一个Key Value对
    - Delete 删除一个Key Value对
    - Delete 删除一个Key值的全部Key Value对
    - Update 将Key Value对中更新为新的Value
    - Range 左闭右开的范围查询
    - Clear 清除索引

lab2-simple分支的额外接口：
- page
  - NodePage 结点页面，利用内部变量区分了不同类型的树结点
    - 构建和析构函数
    - Insert 在当前结点之下，插入一个Key Value对
    - Delete 在当前结点之下，删除一个Key Value对
    - Delete 在当前结点之下，删除一个Key值的全部Key Value对
    - Update 在当前结点之下，将Key Value对中更新为新的Value
    - Range 在当前结点之下，左闭右开的范围查询
    - Clear 在当前结点之下，清除索引
    - Load 从页面加载一个结点的数据
    - Store 将结点数据保存到存储页面当中
    - LowerBound/UpperBound/LessBound 各种边界函数
    - PopHalf 用于结点分裂

# Lab2 报告说明

请在报告中描述你所做的工作，如代码实现思路（重点放在代码注释中没有覆盖的部分），**难点**和**关键点**等。如果你发现了实验框架的任何问题，也欢迎在报告中提出，对于有价值的问题我们会给予一定的加分奖励，并在之后的框架中进行改进。

本次实验中，选择使用lab2-clean分支开始实现的同学可以将代码**实现思路**作为重点；选择lab2-simple分支的同学可以考虑将**实现难点**和**框架问题**和**对于Node Page的模块测试测例**和**攻击给定算法性能的测例**作为重点。

# Lab2 截至时间

2021年4月25日（第9周周日）晚23:59分前。
届时网络学堂会布置作业，在作业中提交报告和实验1最终版本 CI 测试的 job id。
