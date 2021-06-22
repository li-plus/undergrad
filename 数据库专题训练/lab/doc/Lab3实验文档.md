# Lab3 实验文档

## Lab3 实验说明

*考虑到本次实验需要实现的JOIN算法的多样性，本次实验的开放性较高，具有很高的拓展性和开放性，建议同学们尽早开始任务。*

本次实验有2个任务：
1. 合并Lab3新增的实验框架代码，保证合并后不影响Lab1,2原始测试
2. 实现一种或多种JOIN算法以完成至少支持**2表JOIN**的过程。

**Sort Merge Join的实现难度相对较低，同时可以取得一个不错的性能。如果能结合部分列上存在索引，能进一步优化Sort Merge Join性能**

## Lab3 实验合并

由于JOIN算法的多样性以及不同JOIN算法的不同适用场景，本次实验采用了类似lab2-clean版本的框架模式。本次实验的合并分支为**lab3**。

考虑到框架中没有添加**列选择子**和**别名**的功能，所以实验对于JOIN条件有所限制。
1. 不需要支持表自己JOIN自己的操作

实验合并的方式直接仿照实验debug更新的方法合并对应分支并解决合并冲突即可。需要注意，合并完成后，建议清除原始数据库并重新运行Lab1,2实验测试。应该保证Lab1,2测试仍能正确运行的情况下开始本次实验。

```bash
cd build            
cmake ..
make
./bin/thdb_clear    # 清除原有数据库
```

## Lab3 更新内容和相关模块

1. record模块添加了一些函数，方便Record之间进行JOIN操作
2. system模块中添加了JOIN的顶层接口
3. parser中对于SELECT语句执行过程进行了修改，以支持JOIN操作同时不影响单表查询过程

新增支持的语法：
1. SELECT * FROM A, B WHERE A.AID = B.AID; # A,B 表直接按照AID列进行JOIN

实现思路：
1. 合并分支，保持Lab1,2测试仍能通过
2. 设计合并记录的函数
3. 选择一种Join算法实现Join算法

## Lab3 样例

两个表的JOIN示例
```sh
> SELECT * FROM students;
 +--------+------------+-----------+
 | stu_id | first_name | last_name | 
 +--------+------------+-----------+
 |      1 |      James |     Smith | 
 |      2 |    Michael |   Johnson | 
 |      3 |     Thomas |     Brown | 
 +--------+------------+-----------+

> SELECT * FROM students_courses;
 +--------+-----------+
 | stu_id | course_id | 
 +--------+-----------+
 |      1 |         1 | 
 |      2 |         3 | 
 |      3 |         2 | 
 |      1 |         3 | 
 +--------+-----------+

> SELECT * FROM students, students_courses WHERE students.stu_id = students_courses.stu_id;
 +--------+------------+-----------+--------+-----------+
 | stu_id | first_name | last_name | stu_id | course_id | 
 +--------+------------+-----------+--------+-----------+
 |      1 |      James |     Smith |      1 |         1 | 
 |      1 |      James |     Smith |      1 |         3 | 
 |      2 |    Michael |   Johnson |      2 |         3 | 
 |      3 |     Thomas |     Brown |      3 |         2 | 
 +--------+------------+-----------+--------+-----------+

```

## Lab3 需要实现的接口

必要的顶层接口：
- system
  - instance
    - Join: 实现JOIN算法的顶层接口

## Lab3 报告说明

由于JOIN算法的多样性，本次实验的开放性很高，不便于限制同学们使用哪种方式进行实现。所以本次实验报告可以将重点放在**执行计划生成**和**实现的JOIN算法类型**以及**JOIN算法的实现思路**这些方面。

## Lab3 截至时间

2021年5月9日（第11周周日）晚23:59分前。
届时网络学堂会布置作业，在作业中提交报告和实验3最终版本 CI 测试的 job id。
