# 数据库系统概论：作业二

使用 Ubuntu 20.04 系统，首先安装 MySQL Server 并启动服务，

```sh
$ sudo apt install mysql-server
$ sudo systemctl start mysql
```

查看版本信息 `mysqladmin --version`

```
mysqladmin  Ver 8.0.22-0ubuntu0.20.04.2 for Linux on x86_64 ((Ubuntu))
```

启动 MySQL Client

```sh
$ sudo mysql
```

**1、请用一条完整的SQL 语句表达以下查询，至少使用三种不同的方法：求关系S(a,b)中不同元组的个数。**

首先创建表 `S(a,b)` 并插入重复数据。

```sql
create database hw2;
use hw2;

create table S (a int, b int);

insert into S (a, b) values (1, 1);
insert into S (a, b) values (1, 2);
insert into S (a, b) values (2, 1);
insert into S (a, b) values (2, 2);

insert into S (a, b) values (1, 1);
insert into S (a, b) values (1, 2);
insert into S (a, b) values (2, 1);
insert into S (a, b) values (2, 2);

insert into S (a, b) values (1, 1);
insert into S (a, b) values (1, 2);
insert into S (a, b) values (2, 1);
insert into S (a, b) values (2, 2);
```

列出表中的所有数据：

```sql
select * from S;
```

```
+------+------+
| a    | b    |
+------+------+
|    1 |    1 |
|    1 |    2 |
|    2 |    1 |
|    2 |    2 |
|    1 |    1 |
|    1 |    2 |
|    2 |    1 |
|    2 |    2 |
|    1 |    1 |
|    1 |    2 |
|    2 |    1 |
|    2 |    2 |
+------+------+
```

接下来用三种方法查询关系 `S(a,b)` 中不同元组的个数。

方法一：使用 count distinct 语句。

```sql
select count(distinct a, b) from S;
```

方法二：嵌套查询，先找到 distinct 记录，再计算 count。

```sql
select count(*) from (select distinct a, b from S) as S;
```

方法三：嵌套查询，使用 group by 语句。

```sql
select count(*) from (select * from S group by a, b) as S;
```

三种方法都得出了相同的结果：关系 `S(a,b)` 中不同元组的个数为 4。

```
+----------+
| count(*) |
+----------+
|        4 |
+----------+
```

**2、我们知道，在SELECT-FROM-WHERE表达式中，FROM子句可以嵌套另外一个SELECT-FROM-WHERE表达式（子查询），请在你的机器上做实验，测试这样的嵌套可以有多少层（直到系统崩溃）？给出完整的测试用力和测试过程。**

在第 1 题的表 `S(a,b)` 上做进一步实验，探索 select-from-where 表达式的最大嵌套层数。经过探索，我的机器上最多可以有 64 个 select 嵌套语句，如下所示。

```sql
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from ( select * from (
select * from ( select * from ( select * from (
    select * from S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S ) as S
) as S ) as S ) as S;
```

查询结果为：

```
+------+------+
| a    | b    |
+------+------+
|    1 |    1 |
|    1 |    2 |
|    2 |    1 |
|    2 |    2 |
|    1 |    1 |
|    1 |    2 |
|    2 |    1 |
|    2 |    2 |
|    1 |    1 |
|    1 |    2 |
|    2 |    1 |
|    2 |    2 |
+------+------+
```

如果再嵌套多一层，系统将报如下错误：

```
ERROR 1473 (HY000): Too high level of nesting for select
```
