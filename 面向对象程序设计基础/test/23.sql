create database oop;
use oop;
create table t(a char, b double, c int not null, primary key(c)); 
insert into t(a, b, c) values('a', 1, 1);
insert into t(a, c) values('b', 2); 
insert into t(b, c) values(3, 3); 
insert into t(a, c) values('d', 4); 
select * from t; 
/********************** Sample Output *************************
a       b       c
a       1.0000  1
b       NULL    2
NULL    3.0000  3
d       NULL    4
**************************************************************/
select count(*), count(a), count(b), count(c) from t; 
/********************** Sample Output *************************
COUNT(*)        COUNT(a)        COUNT(b)        COUNT(c)
4       3       2       4
**************************************************************/
drop database oop;
