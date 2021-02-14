create database oop; 
use oop;
create table t(a int not null, b char);
insert into t(a, b) values (1, "x");
insert into t(a, b) values (1, "x");
insert into t(a, b) values (2, "x");
insert into t(a, b) values (3, "y");
insert into t(a, b) values (3, "z");
insert into t(a, b) values (3, "z");
select b, count(*) from t group by b order by count(*);
/********************** Sample Output *************************
b       COUNT(*)
y       1
z       2
x       3
**************************************************************/
select a, b, count(*) from t group by a, b order by count(*); 
/********************** Sample Output *************************
a       b       COUNT(*)
2       x       1
3       y       1
1       x       2
3       z       2
**************************************************************/
drop database oop;