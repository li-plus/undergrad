create database oop;
use oop;
create table t(a int, b double not null, c char, primary key(b)); 
insert into t(a,b,c) values(1,1,'a'); 
insert into t(b) values(3); 
select * from t; 
/********************** Sample Output *************************
a       b       c       
1       1.0000  a       
NULL    3.0000  NULL   
**************************************************************/
update t set a=4, c='c', b=2 where b=3; 
select * from t;
/********************** Sample Output *************************
a       b       c       
1       1.0000  a       
4       2.0000  c  
**************************************************************/
drop database oop; 