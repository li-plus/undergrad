create database oop;
use oop;
create table t(a int, b double not null, c char, primary key(b)); 
insert into t(a,b,c) values(1,1,'a'); 
insert into t(a,b,c) values(0,2,'b'); 
insert into t(b) values(3); 
insert into t(a,b) values(3,4); 
select * from t; 
/********************** Sample Output *************************
a       b       c       
1       1.0000  a       
0       2.0000  b       
NULL    3.0000  NULL    
3       4.0000  NULL   
**************************************************************/
delete from t where c = 'a' or b > 1 and a = 0; 
select * from t; 
/********************** Sample Output *************************
a       b       c       
NULL    3.0000  NULL    
3       4.0000  NULL  
**************************************************************/
update t set a=4 where b=3; 
select * from t;
/********************** Sample Output *************************
a       b       c       
4       3.0000  NULL    
3       4.0000  NULL 
**************************************************************/
drop database oop; 