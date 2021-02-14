create database oop;
use oop;
create table t(a int, b double, primary key(b)); 
insert into t(a, b) values(4, 1); 
insert into t(a, b) values(5, 2); 
insert into t(a, b) values(3, 3); 
insert into t(a, b) values(1, 4); 
insert into t(a, b) values(2, 5); 
select * from t order by a; 
/********************** Sample Output *************************
a       b       
1       4.0000  
2       5.0000  
3       3.0000  
4       1.0000  
5       2.0000  
**************************************************************/
select * from t order by b; 
/********************** Sample Output *************************
a       b       
4       1.0000  
5       2.0000  
3       3.0000  
1       4.0000  
2       5.0000  
**************************************************************/
drop database oop;