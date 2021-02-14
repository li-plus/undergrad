create database oop;
use oop;
create table t(a int, b int); 
insert into t(a,b) values(3, 2); 
insert into t(a,b) values(3, 0); 
insert into t(a,b) values(0, 3); 
select a, b, a+b, a-b, a*b, a/b, a%b from t order by b; 
/********************** Sample Output *************************
a       b       (a+b)   (a-b)   (a*b)   (a/b)   (a%b)   
3       0       3       3       0       NULL    NULL    
3       2       5       1       6       1.5000  1       
0       3       3       -3      0       0.0000  0       
**************************************************************/
drop database oop;