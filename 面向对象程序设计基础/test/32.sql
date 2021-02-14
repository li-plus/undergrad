create database oop;
use oop;
create table t(a int, b double); 
insert into t(a,b) values(0, 1); 
insert into t(a,b) values(1, 0); 
create table s(a int, b double); 
insert into s(a,b) values(1, 0); 
insert into s(a,b) values(2, 3); 
create table r(a int, b double); 
insert into r(a,b) values(2, 3); 
insert into r(a,b) values(3, 5); 
select a, b from t union all select a, b from s order by b; 
/********************** Sample Output *************************
a       b       
1       0.0000  
1       0.0000  
0       1.0000  
2       3.0000  
**************************************************************/
select a, b from t union select a, b from s order by b; 
/********************** Sample Output *************************
a       b       
1       0.0000  
0       1.0000  
2       3.0000 
**************************************************************/
select a from t union all select a from s union select a from r order by a; 
/********************** Sample Output ************************
a       
0       
1       
2       
3   
**************************************************************/
select a from t union select a from s union all select a from r order by a; 
/********************** Sample Output *************************
a       
0       
1       
2       
2       
3     
**************************************************************/
drop database oop;