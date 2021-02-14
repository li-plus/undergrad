create database oop;
use oop;
create table t(a int, b int, primary key(a));
insert into t(a,b) values(-2,3);
insert into t(a,b) values(1,-1);
insert into t(a,b) values(3,0);
create table s(a int, b int, primary key(a));
insert into s(a,b) values(0,-2);
insert into s(a,b) values(1,1);
insert into s(a,b) values(-2,0);
select * from t, s order by t.a; 
/********************** Sample Output *************************
t.a     t.b     s.a     s.b     
-2      3       0       -2      
-2      3       1       1       
-2      3       -2      0       
1       -1      0       -2      
1       -1      1       1       
1       -1      -2      0       
3       0       0       -2      
3       0       1       1       
3       0       -2      0     
**************************************************************/
select * from t, s where t.a = s.a order by t.a; 
/********************** Sample Output *************************
t.a     t.b     s.a     s.b     
-2      3       -2      0       
1       -1      1       1   
**************************************************************/
drop database oop; 