create database oop;
use oop;
create table t(a int); 
insert into t(a) values(0);
insert into t(a) values(1);
insert into t(a) values(2);
insert into t(a) values(3);
create table r(a int); 
insert into r(a) values(1);
insert into r(a) values(2);
insert into r(a) values(3);
insert into r(a) values(4);
create table s(a int); 
insert into s(a) values(2);
insert into s(a) values(3);
insert into s(a) values(4);
insert into s(a) values(5);
select * from t inner join r order by t.a; 
/********************** Sample Output *************************
t.a     r.a     
0       1       
0       2       
0       3       
0       4       
1       1       
1       2       
1       3       
1       4       
2       1       
2       2       
2       3       
2       4       
3       1       
3       2       
3       3       
3       4      
**************************************************************/
select * from t inner join r on t.a = r.a order by t.a; 
/********************** Sample Output *************************
t.a     r.a     
1       1       
2       2       
3       3   
**************************************************************/
select * from t left join r on t.a = r.a order by t.a; 
/********************** Sample Output *************************
t.a     r.a     
0       NULL    
1       1       
2       2       
3       3     
**************************************************************/
select * from t right join r on t.a = r.a order by r.a; 
/********************** Sample Output *************************
t.a     r.a     
1       1       
2       2       
3       3       
NULL    4  
**************************************************************/
select * from t left join r on t.a = r.a inner join s on r.a = s.a order by t.a; 
/********************** Sample Output *************************
t.a     r.a     s.a     
2       2       2       
3       3       3    
**************************************************************/
select * from t inner join r on t.a = r.a left join s on r.a = s.a order by t.a; 
/********************** Sample Output *************************
t.a     r.a     s.a     
1       1       NULL    
2       2       2       
3       3       3    
**************************************************************/
drop database oop;