create database oop;
use oop;
create table t(a text); 
insert into t(a) values("hello world"); 
insert into t(a) values("goodbye world");
insert into t(a) values("hell no");
insert into t(a) values("good morning");
insert into t(a) values("hello+-*()[]{}^.?$|\/"); 
select a from t where a like "hell%"; 
/********************** Sample Output *************************
a       
hell no 
hello world     
hello+-*()[]{}^.?$|\/   
**************************************************************/
select a from t where a like "%wor%";
/********************** Sample Output *************************
a       
goodbye world   
hello world 
**************************************************************/
select a from t where a like "%+-*()[]{}^.?$|\/";
/********************** Sample Output *************************
a       
hello+-*()[]{}^.?$|\/   
**************************************************************/
drop database oop;