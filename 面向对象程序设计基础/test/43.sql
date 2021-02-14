create database oop;
use oop;
create table a(a int); 
create table b(a int); 
create table c(a int); 
create table d(a int); 
show tables; 
/********************** Sample Output *************************
Tables_in_oop
a
b
c
d
**************************************************************/
drop table a, c, d; 
show tables; 
/********************** Sample Output *************************
Tables_in_oop
b
**************************************************************/
drop database oop; 