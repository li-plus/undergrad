create database oop; 
use oop;
create table aa(a double not null, primary key(a), b int not null, c char); 
create table bb(a double not null, primary key(a), b int not null, c char); 
create table cc(a double not null, primary key(a), b int not null, c char); 
show columns from aa; 
/********************** Sample Output *************************
Field   Type    Null    Key     Default Extra
a       double  NO      PRI     NULL
b       int(11) NO              NULL
c       char(1) YES             NULL
**************************************************************/
show tables;
/********************** Sample Output *************************
Tables_in_oop
aa
bb
cc
**************************************************************/
drop table aa;
create table dd(a double not null, primary key(a), b int not null, c char); 
show tables; 
/********************** Sample Output *************************
Tables_in_oop
bb
cc
dd
**************************************************************/
drop database oop;
