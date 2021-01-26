-- system statement
show databases;

-- database statement
create database abc;
drop database a_0_bc_12;
use abc;
show tables;

-- table stmt
create table a (a int(4));
create table aaa (a int(4), b varchar(256), c float, primary key(a, b));
drop table abc;
desc abc;
insert into ab values (1, 2);
insert into ab values (1, 1), (2,2);
delete from ab where a = 1 and b <= 1 and c is null;
update ab set a = 1, b = 2 where a < 1;
select * from ab where a > 1 and b is not null;
select a.b, b.a, c from ab, cd where a.b < 1 and c is null;

-- idx stmt
create index idx ON tb (a);
drop index idx;
alter table tb add index idx (a,b);
alter table tb drop index idx;

-- alter stmt
alter table tb add primary key (a);
alter table tb add b int(4);
alter table tb drop a;
alter table tb change a primary key (a);
alter table tb rename to new_tb;
alter table tb drop primary key;
alter table tb add constraint pk primary key (a, b);
alter table tb drop primary key pk;
alter table tb add constraint fk foreign key (a, b) references re(x, y);
alter table tb drop foreign key fk;
