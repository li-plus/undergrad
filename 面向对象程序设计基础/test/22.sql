create database oop;
use oop;
create table oop_info(stu_id int not null, stu_name char, primary key(stu_id));
insert into oop_info(stu_id, stu_name) values (2018011343, "a");
insert into oop_info(stu_id, stu_name) values (2018011344, "b");
insert into oop_info(stu_id, stu_name) values (2018011345, "c");
select * into outfile '2_out_file' from oop_info;
drop database oop;