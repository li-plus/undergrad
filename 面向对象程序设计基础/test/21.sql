create database oop;
use oop;
create table oop_info(stu_id int not null, stu_name char, primary key(stu_id)); 
load data infile '1_in_file' into table oop_info(stu_id, stu_name);
select * from oop_info; 
/********************** Sample Output *************************
stu_id  stu_name        
2018011343      a       
2018011344      b       
2018011345      c    
**************************************************************/
drop database oop;