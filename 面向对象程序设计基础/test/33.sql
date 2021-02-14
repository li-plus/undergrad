create database oop;
use oop;
create table t(a double, b int); 
insert into t(a, b) values(-1,  -1); 
insert into t(a, b) values(-0.5, 0); 
insert into t(a, b) values(0, 	 0); 
insert into t(a, b) values(0.5,	 1); 
insert into t(a, b) values(1, 	 1); 
insert into t(a, b) values(1.5,	 2); 
insert into t(a, b) values(2,	 2); 
select a, abs(a), exp(a), ln(a), log10(a), ceil(a), floor(a) from t;
/********************** Sample Output *************************
a       ABS(a)  EXP(a)  LN(a)   LOG10(a)        CEIL(a) FLOOR(a) 
-1.0000 1.0000  0.3679  NULL    NULL    -1      -1      
-0.5000 0.5000  0.6065  NULL    NULL    0       -1      
0.0000  0.0000  1.0000  NULL    NULL    0       0       
0.5000  0.5000  1.6487  -0.6931 -0.3010 1       0       
1.0000  1.0000  2.7183  0.0000  0.0000  1       1       
1.5000  1.5000  4.4817  0.4055  0.1761  2       1       
2.0000  2.0000  7.3891  0.6931  0.3010  2       2      
**************************************************************/
select a, sin(a), cos(a), tan(a), asin(a), acos(a), atan(a) from t;
/********************** Sample Output *************************
a       SIN(a)  COS(a)  TAN(a)  ASIN(a) ACOS(a) ATAN(a) 
-1.0000 -0.8415 0.5403  -1.5574 -1.5708 3.1416  -0.7854 
-0.5000 -0.4794 0.8776  -0.5463 -0.5236 2.0944  -0.4636 
0.0000  0.0000  1.0000  0.0000  0.0000  1.5708  0.0000  
0.5000  0.4794  0.8776  0.5463  0.5236  1.0472  0.4636  
1.0000  0.8415  0.5403  1.5574  1.5708  0.0000  0.7854  
1.5000  0.9975  0.0707  14.1014 NULL    NULL    0.9828  
2.0000  0.9093  -0.4161 -2.1850 NULL    NULL    1.1071  
**************************************************************/
select a, b, a-b/(a+sin(pi()/2))*cos(pi()) from t where a = b or (sin(pi()/2) = cos(0) xor log10(100) = abs(-2)) and not (exp(0) = tan(pi()/4) and ceil(1.5) != floor(2.4)) order by a;
/********************** Sample Output *************************
a       b       (a-((b/(a+SIN((3.1416/2))))*COS(3.1416)))       
-1.0000 -1      NULL    
0.0000  0       0.0000  
1.0000  1       1.5000  
2.0000  2       2.6667  
**************************************************************/
drop database oop;