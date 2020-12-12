
-- full_adder_1.vhd --

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.STD_LOGIC_ARITH.ALL;
USE IEEE.STD_LOGIC_UNSIGNED.ALL;

entity full_adder_1 is -- 1 bit full adder
	port(
		a,b,cin:in std_logic; -- a,b: 1-bit number. cin: carry input 
		s,cout:out std_logic; -- s: sum result. cout: carry output 
		p,g:buffer std_logic  -- p: carry propagate function. g: carry generate function. 
	);
end full_adder_1;

architecture add of full_adder_1 is 
begin	
	process(a,b)
	begin
		p <= a xor b; -- compute carry propagate function.
		g <= a and b; -- compute carry generate function. 
	end process;
	process(cin,p,g)
	begin
		s <= p xor cin; -- compute sum result
		cout <= (p and cin) or g; -- compute carry output 
	end process; 
end add; 

-- full_adder_4.vhd --

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.STD_LOGIC_ARITH.ALL;
USE IEEE.STD_LOGIC_UNSIGNED.ALL;


entity full_adder_4 is -- 4-bit full adder
	port(
		a,b:in std_logic_vector(3 downto 0); -- a,b: 4-bit number. 
		cin:in std_logic; -- cin: carry input. 
		s:out std_logic_vector(3 downto 0); -- s: sum results. 
		cout:out std_logic -- cout: carry output.
	);
end full_adder_4;


--architecture successive of full_adder_4 is
--	component full_adder_1
--		port(
--			a,b,cin:in std_logic;
--			s,cout:out std_logic;
--			p,g:buffer std_logic
--		);
--	end component;
--	signal c:std_logic_vector(3 downto 0);
--	
--begin
--	fa1_0:full_adder_1 port map(a(0),b(0),cin,s(0),c(0));
--	fa1_1:full_adder_1 port map(a(1),b(1),c(0),s(1),c(1));
--	fa1_2:full_adder_1 port map(a(2),b(2),c(1),s(2),c(2));
--	fa1_3:full_adder_1 port map(a(3),b(3),c(2),s(3),cout);	
--end successive;


architecture lookahead of full_adder_4 is 
	component full_adder_1
		port(
			a,b,cin:in std_logic;
			s,cout:out std_logic;
			p,g:buffer std_logic
		);
	end component;
	signal p,g,c: std_logic_vector(3 downto 0); -- p: prop func buffer. g: gen func buffer. c: carry buffer. 
	
begin 
	fa1_0:full_adder_1 port map(a(0),b(0),cin,s(0),p=>p(0),g=>g(0)); 
	fa1_1:full_adder_1 port map(a(1),b(1),c(0),s(1),p=>p(1),g=>g(1)); 
	fa1_2:full_adder_1 port map(a(2),b(2),c(1),s(2),p=>p(2),g=>g(2));
	fa1_3:full_adder_1 port map(a(3),b(3),c(2),s(3),p=>p(3),g=>g(3)); 
	process(p,g,c)
	begin -- compute look-ahead carry 
		c(0) <= g(0) or (p(0) and cin);
		c(1) <= g(1) or (p(1) and g(0)) or (p(1) and p(0) and cin);
		c(2) <= g(2) or (p(2) and g(1)) or (p(2) and p(1) and g(0)) or (p(2) and p(1) and p(0) and cin);
		cout <= g(3) or (p(3) and g(2)) or (p(3) and p(2) and g(1)) or (p(3) and p(2) and p(1) and g(0)) or (p(3) and p(2) and p(1) and p(0) and cin);
	end process; 
end lookahead; 


--architecture system of full_adder_4 is
--	signal result_full: std_logic_vector(4 downto 0); 
--begin
--	process (a,b,cin)
--	begin 
--		result_full <= "00000" + a + b + cin; -- using operator '+' defined in system
--		s <= result_full(3 downto 0); -- get sum result
--		cout <= result_full(4); -- get the carry output
--	end process; 
--end system; 

