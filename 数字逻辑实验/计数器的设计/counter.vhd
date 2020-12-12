-- flip flop 

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.STD_LOGIC_ARITH.ALL;
USE IEEE.STD_LOGIC_UNSIGNED.ALL;

entity ff is 
	port(
		enable: in std_logic:='0'; 
		clk: in std_logic; 
		rst: in std_logic; 
		d: in std_logic:='0'; 
		q: out std_logic:='0'
	);
end ff;

architecture clock of ff is 
begin	
	process(clk, rst)
	begin
		if(enable = '1') then -- if enable 
			if(clk'event and clk = '1') then -- if clock rising 
				q <= d; 
			end if; 
			if(rst = '1') then -- reset 
				q <= '0'; 
			end if; 
		end if; 
	end process; 
end clock; 


-- counter 4 bit

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.STD_LOGIC_ARITH.ALL;
USE IEEE.STD_LOGIC_UNSIGNED.ALL;

entity counter_4bit is 
	port(
		enable: in std_logic; 
		clk: in std_logic; 
		rst: in std_logic; 
		num: out std_logic_vector(3 downto 0) -- output 4-bit number 
	);
end counter_4bit;

architecture count of counter_4bit is 
	component ff
		port(
			enable: in std_logic; 
			clk: in std_logic; 
			rst: in std_logic; 
			d: in std_logic; 
			q: out std_logic
		);
	end component;
	signal sig_q: std_logic_vector(3 downto 0); 
	signal sig_d: std_logic_vector(3 downto 0); 
	
begin 
	ff_0: ff port map(enable=>enable, clk=>clk, rst=>rst, d=>sig_d(0), q=>sig_q(0)); 
	ff_1: ff port map(enable=>enable, clk=>clk, rst=>rst, d=>sig_d(1), q=>sig_q(1)); 
	ff_2: ff port map(enable=>enable, clk=>clk, rst=>rst, d=>sig_d(2), q=>sig_q(2)); 
	ff_3: ff port map(enable=>enable, clk=>clk, rst=>rst, d=>sig_d(3), q=>sig_q(3)); 
	process(clk)
	begin 
		sig_d(0) <= not sig_q(0); 
		sig_d(1) <= sig_q(1) xor sig_q(0); 
		sig_d(2) <= sig_q(2) xor (sig_q(1) and sig_q(0)); 
		sig_d(3) <= sig_q(3) xor (sig_q(2) and sig_q(1) and sig_q(0)); 
		num <= sig_q; 
	end process; 
end count; 





-- display raw

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.STD_LOGIC_ARITH.ALL;
USE IEEE.STD_LOGIC_UNSIGNED.ALL;


entity display_raw is 
	port(
		buf_num: in std_logic_vector(3 downto 0); 
		display_7: out std_logic_vector(6 downto 0)
	);
end display_raw;

architecture display of display_raw is 
begin 
	process(buf_num) 
	begin 
		case buf_num is 
			when "0000"=> display_7 <="1111110"; -- 0
			when "0001"=> display_7 <="0110000"; -- 1
			when "0010"=> display_7 <="1101101"; -- 2
			when "0011"=> display_7 <="1111001"; -- 3
			when "0100"=> display_7 <="0110011"; -- 4
			when "0101"=> display_7 <="1011011"; -- 5
			when "0110"=> display_7 <="1011111"; -- 6
			when "0111"=> display_7 <="1110000"; -- 7
			when "1000"=> display_7 <="1111111"; -- 8
			when "1001"=> display_7 <="1111011"; -- 9
			when "1010"=> display_7 <="1110111"; -- A
			when "1011"=> display_7 <="0011111"; -- B
			when "1100"=> display_7 <="1001110"; -- C
			when "1101"=> display_7 <="0111101"; -- D
			when "1110"=> display_7 <="1001111"; -- E
			when "1111"=> display_7 <="1000111"; -- F			
			when others=> display_7 <="0000000"; -- else 0
		end case;
	end process; 
end display; 




-- counter 

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.STD_LOGIC_ARITH.ALL;
USE IEEE.STD_LOGIC_UNSIGNED.ALL;


entity counter is 
	port(
		clk: in std_logic; 
		rst: in std_logic; 
		pause: in std_logic; 
		display_unit: out std_logic_vector(6 downto 0); 
		display_ten: out std_logic_vector(6 downto 0); 
		
		-- for simulation 
		buf_0: buffer std_logic_vector(3 downto 0); 
		buf_1: buffer std_logic_vector(3 downto 0)
	);
end counter;


architecture auto of counter is 
	component counter_4bit
		port(
			enable: in std_logic; 
			clk: in std_logic; 
			rst: in std_logic; 
			num: out std_logic_vector(3 downto 0)
		);
	end component;
	
	component display_raw
		port(
			buf_num: in std_logic_vector(3 downto 0); 
			display_7: out std_logic_vector(6 downto 0)
		);
	end component; 
	
	signal cnt: integer:=0; 
	signal rst_0: std_logic:='0'; 
	signal rst_1: std_logic:='0';
	signal enable_0: std_logic:='0'; 
	signal enable_1: std_logic:='0'; 
	
begin 
	counter_4bit_0: counter_4bit port map(enable=>enable_0, clk=>clk, rst=>rst_0, num=>buf_0); 
	counter_4bit_1: counter_4bit port map(enable=>enable_1, clk=>clk, rst=>rst_1, num=>buf_1); 
	display_raw_0: display_raw port map(buf_num=>buf_0, display_7=>display_unit); 
	display_raw_1: display_raw port map(buf_num=>buf_1, display_7=>display_ten); 
	
	process(clk)
	begin 
		if(pause = '0') then -- not pause 
			if(clk'event and clk = '1') then 
				if(rst = '1') then -- reset
					rst_0 <= '1'; 
					rst_1 <= '1'; 
				else -- not reset 
					if(cnt < 1000000) then 
						cnt <= cnt + 1; 
						enable_0 <= '0'; 
						enable_1 <= '0'; 
					else -- add one 
						cnt <= 0; 
						enable_0 <= '1'; 
						if(buf_0 = "1001") then -- carry 
							rst_0 <= '1'; 
							enable_1 <= '1'; 
							if(buf_1 = "0101") then -- reset 
								rst_1 <= '1'; 
							else 
								rst_1 <= '0'; 
							end if; 
						else 
							rst_0 <= '0'; 
							enable_1 <= '0'; 
						end if; 
					end if; 
				end if; 
			end if; 
		end if; 
	end process; 
end auto; 



--architecture manual of counter is 
--	component counter_4bit
--		port(
--			enable: in std_logic; 
--			clk: in std_logic; 
--			rst: in std_logic; 
--			num: out std_logic_vector(3 downto 0)
--		);
--	end component;
--	
--	component display_raw
--		port(
--			buf_num: in std_logic_vector(3 downto 0); 
--			display_7: out std_logic_vector(6 downto 0)
--		);
--	end component; 
--	
--	signal rst_0: std_logic:='0'; 
--	signal rst_1: std_logic:='0';
--	signal carry: std_logic; 
--	signal enable_0: std_logic; 
--	signal enable_1: std_logic; 
--	
--begin 
--	counter_4bit_0: counter_4bit port map(enable=>enable_0, clk=>clk, rst=>rst_0, num=>buf_0); 
--	counter_4bit_1: counter_4bit port map(enable=>enable_1, clk=>carry, rst=>rst_1, num=>buf_1); 
--	display_raw_0: display_raw port map(buf_num=>buf_0, display_7=>display_unit); 
--	display_raw_1: display_raw port map(buf_num=>buf_1, display_7=>display_ten); 
--	
--	process(clk)
--	begin 
--		if(pause = '0') then -- not pause 
--			enable_0 <= '1'; 
--			enable_1 <= '1'; 
--			if(clk'event and clk = '1') then 
--				if(rst = '1') then -- reset 
--					rst_0 <= '1'; 
--					rst_1 <= '1'; 
--				else -- not reset 
--					if(buf_0 = "1001") then -- carry 
--						rst_0 <= '1'; 
--						carry <= '1';
--						if(buf_1 = "0101") then -- reset
--							rst_1 <= '1'; 
--						else 
--							rst_1 <= '0'; 
--						end if; 
--					else 
--						carry <= '0'; 
--						rst_0 <= '0'; 
--						rst_1 <= '0'; 
--					end if; 
--				end if; 
--			end if; 
--		else 
--			enable_0 <= '0';
--			enable_1 <= '0';  
--		end if; 
--	end process; 
--end manual; 


