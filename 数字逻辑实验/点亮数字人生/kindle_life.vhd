LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.STD_LOGIC_ARITH.ALL;
USE IEEE.STD_LOGIC_UNSIGNED.ALL;

entity kindle_life is
	port(
		display: out std_logic_vector(6 downto 0); -- without decoder 
		display_4_odd: out std_logic_vector(3 downto 0); -- odd num with decoder 
		display_4_even: out std_logic_vector(3 downto 0); -- even num with decoder
		clk: in std_logic; -- clock input 
		rst: in std_logic -- reset input
	);
end kindle_life;

architecture bhv of kindle_life is 
	signal bin_4_natural: std_logic_vector(3 downto 0):="0000";	-- binary natural num
	signal bin_4_odd: std_logic_vector(3 downto 0):="0001";	-- binary odd num
	signal bin_4_even: std_logic_vector(3 downto 0):="0000"; -- binary even num
	signal cnt: integer:=0; -- counter 
begin	
	process(clk) 
	begin 
		display_4_odd <= bin_4_odd;	-- output without decoder 
		display_4_even <= bin_4_even; -- output without decoder

		if(clk'event and clk = '1') then 
			if(cnt < 1000000) then -- 1 MHz
				cnt <= cnt + 1;	-- update counter 
			else 
				cnt <= 0; -- reset counter 
				-- process natural 
				if(bin_4_natural = "1001") then
					bin_4_natural <= "0000";
				else 
					bin_4_natural <= bin_4_natural + 1;
				end if;
				-- process odd 
				if(bin_4_odd = "1001") then 
					bin_4_odd <= "0001";
				else 
					bin_4_odd <= bin_4_odd + 2;
				end if;
				-- process even
				if(bin_4_even = "1000") then 
					bin_4_even <= "0000";
				else 
					bin_4_even <= bin_4_even + 2;
				end if;
			end if;
		end if;
		
		if(rst = '1') then -- reset pressed 
			bin_4_natural <= "0000";
			bin_4_odd <= "0001";
			bin_4_even <= "0000";
		end if;
	end process;
	
	process(bin_4_natural) -- decode natural num 
	begin
		case bin_4_natural is 
			when "0000"=> display<="1111110"; -- 0
			when "0001"=> display<="0110000"; -- 1
			when "0010"=> display<="1101101"; -- 2
			when "0011"=> display<="1111001"; -- 3
			when "0100"=> display<="0110011"; -- 4
			when "0101"=> display<="1011011"; -- 5
			when "0110"=> display<="1011111"; -- 6
			when "0111"=> display<="1110000"; -- 7
			when "1000"=> display<="1111111"; -- 8
			when "1001"=> display<="1111011"; -- 9
			when "1010"=> display<="1110111"; -- A
			when "1011"=> display<="0011111"; -- B
			when "1100"=> display<="1001110"; -- C
			when "1101"=> display<="0111101"; -- D
			when "1110"=> display<="1001111"; -- E
			when "1111"=> display<="1000111"; -- F			
			when others=> display<="0000000"; -- else 0
		end case;
	end process;
end bhv; 

