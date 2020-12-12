LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.STD_LOGIC_ARITH.ALL;
USE IEEE.STD_LOGIC_UNSIGNED.ALL;

entity freq is 
	port(
		clk: in std_logic; 
		clk_out: out std_logic 
	);
end freq;

architecture add of freq is
	signal cnt: integer := 0; 
begin	
	process(clk)
	begin
		if(clk'event and clk='1') then 
			if (cnt < 50000) then 
				cnt <= cnt + 1; 
				clk_out <= '0'; 
			elsif(cnt < 100000) then 
				cnt <= cnt + 1; 
				clk_out <= '1'; 
			else 
				cnt <= 0; 
				clk_out <= '0'; 
			end if; 
		end if; 
	end process; 
end add; 
