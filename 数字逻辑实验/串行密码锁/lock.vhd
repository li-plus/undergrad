library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_arith.all; 
use ieee.std_logic_unsigned.all; 

entity lock is
	port(
		rst: in std_logic; 
		clk: in std_logic; 
		code: in std_logic_vector(3 downto 0); 
		mode: in std_logic_vector(1 downto 0); 
		unlock: out std_logic; 
		err: out std_logic;
		alarm: out std_logic
	);
end lock;

architecture state of lock is 
	type password is array(3 downto 0) of std_logic_vector(3 downto 0); 
	signal state: integer:=0; 
	signal pwd: password; 
	signal admin_pwd: password:=("1111", "1111", "1111", "1111"); 
	signal cnt: integer:=0; 
begin	
	process(clk, rst) 
	begin 
		if(rst = '1') then 
			state <= 1; 
			if(cnt < 3) then 
				unlock <= '0'; 
				err <= '0'; 
			end if; 
		elsif (rising_edge(clk)) then -- clock 
			if(cnt > 2) then -- cnt is 3. ban all usage except admin password.  
				if(code = admin_pwd(state)) then -- good bit 
					case state is
					when 1 => state <= 2; 
					when 2 => state <= 3; 
					when 3 => state <= 4; 
					when 4 => state <= 0; cnt <= 0; alarm <= '0'; err <= '0'; unlock <= '1'; 
					when others=> NULL; 
					end case; 
				else -- bad bit 
					state <= 0; 
				end if;
			else  
				case mode is 
				when "00" => -- set password
					case state is 
					when 1 => pwd(0) <= code; state <= 2; 
					when 2 => pwd(1) <= code; state <= 3; 
					when 3 => pwd(2) <= code; state <= 4; 
					when 4 => pwd(3) <= code; state <= 0; unlock <= '1'; cnt <= 0; 
					when others => NULL; 
					end case; 
				when "01" => -- verify password
					case state is 
					when 1 => 
						if (code = pwd(0)) then 
							state <= 2; 
						else 
							state <= 0; 
							err <= '1'; 
							if(cnt > 1) then 
								alarm <= '1'; 
							end if; 
							cnt <= cnt + 1; 
						end if; 
					when 2 => 
						if (pwd(1) = code) then 
							state <= 3; 
						else 
							state <= 0; 
							err <= '1'; 
							if(cnt > 1) then 
								alarm <= '1'; 
							end if; 
							cnt <= cnt + 1; 
						end if; 
					when 3 => 
						if (pwd(2) = code)then 
							state <= 4; 
						else 
							state <= 0; 
							err <= '1';
							if(cnt > 1) then 
								alarm <= '1'; 
							end if; 
							cnt <= cnt + 1; 
						end if; 
					when 4 => 
						if(pwd(3) = code) then 
							state <= 0; 
							unlock <= '1';
							cnt <= 0; 
						else 
							state <= 0; 
							err <= '1';
							if(cnt > 1) then 
								alarm <= '1'; 
							end if; 
							cnt <= cnt + 1; 
						end if; 
					when others => NULL; 				
					end case; 
				when "10" => -- verify admin password 
					case state is 
					when 1 => 
						if (code = admin_pwd(0)) then 
							state <= 2; 
						else 
							state <= 0; 
							err <= '1'; 
							if(cnt > 1) then 
								alarm <= '1'; 
							end if; 
							cnt <= cnt + 1; 
						end if; 
					when 2 => 
						if (code = admin_pwd(1)) then 
							state <= 3; 
						else 
							state <= 0; 
							err <= '1'; 
							if(cnt > 1) then 
								alarm <= '1'; 
							end if; 
							cnt <= cnt + 1; 
						end if; 
					when 3 => 
						if (code = admin_pwd(2))then 
							state <= 4; 
						else 
							state <= 0; 
							err <= '1';
							if(cnt > 1) then 
								alarm <= '1'; 
							end if; 
							cnt <= cnt + 1; 
						end if; 
					when 4 => 
						if(code = admin_pwd(3)) then 
							state <= 0; 
							unlock <= '1';
							cnt <= 0; 
						else 
							state <= 0; 
							err <= '1';
							if(cnt > 1) then 
								alarm <= '1'; 
							end if; 
							cnt <= cnt + 1; 
						end if; 
					when others => NULL; 				
					end case;
				when others => NULL;  	
				end case; 	
			end if; 
		end if; 
	end process;
end state; 