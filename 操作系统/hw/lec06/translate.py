import os
import argparse
import re


def read_data(filename):
    with open(filename, 'r') as f:
        data = [int(byte, 16) for byte in re.split(r'\s+', f.read().strip())]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ram-data', type=str, default='ram.txt')
    parser.add_argument('--disk-data', type=str, default='disk.txt')
    parser.add_argument('--pdbr', type=int, default=0xd80)
    parser.add_argument('--virtual-addr', type=lambda x: int(x, 0), nargs='+', required=True)
    args = parser.parse_args()

    ram = read_data(args.ram_data)
    disk = read_data(args.disk_data)

    for virtual_addr in args.virtual_addr:
        print('\nVirtual Address {:#04x}:'.format(virtual_addr))

        if virtual_addr > 0x1fff:
            print('  --> Invalid virtual address')
            continue

        pde_index = (virtual_addr >> 10) & 0b11111
        pde_contents = ram[args.pdbr + pde_index]
        pde_valid = (pde_contents >> 7) & 0b1
        pfn = pde_contents & 0b1111111

        print('  --> pde index:{:#02x}  pde contents:(valid {}, pfn {:#02x})'.format(pde_index, pde_valid, pfn))

        if not pde_valid and pfn == 0x7f:
            print('    --> Fault (page directory entry not valid)')
            continue

        pte_index = (virtual_addr >> 5) & 0b11111
        pte_contents = ram[(pfn << 5) + pte_index] if pde_valid else disk[(pfn << 5) + pte_index]
        pte_valid = (pte_contents >> 7) & 0b1
        pfn = pte_contents & 0b1111111

        print('    --> pte index:{:#02x}  pte contents:(valid {}, pfn {:#02x})'.format(pte_index, pte_valid, pfn))

        if not pte_valid and pfn == 0x7f:
            print('      --> Fault (page table entry not valid)')
            continue

        physical_addr = (pfn << 5) + (virtual_addr & 0b11111)
        value = ram[physical_addr] if pte_valid else disk[physical_addr]
        
        print('      --> Translates to Physical Address {:#03x} --> Value: {:#02x}'.format(physical_addr, value))


if __name__ == "__main__":
    main()
