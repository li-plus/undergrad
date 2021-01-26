import sqlite3
import time
import subprocess
import random
import string


def parse_select(out_msg: str):
    sep_cnt = 0
    table = []
    lines = out_msg.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    for line in lines:
        if line.startswith('+------------'):
            sep_cnt = (sep_cnt + 1) % 3
            if sep_cnt == 0:
                yield table
                table = []
        elif sep_cnt == 2:
            row = line.split('|')[1:-1]  # ignore leading & trailing empty strings
            row = [col.strip() for col in row]
            table.append(row)


def rand_insert():
    a = random.randint(0, 0xffff)
    b = random.randint(0, 0xffff) / 100
    str_len = random.randint(1, 10)
    c = ''.join(random.choices(string.ascii_lowercase + string.digits, k=str_len))
    sql = f'insert into tb values ({a}, {b}, \'{c}\');'
    return sql


def rand_eq():
    col = random.choice(['a', 'b', 'c'])
    if col == 'a':
        val = random.randint(0, 0xffff)
    elif col == 'b':
        val = random.randint(0, 0xffff) / 100
    else:
        str_len = random.randint(1, 10)
        val = ''.join(random.choices(string.ascii_lowercase + string.digits, k=str_len))
        val = f'\'{val}\''
    return f'{col} = {val}'


def rand_delete():
    cond = rand_eq()
    sql = f'delete from tb where {cond};'
    return sql


def rand_update():
    cond = rand_eq()
    set_clause = rand_eq()
    sql = f'update tb set {set_clause} where {cond};'
    return sql


def main():
    random.seed(0)

    num_rec = 100000

    sqls = ['create table tb (a int(4), b float, c varchar(16));',
            'create index iii on tb(a);']

    query_sql = [
        'select * from tb;',
        'select * from tb where a > 10000;',
        'select * from tb where a <= 20000;',
        'select * from tb where b >= 100. and a < 30000;',
        'select * from tb where a <> 100 and a <> 200 and b <> 50.00;',
        'select * from tb where c > \'g\';',
        'select * from tb where c < \'h\';',
    ]

    # build table
    for i in range(num_rec):
        sql = rand_insert()
        sqls.append(sql)

    # query
    sqls += query_sql

    # create another index
    sqls.append('create index ccc on tb(b);')

    # random update / delete / insert
    for i in range(1000):
        choice = random.randint(0, 2)
        if choice == 0:
            sql = rand_insert()
        elif choice == 1:
            sql = rand_delete()
        elif choice == 2:
            sql = rand_update()
        else:
            assert False
        sqls.append(sql)

    sqls += query_sql

    # mock answer
    ans = []

    conn = sqlite3.connect(':memory:')
    c = conn.cursor()

    start = time.time()
    for sql in sqls:
        c.execute(sql)
        if sql.startswith('select'):
            ans.append(c.fetchall())
    print(f'SQLite3 spent {time.time() - start:.4f}s')
    conn.close()

    # program output
    sqls = [
        'drop database db;',
        'create database db;',
        'use db;'
    ] + sqls

    exe_path = 'build/src/main'
    start = time.time()
    p = subprocess.Popen([exe_path], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    sql = '\n'.join(sqls).encode('utf-8')
    out_msg, err_msg = p.communicate(sql)
    print(f'My program spent {time.time() - start:.4f}s')

    # check equal
    out_msg = out_msg.decode('utf-8')
    out = list(parse_select(out_msg))
    assert len(out) == len(ans)
    for i, (out_tb, ans_tb) in enumerate(zip(out, ans)):
        assert len(out_tb) == len(ans_tb)
        out_tb_final = []
        ans_tb_final = []

        for out_line, ans_line in zip(out_tb, ans_tb):
            assert len(out_line) == len(ans_line)
            col_types = [type(col) for col in ans_line]
            out_line = [col_type(col) for col, col_type in zip(out_line, col_types)]
            out_tb_final.append(tuple(out_line))
            ans_tb_final.append(ans_line)

        out_tb_final = sorted(out_tb_final)
        ans_tb_final = sorted(ans_tb_final)
        assert out_tb_final == ans_tb_final
        print(f'Test #{i}: PASSED')


if __name__ == '__main__':
    main()
