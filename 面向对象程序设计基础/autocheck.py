import os
import re
import argparse

result = []


def check_answer(test_out_file, std_out_file):
    with open(test_out_file, 'r') as f:
        test_out = f.readlines()
    with open(std_out_file, 'r') as f:
        std_out = f.readlines()
    test_out = [re.sub(r'\s+', ' ', x.strip()) for x in test_out]
    std_out = [re.sub(r'\s+', ' ', x.strip()) for x in std_out]
    return "".join(test_out).strip() == "".join(std_out).strip()


def run_test(prog, data_dir):
    test_cases = [x.split('.')[0] for x in os.listdir(
        data_dir) if re.match(r".*\.sql", x)]
    for fname in test_cases:
        os.system(
            r"{prog} < {sql} > test_out.txt".format(prog=prog, sql=os.path.join(data_dir, fname + '.sql')))
        is_correct = check_answer(os.path.join(data_dir, fname + '.out'), "test_out.txt")
        print(r"case {fname}".format(fname=fname), is_correct)
        result.append(is_correct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='The directory containing test cases')
    parser.add_argument('--prog', required=True, help='The path to main program')
    params = parser.parse_args()

    run_test(params.prog, params.dir)
    os.remove("test_out.txt")
    print("accuracy", sum(result) / len(result))
