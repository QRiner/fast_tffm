import sys
import math

def tied_rank(x):
    sorted_x = sorted(zip(x, range(len(x))))
    r = [0 for _ in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank + 1 + i) / 2.0
            last_rank = i
        if i == len(sorted_x) - 1:
            for j in range(last_rank, i + 1):
                r[sorted_x[j][1]] = float(last_rank + i + 2) / 2.0
    return r


def auc(actual, pred):
    r = tied_rank(pred)
    num_positive = len([0 for x in actual if x == 1])
    num_negative = len(actual) - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i] == 1])
    return ((sum_positive - num_positive * (num_positive + 1) / 2.0) /
            (num_negative * num_positive))

def main():
    f_file = sys.argv[1]
    s_file = sys.argv[2]
    with open(f_file, 'r') as ff, open(s_file, 'r') as sf, open('comb', 'w') as combined:
        expected_list = []
        pred_list = []
        for f_line in ff.read().splitlines():
            expected_list.append(float(f_line.split(' ')[0]))
        for f_line in sf.read().splitlines():
            pred_list.append(float(f_line.split(' ')[0]))
        for e, p in zip(expected_list, pred_list):
            try:
                pred = 1/(1+ math.exp(-p))
            except Exception:
                pred = 1/(1+ float('inf'))
            combined.write('{} {}\n'.format(e, pred))
        print('expected {}, pred {}'.format(len(expected_list), len(pred_list)))
        print(auc(expected_list, pred_list))

if __name__ == '__main__':
    main()
