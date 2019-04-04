import sys
from fm.config import *
from fm.runner import train, predict

cmd_instruction = '''Usage:
  1. Local training.
    python cmdline.py trai <cfg_file>
  2. Distributed training.
    python cmdline.py dist_train <cfg_file> <job_name> <task_idx>
  3. Local predicting.
    python cmdline.py predict <cfg_file>
  4. Distributed predicting.
    python cmdline.py dist_predict <cfg_file> <job_name> <task_idx>
Arguments:
  <cfg_file>: configuartion file path. See sample.cfg for example.
  <job_name>: 'worker' or 'ps'. Launch as a worker or a parameter server
  <task_idx>: Task index.
'''


def check_argument_error(condition):
    if not condition:
        sys.stderr.write('''Invalid arguments\n''')
        sys.stderr.write(cmd_instruction)
        sys.exit(1)


def main(args):
    argc = len(args)
    if argc == 1:
        print(cmd_instruction, )
        sys.exit(1)
    check_argument_error(argc >= 3)
    mode = args[1]
    cfg_file = args[2]
    job_name = ''
    task_idx = 0
    if mode == 'train' or mode == 'predict':
        check_argument_error(argc == 3)
    elif mode == 'dist_train' or mode == 'dist_predict':
        check_argument_error(argc == 5)
        job_name = args[3]
        task_idx = int(args[4])
    else:
        check_argument_error(False)
    conf_parser = RunnerConfigParser(cfg_file, mode, job_name, task_idx)
    config = conf_parser.parse_config()
    if mode == 'train' or mode == 'dist_train':
        train(config)
    elif mode == 'predict' or mode == 'dist_predict':
        predict(config)


if __name__ == '__main__':
    main(sys.argv)
