import configparser
from enum import Enum


class Modes(Enum):
    train = 1
    dist_train = 2
    predict = 3
    dist_predict = 4


class RunnerConfig:
    MODES = ['train', 'dist_train', 'predict', 'dist_predict']

    @property
    def mode(self):
        return self.mode

    @mode.setter
    def mode(self, value):
        if value in Modes.__members__:
            self.mode = Modes[value]
        else:
            raise ValueError('mode:[{}] not in allowed modes:[{}]'.format(value, ','.join(Modes.__members__)))

    @property
    def job_name(self):
        return self.job_name

    @job_name.setter
    def job_name(self, value):
        self.job_name = value

    @property
    def task_idx(self):
        return self.task_idx

    @task_idx.setter
    def task_idx(self, value):
        self.task_idx = value

    @property
    def mode(self):
        return self.mode

    @mode.setter
    def mode(self, value):
        if value in RunnerConfig:
            self.mode = value
        else:
            raise ValueError('mode:[{}] not in allowed modes:[{}]'.format(value, ','.join(RunnerConfig.MODES)))

    @property
    def factor_num(self):
        return self.factor_num

    @property
    def vocabulary_size(self):
        return self.vocabulary_size

    @property
    def vocabulary_block_num(self):
        return self.vocabulary_block_num

    @property
    def model_file(self):
        return self.model_file

    @property
    def hash_feature_id(self):
        return self.hash_feature_id

    @property
    def ps_hosts(self):
        return self.ps_hosts

    @property
    def worker_hosts(self):
        return self.worker_hosts

    @property
    def batch_size(self):
        return self.batch_size

    @property
    def init_value_range(self):
        return self.init_value_range

    @property
    def factor_lambda(self):
        return self.factor_lambda

    @property
    def bias_lambda(self):
        return self.bias_lambda

    @property
    def thread_num(self):
        return self.thread_num

    @property
    def epoch_num(self):
        return self.epoch_num

    @property
    def train_files(self):
        return self.train_files

    @property
    def weight_files(self):
        return self.weight_files

    @property
    def validation_files(self):
        return self.validation_files

    @property
    def learning_rate(self):
        return self.learning_rate

    @property
    def adagrad_init_accumulator(self):
        return self.adagrad_init_accumulator

    @property
    def loss_type(self):
        return self.loss_type

    @property
    def predict_files(self):
        return self.predict_files

    @property
    def score_path(self):
        return self.score_path

    @factor_num.setter
    def factor_num(self, value):
        self.factor_num = value

    @vocabulary_size.setter
    def vocabulary_size(self, value):
        self.vocabulary_size = value

    @vocabulary_block_num.setter
    def vocabulary_block_num(self, value):
        self.vocabulary_block_num = value

    @model_file.setter
    def model_file(self, value):
        self.model_file = value

    @hash_feature_id.setter
    def hash_feature_id(self, value):
        self.hash_feature_id = value

    @ps_hosts.setter
    def ps_hosts(self, value):
        self.ps_hosts = value

    @worker_hosts.setter
    def worker_hosts(self, value):
        self.worker_hosts = value

    @batch_size.setter
    def batch_size(self, value):
        self.batch_size = value

    @init_value_range.setter
    def init_value_range(self, value):
        self.init_value_range = value

    @factor_lambda.setter
    def factor_lambda(self, value):
        self.factor_lambda = value

    @bias_lambda.setter
    def bias_lambda(self, value):
        self.bias_lambda = value

    @thread_num.setter
    def thread_num(self, value):
        self.thread_num = value

    @epoch_num.setter
    def epoch_num(self, value):
        self.epoch_num = value

    @train_files.setter
    def train_files(self, value):
        self.train_files = value

    @weight_files.setter
    def weight_files(self, value):
        self.weight_files = value

    @validation_files.setter
    def validation_files(self, value):
        self.validation_files = value

    @learning_rate.setter
    def learning_rate(self, value):
        self.learning_rate = value

    @adagrad_init_accumulator.setter
    def adagrad_init_accumulator(self, value):
        self.adagrad_init_accumulator = value

    @loss_type.setter
    def loss_type(self, value):
        self.loss_type = value

    @predict_files.setter
    def predict_files(self, value):
        self.predict_files = value

    @score_path.setter
    def score_path(self, value):
        self.score_path = value


class RunnerConfigParser:
    GENERAL_SECTION = 'General'
    TRAIN_SECTION = 'Train'
    PREDICT_SECTION = 'Predict'
    CLUSTER_SPEC_SECTION = 'ClusterSpec'
    STR_DELIMITER = ','

    def __init__(self, conf_file_path, mode, job_name='', task_idx=0):
        self.config = configparser.ConfigParser()
        self.config.read(conf_file_path)
        self.mode = mode
        self.job_name = job_name
        self.task_idx = task_idx

    def _read_config(self, section, option, not_null=True):
        if not self.config.has_option(section, option):
            if not_null:
                raise ValueError("%s is undefined." % option)
            else:
                return None
        else:
            value = self.config.get(section, option)
            print('  {0} = {1}'.format(option, value))
            return value

    def _read_strs_config(self, section, option, not_null=True):
        val = self._read_config(section, option, not_null)
        if val != None:
            return [s.strip() for s in val.split(RunnerConfigParser.STR_DELIMITER)]
        return None

    def parse_config(self):
        rc = RunnerConfig()
        rc.mode = self.mode
        rc.job_name = self.job_name
        rc.task_idx = self.task_idx
        rc.factor_num = int(self._read_config(RunnerConfigParser.GENERAL_SECTION, 'factor_num'))
        rc.vocabulary_size = int(self._read_config(RunnerConfigParser.GENERAL_SECTION, 'vocabulary_size'))
        rc.vocabulary_block_num = int(self._read_config(RunnerConfigParser.GENERAL_SECTION, 'vocabulary_block_num'))
        rc.model_file = self._read_config(RunnerConfigParser.GENERAL_SECTION, 'model_file')
        rc.hash_feature_id = self._read_config(RunnerConfigParser.GENERAL_SECTION,
                                               'hash_feature_id').strip().lower() == 'true'
        rc.ps_hosts = self._read_strs_config(RunnerConfigParser.CLUSTER_SPEC_SECTION, 'ps_hosts')
        rc.worker_hosts = self._read_strs_config(RunnerConfigParser.CLUSTER_SPEC_SECTION, 'worker_hosts')
        rc.batch_size = int(self._read_config(RunnerConfigParser.TRAIN_SECTION, 'batch_size'))
        rc.init_value_range = float(self._read_config(RunnerConfigParser.TRAIN_SECTION, 'init_value_range'))
        rc.factor_lambda = float(self._read_config(RunnerConfigParser.TRAIN_SECTION, 'factor_lambda'))
        rc.bias_lambda = float(self._read_config(RunnerConfigParser.TRAIN_SECTION, 'bias_lambda'))
        rc.thread_num = int(self._read_config(RunnerConfigParser.TRAIN_SECTION, 'thread_num'))
        rc.epoch_num = int(self._read_config(RunnerConfigParser.TRAIN_SECTION, 'epoch_num'))
        rc.train_files = self._read_strs_config(RunnerConfigParser.TRAIN_SECTION, 'train_files')
        rc.weight_files = self._read_strs_config(RunnerConfigParser.TRAIN_SECTION, 'weight_files', False)
        if rc.weight_files is not None and len(rc.train_files) != len(rc.weight_files):
            raise ValueError('The numbers of train files and weight files do not match.')
        rc.validation_files = self._read_strs_config(RunnerConfigParser.TRAIN_SECTION, 'validation_files', False)
        rc.learning_rate = float(self._read_config(RunnerConfigParser.TRAIN_SECTION, 'learning_rate'))
        rc.adagrad_init_accumulator = float(
            self._read_config(RunnerConfigParser.TRAIN_SECTION, 'adagrad.initial_accumulator'))
        rc.loss_type = self._read_config(RunnerConfigParser.TRAIN_SECTION, 'loss_type').strip().lower()
        if rc.loss_type is not None and not rc.loss_type in ['logistic', 'mse']:
            raise ValueError('Unsupported loss type: %s' % rc.loss_type)
        rc.predict_files = self._read_config(RunnerConfigParser.PREDICT_SECTION, 'predict_files').split(',')
        rc.score_path = self._read_config(RunnerConfigParser.PREDICT_SECTION, 'score_path')
        return rc
