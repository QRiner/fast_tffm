import threading, time, random, os
from fast_tffm.fm import RunnerConfig
from fast_tffm.fm import LocalFmModel, DistFmModel
import tensorflow as tf

PREDICT_BATCH_SIZE = 10000


class _TrainStats:
    pass


def _train(sess, supervisor, worker_num, is_master_worker, need_to_init, model, train_files, weight_files,
           validation_files, epoch_num, thread_num, model_file, output_progress_every_n_examples=10000):
    with sess as sess:
        if is_master_worker:
            if weight_files != None:
                train_and_weight_files = [item for item in zip(train_files, weight_files)]
            else:
                train_and_weight_files = [item for item in zip(train_files, ["" for i in range(len(train_files))])]
            if need_to_init:
                sess.run(model.init_vars)
            for i in range(epoch_num):
                random.shuffle(train_and_weight_files)
                for data_file, weight_file in train_and_weight_files:
                    sess.run(model.file_enqueue_op,
                             feed_dict={model.epoch_id: i, model.is_training: True, model.data_file: data_file,
                                        model.weight_file: weight_file})
                if validation_files != None:
                    for validation_file in validation_files:
                        sess.run(model.file_enqueue_op, feed_dict={model.epoch_id: i, model.is_training: False,
                                                                   model.data_file: validation_file,
                                                                   model.weight_file: ""})
            sess.run(model.file_close_queue_op)
        try:
            fid = 0
            while True:
                epoch_id, is_training, data_file, weight_file = sess.run(model.file_dequeue_op)
                train_stats = _TrainStats()
                train_stats.processed_example_num = 0
                train_stats.lock = threading.Lock()
                start_time = time.time()
                print('[Epoch %d] Task: %s; Data File: %s' % (
                    epoch_id, 'Training' if is_training else 'Validation', data_file),
                      '; Weight File: %s .' % weight_file if weight_file != '' else '.')

                def run():
                    try:
                        while not coord.should_stop() and not (supervisor != None and supervisor.should_stop()):
                            if is_training:
                                _, loss, example_num = sess.run([model.opt, model.loss, model.example_num],
                                                                feed_dict={model.file_id: fid,
                                                                           model.data_file: data_file,
                                                                           model.weight_file: weight_file})
                                global_loss, global_example_num = model.training_stat[epoch_id].update(sess, loss,
                                                                                                       example_num)
                            else:
                                loss, example_num = sess.run([model.loss, model.example_num],
                                                             feed_dict={model.file_id: fid, model.data_file: data_file,
                                                                        model.weight_file: weight_file})
                                global_loss, global_example_num = model.validation_stat[epoch_id].update(sess, loss,
                                                                                                         example_num)
                            if example_num == 0:
                                break
                            train_stats.lock.acquire()
                            train_stats.processed_example_num += example_num
                            if train_stats.processed_example_num % output_progress_every_n_examples < example_num:
                                t = time.time() - start_time
                                print('-- Ex num: %d; Avg loss: %.5f; Time: %.4f; Speed: %.1f ex/sec.' % (
                                    global_example_num, global_loss / global_example_num, t,
                                    train_stats.processed_example_num / t))
                            train_stats.lock.release()
                    except Exception as ex:
                        coord.request_stop(ex)
                        if supervisor != None:
                            supervisor.request_stop(ex)
                        raise

                coord = tf.train.Coordinator()
                threads = [threading.Thread(target=run) for i in range(thread_num)]
                for th in threads: th.start()
                coord.join(threads, stop_grace_period_secs=5)
                if is_training:
                    global_loss, global_example_num = model.training_stat[epoch_id].eval(sess)
                else:
                    global_loss, global_example_num = model.validation_stat[epoch_id].eval(sess)
                print('Finish Processing. Ex num: %d; Avg loss: %.5f.' % (
                    global_example_num, global_loss / global_example_num))
                fid += 1
        except tf.errors.OutOfRangeError:
            pass
        except Exception as ex:
            if supervisor != None:
                supervisor.request_stop(ex)
            raise

        sess.run(model.incre_finshed_worker_num)
        if is_master_worker:
            print('Waiting for other workers to finish ...')
            while True:
                finished_worker_num = sess.run(model.finished_worker_num)
                if finished_worker_num == worker_num: break
                time.sleep(1)
            print('Avg. Loss Summary:')
            for i in range(epoch_num):
                training_loss, training_example_num = model.training_stat[i].eval(sess)
                validation_loss, validation_example_num = model.validation_stat[i].eval(sess)
                print('-- [Epoch %d] Training: %.5f' % (i, training_loss / training_example_num), )
                if validation_example_num != 0:
                    print('; Validation: %.5f' % (validation_loss / validation_example_num))
                else:
                    print()
            model.saver.save(sess, model_file, write_meta_graph=False)
            print('Model saved to', model_file)


def _queue_size(train_files, validation_files, epoch_num):
    qsize = len(train_files)
    if validation_files != None:
        qsize += len(validation_files)
    return qsize * epoch_num


def train(conf: RunnerConfig):
    optimizer = tf.train.AdagradOptimizer(conf.learning_rate, conf.adagrad_init_accumulator)
    queue_size = _queue_size(conf.train_files, conf.validation_files, conf.epoch_num)
    if conf.mode == 'train':
        model = LocalFmModel(queue_size, conf.epoch_num, conf.vocabulary_size, conf.vocabulary_block_num,
                             conf.hash_feature_id, conf.factor_num, conf.init_value_range, conf.loss_type,
                             optimizer, conf.batch_size, conf.factor_lambda, conf.bias_lambda)
        _train(tf.Session(), None, 1, True, True, model, conf.train_files, conf.weight_files, conf.validation_files,
               conf.epoch_num, conf.thread_num, conf.model_file)
    elif conf.mode == 'dist_train':
        cluster = tf.train.ClusterSpec({'ps': conf.ps_hosts, 'worker': conf.worker_hosts})
        server = tf.train.Server(cluster, job_name=conf.job_name, task_index=conf.task_idx)
        if conf.job_name == 'ps':
            server.join()
        elif conf.job_name == 'worker':
            model = DistFmModel(queue_size, cluster, conf.task_idx, conf.epoch_num, conf.vocabulary_size,
                                conf.vocabulary_block_num, conf.hash_feature_id, conf.factor_num, conf.init_value_range,
                                conf.loss_type, optimizer, conf.batch_size, conf.factor_lambda, conf.bias_lambda)
            sv = tf.train.Supervisor(is_chief=(conf.task_idx == 0), init_op=model.init_vars)
            _train(sv.managed_session(server.target, config=tf.ConfigProto(log_device_placement=False)), sv,
                   len(conf.worker_hosts), conf.task_idx == 0, False, model, conf.train_files, conf.weight_files,
                   conf.validation_files, conf.epoch_num, conf.thread_num, conf.model_file)


def _predict(sess, supervisor, is_master_worker, model, model_file, predict_files, score_path, need_to_init):
    with sess as sess:
        if is_master_worker:
            if need_to_init:
                sess.run(model.init_vars)
            if not os.path.exists(score_path):
                os.mkdir(score_path)
            model.saver.restore(sess, model_file)
            for fname in predict_files:
                sess.run(model.file_enqueue_op,
                         feed_dict={model.epoch_id: 0, model.is_training: False, model.data_file: fname,
                                    model.weight_file: ''})
            sess.run(model.file_close_queue_op)
            sess.run(model.set_model_loaded)
        try:
            while not sess.run(model.model_loaded):
                print('Waiting for the model to be loaded.')
                time.sleep(1)
            fid = 0
            while True:
                _, _, fname, _ = sess.run(model.file_dequeue_op)
                score_file = score_path + '/' + os.path.basename(fname) + '.score'
                print('Start processing %s, scores written to %s ...' % (fname, score_file))
                with open(score_file, 'w') as o:
                    while True:
                        pred_score, example_num = sess.run([model.pred_score, model.example_num],
                                                           feed_dict={model.file_id: fid, model.data_file: fname,
                                                                      model.weight_file: ''})
                        if example_num == 0: break
                        for score in pred_score:
                            o.write(str(score) + '\n')
                fid += 1
        except tf.errors.OutOfRangeError:
            pass
        except Exception as ex:
            if supervisor != None:
                supervisor.request_stop(ex)
            raise


def predict(conf: RunnerConfig):
    if conf.mode == 'predict':
        model = LocalFmModel(len(conf.predict_files), 0, conf.vocabulary_size, conf.vocabulary_block_num,
                             conf.hash_feature_id, conf.factor_num, 0, None, None, PREDICT_BATCH_SIZE, 0, 0)
        _predict(tf.Session(), None, True, model, conf.model_file, conf.predict_files, conf.score_path, True)
    else:
        cluster = tf.train.ClusterSpec({'ps': conf.ps_hosts, 'worker': conf.worker_hosts})
        server = tf.train.Server(cluster, job_name=conf.job_name, task_index=conf.task_idx)
        if conf.job_name == 'ps':
            server.join()
        elif conf.job_name == 'worker':
            model = DistFmModel(len(conf.predict_files), cluster, conf.task_idx, 0, conf.vocabulary_size,
                                conf.vocabulary_block_num, conf.hash_feature_id, conf.factor_num, 0, None, None,
                                PREDICT_BATCH_SIZE, 0, 0)
            sv = tf.train.Supervisor(is_chief=(conf.task_idx == 0), init_op=model.init_vars)
            _predict(sv.managed_session(server.target, config=tf.ConfigProto(log_device_placement=False)), sv,
                     conf.task_idx == 0, model, conf.model_file, conf.predict_files, conf.score_path, False)
        else:
            sys.stderr.write('Invalid Job Name: %s' % conf.job_name)
            raise Exception
