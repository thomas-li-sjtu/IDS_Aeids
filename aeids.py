from BufferedPackets import WINDOW_SIZE
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.models import model_from_json
# from LibNidsReaderThread import LibNidsReaderThread
# from PcapReaderThread import PcapReaderThread
from StreamReaderThread import StreamReaderThread
from tensorflow import Tensor

import binascii
import math
import numpy
import os
import psycopg2
import psycopg2.extras
import sys
import time
import traceback

tensorboard_log_enabled = False
backend = "tensorflow"
done = False  # 完成的标志
prt = None
conf = {}
activation_functions = ["elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear",
                        "softmax"]
conn = None

# possible values: mean, median, zscore
threshold = "median"


def main(argv):  # 传入参数
    try:
        # 查看命令参数
        if argv[1] != "training" and argv[1] != "predicting" and argv[1] != "testing" and argv[1] != "counting":
            raise IndexError("Phase {} does not exist.".format(argv[1]))
        else:
            phase = argv[1]

        if argv[2] != "tcp" and argv[2] != "udp":  # 协议
            raise IndexError("Protocol {} is not supported.".format(argv[3]))
        else:
            protocol = argv[2]

        if not argv[3].isdigit():  # 端口号
            raise IndexError("Port must be numeric.")
        else:
            port = argv[3]

        if phase != "counting":
            try:
                hidden_layers = argv[4].split(",")
                for neurons in hidden_layers:
                    if not neurons.isdigit():
                        raise IndexError("Hidden layers must be comma separated numeric values")
            except ValueError:
                raise IndexError("Hidden layers must be comma separated numeric values")

            if argv[5] not in activation_functions:
                raise IndexError("Activation function must be one of the following list")
            else:
                activation_function = argv[5]  # 激活函数

            try:
                dropout = float(argv[6])  # dropout层
            except ValueError:
                raise IndexError("Dropout must be numeric.")

            if phase == "training" and not argv[8].isdigit():
                raise IndexError("Batch size must be numeric.")
            elif phase == "training" or phase == "predicting":
                batch_size = int(argv[8])  # 得到batch size

            filename = argv[7]
            if phase == "testing":
                aeids(phase, filename, protocol, port, hidden_layers, activation_function, dropout, argv[8])
            else:
                aeids(phase, filename, protocol, port, hidden_layers, activation_function, dropout,
                      batch_size=batch_size)
        else:
            count_byte_freq(argv[4], protocol, port)  # 计算字节频率

    except IndexError as e:
        print(
            "Usage: python aeids.py <training|predicting|testing|counting> <tcp|udp> <port> <hidden_layers> <activation_function> <dropout> <training filename> [batch_size] [testing filename]")
        print(traceback.print_exc())
        exit(0)
    except KeyboardInterrupt:
        print("Interrupted")
        if prt is not None:
            prt.done = True
    except BaseException as e:
        print(traceback.print_exc())
        if prt is not None:
            prt.done = True


def aeids(phase="training", filename="", protocol="tcp", port="80", hidden_layers=[200, 100],
          activation_function="relu", dropout=0.0, testing_filename="", batch_size=1):
    global done
    global prt
    read_conf()

    if phase == "training":
        numpy.random.seed(666)

        autoencoder = init_model(hidden_layers, activation_function, dropout)  # 初始化

        if "{}-{}".format(filename, port) in conf["training_filename"]:  # 每一轮的迭代次数
            steps_per_epoch = conf["training_filename"]["{}-{}".format(filename, port)] / batch_size
        else:
            steps_per_epoch = conf["training_filename"]["default-80"] / batch_size

        if tensorboard_log_enabled and backend == "tensorflow":
            tensorboard_callback = TensorBoard(log_dir="./logs", batch_size=10000, write_graph=True, write_grads=True,
                                               histogram_freq=1)  # tensorboard
            autoencoder.fit_generator(byte_freq_generator(filename, protocol, port, batch_size), steps_per_epoch=100,
                                      epochs=100, verbose=1, callbacks=[tensorboard_callback])  # 训练
            check_directory(filename, "models")
            autoencoder.save("models/{}/aeids-with-log-{}-hl{}-af{}-do{}.hdf5".format(filename, protocol + port,
                                                                                      ",".join(hidden_layers),
                                                                                      activation_function, dropout),
                             overwrite=True)
        else:
            autoencoder.fit_generator(byte_freq_generator(filename, protocol, port, batch_size),
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=10, verbose=1)
            check_directory(filename, "models")
            autoencoder.save(
                "models/{}/aeids-{}-hl{}-af{}-do{}.hdf5".format(filename, protocol + port, ",".join(hidden_layers),
                                                                activation_function, dropout), overwrite=True)

        print("Training autoencoder finished. Calculating threshold...")
        predict_byte_freq_generator(autoencoder, filename, protocol, port, hidden_layers, activation_function, dropout,
                                    phase)
        done = True
        prt.cleanup_all_buffers()
        prt = None
        print("\nFinished.")
    elif phase == "predicting":
        autoencoder = load_autoencoder(filename, protocol, port, hidden_layers, activation_function, dropout)  # 加载模型
        predict_byte_freq_generator(autoencoder, filename, protocol, port, hidden_layers, activation_function, dropout,
                                    phase)  # 预测结果
        done = True
        print("\nFinished.")
    elif phase == "testing":  # 测试
        autoencoder = load_autoencoder(filename, protocol, port, hidden_layers, activation_function, dropout)  # 加载模型
        predict_byte_freq_generator(autoencoder, filename, protocol, port, hidden_layers, activation_function, dropout,
                                    phase, testing_filename)
        prt = None
        print("\nFinished.")
    else:
        raise IndexError


def read_conf():  # 读取参数
    global conf

    fconf = open("aeids.conf", "r")  # 读入参数存储文件
    if not fconf:
        print("File aeids.conf does not exist.")
        exit(-1)

    conf["root_directory"] = []
    conf["training_filename"] = {"default-80": 100000}
    lines = fconf.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        split = line.split("=", 2)
        print(split)
        if split[0] == "root_directory":
            conf["root_directory"].append(split[1].strip())
        elif split[0] == "training_filename":
            tmp = split[1].split(":")
            conf["training_filename"]["{}-{}".format(tmp[0], tmp[1])] = int(tmp[2])

    fconf.close()


def init_model(hidden_layers=[200, 100], activation_function="relu", dropout=0):  # 模型结构设置
    input_dimension = 256
    input = Input(shape=(input_dimension,))  # 输入层，维度256

    for i in range(0, len(hidden_layers)):  # encode层
        if i == 0:
            encoded = Dense(int(hidden_layers[i]), activation=activation_function)(input)
        else:
            encoded = Dense(int(hidden_layers[i]), activation=activation_function)(encoded)

        encoded = Dropout(dropout)(encoded)  # 层之间dropout

    for i in range(len(hidden_layers) - 1, -1, -1):  # decode层
        if i == len(hidden_layers) - 1:
            decoded = Dense(int(hidden_layers[i]), activation=activation_function)(encoded)
        else:
            decoded = Dense(int(hidden_layers[i]), activation=activation_function)(decoded)

        decoded = Dropout(0.2)(decoded)

    if len(hidden_layers) == 1:
        decoded = Dense(input_dimension, activation="sigmoid")(encoded)
    else:
        decoded = Dense(input_dimension, activation="sigmoid")(decoded)
    autoencoder = Model(outputs=decoded, inputs=input)
    autoencoder.compile(loss="binary_crossentropy", optimizer="adadelta")

    return autoencoder


def load_autoencoder(filename, protocol, port, hidden_layers, activation_function, dropout):  # 加载模型
    autoencoder = load_model(
        "models/{}/aeids-{}-hl{}-af{}-do{}.hdf5".format(filename, protocol + port, ",".join(hidden_layers),
                                                        activation_function, dropout))
    return autoencoder


def byte_freq_generator(filename, protocol, port, batch_size):  # 是一个生成器，一个函数定义中包含yield关键字，那么这个函数就不再是一个普通函数
    global prt
    global conf
    global done
    prt = StreamReaderThread(get_pcap_file_fullpath(filename), protocol, port)
    prt.start()  # 线程开始活动
    counter = 0
    done = False

    while not done:
        while not prt.done or prt.has_ready_message():
            if not prt.has_ready_message():  # 没有信息
                prt.wait_for_data()
                continue
            else:
                buffered_packets = prt.pop_connection()
                if buffered_packets is None:
                    time.sleep(0.0001)
                    continue
                if buffered_packets.get_payload_length("server") > 0:
                    byte_frequency = buffered_packets.get_byte_frequency("server")
                    X = numpy.reshape(byte_frequency, (1, 256))

                    if counter == 0 or counter % batch_size == 1:
                        dataX = X
                    else:
                        dataX = numpy.r_["0,2", dataX, X]  # 按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()

                    counter += 1

                    if counter % batch_size == 0:
                        yield dataX, dataX

        if dataX.shape[0] > 0:
            yield dataX, dataX

        prt.reset_read_status()


def predict_byte_freq_generator(autoencoder, filename, protocol, port, hidden_layers, activation_function, dropout,
                                phase="training", testing_filename=""):
    global prt
    global threshold
    print("test_filename", testing_filename)
    if prt is None:
        if phase == "testing":
            prt = StreamReaderThread(get_pcap_file_fullpath(testing_filename), protocol, port)
            print("testing filename: " + testing_filename)
        else:
            prt = StreamReaderThread(get_pcap_file_fullpath(filename), protocol, port)

        prt.delete_read_connections = True
        prt.start()  # start() 方法是启动一个子线程，线程名就是自己定义的name
    else:
        prt.reset_read_status()
        prt.delete_read_connections = True

    errors_list = []  # 报错列表
    counter = 0
    print("predict")

    if phase == "testing":
        t1, t2 = load_threshold(filename, protocol, port, hidden_layers, activation_function, dropout)  # 载入测试的阈值
        check_directory(filename, "results")  # 建立目录
        # fresult = open("results/{}/result-{}-hl{}-af{}-do{}-{}.csv".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout, testing_filename), "w")
        open_conn()  # 数据库
        experiment_id = create_experiment(filename, testing_filename, protocol, port, ",".join(hidden_layers),
                                          activation_function, dropout)  # 数据库环境配置
        # if fresult is None:
        #     raise Exception("Could not create file")

    # ftemp = open("results/data.txt", "wb")
    # fcsv = open("results/data.csv", "wb")
    # a = csv.writer(fcsv, quoting=csv.QUOTE_ALL)
    # time.sleep(2)
    i_counter = 0
    while (not prt.done) or (prt.has_ready_message()):
        if not prt.has_ready_message():
            prt.wait_for_data()
        else:
            buffered_packets = prt.pop_connection()
            if buffered_packets is None:
                continue
            if buffered_packets.get_payload_length("server") == 0:
                continue

            i_counter += 1
            # print "{}-{}".format(i_counter, buffered_packets.id)
            # print "{}-{}: {}".format(i_counter, buffered_packets.id, buffered_packets.get_payload("server")[:100])
            byte_frequency = buffered_packets.get_byte_frequency("server")
            # ftemp.write(buffered_packets.get_payload())
            # a.writerow(byte_frequency)
            data_x = numpy.reshape(byte_frequency, (1, 256))
            decoded_x = autoencoder.predict(data_x)  #
            # a.writerow(decoded_x[0])

            # fcsv.close()
            error = numpy.mean((decoded_x - data_x) ** 2, axis=1)  # 求方差
            # ftemp.write("\r\n\r\n{}".format(error))
            # ftemp.close()
            if phase == "training" or phase == "predicting":
                errors_list.append(error)
            elif phase == "testing":  # 测试，计入数据库
                decision = decide(error[0], t1, t2)
                # fresult.write("{},{},{},{},{},{}\n".format(buffered_packets.id, error[0], decision[0], decision[1], decision[2], buffered_packets.get_hexlify_payload()))
                write_results_to_db(experiment_id, buffered_packets, error, decision)

            counter += 1
            sys.stdout.write("\rCalculated {} connections.".format(counter))  # 等价于没有换行符的print
            sys.stdout.flush()  # 刷新stdout，以实时看到输出信息


    errors_list = numpy.reshape(errors_list, (1, len(errors_list)))
    if phase == "training" or phase == "predicting":
        save_mean_stdev(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list)
        save_q3_iqr(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list)
        save_median_mad(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list)
    elif phase == "testing":
        # fresult.close()
        return


def count_byte_freq(filename, protocol, port):
    global prt
    global conf

    read_conf()

    prt = StreamReaderThread(get_pcap_file_fullpath(filename), protocol, port)
    prt.start()
    prt.delete_read_connections = True
    counter = 0
    missed_counter = 0

    while not prt.done or prt.has_ready_message():
        if not prt.has_ready_message():
            # print(1)
            # time.sleep(0.0001)
            missed_counter += 1
            sys.stdout.write(
                "\r1-{} flows. Missed: {}. {} items in buffer. packets: {}. last ts: {}".format(counter, missed_counter,
                                                                                                len(prt.tcp_buffer),
                                                                                                prt.packet_counter,
                                                                                                prt.last_timestamp))
            sys.stdout.flush()
            prt.wait_for_data()
            continue
        else:
            start = time.time()
            buffered_packets = prt.pop_connection()
            end = time.time()
            if buffered_packets is None:
                # print(2)
                # time.sleep(0.0001)
                missed_counter += 1
                sys.stdout.write("\r2-{} flows. Missed: {}. Time: {}".format(counter, missed_counter, end - start))
                sys.stdout.flush()
                prt.wait_for_data()
                continue
            elif buffered_packets.get_payload_length("server") > 0:
                counter += 1
                sys.stdout.write("\r3-{} flows. Missed: {}. Time: {}".format(counter, missed_counter, end - start))
                sys.stdout.flush()
            else:
                missed_counter += 1
                sys.stdout.write("\r4-{} flows. Missed: {}. Time: {}".format(counter, missed_counter, end - start))
                sys.stdout.flush()

    print("Total flows: {}".format(counter))


def save_mean_stdev(filename, protocol, port, hidden_layers, activation_function, dropout,
                    errors_list):  # errors_list均值与方差
    mean = numpy.mean(errors_list)
    stdev = numpy.std(errors_list)
    fmean = open("models/{}/mean-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers),
                                                               activation_function, dropout), "w")
    fmean.write("{},{}".format(mean, stdev))
    fmean.close()


def save_q3_iqr(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list):
    qs = numpy.percentile(errors_list, [100, 75, 50, 25, 0])  # 计算一个多维数组的任意百分比分位数
    iqr = qs[1] - qs[3]  # 四分位距
    MC = ((qs[0] - qs[2]) - (qs[2] - qs[4])) / (qs[0] - qs[4])
    if MC >= 0:
        constant = 3
    else:
        constant = 4
    iqrplusMC = 1.5 * math.pow(math.e, constant * MC) * iqr  #
    print("IQR: {}\nMC: {}\nConstant: {}".format(iqr, MC, constant))
    fmean = open("models/{}/median-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers),
                                                                 activation_function, dropout), "w")
    fmean.write("{},{}".format(qs[1], iqrplusMC))
    fmean.close()


def save_median_mad(filename, protocol, port, hidden_layers, activation_function, dropout,
                    errors_list):  # errorlist中位数和median absolute deviation
    median = numpy.median(errors_list)
    mad = numpy.median([numpy.abs(error - median) for error in errors_list])

    fmean = open("models/{}/zscore-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers),
                                                                 activation_function, dropout), "w")
    fmean.write("{},{}".format(median, mad))
    fmean.close()


def load_threshold(filename, protocol, port, hidden_layers, activation_function, dropout):  # 读取上面三个函数的输出文件
    t1 = []
    t2 = []

    fmean = open(
        "models/{}/mean-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers),
                                                      activation_function, dropout), "r")
    line = fmean.readline()
    split = line.split(",")
    t1.append(split[0])
    t2.append(split[1])
    fmean.close()

    fmean = open(
        "models/{}/median-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers),
                                                        activation_function, dropout), "r")
    line = fmean.readline()
    split = line.split(",")
    t1.append(split[0])
    t2.append(split[1])
    fmean.close()

    fmean = open(
        "models/{}/zscore-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers),
                                                        activation_function, dropout), "r")
    line = fmean.readline()
    split = line.split(",")
    t1.append(split[0])
    t2.append(split[1])
    fmean.close()

    return t1, t2


def get_threshold(threshold_method, t1, t2):  # 计算阈值
    if threshold_method == "mean":
        return (float(t1[0]) + 2 * float(t2[0]))
    elif threshold_method == "median":
        return (float(t1[1]) + float(t2[1]))
    elif threshold_method == "zscore":
        return 3.5


def decide(mse, t1, t2):
    decision = []

    if mse > (float(t1[0]) + 2 * float(t2[0])):
        decision.append(True)
    else:
        decision.append(False)

    if mse > (float(t1[1]) + float(t2[1])):
        decision.append(True)
    else:
        decision.append(False)

    zscore = 0.6745 * (mse - float(t1[2])) / float(t2[2])
    if zscore > 3.5 or zscore < -3.5:
        decision.append(True)
    else:
        decision.append(False)

    return decision


def check_directory(filename, root="models"):  # 建立目录
    if not os.path.isdir("./{}/{}".format(root, filename)):
        os.mkdir("./{}/{}".format(root, filename))


def get_pcap_file_fullpath(filename):  # 获得pcap文件的路径
    global conf
    for i in range(0, len(conf["root_directory"])):
        if os.path.isfile(conf["root_directory"][i] + filename):
            return conf["root_directory"][i] + filename


def open_conn():  # psycopg2 库是 python 用来操作 postgreSQL 数据库的第三方库      连接数据库
    global conn

    conn = psycopg2.connect(host="localhost", database="aeids", user="postgres", password="postgres")
    conn.set_client_encoding('Latin1')


def create_experiment(training_filename, testing_filename, protocol, port, hidden_layer, activation_function, dropout):  # 创建数据库
    global conn

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(
        "SELECT * FROM experiments WHERE training_filename=%s AND testing_filename=%s AND protocol=%s AND port=%s AND hidden_layers=%s AND activation_function=%s AND dropout=%s",
        (training_filename, testing_filename, protocol, port, hidden_layer, activation_function, dropout))

    if cursor.rowcount > 0:  # There is an existing experiment, get the ID
        row = cursor.fetchone()
        return row["id"]
    else:
        cursor.execute(
            "INSERT INTO experiments(training_filename, testing_filename, protocol, port, hidden_layers, activation_function, dropout) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (training_filename, testing_filename, protocol, port, hidden_layer, activation_function, dropout))
        if cursor.rowcount == 1:
            row = cursor.fetchone()
            conn.commit()
            return row["id"]
        else:
            raise Exception("Cannot insert a new experiment")


def get_message_id(buffered_packet):
    global conn

    tmp = buffered_packet.id.split("-")
    src_addr = tmp[0]
    src_port = tmp[1]
    dst_addr = tmp[2]
    dst_port = tmp[3]
    protocol = tmp[4]
    start_time = buffered_packet.get_start_time()
    stop_time = buffered_packet.get_stop_time()

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute("SELECT * FROM messages WHERE src_ip=%s AND src_port=%s AND dst_ip=%s AND dst_port=%s AND "
                   "protocol=%s AND window_size=%s AND start_time=%s AND stop_time=%s",
                   (src_addr, src_port, dst_addr, dst_port, protocol, WINDOW_SIZE, start_time, stop_time))

    if cursor.rowcount > 0:
        row = cursor.fetchone()
        return row["id"]
    else:
        cursor.execute("""INSERT INTO messages (src_ip, src_port, dst_ip, dst_port, protocol, start_time, stop_time, """
                       """payload, window_size) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                       (src_addr, src_port, dst_addr, dst_port, protocol, start_time, stop_time,
                        psycopg2.Binary(buffered_packet.get_payload("server")), WINDOW_SIZE))
        if cursor.rowcount == 1:
            row = cursor.fetchone()
            conn.commit()
            return row["id"]
        else:
            raise Exception("Cannot insert a new message")


def write_results_to_db(experiment_id, buffered_packet, error, decision):  # 写入数据库
    global conn

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    message_id = get_message_id(buffered_packet)

    cursor.execute(
        "UPDATE mse_results SET mse=%s, decision_mean=%s, decision_median=%s, decision_zscore=%s WHERE messages_id=%s AND experiments_id=%s",
        (error[0], decision[0], decision[1], decision[2], message_id, experiment_id))
    if cursor.rowcount == 0:  # The row doesn't exist
        cursor.execute(
            "INSERT INTO mse_results (experiments_id, messages_id, mse, decision_mean, decision_median, decision_zscore) VALUES (%s, %s, %s, %s, %s, %s)",
            (experiment_id, message_id, error[0], decision[0], decision[1], decision[2]))

    conn.commit()


if __name__ == '__main__':
    main(sys.argv)
