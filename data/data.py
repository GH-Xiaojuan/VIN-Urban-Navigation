import numpy as np
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_data_bucks(imsize, data_buck_dir, buckSize):
    # allocate tensorSize
    X = np.empty(shape=[0, 9, imsize, imsize])
    S1 = np.empty(shape=[0, 3])
    S2 = np.empty(shape=[0, 3])
    Y = np.empty(shape=[0, 3])

    for i in range(buckSize):
        file_name = data_buck_dir + '/' + str(i) + '.dat'
        print('Load file: ', file_name)
        file = open(file_name, 'rb')
        data = pickle.load(file)
        X_ = data['X']
        Y_ = data['Y']
        S1_ = data['S1']
        S2_ = data['S2']

        X = np.concatenate([X, X_], 0)
        Y = np.concatenate([Y, Y_], 0)
        S1 = np.concatenate([S1, S1_], 0)
        S2 = np.concatenate([S2, S2_], 0)
        file.close()

    X = np.transpose(X, [0, 2, 3, 1])

    # training,  test sets
    all_training_samples = int(6 / 7.0 * X.shape[0])
    training_samples = all_training_samples
    Xtrain = X[0: training_samples]
    ytrain = Y[0: training_samples]
    S1train = S1[0: training_samples]
    S2train = S2[0: training_samples]

    Xtest = X[all_training_samples:]
    ytest = Y[all_training_samples:]
    S1test = S1[all_training_samples:]
    S2test = S2[all_training_samples:]
    ytest = ytest.flatten()

    sortinds = np.random.permutation(training_samples)
    Xtrain = Xtrain[sortinds]
    S1train = S1train[sortinds]
    S2train = S2train[sortinds]
    ytrain = ytrain[sortinds]
    ytrain = ytrain.flatten()

    return Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest


def data_init(args):
    rewards_path, decision_path, out_path, start, end, M, N = args[0], args[1], args[2], \
                                                              args[3], args[4], args[5], args[6]

    process_name = multiprocessing.current_process().name
    print('[%s] process data: (%d, %d), output: %s' % (process_name, start, end, out_path))
    with open(rewards_path, 'rb') as rf:
        rewardsMap = pickle.load(rf)
    with open(decision_path, 'rb') as df:
        decisions = pickle.load(df)

    X = np.empty(shape=[0, 9, M, N])
    S1 = np.empty(shape=[0, 3])
    S2 = np.empty(shape=[0, 3])
    Y = np.empty(shape=[0, 3])
    end = end if end < len(decisions) else len(decisions)
    for i in range(start, end):
        if i != 0 and i % 1000 == 0:
            print('[%s] cur: %d' % (process_name, i - start))

        mapPtr = decisions[i][0]
        Xtemp = abs(rewardsMap[mapPtr])
        maxX = np.max(Xtemp)
        Xtemp = np.where(Xtemp == 0, maxX, Xtemp)
        minX = np.min(Xtemp)
        # normalization to [-1, 0]
        Xtemp = - (Xtemp - minX) * 1.0 / (maxX - minX)
        # print("norm",Xtemp)

        # add goal layer all -1
        goalLayer = - np.ones(M * N).reshape(1, M, N)
        x, y = decisions[i][1:3]
        # set destination reward is +1
        goalLayer[0, x, y] = 1
        Xtemp = np.concatenate([Xtemp.reshape(8, M, N), goalLayer], 0)
        # print("con",np.shape(Xtemp))

        # add to X,S1,S2,Y
        X = np.concatenate([X, Xtemp.reshape(1, 9, M, N)], 0)
        S1 = np.concatenate([S1, np.array(decisions[i][3:6]).reshape(1, 3)], 0)
        S2 = np.concatenate([S2, np.array(decisions[i][6:9]).reshape(1, 3)], 0)
        Y = np.concatenate([Y, np.array(decisions[i][9:12]).reshape(1, 3)], 0)

    feed = {'X': X, 'S1': S1, 'S2': S2, 'Y': Y}
    with open(out_path, 'wb') as f:
        pickle.dump(feed, f)
    print('[%s] completed!' % (process_name))

if __name__ == '__main__':
    process = 7
    M, N = 20, 20
    print('Total process: %d' % (process))
    start = time.time()
    rewards_path = './rewards.data'
    decisions_path = './decisions.data'
    with open(decisions_path, 'rb') as df:
        decisions = pickle.load(df)
        num = len(decisions)
        print('num of decisions: %d' % num)
    outs = []
    starts = []
    ends = []
    rpaths = []
    dpaths= []
    Ms = []
    Ns = []
    step = 10000
    num = step * 10
    for i in range(0, num, step):
        outs.append('./data_buckets/' + str(i // step) + '.dat')
        starts.append(i)
        ends.append(i + step)
        rpaths.append(rewards_path)
        dpaths.append(decisions_path)
        Ms.append(M)
        Ns.append(N)
    pool = ProcessPoolExecutor(max_workers=process)
    args = zip(rpaths, dpaths, outs, starts, ends, Ms, Ns)
    for _ in pool.map(data_init, args):
        pass

    end = time.time()
    print('Finally took %.3f second' % (end - start))