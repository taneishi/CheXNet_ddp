import pandas as pd

from model import CLASS_NAMES

def main():
    stats = []

    train = pd.read_csv('labels/train_list.txt', sep=' ', header=None, index_col=0)
    train.columns = CLASS_NAMES
    stats.append(train.sum() / train.shape[0] * 100.)

    val = pd.read_csv('labels/val_list.txt', sep=' ', header=None, index_col=0)
    val.columns = CLASS_NAMES
    stats.append(val.sum() / val.shape[0] * 100.)

    test = pd.read_csv('labels/test_list.txt', sep=' ', header=None, index_col=0)
    test.columns = CLASS_NAMES
    stats.append(test.sum() / test.shape[0] * 100.)

    bmt = pd.read_csv('labels/bmt_list.txt', sep=' ', header=None, index_col=0)
    bmt.columns = CLASS_NAMES
    stats.append(bmt.sum() / bmt.shape[0] * 100.)

    stats = pd.concat(stats, axis=1)
    stats.columns = ['train', 'val', 'test', 'bmt']
    print(stats.round(3))

if __name__ == '__main__':
    main()
