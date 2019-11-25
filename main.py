import pickle as pkl
import pprint as pp

CIFAR_10_BATCH_PATH = "./dataset/cifar-10-python/cifar-10-batches-py"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pkl.load(fo, encoding='bytes')
    return dict


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pkl.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels

def keys_of_cfar10(cifar10_path_folder_path, batch_id):
    with open(cifar10_path_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pkl.load(file, encoding='latin1')
    return batch.keys()


#pp.pprint(unpickle("./dataset/cifar-10-python/cifar-10-batches-py/batches.meta"))
#print(unpickle("./dataset/cifar-10-python/cifar-10-batches-py/data_batch_1"))
#print(load_cfar10_batch("./dataset/cifar-10-python/cifar-10-batches-py",1)[0])
print(keys_of_cfar10(CIFAR_10_BATCH_PATH,1))
