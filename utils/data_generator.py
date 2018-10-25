import numpy as np
import copy
            
class VanillaDataGenerator(object):

    def __init__(self, x, y, batch_size, shuffle=True, seed=1234):
        """
        Args:
          x: ndarray
          y: 2darray
          batch_size: int
          shuffle: bool
          seed: int
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rs = np.random.RandomState(seed)
        
    def generate(self, max_iteration=None):
        
        batch_size = self.batch_size
        
        samples_num = len(self.x)
        indexes = np.arange(samples_num)
        
        if self.shuffle:
            self.rs.shuffle(indexes)
        
        iteration = 0
        pointer = 0
        
        while True:
            
            if iteration == max_iteration:
                break
            
            # Get batch indexes
            batch_idxes = indexes[pointer : pointer + batch_size]
            pointer += batch_size
            
            # Reset pointer
            if pointer >= samples_num:
                pointer = 0
                
                if self.shuffle:
                    self.rs.shuffle(indexes)
            
            iteration += 1
            
            yield self.x[batch_idxes], self.y[batch_idxes]
            
            
class BalancedDataGenerator(object):
    """Balanced data generator. Each mini-batch is balanced with approximately 
    the same number of samples from each class. 
    """
    
    def __init__(self, x, y, batch_size, labels_map, shuffle=True, seed=1234, verbose=0):
        """
        Args:
          x: ndarray
          y: 2darray
          batch_size: int
          shuffle: bool
          seed: int
          verbose: int
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.labels_map = labels_map
        self.shuffle = shuffle
        self.rs = np.random.RandomState(seed)
        self.verbose = verbose
        
        assert self.y.ndim == 2, "y must have dimension of 2!"
            
    def get_classes_set(self, samples_num_of_classes):
        
        classes_num = len(samples_num_of_classes)
        classes_set = []
        
        for k in range(classes_num):
            classes_set += [self.labels_map[k]]
            
        return classes_set
        
    def generate(self, max_iteration=None):
        
        y = self.y
        batch_size = self.batch_size

        (samples_num, classes_num) = y.shape
        
        samples_num_of_classes = np.sum(y, axis=0)

        # for ii in self.labels_map:
        #     print(samples_num_of_classes[ii])
        # print(samples_num_of_classes)
        # [226   1   1   4   0   0   2   0   0   0   0   0   0   0   0   0   0   1
        #    3   0   0   0  10  52   7   0   0   0   0   0   0   0   0   0   0   1
        #    0   0   0   0   0   2   1   0   0   1   0   0   0   0   1   0   0   0
        #    0   0   1   0   0   0   2   1   0   0   0   0   0   0   0   0   0   0
        #   47  47  48  54   6   4  20   7   5   1   1   1   0   0   1   0   0   0
        #    0   0   0   0   0   1   0   0   1   0   0   0   0   0   0   0   1   1
        #    0   0   1   0   0   0   1   0   0   0   0   0   0   0  11   0   0   0
        #    0   0   0   0   0   0   0   0   0   0   0 106   1   1   2   1   1   0
        #    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0
        #    0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        #    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        #    0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0
        #    0   1   0   1   1   0   0   0   0   0   1   0   0   0   0   0   0   0
        #    0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0
        #    1   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0
        #    0   0   0   0   0   0   0   0   0   0   0   0   0   3   0   2   0   0
        #  253   1   1   1   6   2   1   1   0   0   1   0   7   2   1   0   1   0
        #    0   4   2   1  24   0   0   0   1   0   1   0   0   0   0   0 197  86
        #  129  67   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        #    0   2   1   0   0   0   0   0   0   0   1   0   6   3   1   0   2  46
        #    2   0   0   0   0   0   0   0   0   0  76  58  37   0  49   0   0   0
        #    0   0   0   0   0   0  65   1  31   0   8  10   9  54   0   0   0  59
        #  195   3   2  59  11   0   0   0   0   0   4   0   0   0   0   0   0   0
        #    0   0   0   0   1   0   0   0   0   0   1   1   4 182  92   1  44  54
        #    1   1   0   0   1   0   0   0   3   6   0   3   7  11   5   0   3   7
        #    6   6  13   0  47   0   1   0   0   0   0   2   0   0   0   0   0   0
        #    0   0  56   0   0   0   0   0   0   0   0   0   0   5   0   0   0   0
        #    0   0   0   0   2   0   0   0   0   0   0   0   0   0   3   0   0   0
        #   10   0  38   2   0   6   3   0   0   1   0   0   0   0   0   0   0   0
        #    0   0   0   0   0]

        # E.g. [0, 1, 2, ..., K]
        classes_set = self.get_classes_set(self.labels_map)

        if self.verbose:
            print("samples_num_of_classes: {}".format(samples_num_of_classes))
            print("classes_set: {}".format(classes_set))
        
        # E.g. [[0, 1, 2], [3, 4, 5, 6], [7, 8], ...]
        indexes_of_classes = []
        
        for k in range(classes_num):
            indexes_of_classes.append(np.where(y[:, k] == 1)[0])
            
        # Shuffle indexes
        if self.shuffle:
            for k in range(classes_num):
                self.rs.shuffle(indexes_of_classes[k])
        
        queue = []
        iteration = 0
        pointers_of_classes = [0] * classes_num

        while True:
            
            if iteration == max_iteration:
                break
            
            # Get a batch containing classes from a queue
            while len(queue) < batch_size:
                # self.rs.shuffle(classes_set)
                queue += classes_set
            self.rs.shuffle(queue)

            batch_classes = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            samples_num_of_classes_in_batch = [batch_classes.count(k) for k in range(classes_num)]
            batch_idxes = []
            
            # Get index of data from each class
            for k in range(classes_num):
                
                bgn_pointer = pointers_of_classes[k]
                fin_pointer = pointers_of_classes[k] + samples_num_of_classes_in_batch[k]
                
                per_class_batch_idxes = indexes_of_classes[k][bgn_pointer : fin_pointer]
                batch_idxes.append(per_class_batch_idxes)

                pointers_of_classes[k] += samples_num_of_classes_in_batch[k]
                
                if pointers_of_classes[k] >= samples_num_of_classes[k]:
                    pointers_of_classes[k] = 0
                    
                    if self.shuffle:
                        self.rs.shuffle(indexes_of_classes[k])
                
            batch_idxes = np.concatenate(batch_idxes, axis=0)
            
            iteration += 1
            
            time_slice = np.random.randint(0, 9, size=batch_size)       # random seperate a slice, and concat in reverse.
            batch_inner_shuffle = []                                    # batch_inner_shuffle = copy.deepcopy(self.x[batch_idxes])
            for ii,item in enumerate(self.x[batch_idxes]):
                # print(item)
                batch_inner_shuffle.append(item)            # batch_inner_shuffle.append( np.concatenate([item[time_slice[ii]:,:,:], item[:time_slice[ii],:,:]], axis=0))     

            yield batch_inner_shuffle, self.y[batch_idxes]
            
            
if __name__ == '__main__':
    
    x = np.ones((1000, 784))
    y = np.ones((1000, 10))
    
    gen = BalancedDataGenerator(x, y, batch_size=128, shuffle=True, seed=1234)
    
    for (batch_x, batch_y) in gen.generate(max_iteration=3):
        print(batch_x.shape)