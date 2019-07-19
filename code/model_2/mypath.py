class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'jingwei':
            return '../../data/train_model_2'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
