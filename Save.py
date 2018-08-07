import shelve
import os.path
import shutil
import copy
import Misc


class DataSaver:
    def __init__(self, folder):
        self.folder = os.path.join('output', folder)
        if not os.path.exists(self.folder): os.makedirs(self.folder)

    @property
    def data_file_name(self):
        return os.path.join(self.folder, 'data.csv')

    def clear_folder(self):
        shutil.rmtree(self.folder)
        os.makedirs(self.folder)

    def save_objects(self, bat, data, parameters=[], prefix='save'):
        name = prefix + '_'
        for x in parameters: name += str(x) + '_'
        name = name.rstrip('_')
        filename = os.path.join(self.folder, name)
        my_shelf = shelve.open(filename)
        data['name'] = copy.copy(name)
        my_shelf['bat'] = bat
        my_shelf['data'] = data
        my_shelf.close()
        name = copy.copy(name)
        return name

    def read_objects(self, name):
        filename = os.path.join(self.folder, name)
        print(filename)
        my_shelf = shelve.open(filename)
        bat = my_shelf['bat']
        data = my_shelf['data']
        my_shelf.close()
        return bat, data

    def add_data(self, data):
        filename = self.data_file_name
        Misc.append2csv(data, filename)
