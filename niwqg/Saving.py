# Generic methods for writing to disk

import os
import h5py

def init_save_snapshots(self,path):

    self.fno = path

    if (not os.path.isdir(self.fno)) and self.save_to_disk:
        os.makedirs(self.fno)
        os.makedirs(self.fno+"/snapshots/")

def file_exist(fno,overwrite=True):
    if os.path.exists(fno):
        if overwrite:
            os.remove(fno)
        else: raise IOError("File exists: {0}".format(fno))

def save_setup(self,):

    """Save setup  """

    if self.save_to_disk:

        fno = self.fno + '/setup.h5'

        file_exist(fno,overwrite=self.overwrite)

        h5file = h5py.File(fno, 'w')
        h5file.create_dataset("grid/nx", data=(self.nx),dtype=int)
        h5file.create_dataset("grid/x", data=(self.x))
        h5file.create_dataset("grid/y", data=(self.y))
        h5file.create_dataset("grid/wv", data=self.wv)
        h5file.create_dataset("grid/k", data=self.kk)
        h5file.create_dataset("grid/l", data=self.ll)
        # h5file.create_dataset("constants/f0", data=(self.f))
        h5file.close()

def save_snapshots(self, fields=['t','q','p']):

    """ Save snapshots of fields """

    if ( ( not (self.tc%self.tsnaps) ) & (self.save_to_disk) ):

        fno = self.fno + '/snapshots/{:015.0f}'.format(self.t)+'.h5'

        file_exist(fno)

        h5file = h5py.File(fno, 'w')

        for field in fields:
            if field == 't':
                h5file.create_dataset(field, data=(self.t))
            else:
                h5file.create_dataset(field, data=eval("self."+field))

        h5file.close()
    else:
        pass

def _save_diagnostics(self):

    """ Save diagnostics """

    fno = self.fno + 'diagnostics.h5'

    file_exist(fno,overwrite=self.overwrite)

    h5file = h5py.File(fno, 'w')

    for diagnostic in self.diagnostics_list:
        h5file.create_dataset(diagnostic, data=eval("self."+diagnostic))

    h5file.close()
