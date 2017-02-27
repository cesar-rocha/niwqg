# Generic diagnostics methods used from by QGModel and QGNIW Kernel
# These methods are adaptations from pyqg (https://doi.org/10.5281/zenodo.30517)

import numpy as np

def get_diagnostic(self, dname):
    return (self.diagnostics[dname]['value'] /
            self.diagnostics[dname]['count'])
    def _set_active_diagnostics(self, diagnostics_list):
        for d in self.diagnostics:
            self.diagnostics[d]['active'] == (d in diagnostics_list)

def add_diagnostic(self, diag_name, description=None, units=None,types='scalar', function=None):

    assert hasattr(function, '__call__')
    assert isinstance(diag_name, str)

    self.diagnostics[diag_name] = {
       'description': description,
       'units': units,
       'active': True,
       'count': 0,
       'type': types,
       'function': function,}

def describe_diagnostics(self):
    """Print a human-readable summary of the available diagnostics."""
    diag_names = self.diagnostics.keys()
    diag_names.sort()
    print('NAME               | DESCRIPTION')
    print(80*'-')
    for k in diag_names:
        d = self.diagnostics[k]
        print('{:<10} | {:<54}').format(
             *(k,  d['description']))

def _set_active_diagnostics(self, diagnostics_list):
    for d in self.diagnostics:
        self.diagnostics[d]['active'] == (d in diagnostics_list)

def increment_diagnostics(self):

    if  ( not (self.tc%self.tdiags) ):
        self._calc_derived_fields()

        for dname in self.diagnostics:
            res = self.diagnostics[dname]['function'](self)
            try:
                if self.diagnostics[dname]['type'] == 'scalar':
                    self.diagnostics[dname]['value'] = np.hstack([ self.diagnostics[dname]['value'],res])
                else:
                    self.diagnostics[dname]['value'] += res
                    self.diagnostics[dname]['value'] *= 0.5
            except:
                if self.diagnostics[dname]['type'] == 'scalar':
                    self.diagnostics[dname]['value'] = np.array(res)
                else:
                    self.diagnostics[dname]['value'] = res
