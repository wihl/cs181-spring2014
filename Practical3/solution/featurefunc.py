import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import util

class MetricType:
    process = 1
    thread = 2

class Dataset(object):
    '''
    Create and manage datasets
    '''
    def __init__(self, metricType = MetricType.process):
        self.featureDict = None
        self.metricType = metricType
        if self.metricType == MetricType.process:
            print "using process metrics"
        else:
            print "using thread metrics"
        self.ffs = self.getFeatures()

    def getFeatureDict(self):
        return self.featureDict

    def getFeatures(self):
        if self.metricType == MetricType.process:
            return [first_last_system_call_feats, system_call_count_feats, process_metrics]
        else:
            return [basic_thread_features]

    def getDataset(self,directory):
        self.ids = []
        self.directory = directory
        self.fds = []
        y = []
        for datafile in os.listdir(self.directory):
            # extract id and true class (if available) from filename
            id_str,clazz = datafile.split('.')[:2]
            self.ids.append(id_str)
            if clazz != "X":
                actualValue = util.malware_classes.index(clazz) 
            else:
                actualValue = None

            if self.metricType == MetricType.process:
                # add target class if this is training data
                if actualValue is not None: y.append(actualValue)
                self.extractFeaturesByFile(datafile)
            else:
                # we will have one row per system call, rather than per process
                numRows = self.extractThreadFeatures(datafile,actualValue)
                if actualValue is not None: y.extend([actualValue] * numRows)

        X = self.makeDesignMat()

        return X, y, self.ids


    def extractFeaturesByFile(self, datafile):
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(self.directory,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in self.ffs]
        self.fds.append(rowfd)
        return

    def extractThreadFeatures(self, datafile, actualValue):
        numRows = 0
        return numRows

    def makeDesignMat(self):
        if self.featureDict is None:
            all_feats = set()
            [all_feats.update(fd.keys()) for fd in self.fds]
            feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
        else:
            feat_dict = self.featureDict
            
        cols = []
        rows = []
        data = []        
        for i in xrange(len(self.fds)):
            temp_cols = []
            temp_data = []
            for feat,val in self.fds[i].iteritems():
                try:
                    # update temp_cols iff update temp_data
                    temp_cols.append(feat_dict[feat])
                    temp_data.append(val)
                except KeyError as ex:
                    if self.featureDict is not None:
                        pass  # new feature in test data; nbd
                    else:
                        raise ex

            # all fd's features in the same row
            k = len(temp_cols)
            cols.extend(temp_cols)
            data.extend(temp_data)
            rows.extend([i]*k)

        assert len(cols) == len(rows) and len(rows) == len(data)
    

        self.featureDict = feat_dict
        X = sparse.csr_matrix((np.array(data),
                           (np.array(rows), np.array(cols))),
                              shape=(len(self.fds), len(self.featureDict)))
        return X


def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            #last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    #c["last_call-"+last_call] = 1
    return c

def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
            # prune off noisy features
            if el.tag not in ['create_process','query_value', 'get_host_by_name']:
                c['num_'+el.tag] += 1
    return c

def process_metrics(tree):
    c = Counter()
    for el in tree.iter():
        if el.tag == "process":
            # process attributes of interest
            for r in ['startreason', 'terminationreason', 'username', 'executionstatus', 'applicationtype']:
                if el.get(r) != None:
                    c[r+'-'+el.attrib[r]] = c.get(r+'-'+el.attrib[r],0) + 1
    return c

def basic_thread_features(tree):
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['action'] = hash(el.tag)
            # prune off noisy features
            if el.tag not in ['create_process','query_value', 'get_host_by_name']:
                c['num_'+el.tag] += 1
    return c
    
