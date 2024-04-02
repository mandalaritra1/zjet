import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
print("awkward version ", ak.__version__)
print("coffea version ", coffea.__version__)
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict
import pickle
import dask
#from distributed.diagnostics.plugin import UploadDirectory
import os
#from cms_utils import *
print("numpy version", np.__version__)
print("dask version", dask.__version__)

from smp_utils import *
from cms_utils import *
from response_maker_nanov9_mc_lib import *
from response_maker_nanov9 import *


from distributed import Client
from lpcjobqueue import LPCCondorCluster

cluster = LPCCondorCluster(transfer_input_files = ["data", "correctionFiles", "jsonpog-integration","samples",  "response_maker_nanov9_lib.py","smp_utils.py","cms_utils.py", "weight_class.py", "response_maker_nanov9_mc_lib.py"], 
                           ship_env = False,
                          memory = '4GB')
cluster.adapt(minimum=0, maximum=10000)


client = Client(cluster)
eras_mc = ['UL16NanoAODv9']
for era in eras_mc:
    response_maker_nanov9(testing=False, do_gen=True, client=client, prependstr="root://cmsxrootd.fnal.gov/", eras_mc=[era] )
    #response_maker_nanov9(testing=False, do_gen=False, client=client)
    print("Done running 2016")
    