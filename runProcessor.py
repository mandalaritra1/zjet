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

from python.response_maker_nanov9_lib import *
from python.response_maker_nanov9 import *
from python.smp_utils import *
from python.cms_utils import *


from distributed import Client
from lpcjobqueue import LPCCondorCluster

cluster = LPCCondorCluster(transfer_input_files = [ "correctionFiles", "samples", "python"], 
                           ship_env = False,
                           memory = "5GB",
                          scheduler_options={"dashboard_address": ":2019"})
cluster.adapt(minimum=0, maximum=10000)


client = Client(cluster)

eras_mc = ['UL16NanoAODv9','UL17NanoAODv9','UL18NanoAODv9']

response_maker_nanov9(testing=False, do_gen=False, client=client, prependstr="root://cmsxrootd.fnal.gov/", eras_mc=eras_mc, do_syst = False, do_jk = False , dask = True, fname_out = None)
#response_maker_nanov9(testing=False, do_gen=False, client=client)
print("Done running All")