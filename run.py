import argparse
import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict
import pickle
import dask
import os

from python.response_maker_nanov9_lib import *
from python.response_maker_nanov9 import *
from python.smp_utils import *
from python.cms_utils import *

jet_systematics_all = ['nominal', 'JERUp', 'JERDown', 'hem',
 'JES_AbsoluteMPFBiasUp', 'JES_AbsoluteMPFBiasDown', 'JES_AbsoluteScaleUp', 'JES_AbsoluteScaleDown', 
 'JES_AbsoluteStatUp', 'JES_AbsoluteStatDown', 'JES_FlavorQCDUp', 'JES_FlavorQCDDown', 'JES_FragmentationUp', 
 'JES_FragmentationDown', 'JES_PileUpDataMCUp', 'JES_PileUpDataMCDown', 'JES_PileUpPtBBUp', 'JES_PileUpPtBBDown', 
 'JES_PileUpPtEC1Up', 'JES_PileUpPtEC1Down', 'JES_PileUpPtEC2Up', 'JES_PileUpPtEC2Down', 'JES_PileUpPtHFUp', 'JES_PileUpPtHFDown', 
 'JES_PileUpPtRefUp', 'JES_PileUpPtRefDown', 'JES_RelativeFSRUp', 'JES_RelativeFSRDown', 'JES_RelativeJEREC1Up', 'JES_RelativeJEREC1Down',
 'JES_RelativeJEREC2Up', 'JES_RelativeJEREC2Down', 'JES_RelativeJERHFUp', 'JES_RelativeJERHFDown', 'JES_RelativePtBBUp', 'JES_RelativePtBBDown',
 'JES_RelativePtEC1Up', 'JES_RelativePtEC1Down', 'JES_RelativePtEC2Up', 
 'JES_RelativePtEC2Down', 'JES_RelativePtHFUp', 'JES_RelativePtHFDown', 'JES_RelativeBalUp', 
 'JES_RelativeBalDown', 'JES_RelativeSampleUp', 'JES_RelativeSampleDown', 'JES_RelativeStatECUp', 'JES_RelativeStatECDown',
 'JES_RelativeStatFSRUp', 'JES_RelativeStatFSRDown', 'JES_RelativeStatHFUp', 'JES_RelativeStatHFDown', 'JES_SinglePionECALUp', 'JES_SinglePionECALDown', 
 'JES_SinglePionHCALUp', 'JES_SinglePionHCALDown', 'JES_TimePtEtaUp', 'JES_TimePtEtaDown', 'JMRUp', 'JMRDown', 'JMSUp', 'JMSDown']

jet_systematics_mass = [ 'nominal', 'JMRUp', 'JMRDown', 'JMSUp', 'JMSDown']

systematics_all =  ['nominal', 'puUp', 'puDown' , 'elerecoUp', 'elerecoDown', 
                    'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp', 'murecoDown', 
                    'muidUp', 'muidDown', 'mutrigUp', 'mutrigDown', 
                    'pdfUp', 'pdfDown', 'q2Up', 'q2Down',
                    'prefiringUp', 'prefiringDown'] 

systematics = ['nominal']

def main(test, do_gen, dask, systematic, do_herwig, output):
    # Set the output filename based on the options
    if test:
        output_prefix = "test_"
    else:
        output_prefix = ""
        
    if do_gen and do_herwig:
        output_filename = f"{output_prefix}mc_herwig.pkl"
    elif do_gen and not do_herwig:
        output_filename = f"{output_prefix}mc_pythia.pkl"
    else:
        output_filename = f"{output_prefix}data.pkl"
    
    # Adjust systematic settings
    if systematic == "1":
        syst_list = ['nominal']
        jet_syst_list = ['nominal']
    elif systematic == "2":
        syst_list = ['nominal']
        jet_syst_list = jet_systematics_mass
    elif systematic == "3":
        syst_list = systematics_all
        jet_syst_list = jet_systematics_all
    else:
        raise ValueError("Invalid option for systematic. Choose from '1', '2', or '3'.")

    # Assuming you have a dask client setup if required
    client_1 = None
    if dask:
        client_1 = client

    # Call your function with the dynamic values
    response_maker_nanov9(
        testing=test,
        do_gen=do_gen,
        client=client_1,
        prependstr="root://cmsxrootd.fnal.gov/",
        eras_mc=['UL16NanoAODAPVv9', 'UL16NanoAODv9','UL17NanoAODv9', 'UL18NanoAODv9'],
        do_syst=(systematic != "1"),
        do_jk=False,
        dask=dask,
        do_herwig=do_herwig,
        syst_list=syst_list,
        jet_syst_list=jet_syst_list,
        fname_out=output_filename
    )
    
    print(f"Done running All. Output saved to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential Jet Mass Analysis for Z+Jets events with NanoAODv9")

    # Define parser arguments for each main() parameter
    parser.add_argument("-t", "--test", action="store_true", help="Run in test mode (Boolean)")
    parser.add_argument("-mc", "--do_gen", action="store_true", help="Generate MC data (Boolean)")
    parser.add_argument("--dask", action="store_true", help="Enable Dask (Boolean)")
    parser.add_argument("-s", "--systematic", choices=["1", "2", "3"], default="1", help="Systematic mode (1: No systematics, 2: Minimal jet systematics, 3: All systematics)")
    parser.add_argument("--herwig", action="store_true", help="Use Herwig (Boolean)")
    parser.add_argument("-o", "--output", required=False, help="Path to save output (overrides default name)")

    args = parser.parse_args()
    client = None
    if args.dask:
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster
        
        cluster = LPCCondorCluster(
            transfer_input_files=["correctionFiles", "samples", "python"],
            ship_env=False,
            memory="5GB",
            scheduler_options={"dashboard_address": ":2018"}
        )
        cluster.adapt(minimum=0, maximum=500)
        client = Client(cluster)
        print("Dask client has been set up")
        print("Observe progress at http://127.0.0.1:8883/proxy/2018/status")
    # Call main with parsed arguments
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        main(test=args.test, do_gen=args.do_gen, dask=args.dask, systematic=args.systematic, do_herwig=args.herwig, output=args.output)
