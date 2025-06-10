# Can grab a file on cmslpc from 
# /store/group/lpctlbsm/NanoAODJMAR_2019_V1/Production/CRAB/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/DYJetsToLLM-50TuneCUETP8M113TeV-madgraphMLM-pythia8RunIISummer16MiniAODv3-PUMoriond17_ext2-v2/190513_171710/0000/*.root
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
import glob
import pickle
import dask



from python.response_maker_nanov9_lib_v4 import *



def response_maker_nanov9(testing=False, do_gen=True, client=None, prependstr = "root://xcache/",
                          skimfilename=None, eras_mc = None, eras_data = None, do_syst = False , dask = False, do_jk = False,
                          do_herwig = False, do_background = False, fname_out = None, syst_list = None, jet_syst_list = None ): 


    if do_jk == True:
        do_syst = False
        do_background = False
    filedir = "samples/"

    # eras_data = [
    #     'UL16NanoAOD', 
    #     'UL16NanoAODAPV', 
    #     'UL17NanoAOD', 
    #     'UL18NanoAOD'
    #        ]
    eras_mc = eras_mc
    
    
    if not testing: 
        nworkers = 1
        if do_syst or do_jk:
            chunksize = 2000000
        else:
            chunksize = 1000000
        maxchunks = None
    elif dask and (client != None):
        chunksize = 100000
        maxchunks = None
    else:
        client = None
        nworkers = 1
        if do_gen: 
            chunksize = 200
        else:
            chunksize=1000000
        maxchunks = None

    print("Chunk Size ", chunksize)
    print("Max chunks", maxchunks)
    fileset = {}
    if not testing: 
        
        if do_gen and (not do_herwig) and (not do_background):
            print("Running over PYTHIA MC")
            dy_mc_filestr = "DYJetsToLL_M-50_HT_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_%s_files.txt"
            #dy_mc_filestr = "pythia_%s.txt"

            for era in eras_mc: 
                filename = filedir + dy_mc_filestr % (era)
                with open(filename) as f:
                    dy_mc_files = [prependstr + i.rstrip() for i in f.readlines() if i[0] != "#"  ] 
                    fileset[era] = dy_mc_files
        elif do_gen and do_herwig:
            print("Running over Herwig MC")
            dy_mc_filestr = "DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7_%s.txt"

            for era in eras_mc: 
                filename = filedir + dy_mc_filestr % (era)
                with open(filename) as f:
                    dy_mc_files = [prependstr + i.rstrip() for i in f.readlines() if i[0] != "#"  ] 
                    fileset[era] = dy_mc_files
        elif do_gen and do_background:
            print("Running over Background")
            bg_cat_list = [ 'ww', 'wz', 'zz', 'ttjets']
            for bg in bg_cat_list:
                filestr = bg 
                for era in eras_mc:
                    filename = filedir + filestr + era + ".txt"
                    with open(filename) as f:
                        files =  [prependstr + i.rstrip() for i in f.readlines() if i[0] != "#"  ] 
                        fileset[bg + '_' +era] = files
        else: 
            print("Running over Data")

            # datasets_list = [['SingleElectron_UL2016','SingleMuon_UL2016'],]
            #                  #['SingleElectron_UL2016APV','SingleMuon_UL2016APV'],]
            #                  #['SingleElectron_UL2017','SingleMuon_UL2017'],]
            #                  #['EGamma_UL2018','SingleMuon_UL2018']]

            # fname_out_list = ['2016',]
            #                   #'2016APV',]
            #                   #'2017',]
            #                   #'2018']

            datasets_list = []
            fname_out_list = []

            

            for era in eras_data:
                print("Era is ", era)
                if era == '2016':
                    datasets_list.append(['SingleElectron_UL2016','SingleMuon_UL2016'])
                    fname_out_list.append(era)
                if era == '2016APV':
                    datasets_list.append(['SingleElectron_UL2016APV','SingleMuon_UL2016APV'])
                    fname_out_list.append(era)
                if era == '2017':
                    datasets_list.append(['SingleElectron_UL2017','SingleMuon_UL2017'])
                    fname_out_list.append(era)
                if era == '2018':
                    datasets_list.append(['EGamma_UL2018','SingleMuon_UL2018'])
                    fname_out_list.append(era)
            # datasets_data = [
            #     'SingleElectron_UL2016APV',
            #     'SingleElectron_UL2016',
            #     'SingleElectron_UL2017',
            #     'EGamma_UL2018',
            #     'SingleMuon_UL2016APV',
            #     'SingleMuon_UL2016',
            #     'SingleMuon_UL2017',
            #     'SingleMuon_UL2018',
            # ]
            
            fileset_data_list = []
            for datasets_data in datasets_list: 
                fileset = {}
                for dataset in datasets_data: 
                    filename = filedir + dataset + '_NanoAODv9_files.txt'
                    with open(filename) as f:
                        data_files = [prependstr + i.rstrip() for i in f.readlines()  if i[0] != "#" ]
                        fileset[dataset] = data_files
                fileset_data_list.append(fileset)
    else: ## Testinng
        if do_gen :
            
            if ( not do_herwig) and (not do_background):
                filename = filedir+"subset2016mc.txt"
                #fileset["UL2018"] = [prependstr+'/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/00EA9563-5449-D24E-9566-98AE8E2A61AE.root']
                with open(filename) as f:
                    fileset["UL18NanoAODv9"] = [prependstr + i.rstrip() for i in f.readlines() if i[0] != "#" ]
            elif ( not do_herwig) and do_background:
                print("Doing bg test")
                fileset["wz_2017"] = [prependstr + "/store/data/Run2017H/SingleMuon/NANOAOD/UL2017_MiniAODv2_NanoAODv9_GT36-v1/2530000/99416BAF-C9A9-C24F-98C6-0A22EB8EAFE5.root"]
            else:
                print("Doing Herwig Test")
                fileset["UL17NanoAODv9"] = [prependstr + "/store/data/Run2017G/SingleMuon/NANOAOD/UL2017_MiniAODv2_NanoAODv9_GT36-v1/70000/216F0004-2555-CA48-A585-78156C4E7BB5.root"]
        else: 
            fileset["UL2018"] = [prependstr + "/store/data/Run2018A/SingleMuon/NANOAOD/UL2018_MiniAODv2_NanoAODv9_GT36-v1/2820000/AB2DCE09-D88B-124F-BB18-89C6D59D04A8.root"]
            fileset["UL2017"] = [prependstr +"/store/data/Run2017B/SingleElectron/NANOAOD/UL2017_MiniAODv2_NanoAODv9-v1/120000/46E53FF3-D096-C647-83A1-8112BC83D056.root"]

        print(fileset)

                

    if client == None:  
        print("Fileset keys ", fileset.keys())

        run = processor.Runner(
            executor = processor.FuturesExecutor(compression=None, workers=nworkers),
            schema=NanoAODSchema,
            chunksize=chunksize,
            maxchunks=maxchunks,
            skipbadfiles=True
        )
    else: 
        run = processor.Runner(
            executor = processor.DaskExecutor(client=client, 
                                              retries=3, 
                                              treereduction=6, 
                                              status=True),
            schema=NanoAODSchema,
            chunksize=chunksize,
            maxchunks=maxchunks,
            
            skipbadfiles=True
        )

    
    print("Running...")
    # print(fileset)
    # if client == None or testing == True:
    #     dataset_runnable, dataset_updated = preprocess(
    #         fileset,
    #         align_clusters=False,
    #         step_size=100_000,
    #         files_per_batch=1,
    #         skip_bad_files=True,
    #         save_form=False,
    #     )
    #     to_compute = apply_to_fileset(
    #             QJetMassProcessor(do_gen=do_gen, skimfilename=skimfilename),
    #             max_chunks(dataset_runnable, 1000),
    #             schemaclass=NanoAODSchema,
    #         )
    #     (output, ) = dask.compute(to_compute) 
    def run_over_fileset(fileset, fname_out = fname_out):    
        output = run(
            fileset,
            "Events",
            processor_instance=QJetMassProcessor(do_gen=do_gen, skimfilename=skimfilename, do_syst = do_syst, do_background  = do_background,
                                                 do_jk = do_jk, syst_list = syst_list, jet_syst_list = jet_syst_list ),
        )
    
        print("Done running")
        if fname_out == None:
            if do_gen:
                if testing == False:
                    if len(eras_mc)==1:
                        fname_out = 'qjetmass_zjets_gen_'+eras_mc[0]+'_' +"all_syst" +'.pkl'
                    else:
                        fname_out = 'qjetmass_zjets_gen_'+'_' +"all_syst" +'.pkl'
                else:
                    fname_out = 'test_qjetmass_zjets_gen_'+eras_mc[0]+'_' +"all_syst" +'.pkl'
            else:
                if testing == True:
                    fname_out = 'test_qjetmass_zjets_reco.pkl'
                else:
                    fname_out = 'qjetmass_zjets_reco.pkl'
            if do_jk:
                fname_out = 'jackknife_output.pkl'
        with open(fname_out, "wb") as f:
            pickle.dump( output, f )
        print(fname_out ," was created.")
    import datetime
    dt = datetime.datetime.now()
    dtstr = dt.strftime("%d%m")
    if testing:
        run_over_fileset(fileset)
    elif ((not testing) & (not do_gen)):
        print("Running over DATA")

        i_name = 0
        for fileset in fileset_data_list:
            if fname_out == None:
                print(f"Now using files from {fname_out_list[i_name]}")
                print(f"Output file will be saved at {'outputs/data_'+dtstr+'_'+fname_out_list[i_name] + '.pkl'}")
                run_over_fileset(fileset, fname_out = 'outputs/data_'+dtstr+'_'+fname_out_list[i_name] + '.pkl')
                i_name += 1
            else:
                print(f"Now using files from {fname_out_list[i_name]}")
                print(f"Output file will be saved at {fname_out}")
                run_over_fileset(fileset, fname_out = fname_out)
                i_name += 1
    else:
        run_over_fileset(fileset)
            
        
        
    
