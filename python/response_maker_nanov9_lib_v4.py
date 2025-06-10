import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
import os
import pandas as pd
import time
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.analysis_tools import PackedSelection
from collections import defaultdict

import tokenize as tok
import re

from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory

from python.weight_class import Weights
from python.cms_utils import *
from python.smp_utils import *
from copy import deepcopy

class QJetMassProcessor(processor.ProcessorABC):
    '''
    Processor to run a Z+jets jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, do_gen=True, ptcut=200., etacut = 2.5, ptcut_ee = 40., ptcut_mm = 29., skimfilename=None, 
                 do_syst = False, do_jk = False, syst_list = None, jet_syst_list = None, do_background = False):
        
        self.lumimasks = getLumiMaskRun2()
        
        # should have separate lower ptcut for gen
        self.do_gen=do_gen
        self.ptcut = ptcut
        self.etacut = etacut        
        self.lepptcuts = [ptcut_ee, ptcut_mm]
    

        self.do_jk = do_jk
        self.do_syst = do_syst
        self.do_background = do_background
        
        self.minimal = True
        if do_jk:
            self.minimal = False
        binning = util_binning()
        
        ptreco_axis = binning.ptreco_axis
        mreco_axis = binning.mreco_axis
        ptgen_axis = binning.ptgen_axis     
        mgen_axis = binning.mgen_axis

        dataset_axis = binning.dataset_axis
        lep_axis = binning.lep_axis
        n_axis = binning.n_axis
        mass_axis = binning.mass_axis
        zmass_axis = binning.zmass_axis
        pt_axis = binning.pt_axis
        frac_axis = binning.frac_axis
        dr_axis = binning.dr_axis
        dr_fine_axis = binning.dr_fine_axis
        dphi_axis = binning.dphi_axis    
        syst_axis = binning.syst_axis
        eta_axis = binning.eta_axis
        phi_axis = binning.phi_axis
        ptfine_axis = binning.ptfine_axis
        
        ptgen_axis_fine  = binning.ptgen_axis_fine 

        
        mptreco_axis = binning.mreco_over_pt_axis
        mptgen_axis = binning.mgen_over_pt_axis

        

        ht_axis = hist.axis.StrCategory([],growth = True, name = "ht_bin", label = "h_T bin")
        channel_axis = hist.axis.StrCategory([],growth = True, name = "channel", label = "Channel")
        
        weight_axis = hist.axis.Regular(100, 0, 10, name="corrWeight", label=r"Weight")

        mcut_reco_u_axis = binning.mcut_reco_u_axis
        mcut_reco_g_axis = binning.mcut_reco_g_axis

        mcut_gen_u_axis = binning.mcut_gen_u_axis
        mcut_gen_g_axis = binning.mcut_gen_g_axis
        #### weight to check what is causing this
        
        self.gen_binning = binning.gen_binning
        self.reco_binning = binning.reco_binning

        self.hists = {}

        self.mode = "minimal"

        if self.mode == "all":        
            register_hist(self.hists, "njet_gen", [dataset_axis, n_axis])
            register_hist(self.hists, "njet_reco", [dataset_axis, n_axis])
            register_hist(self.hists, "ptjet_gen_pre", [dataset_axis, pt_axis])
            register_hist(self.hists, "ptjet_reco_over_gen", [dataset_axis, frac_axis])
            register_hist(self.hists, "drjet_reco_gen", [dataset_axis, dr_fine_axis])
            register_hist(self.hists, "ptz_gen", [dataset_axis, pt_axis])
            register_hist(self.hists, "ptz_reco", [dataset_axis, pt_axis])
            register_hist(self.hists, "mz_gen", [dataset_axis, zmass_axis])
            register_hist(self.hists, "mz_reco", [dataset_axis, zmass_axis])
            register_hist(self.hists, "mz_reco_over_gen", [dataset_axis, frac_axis])
            register_hist(self.hists, "dr_z_jet_gen", [dataset_axis, dr_axis])
            register_hist(self.hists, "dr_z_jet_reco", [dataset_axis, dr_axis])
            register_hist(self.hists, "dphi_z_jet_gen", [dataset_axis, dphi_axis])
            register_hist(self.hists, "dphi_z_jet_reco", [dataset_axis, dphi_axis])
            register_hist(self.hists, "dr_ll_gen", [channel_axis, dr_axis])
            register_hist(self.hists, "dr_ll_reco", [channel_axis, dr_axis])
            register_hist(self.hists, "dphi_lep_jet_reco", [dataset_axis, dphi_axis])
            
            register_hist(self.hists, "ptasym_z_jet_gen", [dataset_axis, frac_axis])
            register_hist(self.hists, "ptasym_z_jet_reco", [dataset_axis, frac_axis])
            register_hist(self.hists, "ptfrac_z_jet_gen", [dataset_axis, ptreco_axis, frac_axis])
            register_hist(self.hists, "ptfrac_z_jet_reco", [dataset_axis, ptreco_axis, frac_axis])
            register_hist(self.hists, "dr_gen_subjet", [dataset_axis, dr_axis])
            register_hist(self.hists, "dr_reco_to_gen_subjet", [dataset_axis, dr_axis])
            
            register_hist(self.hists, "jet_mass_u_presel", [dataset_axis, mreco_axis, ht_axis])
            register_hist(self.hists, "jet_mass_u_postsel", [dataset_axis, mreco_axis, ht_axis])
            register_hist(self.hists, "jet_mass_g_presel", [dataset_axis, mreco_axis, ht_axis])
            register_hist(self.hists, "jet_mass_g_postsel", [dataset_axis, mreco_axis, ht_axis])
            
            register_hist(self.hists, "m_z_reco", [dataset_axis, zmass_axis, syst_axis])
            register_hist(self.hists, "eta_z_reco", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "pt_z_reco", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "phi_z_reco", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "eta_phi_z_reco", [dataset_axis, eta_axis, phi_axis, syst_axis])
            register_hist(self.hists, "m_jet_reco", [dataset_axis, mreco_axis, syst_axis])
            register_hist(self.hists, "eta_jet_reco", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "pt_jet_reco", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "phi_jet_reco", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "eta_phi_jet_reco_preveto", [dataset_axis, eta_axis, phi_axis, syst_axis])
            register_hist(self.hists, "eta_phi_jet_reco", [dataset_axis, eta_axis, phi_axis, syst_axis])
            
            register_hist(self.hists, "fakes_u", [dataset_axis, ptreco_axis, mreco_axis])
            register_hist(self.hists, "misses_u", [dataset_axis, ptgen_axis, mgen_axis])
            register_hist(self.hists, "fakes_g", [dataset_axis, ptreco_axis, mreco_axis])
            register_hist(self.hists, "misses_g", [dataset_axis, ptgen_axis, mgen_axis])
            
            register_hist(self.hists, "ptjet_mjet_u_reco", [dataset_axis, ptreco_axis, mreco_axis, syst_axis])
            register_hist(self.hists, "ptjet_mjet_g_reco", [dataset_axis, ptreco_axis, mreco_axis, syst_axis])
            register_hist(self.hists, "ptjet_mjet_u_gen", [dataset_axis, ptgen_axis, mgen_axis])
            register_hist(self.hists, "ptjet_mjet_g_gen", [dataset_axis, ptgen_axis, mgen_axis, syst_axis])
            
            register_hist(self.hists, "m_u_jet_reco_over_gen", [dataset_axis, ptgen_axis, mgen_axis, frac_axis])
            register_hist(self.hists, "m_g_jet_reco_over_gen", [dataset_axis, ptgen_axis, mgen_axis, frac_axis])
            register_hist(self.hists, "delta_m_u", [dataset_axis, ptgen_axis, mgen_axis, binning.diff_axis])
            register_hist(self.hists, "delta_m_g", [dataset_axis, ptgen_axis, mgen_axis, binning.diff_axis])
            
            register_hist(self.hists, "pt_u_jet_reco_over_gen", [dataset_axis, ptgen_axis, mgen_axis, frac_axis])
            register_hist(self.hists, "pt_g_jet_reco_over_gen", [dataset_axis, ptgen_axis, mgen_axis, frac_axis])
            register_hist(self.hists, "delta_pt_u", [dataset_axis, ptgen_axis, mgen_axis, binning.diff_axis_large])
            register_hist(self.hists, "delta_pt_g", [dataset_axis, ptgen_axis, mgen_axis, binning.diff_axis_large])
            
            register_hist(self.hists, "response_matrix_u", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])
            register_hist(self.hists, "response_matrix_g", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])
            
            register_hist(self.hists, "jk_response_matrix_u", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis, binning.jackknife_axis])
            register_hist(self.hists, "jk_response_matrix_g", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis, binning.jackknife_axis])
            register_hist(self.hists, "jk_ptjet_mjet_g_reco", [ptreco_axis, mreco_axis, binning.jackknife_axis])
            register_hist(self.hists, "jk_ptjet_mjet_u_reco", [ptreco_axis, mreco_axis, binning.jackknife_axis])
            
            register_hist(self.hists, "m_over_pt_g", [dataset_axis, ptreco_axis, mptreco_axis], label="A.U.")
            register_hist(self.hists, "m_over_pt_u", [dataset_axis, ptreco_axis, mptreco_axis], label="A.U.")
            register_hist(self.hists, "resp_mpt_g", [dataset_axis, ptreco_axis, mptreco_axis, ptgen_axis, mptgen_axis, syst_axis], label="A.U.")
            register_hist(self.hists, "resp_mpt_u", [dataset_axis, ptreco_axis, mptreco_axis, ptgen_axis, mptgen_axis, syst_axis], label="A.U.")

        elif self.mode == "minimal":
            register_hist(self.hists, "ptjet_mjet_u_reco", [dataset_axis,channel_axis, ptreco_axis, mreco_axis, syst_axis])
            register_hist(self.hists, "ptjet_mjet_g_reco", [dataset_axis,channel_axis, ptreco_axis, mreco_axis, syst_axis])
            #register_hist(self.hists, "ptjet_mjet_u_gen", [dataset_axis, ptgen_axis, mgen_axis])
            #register_hist(self.hists, "ptjet_mjet_g_gen", [dataset_axis, ptgen_axis, mgen_axis, syst_axis])

            register_hist(self.hists, "response_matrix_u", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])
            register_hist(self.hists, "response_matrix_g", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])

            register_hist(self.hists, "fakes_u", [dataset_axis, ptreco_axis, mreco_axis, channel_axis])
            register_hist(self.hists, "misses_u", [dataset_axis, ptgen_axis, mgen_axis, channel_axis])
            register_hist(self.hists, "fakes_g", [dataset_axis, ptreco_axis, mreco_axis, channel_axis])
            register_hist(self.hists, "misses_g", [dataset_axis, ptgen_axis, mgen_axis, channel_axis])

            register_hist(self.hists, "eta_phi_jet_reco_preveto", [dataset_axis, eta_axis, phi_axis])
            register_hist(self.hists, "eta_phi_jet_reco_postveto", [dataset_axis, eta_axis, phi_axis])
            #register_hist(self.hists, "eta_phi_jet_reco", [dataset_axis, eta_axis, phi_axis, syst_axis])

            register_hist(self.hists, "pt_z_reco", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "pt_jet_reco", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "weird_eta", [eta_axis])

            #register_hist(self.hists, "ak4_jet_puID", [n_axis])
            #register_hist(self.hists, "ak4_jet_puIdDisc", [frac_axis])
            #register_hist(self.hists, "ak4_jet_neHEF", [frac_axis])
            #register_hist(self.hists, "ak4_jet_neEmEF", [frac_axis])
            #register_hist(self.hists, "problematic_dr", [dr_axis])
            
        
            
        elif self.mode == "jk_mc":
            register_hist(self.hists, "jk_response_matrix_u", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis, binning.jackknife_axis])
            register_hist(self.hists, "jk_response_matrix_g", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis, binning.jackknife_axis])
        elif self.mode == "jk_data":
            register_hist(self.hists, "jk_ptjet_mjet_g_reco", [ptreco_axis, mreco_axis, binning.jackknife_axis])
            register_hist(self.hists, "jk_ptjet_mjet_u_reco", [ptreco_axis, mreco_axis, binning.jackknife_axis])
            
        elif self.mode == "mpt":
            print("entering")
            register_hist(self.hists, "m_over_pt_g", [dataset_axis, ptreco_axis, mptreco_axis,  syst_axis, mcut_reco_g_axis], label="A.U.")
            register_hist(self.hists, "m_over_pt_u", [dataset_axis, ptreco_axis, mptreco_axis, mcut_reco_u_axis, syst_axis], label="A.U.")
            register_hist(self.hists, "resp_mpt_g", [dataset_axis, ptreco_axis, mptreco_axis, ptgen_axis, mptgen_axis,mcut_reco_g_axis, mcut_gen_g_axis,  syst_axis], label="A.U.")
            register_hist(self.hists, "resp_mpt_u", [dataset_axis, ptreco_axis, mptreco_axis, ptgen_axis, mptgen_axis,mcut_reco_u_axis,  mcut_gen_u_axis, syst_axis], label="A.U.")
            print("marker 1")
            register_hist(self.hists, "fakes_mpt_u" ,[ptreco_axis, mptreco_axis, mcut_reco_u_axis], label="Counts")
            register_hist(self.hists, "misses_mpt_u"  , [ptgen_axis, mptgen_axis,  mcut_gen_u_axis], label="Counts")
     
            register_hist(self.hists, "fakes_mpt_g",[ptreco_axis, mptreco_axis, mcut_reco_g_axis],  label="Counts")
            register_hist(self.hists, "misses_mpt_g", [ptgen_axis, mptgen_axis, mcut_gen_g_axis],  label="Counts")
            print("marker 2")
        
        cutflow = {}

        self.hists["cutflow"] = cutflow
        jackknife_total = { '0': 0, '1': 0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0 }


        
        
        # if self.minimal:
        #     self.hists = {
            
        #     "njet_gen":h_njet_gen,
        #     "total_weight": hist.Hist( weight_axis, label = 'Counts'),

        #     "fakes_u": h_fakes_u,
        #     "misses_u": h_misses_u,
        #     "fakes_g": h_fakes_g,
        #     "misses_g": h_misses_g,

        #     "m_u_jet_reco_over_gen":h_m_u_jet_reco_over_gen,
        #     "m_g_jet_reco_over_gen":h_m_g_jet_reco_over_gen,
        #     "delta_m_u":   h_delta_m_u ,
        #     "delta_m_g":   h_delta_m_g ,

        #     "pt_u_jet_reco_over_gen":h_pt_u_jet_reco_over_gen,
        #     "pt_g_jet_reco_over_gen":h_pt_g_jet_reco_over_gen,
        #     "delta_pt_u":   h_delta_pt_u ,
        #     "delta_pt_g":   h_delta_pt_g ,
        #     'm_over_pt_u':h_m_over_pt_u,
        #     'm_over_pt_g':h_m_over_pt_g,

        #     "dr_ll_gen": h_dr_ll_gen,
        #     "dr_ll_reco": h_dr_ll_reco,

        #     'ptjet_mjet_u_reco': h_ptjet_mjet_u_reco,
        #     'ptjet_mjet_g_reco': h_ptjet_mjet_g_reco,

        #     "resp_mpt_u": h_resp_mpt_u,
        #     "resp_mpt_g": h_resp_mpt_g,
        #     "response_matrix_u":h_response_matrix_u,
        #     "response_matrix_g":h_response_matrix_g,
        #     "cutflow":cutflow,

        # }
            
        # if self.minimal & (not self.do_gen):
        #     self.hists = {
            
        #     "njet_gen":h_njet_gen,
        #     "total_weight": hist.Hist( weight_axis, label = 'Counts'),


        #     "ptjet_mjet_u_reco":h_ptjet_mjet_u_reco, 

        #     "ptjet_mjet_g_reco":h_ptjet_mjet_g_reco, 


        #     'm_over_pt_u':h_m_over_pt_u,
        #     'm_over_pt_g':h_m_over_pt_g,


        #     "dr_ll_reco": h_dr_ll_reco,
        #     "cutflow":cutflow,

        # }
        
        #self.systematics = ['nominal', 'puUp', 'puDown', "elerecoUp", "elerecoDown" ] 

        
        if do_syst and (syst_list == None): # in case systematic flag is enabled but no list of systematics is provied
            self.systematics = ['nominal', 'puUp', 'puDown' , 'elerecoUp', 'elerecoDown', 
                                'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp', 'murecoDown', 
                                'muidUp', 'muidDown', 'muisoUp' , 'muisoDown', 'mutrigUp', 'mutrigDown', 
                                'pdfUp', 'pdfDown', 'q2Up', 'q2Down',
                                'prefiringUp', 'prefiringDown'] 
        elif do_syst and (syst_list != None):
            self.systematics = syst_list
        else: # no systematics
            self.systematics = ['nominal']

        
        if do_syst and (jet_syst_list == None):
            self.jet_systematics = ["nominal", "JERUp", "JERDown", "hem"]
        elif do_syst and (jet_syst_list != None):
            self.jet_systematics = jet_syst_list
        else:
            self.jet_systematics = ["nominal"]
        

        self.do_syst = do_syst
        
        self.jet_syst_list = jet_syst_list
        self.means_stddevs = defaultdict()

        
    
    @property
    def accumulator(self):
        #return self._histos
        return deepcopy(self.hists)

    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events_all):

        import time
        t0  = time.time()

        dataset = events_all.metadata['dataset']

        filename = events_all.metadata['filename']

        print(self.hists.keys())
        
        print("Initial Total Event ", len(events_all))
            
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'UL2016APV', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')

        
        
        #events_all  = events_all[(ak.num(events_all.SubJet) > 1) & (ak.num(events_all.FatJet) > 0)]
        #####################################
        #### Find the era from the file name
        #### Apply the good lumi mask
        #####################################
        if (self.do_gen):
            era = None
            firstidx = filename.find( "store/mc/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            year = fname_toks[ fname_toks.index("mc") + 1]
            ht_bin = fname_toks[ fname_toks.index("mc") + 2]

            ## Flag used for number of events
            herwig = False
            if 'herwig' in filename: herwig = True
        else:
            firstidx = filename.find( "store/data/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            era = fname_toks[ fname_toks.index("data") + 1]
            channel = fname_toks[ fname_toks.index('NANOAOD') - 1]
            ht_bin = 'all'
            print("IOV ", IOV, ", era ", era)
            lumi_mask = self.lumimasks[IOV](events_all.run, events_all.luminosityBlock)
            print("RUN", events_all.run )
            print("Lumi block", events_all.luminosityBlock )
            print("lumi_mask", lumi_mask)
            print("Len of evnets after mask", len(events_all[lumi_mask]) )
            events_all = events_all[lumi_mask]

            print("Events after lumi mask ", len(events_all))


        


        #### Setting up storage for cutflows ####
        
        if dataset not in self.hists["cutflow"]:
            self.hists["cutflow"][dataset] = {}
        if ht_bin not in self.hists["cutflow"][dataset]:
            self.hists["cutflow"][dataset][ht_bin] = defaultdict(int)

        print("The dataset is ", dataset)
        
        print("Initial Total Event ", len(events_all))

        if self.do_gen:
            if "LHEWeight" in events_all.fields:
                weights = events_all["LHEWeight"].originalXWGTUP
            else:
                weights = np.full( len( events_all ), 1.0 )
    
            if ht_bin not in self.hists["cutflow"][dataset]:
                self.hists["cutflow"][dataset][ht_bin]["sumw"] = 0
            self.hists["cutflow"][dataset][ht_bin]["sumw"] += np.sum(weights)
        

        
        index_list = np.arange(len(events_all)) ## for jackknife resampling
        for jk_index in range(0,10):  ## loops from 0 to 10 in case do_jk flag is enabled, otherwise breaks at 0
            jk_sel = ak.where(index_list%10 == jk_index, False, True) ## splitting events into 10 distinct parts

            if self.do_jk:
                print("Now doing jackknife {}".format(jk_index))
                events0 = events_all[jk_sel]
            else:
                events0 = events_all 
            del jk_sel
                      
            # if self.do_gen:
    
                
            #     #pdf uncertainty systematics len
            #     events0["pdf_N"] = GetPDFweights(events0)
            #     events0["pdf_U"] = GetPDFweights(events0, var="up")
            #     events0["pdf_D"] = GetPDFweights(events0, var="down")
    
    
            #     #q2 uncertainty systematics
            
            #     events0["q2_N"] = GetQ2weights(events0)
            #     events0["q2_U"] = GetQ2weights(events0, var="up")
            #     events0["q2_D"] = GetQ2weights(events0, var="down")
                
            #     #pileup
            #     events0["pu_nominal"] = GetPUSF(IOV, np.array(events0.Pileup.nTrueInt))
            #     events0["pu_U"]    = GetPUSF(IOV, np.array(events0.Pileup.nTrueInt), "up")
            #     events0["pu_D"]    = GetPUSF(IOV, np.array(events0.Pileup.nTrueInt), "down")
    
                
            #     ## L1PreFiringWeight
            #     events0["prefiring_N"] = GetL1PreFiringWeight(IOV, events0)
            #     events0["prefiring_U"] = GetL1PreFiringWeight(IOV, events0, "Up")
            #     events0["prefiring_D"] = GetL1PreFiringWeight(IOV, events0, "Dn")
    
    
                
            #     ## Electron Reco systematics
            #     events0["elereco_N"] = GetEleSF(IOV, "RecoAbove20", events0.Electron.eta, events0.Electron.pt)
            #     events0["elereco_U"] = GetEleSF(IOV, "RecoAbove20", events0.Electron.eta, events0.Electron.pt, "up")
            #     events0["elereco_D"] = GetEleSF(IOV, "RecoAbove20", events0.Electron.eta, events0.Electron.pt, "down")
    
    
    
            #     # Electron ID systematics/projects/TUnfoldExamples/
    
            #     events0["eleid_N"] = GetEleSF(IOV, "Tight", events0.Electron.eta, events0.Electron.pt)
            #     events0["eleid_U"] = GetEleSF(IOV, "Tight", events0.Electron.eta, events0.Electron.pt, "up")
            #     events0["eleid_D"] = GetEleSF(IOV, "Tight", events0.Electron.eta, events0.Electron.pt, "down")


            #     # Electron Trig Systematics


            #     events0["eletrig_N"] = GetEleTrigEff(IOV, events0.Electron.pt, events0.Electron.eta)
                
            #     events0["eletrig_U"] = GetEleTrigEff(IOV, events0.Electron.pt, events0.Electron.eta, var = "up")
            #     events0["eletrig_D"] = GetEleTrigEff(IOV, events0.Electron.pt, events0.Electron.eta, var = "down")
                
            #     # Muon Reco systematics
            #     events0["mureco_N"] = GetMuonSF(IOV, "RECO", np.abs(events0.Muon.eta), events0.Muon.pt) 
            #     events0["mureco_U"] = GetMuonSF(IOV, "RECO", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")
            #     events0["mureco_D"] = GetMuonSF(IOV, "RECO", np.abs(events0.Muon.eta), events0.Muon.pt, "systdown")
                
    
    
            #     ## Muon ID systematics
            #     events0["muid_N"] = GetMuonSF(IOV, "IDISO", np.abs(events0.Muon.eta), events0.Muon.pt)
            #     events0["muid_U"] = GetMuonSF(IOV, "IDISO", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")
            #     events0["muid_D"] = GetMuonSF(IOV, "IDISO", np.abs(events0.Muon.eta), events0.Muon.pt, "systdown")
    
    
            #     print("Muid and reco working")
                
            #     #q2 uncertainty systematics
            
            #     events0["q2_N"] = GetQ2weights(events0)
            #     events0["q2_U"] = GetQ2weights(events0, var="up")
            #     events0["q2_D"] = GetQ2weights(events0, var="down")
                
            #     #Muon Trigger systematics
            #     events0["mutrig_N"] = GetMuonSF(IOV, "HLT", np.abs(events0.Muon.eta), events0.Muon.pt)
            #     events0["mutrig_U"] = GetMuonSF(IOV, "HLT", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")
            #     events0["mutrig_D"] = GetMuonSF(IOV, "HLT", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")
    
                
            
            
            # else:
            #     systematic_list = ['pdf_N', 'pdf_U', 'pdf_D', 'q2_N', 'q2_U', 'q2_D', "pu_nominal", "pu_U", "pu_D", "prefiring_N", "prefiring_U", "prefiring_D", 
            #                   "elereco_N", "elereco_U", "elereco_D", "eleid_N", "eleid_U", "eleid_D" ,"eletrig_N","eletrig_U","eletrig_D" ,"mureco_N", "mureco_U", "mureco_D",  "muid_N", "muid_U", "muid_D", "mutrig_N", "mutrig_U", "mutrig_D"]
                
            #     for systematic in systematic_list:
            #         if "mu" in systematic:
            #             events0[systematic] = ak.ones_like(events0.Muon.pt)
            #         elif "ele" in systematic:
            #             events0[systematic] = ak.ones_like(events0.Electron.pt)
            #         else:
            #             events0[systematic]= ak.ones_like(events0.event, dtype=float) ## setting everything to 1 in case of DATA
    
            #####################################
            ### Initialize selection
            #####################################
            sel = PackedSelection()
            if not self.do_gen:
                sel.add("npv", events0.PV.npvsGood > 0)

            
            #####################################
            ### Trigger selection for data
            #####################################       
           
            if not self.do_gen:
                print("Channel {}".format(channel))
                if "UL2016" in dataset: 
                    if channel == "SingleMuon":
                        trigsel = events0.HLT.IsoMu24  
                    else:
                        trigsel = events0.HLT.Ele27_WPTight_Gsf | events0.HLT.Photon175
                elif "UL2017" in dataset:
                    if channel == "SingleMuon":
                        trigsel = events0.HLT.IsoMu27  
                    else:
                        trigsel = events0.HLT.Ele35_WPTight_Gsf | events0.HLT.Photon200
                elif "UL2018" in dataset:
                    if channel == "SingleMuon":
                        trigsel = events0.HLT.IsoMu24  
                    else:
                        trigsel = events0.HLT.Ele32_WPTight_Gsf | events0.HLT.Photon200
                else:
                    raise Exception("Dataset is incorrect, should have 2016, 2017, 2018: ", dataset)
                sel.add("trigsel", trigsel)    
    
                print("Trigger Selection ", ak.sum(sel.require(trigsel = True)))


            ###################################
            #### Jet corrections
            ###################################
            
            if self.do_syst and (self.jet_syst_list == None):
                for unc_src in (unc_src for unc_src in corr_jets.fields if "JES" in unc_src):
                    #print("Uncertainty source: ", unc_src)
                    #print(corr_jets[unc_src])
                    self.jet_systematics.append(unc_src+"Up")
                    self.jet_systematics.append(unc_src+"Down")
                    
                self.jet_systematics.append("JMRUp")
                self.jet_systematics.append("JMRDown")
                self.jet_systematics.append("JMSUp")
                self.jet_systematics.append("JMSDown")
            elif self.do_syst and (self.jet_syst_list != None):
                self.jet_systematics = self.jet_syst_list
            else:
                self.jet_systematics = ["nominal"]
                
                
            # corr_jets = GetJetCorrections(events0.FatJet, events0, era, IOV, isData = not self.do_gen)  ###### correcting FatJet.mass
            # corr_jets = corr_jets[corr_jets.subJetIdx1 > -1]
            # #print(" Uncorrected subjet mass", events0.SubJet.mass)
            # corr_subjets = GetJetCorrections(events0.SubJet, events0, era, IOV, isData = not self.do_gen, mode = 'AK4')
            # print(" Uncorrected softdrop mass", corr_jets.msoftdrop)
            
            # corr_jets['msoftdrop'] =   (corr_subjets[corr_jets.subJetIdx1] + corr_subjets[corr_jets.subJetIdx2]).mass 


            

    
      
            # for jet_syst in self.jet_systematics:
            #     #print("length of event in loop: " , len(events))
            #     print(" Now doing ", jet_syst)
            #     # print(jet_syst[:-2])
            #     # print(jet_syst[:-2]=="Up")
            #     # print("f")
            #     if jet_syst == "nominal":
            #         if self.do_gen:
            #             #events = ak.with_field(events0,  corr_jets , "FatJet")
                        
            #             events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV, corr_jets)) , "FatJet")
            #         else:
                        
            #             events = ak.with_field(events0, corr_jets , "FatJet")
                    
            #     elif jet_syst == "hem":
                    
            #         events = ak.with_field(events0,  HEMCleaning(IOV,jmrsf(IOV, jmssf(IOV,corr_jets))) , "FatJet")
                    
            #     elif jet_syst == "JERUp":
            #         corr_jets_obj = corr_jets.JER.up
            #         corr_jets_obj['msoftdrop'] = (corr_subjets.JER.up[corr_jets.subJetIdx1] + corr_subjets.JER.up[corr_jets.subJetIdx2]).mass
                    
            #         events = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
            #         del corr_jets_obj
    
            #     elif jet_syst == "JERDown":
            #         corr_jets_obj = corr_jets.JER.down
            #         corr_jets_obj['msoftdrop'] = (corr_subjets.JER.down[corr_jets.subJetIdx1] + corr_subjets.JER.down[corr_jets.subJetIdx2]).mass
                    
            #         events = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
            #         del corr_jets_obj
            #     elif jet_syst == "JMSUp":
            #         events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets, var = "up")) , "FatJet")
            #     elif jet_syst == "JMSDown":
            #         events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets, var = "down")) , "FatJet")
            #     elif jet_syst == "JMRUp":
            #         events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets), var = "up") , "FatJet")
            #     elif jet_syst == "JMRDown":
            #         events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets), var = "down") , "FatJet")
                
            #     elif (jet_syst[-2:]=="Up" and "JES" in jet_syst):
            #         #print(jet_syst)
            #         field = jet_syst[:-2]
            #         #print(field)
            #         corr_jets_obj = corr_jets[field].up
            #         corr_jets_obj['msoftdrop'] = (corr_subjets[field].up[corr_jets.subJetIdx1] + corr_subjets[field].up[corr_jets.subJetIdx2]).mass
                    
            #         events = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
            #         del corr_jets_obj
                    
            #     elif (jet_syst[-4:]=="Down" and "JES" in jet_syst):
            #         field = jet_syst[:-4]
            #         corr_jets_obj = corr_jets[field].down
            #         corr_jets_obj['msoftdrop'] = (corr_subjets[field].down[corr_jets.subJetIdx1] + corr_subjets[field].down[corr_jets.subJetIdx2]).mass
                    
            #         events = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
            #         del corr_jets_obj
                
            #     else:
            #         print("{} is not considered".format(jet_syst))
                    
    
            #####################################
            ### Remove events with very large gen weights (>2 sigma)
            #####################################
            
            if self.do_gen:
                if "LHEWeight"  in events_all.fields:
            
                    if dataset not in self.means_stddevs : 
                        average = np.average( events0["LHEWeight"].originalXWGTUP )
                        stddev = np.std( events0["LHEWeight"].originalXWGTUP )
                        self.means_stddevs[dataset] = (average, stddev)            
                    average,stddev = self.means_stddevs[dataset]
                    vals = (events0["LHEWeight"].originalXWGTUP - average ) / stddev
                    self.hists["cutflow"][dataset][ht_bin]["all events"] = len(events0)
                    events0 = events0[ np.abs(vals) < 2 ]
                    self.hists["cutflow"][dataset][ht_bin]["weights cut"] = len(events0)
                    
                    sel = PackedSelection(dtype='uint64') ## initialise selection for MC
                    sel.add('npv', events0.PV.npvsGood >0)
                    
                    
                    
                    #####################################
                    ### Initialize event weight to gen weight
                    #####################################
                    
                    
                    weights = events0["LHEWeight"].originalXWGTUP
                else:
                    sel = PackedSelection(dtype='uint64') ## initialise selection for MC
                    sel.add('npv', events0.PV.npvsGood >0)
                    weights = ak.ones_like(events0.PV.npvsGood, dtype = np.float64)
                #if herwig: weights = np.full( len( events ), 0.005842 )
                    
                # xs_scale = getXSweight(dataset, IOV, herwig
                if (not herwig) and (not self.do_background):
                    print("Scaling weights")
                    xs_scale = xs_scale_dic[dataset][ht_bin]
                
                    weights = weights*xs_scale
                    ptweight = ak.fill_none(pt_reweight(ak.firsts(events0.FatJet).pt), 1)
                    weights = weights*ptweight
                    del ptweight

            else:
                weights = np.full( len( events0 ), 1.0 )


            

    
    

            #####################################
            ### Gen selection
            #####################################
  
            if self.do_gen:
                #####################################
                ### Events with at least one gen jet
                #####################################
                
    
                


                #events.GenJetAK8 = events.GenJetAK8[(events.GenJetAK8.pt > 200) & (np.abs(events.GenJetAK8.eta) < 2.5)]

                

                #events.GenJetAK8 = events.GenJetAK8[(events.GenJetAK8.pt > 200) & (np.abs(events.GenJetAK8.eta) < 2.5)]

                

                # Only calculate delta_r for events with leptons; otherwise, keep all jets
                
                
                # Use ak.is_none to handle None values explicitly

                ##### working here
                ### Old deltaphi mask

                # Use ak.firsts to get the leading lepton, which will return None for events without leptons
                #leading_lepton = ak.firsts(events0.GenDressedLepton)
                #delta_phi_mask =  (np.abs(events0.GenJetAK8.delta_phi(leading_lepton)) > 0.4)
                #delta_phi_mask = ak.where(ak.is_none(delta_phi_mask), ak.full_like(events0.GenJetAK8.pt, True, dtype=bool), delta_phi_mask)

                ## Old deltaphi mask ends here
                #print(delta_phi_mask)
                
                # Apply the mask to GenJetAK8

                ############

                # print("No of gen electron", ak.sum((events0.GenDressedLepton.pdgId == -11) | (events0.GenDressedLepton.pdgId == 11), axis = 1 ))
                # print("No of gen muon", ak.sum((events0.GenDressedLepton.pdgId == -13) | (events0.GenDressedLepton.pdgId == 13), axis = 1 ))

                # import matplotlib.pyplot as plt
                # plt.hist(ak.num(events0.GenDressedLepton, axis = 1 ), alpha = 0.5)
                # plt.hist(ak.sum((events0.GenDressedLepton.pdgId == -11) | (events0.GenDressedLepton.pdgId == 11), axis = 1 ), alpha = 0.5)
                # plt.hist(ak.sum((events0.GenDressedLepton.pdgId == -13) | (events0.GenDressedLepton.pdgId == 13), axis = 1 ), alpha = 0.5)
                # plt.show()



                ##############

                events0 = ak.with_field(
                events0,
                events0.GenDressedLepton[(events0.GenDressedLepton.pt > self.lepptcuts[1]) 
                                & (np.abs(events0.GenDressedLepton.eta) < 2.5) 

                    ],
                    "GenDressedLepton"
                    )

                
                leptons = ak.pad_none(events0.GenDressedLepton, 2, clip=True)
                
                
                leading_lepton = leptons[:, 0]
                subleading_lepton = leptons[:, 1]

                delta_phi_leading_lepton = np.abs(events0.GenJetAK8.delta_r(leading_lepton)) > 0.8
                # Replace None values with a full_like boolean array (True).
                delta_phi_leading_lepton = ak.where(
                    ak.is_none(delta_phi_leading_lepton),
                    ak.full_like(events0.GenJetAK8.pt, True, dtype=bool),
                    delta_phi_leading_lepton,
                )

                delta_phi_subleading_lepton = np.abs(events0.GenJetAK8.delta_r(subleading_lepton)) > 0.8
                # Replace None values with a full_like boolean array (True).
                delta_phi_subleading_lepton = ak.where(
                    ak.is_none(delta_phi_subleading_lepton),
                    ak.full_like(events0.GenJetAK8.pt, True, dtype=bool),
                    delta_phi_subleading_lepton,
                )

                delta_phi_mask = delta_phi_leading_lepton & delta_phi_subleading_lepton

                events0 = ak.with_field(
                            events0,
                            events0.GenJetAK8[(events0.GenJetAK8.pt > 0)
                                        & (np.abs(events0.GenJetAK8.eta) < 2.5)
                                        & delta_phi_mask
                            ],
                            "GenJetAK8"
                        )

                sel.add("oneGenJet", 
                      ak.sum( (events0.GenJetAK8.pt > 0) & (np.abs(events0.GenJetAK8.eta) < 2.5), axis=1 ) >= 1
                )
                #events.GenJetAK8 = events.GenJetAK8[delta_phi_mask]



                sel.add("oneGenJet_seq", sel.all('npv', 'oneGenJet') )
                ###################################
                ### Events with no misses #########
                ###################################
    
                # matches = ak.all(events.GenJetAK8.delta_r(events.GenJetAK8.nearest(events.FatJet)) < 0.4, axis = -1)
                # misses = ~matches
    
                # sel.add("matches", matches)
                # sel.add("misses", misses)
                #print(len(ak.flatten(events[misses].GenJetAK8.pt)))
                #print(len(ak.flatten(ak.broadcast_arrays( weights[misses], events[misses].GenJetAK8.pt)[0] )))
                
                
    
                #####################################
                ### Make gen-level Z
                #####################################
                z_gen = get_z_gen_selection(events0, sel, self.lepptcuts[0], self.lepptcuts[1], None, None)
                z_ptcut_gen = ak.where( sel.all("twoGen_leptons") & ~ak.is_none(z_gen),  z_gen.pt > 90., False )
                z_mcut_gen = ak.where( sel.all("twoGen_leptons") & ~ak.is_none(z_gen),  (z_gen.mass > 71.) & (z_gen.mass < 111), False )


                ######## dr between two leptons ########
                twoGen_ee_sel = sel.require(twoGen_ee = True)
                twoGen_mm_sel = sel.require(twoGen_mm = True)

                
                
                fill_hist(self.hists, 'dr_ll_gen',channel = r'$\mu\mu$',
                                             dr = events0[twoGen_mm_sel].GenDressedLepton[:,0].delta_r(events0[twoGen_mm_sel].GenDressedLepton[:,1]),
                                             weight = weights[twoGen_mm_sel]
                                            )

                fill_hist(self.hists, 'dr_ll_gen',channel = r'$ee$',
                                             dr = events0[twoGen_ee_sel].GenDressedLepton[:,0].delta_r(events0[twoGen_ee_sel].GenDressedLepton[:,1]),
                                             weight = weights[twoGen_ee_sel]
                                            )


                ########################################
                sel.add("twoGen_leptons_seq", sel.all('twoGen_leptons', 'oneGenJet_seq') )
                
                
                sel.add("z_ptcut_gen", z_ptcut_gen)

                
                
                sel.add("z_mcut_gen", z_mcut_gen)
                sel.add("z_ptcut_gen_seq", sel.all('z_ptcut_gen', "twoGen_leptons_seq" ))
                sel.add("z_mcut_gen_seq", sel.all('z_mcut_gen', "z_ptcut_gen_seq" ))
                
                
    
                #####################################
                ### Get Gen Jet
                #####################################
                #print("zgen len ", len(z_gen))
                #print("events.GenJetAK8 len ", len(events.GenJetAK8))
                
                gen_jet, z_jet_dphi_gen = get_dphi( z_gen, events0.GenJetAK8 )
                z_jet_dr_gen = gen_jet.delta_r(z_gen)
    
                
    
                #####################################
                ### Gen event topology selection
                #####################################        
                z_pt_asym_gen = np.abs(z_gen.pt - gen_jet.pt) / (z_gen.pt + gen_jet.pt)
                z_pt_frac_gen = gen_jet.pt / z_gen.pt
                z_pt_asym_sel_gen =  z_pt_asym_gen < 0.3
                z_jet_dphi_sel_gen = z_jet_dphi_gen > 1.57 #2.8 #np.pi * 0.5
                sel.add("z_jet_dphi_sel_gen", z_jet_dphi_sel_gen)
                sel.add("z_pt_asym_sel_gen", z_pt_asym_sel_gen)

                sel.add("z_jet_dphi_sel_gen_seq", sel.all("z_mcut_gen_seq", "z_jet_dphi_sel_gen") )
                sel.add( "z_pt_asym_sel_gen_seq", sel.all("z_jet_dphi_sel_gen_seq", "z_pt_asym_sel_gen") )
                
                #####################################
                ### Make gen plots with Z and jet cuts
                #####################################
                kinsel_gen = sel.require(twoGen_leptons=True,oneGenJet=True,z_ptcut_gen=True,z_mcut_gen=True)
                sel.add("kinsel_gen", kinsel_gen)
                toposel_gen = sel.require( z_pt_asym_sel_gen=True, z_jet_dphi_sel_gen=True)
                sel.add("toposel_gen", toposel_gen)



                
                # fill_hist(self.hists, "ptz_gen",dataset=dataset,
                #                           pt=z_gen[kinsel_gen].pt,
                #                           weight=weights[kinsel_gen])
                # fill_hist(self.hists, "mz_gen",dataset=dataset,
                #                           mass=z_gen[kinsel_gen].mass,
                #                           weight=weights[kinsel_gen])
                # fill_hist(self.hists, "njet_gen",dataset=dataset,
                #                             n=ak.num(events[kinsel_gen].GenJetAK8),
                #                             weight = weights[kinsel_gen] )
    
                # There are None elements in these arrays when the reco_jet is not found.
                # To make "N-1" plots, we need to reduce the size and remove the Nones
                # otherwise the functions will throw exception.
                weights2 = weights[ ~ak.is_none(gen_jet) & kinsel_gen]
                z_jet_dr_gen2 = z_jet_dr_gen[ ~ak.is_none(gen_jet) & kinsel_gen]
                z_pt_asym_sel_gen2 = z_pt_asym_sel_gen[~ak.is_none(gen_jet) & kinsel_gen]
                z_pt_asym_gen2 = z_pt_asym_gen[~ak.is_none(gen_jet) & kinsel_gen]
                z_jet_dphi_gen2 = z_jet_dphi_gen[~ak.is_none(gen_jet) & kinsel_gen]
                z_pt_frac_gen2 = z_pt_frac_gen[~ak.is_none(gen_jet) & kinsel_gen]
                z_jet_dphi_sel_gen2 = z_jet_dphi_sel_gen[~ak.is_none(gen_jet) & kinsel_gen]
    
                # Making N-1 plots for these three
                # fill_hist(self.hists, "dr_z_jet_gen", dataset=dataset,
                #                                   dr=z_jet_dr_gen2[z_pt_asym_sel_gen2],
                #                                   weight=weights2[z_pt_asym_sel_gen2])
                # fill_hist(self.hists, "dphi_z_jet_gen",dataset=dataset, 
                #                                    dphi=z_jet_dphi_gen2[z_pt_asym_sel_gen2], 
                #                                    weight=weights2[z_pt_asym_sel_gen2])
                # fill_hist(self.hists, "ptasym_z_jet_gen",dataset=dataset, 
                #                                      frac=z_pt_asym_gen2[z_jet_dphi_sel_gen2],
                #                                      weight=weights2[z_jet_dphi_sel_gen2])
                # fill_hist(self.hists, "ptfrac_z_jet_gen",dataset=dataset, 
                #                                      ptreco=z_gen[z_jet_dphi_sel_gen2].pt,
                #                                      frac=z_pt_frac_gen2[z_jet_dphi_sel_gen2],
                #                                      weight=weights2[z_jet_dphi_sel_gen2])
            
                #####################################
                ### Get gen subjets 
                #####################################
                gensubjets = events0.SubGenJetAK8
                groomed_gen_jet, groomedgensel = get_groomed_jet(gen_jet, gensubjets, False)
    
                #####################################
                ### Convenience selection that has all gen cuts
                #####################################
                allsel_gen = sel.all("npv", "kinsel_gen", "toposel_gen" )
                sel.add("allsel_gen", allsel_gen)
                print("How many in GEN?", np.sum(allsel_gen))
                print("How many GEN in mumu", np.sum(allsel_gen & twoGen_mm_sel))
                print("How many GEN in ee", np.sum(allsel_gen & twoGen_ee_sel))
                



            
                #####################################
                ### Plots for gen jets and subjets
                #####################################
                # fill_hist(self.hists, "ptjet_gen_pre",dataset=dataset, 
                #                              pt=gen_jet[allsel_gen].pt, 
                #                              weight=weights[allsel_gen])
                # fill_hist(self.hists, "dr_gen_subjet",dataset=dataset,
                #                                  dr=groomed_gen_jet[allsel_gen].delta_r(gen_jet[allsel_gen]),
                #                                  weight=weights[allsel_gen])

#                     misses = sel.all("npv", "kinsel_gen", "toposel_gen" , "misses")
#                     fill_hist(self.hists, "misses",dataset = dataset, ptgen= ak.flatten(events[misses].GenJetAK8.pt),
#                                                       mgen = ak.flatten(events[misses].GenJetAK8.mass),  weight = ak.flatten(ak.broadcast_arrays( weights[misses], events[misses].GenJetAK8.pt)[0] ) )
                #del z_jet_dr_gen2, z_pt_asym_sel_gen2, z_pt_asym_gen2, z_pt_frac_gen2, z_jet_dphi_sel_gen2

            
            #####################################
            ### Make reco-level Z
            #####################################
            
            #plot_pt_threshold(events0)
            #plot_nlepton(events0)

            ##########################################

            # events0 = ak.with_field(
            #     events0,
            #     events0.Electron[(events0.Electron.pt > self.lepptcuts[0]) 
            #                     & (np.abs(events0.Electron.eta) < 2.5) 
            #                     & (events0.Electron.pfRelIso03_all < 0.25)  # supressing isolation cut here
            #                     & (events0.Electron.cutBased > 3) ## TightId == 4 , mediumId == 3, looseId == 2
            #     ],
            #     "Electron_nocut"
            # )
            # events0 = ak.with_field(
            #     events0,
            #     events0.Muon[(events0.Muon.pt > self.lepptcuts[1]) 
            #                 &(np.abs(events0.Muon.eta) < 2.5)
                
            #     ],
            #     "Muon_nocut"
            # )

            
            events0 = ak.with_field(
                events0,
                events0.Electron[(events0.Electron.pt > self.lepptcuts[0]) 
                                & (np.abs(events0.Electron.eta) < 2.5) 
                                & (events0.Electron.pfRelIso03_all < 0.25)  # supressing isolation cut here
                                & (events0.Electron.cutBased > 3) ## TightId == 4 , mediumId == 3, looseId == 2
                ],
                "Electron"
            )

            

            

            
            events0 = ak.with_field(
                events0,
                events0.Muon[(events0.Muon.pt > self.lepptcuts[1]) 
                            &(np.abs(events0.Muon.eta) < 2.5)
                            &(events0.Muon.pfIsoId > 1) #medium iso, pfIso04 < 0.2 , 2 = loose, 3 = medium # supressing isolation cut here
                            #&(events0.Muon.miniIsoId > 1)
                            &(events0.Muon.mediumId	 == True)
                            #&(events0.Muon.looseId	 == True)
                
                ],
                "Muon"
            )


            

            ########### fitering muon jets end

            ###################################
            # events0 = ak.with_field(
            #     events0,
            #     events0.Electron[(events0.Electron.pt > 20) 
            #                     & (np.abs(events0.Electron.eta) < 2.5) 
            #                     & (events0.Electron.pfRelIso03_all < 0.3)  #_mvaFall17V2Iso_WP90
            #                     #& (events0.Electron.mvaFall17V2Iso_WP80 == True)
            #                     #& (events0.Electron.cutBased > 2) ## TightId == 4 , mediumId == 3, looseId == 2
            #     ],
            #     "Electron"
            # )

            

            
            # events0 = ak.with_field(
            #     events0,
            #     events0.Muon[(events0.Muon.pt > 20) 
            #                 &(np.abs(events0.Muon.eta) < 2.5)
            #                 &(events0.Muon.miniIsoId > 1) #medium iso, pfIso04 < 0.2 , 2 = loose, 3 = medium 
            #                 &(events0.Muon.mediumId	 == True)
            #                 #&(events0.Muon.tightId	 == True)
            #                 #&(events0.Muon.looseId	 == True)
                
            #     ],
            #     "Muon"
            # )
            #########################################################################
            #plot_nlepton(events0)
            #print("Number of electrons avoiding pt cut", ak.sum(ak.num(events0.Electron[ (events0.Electron.pt < self.lepptcuts[0])], axis = 1)))
            z_reco = get_z_reco_selection(events0, sel, self.lepptcuts[0], self.lepptcuts[1], None, None)
            z_ptcut_reco = z_reco.pt > 90
            z_mcut_reco = (z_reco.mass > 71.) & (z_reco.mass < 111.)
            sel.add("z_ptcut_reco", z_ptcut_reco & (sel.require(twoReco_leptons = True) ))
            sel.add("z_mcut_reco", z_mcut_reco & (sel.require(twoReco_leptons = True) ))


            #### dr reco plots ###

            twoReco_ee_sel = sel.require(twoReco_ee = True)
            twoReco_mm_sel = sel.require(twoReco_mm = True)

            fill_hist(self.hists, "dr_ll_reco",channel = r"$\mu\mu$ Pre-sel",
                                          dr = events0[twoReco_mm_sel].Muon[:, 0].delta_r(events0[twoReco_mm_sel].Muon[:, 1]),
                                          weight = weights[twoReco_mm_sel])
            fill_hist(self.hists, "dr_ll_reco",channel = r"$ee$ Pre-sel",
                                          dr = events0[twoReco_ee_sel].Electron[:, 0].delta_r(events0[twoReco_ee_sel].Electron[:, 1]),
                                          weight = weights[twoReco_ee_sel])



            #######################
            ########### filtering muon jets
            ##############


            # For electrons:
            # Pad the Electron array to exactly 2 entries per event.
            electrons = ak.pad_none(events0.Electron, 2, clip=True)

            #print("Number of two electron events", np.sum(ak.num(events0.Electron, axis = 1) == 2))
            #print("Number of two electron events (new object)", np.sum(ak.num(electrons, axis = 1) == 2))
            leading_electron = electrons[:, 0]
            subleading_electron = electrons[:, 1]

            #print("LEading Electron pt", leading_electron.pt[leading_electron.pt > 0 ])

            #print("Subleading Electron pt", subleading_electron.pt[subleading_electron.pt > 0 ])
            
            # Compute delta_phi for the leading electron.
            delta_phi_leading_electron = np.abs(events0.FatJet.delta_r(leading_electron)) > 0.8
            # Replace None values with a full_like boolean array (True).
            delta_phi_leading_electron = ak.where(
                ak.is_none(delta_phi_leading_electron),
                ak.full_like(events0.FatJet.pt, True, dtype=bool),
                delta_phi_leading_electron,
            )
            
            # Compute delta_phi for the subleading electron.
            delta_phi_subleading_electron = np.abs(events0.FatJet.delta_r(subleading_electron)) > 0.8
            # Replace None values with a full_like boolean array (True).
            delta_phi_subleading_electron = ak.where(
                ak.is_none(delta_phi_subleading_electron),
                ak.full_like(events0.FatJet.pt, True, dtype=bool),
                delta_phi_subleading_electron,
            )
            
            # Combine both electron delta_phi masks.
            delta_phi_mask_electron = delta_phi_leading_electron & delta_phi_subleading_electron
            
            
            # For muons:
            # Pad the Muon array to exactly 2 entries per event.
            muons = ak.pad_none(events0.Muon, 2, clip=True)
            leading_muon = muons[:, 0]
            subleading_muon = muons[:, 1]


            #print("LEading Muon pt", leading_muon.pt[leading_muon.pt > 0 ])

            #print("Subleading Muon pt", subleading_muon.pt[subleading_muon.pt > 0 ])

            
            # Compute delta_phi for the leading muon.
            delta_phi_leading_muon = np.abs(events0.FatJet.delta_r(leading_muon)) > 0.8
            # Replace None values with a full_like boolean array (True).
            delta_phi_leading_muon = ak.where(
                ak.is_none(delta_phi_leading_muon),
                ak.full_like(events0.FatJet.pt, True, dtype=bool),
                delta_phi_leading_muon,
            )
            
            # Compute delta_phi for the subleading muon.
            delta_phi_subleading_muon = np.abs(events0.FatJet.delta_r(subleading_muon)) > 0.8
            # Replace None values with a full_like boolean array (True).
            delta_phi_subleading_muon = ak.where(
                ak.is_none(delta_phi_subleading_muon),
                ak.full_like(events0.FatJet.pt, True, dtype=bool),
                delta_phi_subleading_muon,
            )
            
            # Combine both muon delta_phi masks.
            delta_phi_mask_muon = delta_phi_leading_muon & delta_phi_subleading_muon
            
            
            print("Delta phi mask muon", delta_phi_mask_muon)
            ###########
            ##########
            ## ChatGPT End
            ###########


            # Compute delta R between jets and the leading muon
            #print("Delta r mask", delta_phi_mask)
            
            #Define the delta R mask (keep jets far from the leading electron)
            

            #corr_jets = corr_jets[corr_jets.subJetIdx1 > -1]
            
            #Apply selection for recojets along with filtering out jets close to Muons
            recojets = events0.FatJet[
                (events0.FatJet.subJetIdx1 > -1) 
                & (events0.FatJet.mass > 0) 
                &(events0.FatJet.pt > 200) 
                &(np.abs(events0.FatJet.eta) < 2.5) 
                &(events0.FatJet.jetId > 1) 
                &delta_phi_mask_electron 
                &delta_phi_mask_muon
            ]

            

            events0 = ak.with_field(events0,  recojets, "FatJet")
            #applying muon and electron selections after the Jet filtering has been done
            


            
            #register_hist(self.hists, "weird_eta", [eta_axis])
            del delta_phi_mask_electron, delta_phi_mask_muon
            
            #####################################
            ### Reco jet selection
            #####################################
            #recojets = events.FatJet[(events.FatJet.pt > 200) & (np.abs(events.FatJet.eta) < 2.5)& (events.FatJet.jetId > 3)]

            ##########
            ## Aritra's selection
            ###########
            # leading_electron = ak.firsts(events0.Electron)
            # delta_phi_mask_electron = np.abs(events0.FatJet.delta_phi(leading_electron)) > 0.4
            
            # delta_phi_mask_electron = ak.where(ak.is_none(delta_phi_mask_electron), ak.full_like(events0.FatJet.pt, True, dtype=bool), delta_phi_mask_electron)
            # # Compute delta R between jets and the leading electron
            # #print("Delta r mask", delta_phi_mask_electron)

            # leading_muon = ak.firsts(events0.Muon)
            # delta_phi_mask = np.abs(events0.FatJet.delta_phi(leading_muon)) > 0.4
            # #print("Delta r mask", delta_phi_mask)
            # delta_phi_mask_muon = ak.where(ak.is_none(delta_phi_mask), ak.full_like(events0.FatJet.pt, True, dtype=bool), delta_phi_mask)

            ##########
            ## Aritra's selection
            ###########




            
            # recojets = events.FatJet[
            #                     (events.FatJet.pt > 200) &
            #                     (np.abs(events.FatJet.eta) < 2.5) &
            #                     (events.FatJet.jetId > 3) 
            #                 ]
            


            
            #& (events.FatJet.muonIdx3SJ == -1)  ] # &  get_dR( z_reco, events.FatJet )>0.8
            # jetid = reco_jet.jetId > 1
            # sel.add("jetid", jetid)

            hem_sel = HEMVeto(recojets, events0.run)
            
            sel.add("oneRecoJet", 
                 ak.sum( (events0.FatJet.pt > 200) & (np.abs(events0.FatJet.eta) < 2.5)  & (events0.FatJet.jetId == 6) & hem_sel , axis=1 ) >= 1
            )
            
            sel.add("oneRecoJet_seq", sel.all('oneRecoJet', 'npv') )
            sel.add("twoReco_leptons_seq", sel.all('twoReco_leptons', 'oneRecoJet') )
            sel.add("z_ptcut_reco_seq", sel.all('z_ptcut_reco', "twoReco_leptons_seq" ))
            sel.add("z_mcut_reco_seq", sel.all('z_mcut_reco', "z_ptcut_reco_seq" ))

            #recojets = apply_lepton_separation(recojets, events.Muon, events.Electron)

            

            

            #####################################
            # Find reco jet opposite the reco Z
            #####################################
            
    
            reco_jet, z_jet_dphi_reco = get_dphi( z_reco, recojets )


            reco_jet = ak.firsts(recojets)

            
            #print("Special event reco_jet object pt", reco_jet[sel_spl].pt)
            #print("Special event FatJets object pt, eta, jetid", events0[sel_spl].FatJet.pt, events0[sel_spl].FatJet.eta, events0[sel_spl].FatJet.jetId)

            #print("Special event FatJets object pt, eta, jetid", recojets[sel_spl].pt, recojets[sel_spl].eta, recojets[sel_spl].jetId)
            #reco_jet, dr = find_closest_dr( gen_jet, recojets )
            z_jet_dr_reco = reco_jet.delta_r(z_reco)
            z_jet_dphi_reco_values = z_jet_dphi_reco

            ####### MAKE PRESEL PLOTS ######
            filter_sel = sel.all('npv', 'oneRecoJet')
            reco_exists = ~ak.is_none(reco_jet.mass)

            
            #####################################
            ### Reco event topology sel
            #####################################
            z_jet_dphi_sel_reco = (z_jet_dphi_reco > 1.57)  #& (sel.require(twoReco_leptons = True))#np.pi * 0.5
            z_pt_asym_reco = np.abs(z_reco.pt - reco_jet.pt) / (z_reco.pt + reco_jet.pt)
            z_pt_frac_reco = reco_jet.pt / z_reco.pt
            z_pt_asym_sel_reco = (z_pt_asym_reco < 0.3) #& (sel.require(twoReco_leptons = True))
            sel.add("z_jet_dphi_sel_reco", z_jet_dphi_sel_reco)
            sel.add("z_pt_asym_sel_reco", z_pt_asym_sel_reco)


            
            kinsel_reco = sel.require(twoReco_leptons=True,oneRecoJet=True,z_ptcut_reco=True,z_mcut_reco=True)#, jetid = True)
            sel.add("kinsel_reco", kinsel_reco)
            print("Leading RECO Jet matched muon ID", reco_jet.muonIdx3SJ)
            #print("Z-Jet dphi cut ",  sel.require(kinsel_reco= True, z_jet_dphi_sel_reco= True,trigsel= True ).sum())
            #print("Z-Jet pt-asymmetry cut ",  sel.require(kinsel_reco= True,z_jet_dphi_sel_reco= True, z_pt_asym_sel_reco= True,trigsel= True ).sum())

            
            toposel_reco = sel.require( z_pt_asym_sel_reco=True, z_jet_dphi_sel_reco=True)
            sel.add("toposel_reco", toposel_reco)
    
            
            sel.add("z_jet_dphi_sel_reco_seq", sel.all("z_mcut_reco_seq", "z_jet_dphi_sel_reco") )
            sel.add( "z_pt_asym_sel_reco_seq", sel.all("z_jet_dphi_sel_reco_seq", "z_pt_asym_sel_reco") )
            #sel.add("jetid_seq", sel.all("z_pt_asym_sel_reco_seq", "jetid") )

            
            # Note: Trigger is not applied in the MC, so this is 
            # applying the full gen selection here to be in sync with rivet routine
            if self.do_gen:
                presel_reco = sel.all("npv", "allsel_gen", "kinsel_reco")
            else:
                presel_reco = sel.all("npv", "trigsel", "kinsel_reco")

            

            if self.do_gen:
                delta_R = reco_jet.delta_r(gen_jet)
                misses = delta_R > 0.4
                sel.add("misses", misses)
                sel.add("matches", ~misses)
                #misses = sel.all("misses")
                matches = ~misses 
                sel.add("matches", matches)

            #print("How many RECO independent of GEN?", ak.sum(sel.all('npv', 'kinsel_reco', 'toposel_reco')))
            print("How many RECO independent of GEN", ak.sum(sel.all('npv', 'kinsel_reco', 'toposel_reco')))

            
            #allsel_reco = presel_reco & toposel_reco
            sel.add("presel_reco", presel_reco)

            allsel_reco = sel.all('presel_reco', 'toposel_reco' )
            
            sel.add("allsel_reco", allsel_reco)
            if not self.minimal:
                fill_hist(self.hists, "mz_reco",dataset=dataset, mass=z_reco[presel_reco].mass, 
                                           weight=weights[presel_reco])
                if self.do_gen:
                    fill_hist(self.hists, "mz_reco_over_gen",dataset=dataset, 
                                                        frac=z_reco[presel_reco].mass / z_gen[presel_reco].mass, 
                                                        weight=weights[presel_reco] )

            ###### Checking efficiency of each cut GEN vs RECO
            # z_ptcut_reco
            # z_mcut_reco
            # z_jet_dphi_sel_reco
            # z_pt_asym_sel_reco
            
            # There are None elements in these arrays when the reco_jet is not found.
            # To make "N-1" plots, we need to reduce the size and remove the Nones
            # otherwise the functions will throw exception.
            weights3 = weights[ ~ak.is_none(reco_jet)]
            presel_reco3 = presel_reco[~ak.is_none(reco_jet)]
            z_jet_dr_reco3 = z_jet_dr_reco[ ~ak.is_none(reco_jet)]
            z_pt_asym_sel_reco3 = z_pt_asym_sel_reco[~ak.is_none(reco_jet)]
            z_pt_asym_reco3 = z_pt_asym_reco[~ak.is_none(reco_jet)]
            z_pt_frac_reco3 = z_pt_frac_reco[~ak.is_none(reco_jet)]
            z_jet_dphi_reco3 = z_jet_dphi_reco[~ak.is_none(reco_jet)]
            z_jet_dphi_sel_reco3 = z_jet_dphi_sel_reco[~ak.is_none(reco_jet)]

            if not self.minimal:
                # Making N-1 plots for these three
                fill_hist(self.hists, "dr_z_jet_reco", dataset=dataset,
                                                  dr=z_jet_dr_reco3[presel_reco3 & z_pt_asym_sel_reco3],
                                                  weight=weights3[presel_reco3 & z_pt_asym_sel_reco3])
                # fill_hist(self.hists, "dphi_z_jet_reco",dataset=dataset, 
                #                                    dphi=z_jet_dphi_reco3[presel_reco3 & z_pt_asym_sel_reco3], 
                #                                    weight=weights3[presel_reco3 & z_pt_asym_sel_reco3])
                fill_hist(self.hists, "ptasym_z_jet_reco",dataset=dataset, 
                                                     frac=z_pt_asym_reco3[presel_reco3 & z_jet_dphi_sel_reco3],
                                                     weight=weights3[presel_reco3 & z_jet_dphi_sel_reco3])
                fill_hist(self.hists, "ptfrac_z_jet_reco",dataset=dataset, 
                                                     ptreco=z_reco[presel_reco3 & z_jet_dphi_sel_reco3].pt,
                                                     frac=z_pt_frac_reco3[presel_reco3 & z_jet_dphi_sel_reco3],
                                                     weight=weights3[presel_reco3 & z_jet_dphi_sel_reco3])
            

            del z_jet_dr_reco3, z_pt_asym_sel_reco3, z_pt_asym_reco3, z_pt_frac_reco3, z_jet_dphi_sel_reco3, z_jet_dphi_reco3, z_pt_asym_reco, z_pt_asym_sel_reco

            
            ### modify for groomed (by default this will be true, redundant)
            jet_mass_groomed_sel = reco_jet.msoftdrop > -100
            sel.add("jet_mass_groomed_sel", jet_mass_groomed_sel)
            
            
            
            if self.do_gen:
                #fakes = ak.any(ak.is_none(events.FatJet.matched_gen, axis = -1), axis = -1)  ## not true
                fakes = ak.is_none(events0.GenJetAK8[reco_jet.genJetAK8Idx].pt)
                sel.add("fakes", fakes)
                fakes_2 = sel.require(kinsel_reco = True, toposel_reco = True,   kinsel_gen = False, toposel_gen = False, jet_mass_groomed_sel = True )
                fakes_3 = sel.require( z_pt_asym_sel_reco_seq= True,  matches = False)
                sel.add("fakes_2", fakes_2|fakes_3)
                
                print("How many events passed reco, but failed GEN? aka fakes", np.sum(fakes_2|fakes_3)) 
                

                
                
                
                # print("fake reco jets", reco_jet[fakes])
                ## small number of reco jet pt after correction was bleeding into pt<200 region, hard cut to prevent that
                pt200 = reco_jet.pt > 200   
                sel.add("pt200", ~pt200)
                del pt200
                
                fakes = sel.any('fakes', 'fakes_2')
                matched_reco = sel.require(fakes = False)
                sel.add("matched_reco", matched_reco)
                
                fakes_sel = fakes & (~ak.is_none(reco_jet.matched_gen.mass))
                fakes_sel_mm = fakes & (~ak.is_none(reco_jet.matched_gen.mass)) & twoReco_mm_sel
                fakes_sel_ee = fakes & (~ak.is_none(reco_jet.matched_gen.mass)) & twoReco_ee_sel
                
                if np.sum(fakes_sel)>0:
                    print("Enters fakes part")

                    if np.sum(fakes_sel_mm) > 0:
                        fill_hist(self.hists, "fakes_u",dataset = dataset, channel = 'mm',
                                             ptreco = reco_jet[fakes_sel_mm].pt,
                                             mreco = reco_jet[fakes_sel_mm].mass,
                                             weight = weights[fakes_sel_mm])
                        fill_hist(self.hists, "fakes_g",dataset = dataset, channel = 'mm',
                                                 ptreco = reco_jet[fakes_sel_mm].pt,
                                                 mreco = reco_jet[fakes_sel_mm].msoftdrop,
                                                 weight = weights[fakes_sel_mm])
                        
                    if np.sum(fakes_sel_ee) > 0:
                        
                        fill_hist(self.hists, "fakes_u",dataset = dataset, channel = 'ee',
                                             ptreco = reco_jet[fakes_sel_ee].pt,
                                             mreco = reco_jet[fakes_sel_ee].mass,
                                             weight = weights[fakes_sel_ee])
                        fill_hist(self.hists, "fakes_g",dataset = dataset, channel = 'ee',
                                                 ptreco = reco_jet[fakes_sel_ee].pt,
                                                 mreco = reco_jet[fakes_sel_ee].msoftdrop,
                                                 weight = weights[fakes_sel_ee])
    
                        



                    
                    fill_hist(self.hists,"fakes_mpt_u",
                                             ptreco = reco_jet[fakes_sel].pt,
                                             mreco = reco_jet[fakes_sel].mass,
                                             mpt_reco = 2*np.log10(reco_jet[fakes_sel].mass/(reco_jet[fakes_sel].pt*0.8)),
                                             weight = weights[fakes_sel])
                    fill_hist(self.hists,"fakes_mpt_g",
                                             ptreco = reco_jet[fakes_sel].pt,
                                             mreco = reco_jet[fakes_sel].msoftdrop,
                                             mpt_reco = 2*np.log10(reco_jet[fakes_sel].msoftdrop/(reco_jet[fakes_sel].pt*0.8)),
                                             weight = weights[fakes_sel])


                
                misses_1 = sel.require(allsel_gen = True, allsel_reco = False)
                misses_2 = sel.require(allsel_gen = True, misses = True)
                misses_3 = sel.require(allsel_gen = True, pt200 = True)



                
                misses = misses_1 | misses_2 | misses_3  ## basically cosidering the jets which passes both reco & gen however the delta R between them is >0.4
                del misses_1, misses_2, misses_3
                #print("How many event PASSES GEN, fails RECO ", np.sum(misses_1))
                
                print("How many misses? ", np.sum(misses  ))
                print("Event numbers for misses", events0[misses].event)
                

                #print("How many event passes both reco and gen, however fails to pass R<0.4 ", np.sum(sel.require(allsel_reco = True, matches = False)))
                #print("How many event passes gen, however fails to pass R<0.4 ", np.sum(misses_2))
                #print("How many event passes RECO, however fails to pass R<0.4 ", np.sum(sel.require(z_pt_asym_sel_reco_seq= True, matches = False)))
                allsel_reco = sel.require(allsel_reco = True, allsel_gen = True, matched_reco = True, jet_mass_groomed_sel = True, pt200 = False, matches= True) ## final selection
                allsel_reco = allsel_reco & (~ak.is_none(reco_jet.matched_gen.mass))
                sel.add("final_selection", allsel_reco)
                print("How many allsel", np.sum(allsel_reco))
                #print("How many gen_jet does not have mass? ", sel.all("allsel_gen") & ak.is_none(gen_jet.mass))
                miss_sel = (misses  | sel.require(z_pt_asym_sel_reco_seq= True, matches = False) ) & (~ak.is_none(gen_jet.mass)) #& (events.event == 9332270))
                miss_sel_ee = miss_sel & twoGen_ee_sel
                miss_sel_mm = miss_sel & twoGen_mm_sel
                if ak.sum(miss_sel) > 0 :
                    if ak.sum(miss_sel_mm) > 0:
                        fill_hist(self.hists, "misses_u",dataset = dataset, ptgen= gen_jet[miss_sel_ee].pt,channel = 'mm',
                                                  mgen = gen_jet[miss_sel_ee].mass,  weight =  weights[miss_sel_ee] ) 
                        fill_hist(self.hists, "misses_g",dataset = dataset, ptgen= gen_jet[miss_sel_ee].pt,channel = 'mm',
                                              mgen = groomed_gen_jet[miss_sel_ee].mass,  weight =  weights[miss_sel_ee] ) 

                    if ak.sum(miss_sel_ee) > 0:
                        fill_hist(self.hists, "misses_u",dataset = dataset, ptgen= gen_jet[miss_sel].pt, channel = 'ee',
                                                  mgen = gen_jet[miss_sel].mass,  weight =  weights[miss_sel] ) 
                        fill_hist(self.hists, "misses_g",dataset = dataset, ptgen= gen_jet[miss_sel].pt, channel = 'ee',
                                                  mgen = groomed_gen_jet[miss_sel].mass,  weight =  weights[miss_sel] ) 

                    fill_hist(self.hists,"misses_mpt_u", ptgen= gen_jet[miss_sel].pt,
                                              mgen = gen_jet[miss_sel].mass, mpt_gen = gen_jet[miss_sel].mass/(gen_jet[miss_sel].pt*0.8) , weight =  weights[miss_sel] ) 
                    fill_hist(self.hists,"misses_mpt_g", ptgen= gen_jet[miss_sel].pt,
                                              mgen = groomed_gen_jet[miss_sel].mass, mpt_gen = groomed_gen_jet[miss_sel].mass/(gen_jet[miss_sel].pt*0.8), weight =  weights[miss_sel] ) 
                    
                    del miss_sel_ee, miss_sel_mm         
            else:
                allsel_reco = sel.all("allsel_reco", "jet_mass_groomed_sel") 
                sel.add("final_selection", allsel_reco)
            
            
            if self.do_gen:
                print("How many of these are not matches? ", ak.sum((reco_jet[allsel_reco ].matched_gen.pt - gen_jet[allsel_reco ].pt) > 1 ))
            ### Cut down arrays after final reco selection
            
            # print("Events that pass GEN, fail RECO")
            # print(" --------------------------------------- ")
            # print(" --------------------------------------- ")
            # miss_sel = misses
            # events_m = events0[miss_sel ]
            # weights_m = weights[miss_sel]
            # z_reco_m = z_reco[miss_sel]
            # reco_jet_m = reco_jet[miss_sel]

            # print("Missed RECO jets matched muon ID", reco_jet_m.muonIdx3SJ)


            # #weights = weights[allsel_reco]
            # if self.do_gen:
            #     z_gen_m = z_gen[miss_sel]
            #     gen_jet_m = gen_jet[miss_sel]
            #     groomed_gen_jet_m = groomed_gen_jet[miss_sel]

            
            # print("Number of electrons avoiding pt cut", ak.sum(ak.num(events_m.Electron[ (events_m.Electron.pt < self.lepptcuts[0])], axis = 1)))
            # print("Lepton pt", events_m.GenDressedLepton.pt)
            # #print("Muon pt", events.Muon.pt)
            # print("Z pt", z_gen_m.pt)
            # print("Jet pt", gen_jet_m.pt)

            # #print("Events run number", events.event)
            # print("Lepton phi", events_m.GenDressedLepton.phi)
            # #print("Muon phi", events.Muon.phi)
            # print("Z phi", z_gen_m.phi)
            # print("Jet phi", gen_jet_m.phi)

            
            # print("Reco Now")
            # print(" --------------------------------------- ")
            # #print("Events run number", events.event)
            # print("Electron pt", events_m.Electron.pt)
            # print("Muon pt", events_m.Muon.pt)
            # print("Z pt", z_reco_m.pt)
            # print("Z mass", z_reco_m.mass)
            # print("Jet pt", reco_jet_m.pt)
            # print("All jets  jetID?", recojets[miss_sel].jetId)
            # print("All Jets pt", recojets[miss_sel].pt.to_list())
            # print("All Jets phi", recojets[miss_sel].phi.to_list())
            #print("Event number", events_m.event)

            # #print("Events run number", events.event)
            # print("Electron phi", events_m.Electron.phi)
            # print("Muon phi", events_m.Muon.phi)
            # print("Z phi", z_reco_m.phi)
            # print("Jet phi", reco_jet_m.phi)

    


            # # Testing out which events fail

            # import pickle
            
            # # Assuming events_m, gen_jet_m, z_gen, z_reco_m, reco_jet_m are already defined
            # # These objects should be arrays from a framework like uproot or coffea, containing pt and phi properties
            
            # # Unflatten gen and reco attributes for jets and Z bosons to match the event structure
            # num_particles_per_event = len(events_m)  # Number of entries per event for unflattening
            
            # # Unflatten gen and reco jet and Z boson arrays to be consistent with event structure
            # gen_jet_pt = gen_jet_m.pt
            # gen_jet_phi = gen_jet_m.phi
            
            # reco_jet_pt = reco_jet_m.pt
            # reco_jet_phi = reco_jet_m.phi
            
            # z_gen_pt = z_gen_m.pt
            # z_gen_phi = z_gen_m.phi
            
            # z_reco_pt = z_reco_m.pt
            # z_reco_phi = z_reco_m.phi
            
            # # Define separate dictionaries for leptons and for jets and Z bosons
            # lepton_data = {
            #     "GenDressedLepton": {
            #         "pt": events_m.GenDressedLepton.pt,
            #         "phi": events_m.GenDressedLepton.phi
            #     },
            #     "Electron": {
            #         "pt": events_m.Electron.pt,
            #         "phi": events_m.Electron.phi,
            #         "eta": events_m.Electron_nocut.eta,
            #         "id": events_m.Electron_nocut.cutBased,
            #         "mvaisoid": events_m.Electron_nocut.mvaFall17V2Iso_WP90,
            #         #"isoid": events_m.Electron_nocut.pfIsoId,
            #         "miniisoid": events_m.Electron_nocut.miniPFRelIso_all
            #     },
            #     "Muon": {
            #         "pt": events_m.Muon.pt,
            #         "phi": events_m.Muon.phi,
            #         "id": events_m.Muon_nocut.mediumId,
            #         "isoid": events_m.Muon_nocut.pfIsoId,
            #         "miniisoid": events_m.Muon_nocut.miniIsoId
            #     },
            #     "RecoJets" : {
            #         "pt": events_m.FatJet.pt,
            #         "phi": events_m.FatJet.phi
            #     }
            # }
            
            # jet_z_data = {
            #     "GenJet": {
            #         "pt": gen_jet_pt,
            #         "phi": gen_jet_phi
            #     },
            #     "Z_gen": {
            #         "pt": z_gen_pt,
            #         "phi": z_gen_phi
            #     },
            #     "Z_reco": {
            #         "pt": z_reco_pt,
            #         "phi": z_reco_phi
            #     },
            #     "RecoJet": {
            #         "pt": reco_jet_pt,
            #         "phi": reco_jet_phi
            #     },
            #     "event": {
            #         "event": events_m.event}
            # }
            
            # # Convert dictionaries to awkward arrays with nested structure
            # ak_lepton_data = ak.Array(lepton_data)
            # ak_jet_z_data = ak.Array(jet_z_data)
            
            # # Save the awkward arrays to separate pickle files
            # with open("lepton_data.pkl", "wb") as f:
            #     pickle.dump(ak_lepton_data, f)
            
            # with open("jet_z_data.pkl", "wb") as f:
            #     pickle.dump(ak_jet_z_data, f)
            
            # print("Lepton data saved to lepton_data.pkl")
            # print("Jet and Z boson data saved to jet_z_data.pkl")
            
            #######################################
            ###########################################

            

            
            
            events0 = events0[allsel_reco ]
            weights = weights[allsel_reco]
            z_reco = z_reco[allsel_reco]
            reco_jet = reco_jet[allsel_reco]

            
             





            #weights = weights[allsel_reco]
            if self.do_gen:
                z_gen = z_gen[allsel_reco]
                gen_jet = gen_jet[allsel_reco]
                groomed_gen_jet = groomed_gen_jet[allsel_reco]


                
                # subjet1 = problematic_events.SubJet[problematic_reco_jet.subJetIdx1]
                # print("subjet1.mass ", subjet1.mass)
                # print("len of subjet1", len(subjet1))
                # closest_jet, dr = find_closest_dr(subjet1, problematic_events.Jet)

                
                # fill_hist(self.hists, "problematic_dr", dr = dr)
                # fill_hist(self.hists, "ak4_jet_puID", n = closest_jet.puId)
                # fill_hist(self.hists, "ak4_jet_puIdDisc", frac = closest_jet.puIdDisc)
                # fill_hist(self.hists, "ak4_jet_neHEF", frac = closest_jet.neHEF)
                # fill_hist(self.hists, "ak4_jet_neEmEF", frac = closest_jet.neEmEF)

                
                


                # print("Fatjet muon Id", reco_jet.muonIdx3SJ)
                #                #print("Events run number", events.event)

                # print("Events that pass both RECO and GEN")
                # print(" --------------------------------------- ")
                # print(" --------------------------------------- ")
                # print("Lepton pt", events.GenDressedLepton.pt)
                # #print("Muon pt", events.Muon.pt)
                # print("Z pt", z_gen.pt)
                # print("Jet pt", gen_jet.pt)

                # #print("Events run number", events.event)
                # print("Lepton phi", events.GenDressedLepton.phi)
                # #print("Muon phi", events.Muon.phi)
                # print("Z phi", z_gen.phi)
                # print("Jet phi", gen_jet.phi)

                
                # print("Reco Now")
                # #print("Events run number", events.event)
                # print("Electron pt", events.Electron.pt)
                # print("Muon pt", events.Muon.pt)
                # print("Z pt", z_reco.pt)
                # print("Jet pt", reco_jet.pt)

                # #print("Events run number", events.event)
                # print("Electron phi", events.Electron.phi)
                # print("Muon phi", events.Muon.phi)
                # print("Z phi", z_reco.phi)
                # print("Jet phi", reco_jet.phi)
                ##################################
                #### Apply Jet Veto Map
                ##################################




            ########### Crack eta ##################
            # crack_eta = (np.abs(reco_jet.eta) > 1.57) | (np.abs(reco_jet.eta) <= 1.3)

            # events0 = events0[crack_eta ]
            # weights = weights[crack_eta]
            # z_reco = z_reco[crack_eta]
            # reco_jet = reco_jet[crack_eta]

            # if len(events0) == 0:
            #     return self.hists
            
            # #print("Len of reco jet 1", len(reco_jet.pt))
            # #print("How many none is reco jet 1", ak.sum(ak.is_none(reco_jet.pt)))
            # #print("How many none is reco jet 1 mass", ak.sum(ak.is_none(reco_jet.mass)))
            # #print("RECO jet phi 1", reco_jet.phi )
            # if self.do_gen:
            #     z_gen = z_gen[crack_eta]
            #     gen_jet = gen_jet[crack_eta]
            #     groomed_gen_jet = groomed_gen_jet[crack_eta]

            crack_eta = reco_jet.mass > -10 ## Setting to TRUE for groomed
            
            ################ end of crack eta #######################
           
            if len(events0) == 0:
                return self.hists
            
            #print("How many none is reco jet 1 mass before jetvetomap", ak.sum(ak.is_none(reco_jet.mass)))  


            #fill_hist(self.hists, 'normal_eta', eta = reco_jet[weird_sel].eta, weight = weights[weird_sel])
            
            #fill_hist(self.hists, 'eta_phi_jet_reco_preveto',dataset = dataset,  eta =reco_jet.eta,  phi = reco_jet.phi, weight = weights)
            #jetvetomap = ApplyVetoMap(IOV, reco_jet, mapname='jetvetomap')
            
            
            print("len events", len(events0))
            if len(events0) == 0:
                return self.hists
                
            # print(ak.sum(~jetvetomap))
            
            # events0 = events0[jetvetomap ]
            # weights = weights[jetvetomap]
            # z_reco = z_reco[jetvetomap]
            # reco_jet = reco_jet[jetvetomap]

           
            # if len(events0) == 0:
            #     return self.hists
            
            # #print("Len of reco jet 1", len(reco_jet.pt))
            # #print("How many none is reco jet 1", ak.sum(ak.is_none(reco_jet.pt)))
            # #print("How many none is reco jet 1 mass", ak.sum(ak.is_none(reco_jet.mass)))
            # #print("RECO jet phi 1", reco_jet.phi )
            # if self.do_gen:
            #     z_gen = z_gen[jetvetomap]
            #     gen_jet = gen_jet[jetvetomap]
            #     groomed_gen_jet = groomed_gen_jet[jetvetomap]

            ## Setting to true for groomed

            jetvetomap = crack_eta 

            fill_hist(self.hists, 'eta_phi_jet_reco_postveto', dataset = dataset,  eta =reco_jet.eta,  phi = reco_jet.phi, weight = weights)

            
            if len(events0.FatJet) > 0:
                print("Entering jet correction")
                
                corr_jets = GetJetCorrections(events0.FatJet, events0, era, IOV, isData = not self.do_gen)  ###### correcting FatJet.mass
                corr_jets = corr_jets[corr_jets.subJetIdx1 > -1]
                #print(" Uncorrected subjet mass", events0.SubJet.mass)
                corr_subjets = GetJetCorrections(events0.SubJet, events0, era, IOV, isData = not self.do_gen, mode = 'AK4')
                print(" Uncorrected softdrop mass", corr_jets.msoftdrop)
                
                corr_jets['msoftdrop'] =   (corr_subjets[corr_jets.subJetIdx1] + corr_subjets[corr_jets.subJetIdx2]).mass 
            else:
                corr_jets = events0.FatJet
            if self.do_gen:
                weird_sel = (reco_jet.mass >= 30) & (gen_jet.mass < 20)
                if ak.sum(weird_sel > 0):
                    
                    fill_hist(self.hists, 'weird_eta', eta = reco_jet[weird_sel].eta, weight = weights[weird_sel])
                del weird_sel
            print("Total selected number of events", np.sum(allsel_reco))
            fill_hist(self.hists, "total_weight", weight = weights)
            #print("How many none is reco jet 1.5 mass", ak.sum(ak.is_none(ak.flatten(ak.firsts(corr_jets.mass), axis = 0))))
            #if self.do_gen:
            
            print(" \n \n \n ")





            if self.do_gen:
    
                
                #pdf uncertainty systematics len
                events0["pdf_N"] = GetPDFweights(events0)
                events0["pdf_U"] = GetPDFweights(events0, var="up")
                events0["pdf_D"] = GetPDFweights(events0, var="down")
    
    
                #q2 uncertainty systematics
            
                events0["q2_N"] = GetQ2weights(events0)
                events0["q2_U"] = GetQ2weights(events0, var="up")
                events0["q2_D"] = GetQ2weights(events0, var="down")
                
                #pileup
                events0["pu_nominal"] = GetPUSF(IOV, np.array(events0.Pileup.nTrueInt))
                events0["pu_U"]    = GetPUSF(IOV, np.array(events0.Pileup.nTrueInt), "up")
                events0["pu_D"]    = GetPUSF(IOV, np.array(events0.Pileup.nTrueInt), "down")
    
                
                ## L1PreFiringWeight
                events0["prefiring_N"] = GetL1PreFiringWeight(IOV, events0)
                events0["prefiring_U"] = GetL1PreFiringWeight(IOV, events0, "Up")
                events0["prefiring_D"] = GetL1PreFiringWeight(IOV, events0, "Dn")
    
    
                
                ## Electron Reco systematics
                events0["elereco_N"] = GetEleSF(IOV, "RecoAbove20", events0.Electron.eta, events0.Electron.pt)
                events0["elereco_U"] = GetEleSF(IOV, "RecoAbove20", events0.Electron.eta, events0.Electron.pt, "up")
                events0["elereco_D"] = GetEleSF(IOV, "RecoAbove20", events0.Electron.eta, events0.Electron.pt, "down")
    
    
    
                # Electron ID systematics/projects/TUnfoldExamples/
    
                events0["eleid_N"] = GetEleSF(IOV, "Tight", events0.Electron.eta, events0.Electron.pt)
                events0["eleid_U"] = GetEleSF(IOV, "Tight", events0.Electron.eta, events0.Electron.pt, "up")
                events0["eleid_D"] = GetEleSF(IOV, "Tight", events0.Electron.eta, events0.Electron.pt, "down")


                # Electron Trig Systematics


                events0["eletrig_N"] = GetEleTrigEff(IOV, events0.Electron.pt, events0.Electron.eta)
                
                events0["eletrig_U"] = GetEleTrigEff(IOV, events0.Electron.pt, events0.Electron.eta, var = "up")
                events0["eletrig_D"] = GetEleTrigEff(IOV, events0.Electron.pt, events0.Electron.eta, var = "down")
                
                # Muon Reco systematics
                events0["mureco_N"] = GetMuonSF(IOV, "RECO", np.abs(events0.Muon.eta), events0.Muon.pt) 
                events0["mureco_U"] = GetMuonSF(IOV, "RECO", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")
                events0["mureco_D"] = GetMuonSF(IOV, "RECO", np.abs(events0.Muon.eta), events0.Muon.pt, "systdown")
                
    
    
                ## Muon ID systematics
                events0["muid_N"] = GetMuonSF(IOV, "ID", np.abs(events0.Muon.eta), events0.Muon.pt)
                events0["muid_U"] = GetMuonSF(IOV, "ID", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")
                events0["muid_D"] = GetMuonSF(IOV, "ID", np.abs(events0.Muon.eta), events0.Muon.pt, "systdown")

                ## Muon ISO systematics
                events0["muiso_N"] = GetMuonSF(IOV, "ISO", np.abs(events0.Muon.eta), events0.Muon.pt)
                events0["muiso_U"] = GetMuonSF(IOV, "ISO", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")
                events0["muiso_D"] = GetMuonSF(IOV, "ISO", np.abs(events0.Muon.eta), events0.Muon.pt, "systdown")
    
    
                print("Muid and reco working")
                
                #q2 uncertainty systematics
            
                events0["q2_N"] = GetQ2weights(events0)
                events0["q2_U"] = GetQ2weights(events0, var="up")
                events0["q2_D"] = GetQ2weights(events0, var="down")
                
                #Muon Trigger systematics
                events0["mutrig_N"] = GetMuonSF(IOV, "HLT", np.abs(events0.Muon.eta), events0.Muon.pt)
                events0["mutrig_U"] = GetMuonSF(IOV, "HLT", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")
                events0["mutrig_D"] = GetMuonSF(IOV, "HLT", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")


            else:
                systematic_list = ['pdf_N', 'pdf_U', 'pdf_D', 
                                   'q2_N', 'q2_U', 'q2_D', 
                                   "pu_nominal", "pu_U", "pu_D",
                                   "prefiring_N", "prefiring_U", "prefiring_D", 
                                   "elereco_N", "elereco_U", "elereco_D", 
                                   "eleid_N", "eleid_U", "eleid_D" ,
                                   "eletrig_N","eletrig_U","eletrig_D" ,
                                   "mureco_N", "mureco_U", "mureco_D", 
                                   "muid_N", "muid_U", "muid_D", 
                                   "muiso_N", "muiso_U", "muiso_D",
                                   "mutrig_N", "mutrig_U", "mutrig_D"]
                
                for systematic in systematic_list:
                    if "mu" in systematic:
                        events0[systematic] = ak.ones_like(events0.Muon.pt)
                    elif "ele" in systematic:
                        events0[systematic] = ak.ones_like(events0.Electron.pt)
                    else:
                        events0[systematic]= ak.ones_like(events0.event, dtype=float) ## setting everything to 1 in case of DATA
                
            print("Is this happenning?")
            if len(events0) > 0:
                coffea_weights = Weights(size = len(events0), storeIndividual = True)

                coffea_weights.add("init_weight", weights)

                coffea_weights.add(name = "pu", weight = events0.pu_nominal, weightUp = events0.pu_U, weightDown = events0.pu_D)
                if not self.do_background:
                    coffea_weights.add("q2", events0.q2_N, events0.q2_U, events0.q2_D)
                coffea_weights.add("pdf", events0.pdf_N, events0.pdf_U, events0.pdf_D)
                coffea_weights.add("prefiring", events0.prefiring_N, events0.prefiring_U, events0.prefiring_D)
                w = coffea_weights.weight()

                print('len of z_jet_dphi_reco', len(z_jet_dphi_reco[allsel_reco]))
                print( 'len of reco_jet', len(reco_jet.pt))

                if not self.minimal:
                    fill_hist(self.hists, "dphi_z_jet_reco", dataset=dataset, 
                                                   dphi=z_jet_dphi_reco[allsel_reco][jetvetomap], 
                                                   weight=w)

                    
                
            #print("GEN JET MASS GROOMED", groomed_gen_jet.mass)

            
            ee_sel = sel.all("final_selection", "twoReco_ee")[allsel_reco][crack_eta][jetvetomap ]
            mm_sel = sel.all("final_selection", "twoReco_mm")[allsel_reco][crack_eta][jetvetomap ]

            print("Final Sel", len(events0))
            print("ee_sel", ak.sum(ee_sel))
            print("mm_sel", ak.sum(mm_sel))


            cat_sel_list = {"ee":ee_sel, "mm": mm_sel}

            ee_sys_var_list = [syst for syst in self.systematics if "ele" in syst]
            mm_sys_var_list = [syst for syst in self.systematics if "mu" in syst]
            del allsel_reco

            fill_hist(self.hists, "dr_ll_reco",channel = r"$\mu\mu$ Post-sel",
                                          dr = events0[mm_sel].Muon[:, 0].delta_r(events0[mm_sel].Muon[:, 1]),
                                          weight = weights[mm_sel])
            fill_hist(self.hists, "dr_ll_reco",channel = r"$ee$ Post-sel",
                                          dr = events0[ee_sel].Electron[:, 0].delta_r(events0[ee_sel].Electron[:, 1]),
                                          weight = weights[ee_sel])
            
            
            print("Jet systematics list", self.jet_systematics)
      
            for jet_syst in self.jet_systematics:
                #print("length of event in loop: " , len(events))
                print(" Now doing ", jet_syst)
                # print(jet_syst[:-2])
                # print(jet_syst[:-2]=="Up")
                # print("f")
                if jet_syst == "nominal":
                    if self.do_gen:
                        #events = ak.with_field(events0,  jmssf(IOV, corr_jets) , "FatJet")
                        
                        events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV, corr_jets)) , "FatJet")
                    else:
                        
                        events = ak.with_field(events0, corr_jets , "FatJet")
                    
                elif jet_syst == "hem":
                    
                    events = ak.with_field(events0,  HEMCleaning(IOV,jmrsf(IOV, jmssf(IOV,corr_jets))) , "FatJet")
                    
                elif jet_syst == "JERUp":
                    corr_jets_obj = corr_jets.JER.up
                    corr_jets_obj['msoftdrop'] = (corr_subjets.JER.up[corr_jets.subJetIdx1] + corr_subjets.JER.up[corr_jets.subJetIdx2]).mass
                    
                    events = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
    
                elif jet_syst == "JERDown":
                    corr_jets_obj = corr_jets.JER.down
                    corr_jets_obj['msoftdrop'] = (corr_subjets.JER.down[corr_jets.subJetIdx1] + corr_subjets.JER.down[corr_jets.subJetIdx2]).mass
                    
                    events = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                elif jet_syst == "JMSUp":
                    events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets, var = "up")) , "FatJet")
                elif jet_syst == "JMSDown":
                    events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets, var = "down")) , "FatJet")
                elif jet_syst == "JMRUp":
                    events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets), var = "up") , "FatJet")
                elif jet_syst == "JMRDown":
                    events = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets), var = "down") , "FatJet")
                
                elif (jet_syst[-2:]=="Up" and "JES" in jet_syst):
                    #print(jet_syst)
                    field = jet_syst[:-2]
                    #print(field)
                    corr_jets_obj = corr_jets[field].up
                    corr_jets_obj['msoftdrop'] = (corr_subjets[field].up[corr_jets.subJetIdx1] + corr_subjets[field].up[corr_jets.subJetIdx2]).mass
                    
                    events = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                    
                elif (jet_syst[-4:]=="Down" and "JES" in jet_syst):
                    field = jet_syst[:-4]
                    corr_jets_obj = corr_jets[field].down
                    corr_jets_obj['msoftdrop'] = (corr_subjets[field].down[corr_jets.subJetIdx1] + corr_subjets[field].down[corr_jets.subJetIdx2]).mass
                    
                    events = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                
                else:
                    print("{} is not considered".format(jet_syst))

        
                reco_jet = events.FatJet[:,0]
                
                #print("RECO jet phi 2", reco_jet.phi )
                #print("How many none is reco jet 2", ak.sum(ak.is_none(reco_jet.pt)))
                #print("How many none is gen jet 2", ak.sum(ak.is_none(gen_jet.pt)))
                #print("How many none is reco jet mass  2", ak.sum(ak.is_none(reco_jet.mass)))
                #print("Special event number ", events[ak.is_none(reco_jet.mass)].event)
                ## This part is for additional scaling of jet pt to make the matrix more diagonal, however this does not produce better result and hence excluded for now
                # binning = util_binning()
                # if self.do_gen:
                #     if herwig:
                #         jes = np.loadtxt('correctionFiles/ungroomed_jes_herwig.txt')
                #         jes = 1/jes
                #         jesSF = dense_lookup(jes, [binning.ptgen_axis.edges, binning.mgen_axis.edges])
                #     else:
                #         jes = 1/np.loadtxt('correctionFiles/ungroomed_jes_pythia.txt')
                #         jesSF = dense_lookup(jes, [binning.ptgen_axis.edges, binning.mgen_axis.edges])

                # ## defaulting to HERWIG for data as well
                # else:
                #     jes = 1/np.loadtxt('correctionFiles/ungroomed_jes_pythia.txt')
                #     jesSF = dense_lookup(jes, [binning.ptgen_axis.edges, binning.mgen_axis.edges])    
                    
                # reco_jet = ak.with_field(reco_jet, reco_jet.pt * jesSF(reco_jet.pt, reco_jet.mass), 'pt')

                
                if self.do_gen and herwig:    
                    pt_scale = getpTweight(reco_jet.pt)
                    weights = weights*pt_scale
                    del pt_scale

                if self.do_gen:    
                    pt_scale = getpTweight(reco_jet.pt, herwig)
                    weights = weights*pt_scale
                    del pt_scale


                ## the above portion needs to be uncommented in case we want to reweight to match the RECO pt distribution with DATA.
                # binning = util_binning()
                # jmsSF_list = np.array([1.04860737, 1.03405672, 1.02036575 ,1.00874843])
                # jmsSF = dense_lookup(jmsSF_list, [binning.ptreco_axis.edges])
                # print("jms correction for pt >200 ", jmsSF(201))
                # reco_jet = ak.with_field(reco_jet, reco_jet.mass * jmsSF(reco_jet.pt), 'mass')
                
    
                for cat in cat_sel_list.keys():
                    
                    if jet_syst == "nominal":
                        if cat == "ee":
    
                            events_ee = events[ee_sel]
                            weights_ee = weights[ee_sel]
                            if len(events_ee) > 0:


                                z_reco_ee = z_reco[ee_sel]
                                reco_jet_ee = reco_jet[ee_sel]
                                #weights = weights[allsel_reco]
                                if self.do_gen:
                                    z_gen_ee = z_gen[ee_sel]
                                    gen_jet_ee = gen_jet[ee_sel]
                                    groomed_gen_jet_ee = groomed_gen_jet[ee_sel]

                                
                                coffea_weights = Weights(size = len(events_ee), storeIndividual = True)
    
                                coffea_weights.add("init_weight", weights_ee)
    
                                coffea_weights.add(name = "pu", weight = events_ee.pu_nominal, weightUp = events_ee.pu_U, weightDown = events_ee.pu_D)
                                if not self.do_background:
                                    coffea_weights.add("q2", events_ee.q2_N, events_ee.q2_U, events_ee.q2_D)
                                coffea_weights.add("pdf", events_ee.pdf_N, events_ee.pdf_U, events_ee.pdf_D)
                                coffea_weights.add("prefiring", events_ee.prefiring_N, events_ee.prefiring_U, events_ee.prefiring_D)
        
  
                                
                                elereco_N = events_ee.elereco_N[:,0]*events_ee.elereco_N[:,1]
                                elereco_U = events_ee.elereco_U[:,0]*events_ee.elereco_U[:,1]
                                elereco_D = events_ee.elereco_D[:,0]*events_ee.elereco_D[:,1]
                                
                                coffea_weights.add(name = "elereco", weight = elereco_N, weightUp = elereco_U, weightDown = elereco_D)
                                
                                eleid_N = events_ee.eleid_N[:,0]*events_ee.eleid_N[:,1]
                                eleid_U = events_ee.eleid_U[:,0]*events_ee.eleid_U[:,1]
                                eleid_D = events_ee.eleid_D[:,0]*events_ee.eleid_D[:,1]
    
                                coffea_weights.add(name = "eleid", weight = eleid_N, weightUp = eleid_U, weightDown = eleid_D)
                                #print('eletrig', events_ee.eletrig_N)
   
                                eletrig_N = events_ee.eletrig_N[:,0]
                                eletrig_U = events_ee.eletrig_U[:,0]
                                eletrig_D = events_ee.eletrig_D[:,0]
    
                                coffea_weights.add(name = "eletrig", weight = eletrig_N, weightUp = eletrig_U, weightDown = eletrig_D)

                                coffea_weights.add(name = "mureco",
                                                   weight = ak.ones_like(events_ee.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_ee.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_ee.event, dtype = float))
    
                                coffea_weights.add(name = "muid",
                                                   weight = ak.ones_like(events_ee.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_ee.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_ee.event, dtype = float))
    
                                coffea_weights.add(name = "mutrig",
                                                   weight = ak.ones_like(events_ee.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_ee.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_ee.event, dtype = float))
                                coffea_weights.add(name = "muiso",
                                                   weight = ak.ones_like(events_ee.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_ee.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_ee.event, dtype = float))
    
                                # fill_hist(self.hists, 'puweight',dataset = dataset, corrWeight = events_ee.pu_nominal)  
                                # fill_hist(self.hists, 'q2weight',dataset = dataset, corrWeight = events_ee.q2_N)  
                                # fill_hist(self.hists, 'pdfweight',dataset = dataset, corrWeight = events_ee.q2_N)  
                                # fill_hist(self.hists, 'prefiringweight',dataset = dataset, corrWeight = events_ee.prefiring_N)  
    
                                # #fill_hist(self.hists, 'eleidweight',dataset = dataset, corrWeight = eleid_N)  
                                # fill_hist(self.hists, 'elerecoweight',dataset = dataset, corrWeight = elereco_N)  
                                
                                del elereco_N, elereco_U, elereco_D, eleid_N, eleid_U, eleid_D
    
                                
    
                                
                                
    
                    
                                #systematics = [syst for syst in self.systematics if syst not in mm_sys_var_list]
                                #print("for ee case ", systematics)
                                #print("msoftdrop")
                                #print(reco_jet_ee.msoftdrop)
                                for syst in self.systematics:
                                    print("Syst ", syst)
                                    if syst == "nominal":
                                        w = coffea_weights.weight()


                                        dphi_ele1 = reco_jet_ee.delta_phi(ak.firsts(events_ee.Electron[:,:]))
                                        dphi_ele2 = reco_jet_ee.delta_phi(ak.firsts(events_ee.Electron[:,1:]))

    
                                        if not self.minimal:
                                            fill_hist(self.hists, "dphi_lep_jet_reco", dataset=dataset, 
                                                                                   dphi = dphi_ele1, 
                                                                                   weight=w)
    
                                            fill_hist(self.hists, "dphi_lep_jet_reco", dataset=dataset, 
                                                                                   dphi = dphi_ele2, 
                                                                                   weight=w)

                                        
                                        
                                        if self.do_gen:
                                            fill_hist(self.hists, 'm_u_jet_reco_over_gen',dataset=dataset,
                                                                                     ptgen=gen_jet_ee.pt,
                                                                                     mgen=gen_jet_ee.mass,
                                                                                     frac = (reco_jet_ee.mass - gen_jet_ee.mass)/gen_jet_ee.mass,
                                                                                     weight = w)
                                            fill_hist(self.hists, 'm_g_jet_reco_over_gen',dataset=dataset,
                                                                                     ptgen= gen_jet_ee.pt,
                                                                                     mgen=groomed_gen_jet_ee.mass, 
                                                                                     frac=(reco_jet_ee.msoftdrop-groomed_gen_jet_ee.mass)/groomed_gen_jet_ee.mass,
                                                                                     weight = w)

                                            fill_hist(self.hists, 'delta_m_u', dataset=dataset, ptgen=gen_jet_ee.pt, mgen=gen_jet_ee.mass, diff =  reco_jet_ee.mass - gen_jet_ee.mass, weight = w)
                                            fill_hist(self.hists, 'delta_m_g', dataset=dataset, ptgen= gen_jet_ee.pt, mgen=groomed_gen_jet_ee.mass,  diff =  reco_jet_ee.mass - groomed_gen_jet_ee.mass, weight = w)

                                            fill_hist(self.hists, 'pt_u_jet_reco_over_gen',dataset=dataset,
                                                                                      ptgen=gen_jet_ee.pt,
                                                                                      mgen=gen_jet_ee.mass,
                                                                                      frac = (reco_jet_ee.pt-gen_jet_ee.pt)/gen_jet_ee.pt,
                                                                                      weight = w)
                                            fill_hist(self.hists, 'pt_g_jet_reco_over_gen',dataset=dataset,
                                                                                      ptgen= gen_jet_ee.pt,
                                                                                      mgen=groomed_gen_jet_ee.mass, 
                                                                                      frac=(reco_jet_ee.pt-gen_jet_ee.pt)/gen_jet_ee.pt,
                                                                                      weight = w)

                                            fill_hist(self.hists, 'delta_pt_u', dataset=dataset, ptgen=gen_jet_ee.pt, mgen=gen_jet_ee.mass, diff =  reco_jet_ee.pt - gen_jet_ee.pt, weight = w)
                                            fill_hist(self.hists, 'delta_pt_g', dataset=dataset, ptgen= gen_jet_ee.pt, mgen=groomed_gen_jet_ee.mass,  diff =  reco_jet_ee.pt - gen_jet_ee.pt, weight = w)
                                            if herwig:
                                                syst = 'herwig'
                                        #w = weights
                                    else:
                                        #print(coffea_weights.variations)
                                        w = coffea_weights.weight(modifier = syst)
                                        #w = weights
                                    #print(w)    
                                    print(syst)
                                    fill_hist(self.hists, "ptjet_mjet_u_reco",dataset=dataset,  channel = 'ee', ptreco = reco_jet_ee.pt, mreco = reco_jet_ee.mass, systematic = syst, weight = w)
                                    fill_hist(self.hists, "ptjet_mjet_g_reco",dataset=dataset, channel = 'ee', 
                                                                         ptreco = reco_jet_ee.pt,
                                                                         mreco = reco_jet_ee.msoftdrop, 
                                                                         systematic = syst, weight = w)
                                    #print("log rho values ",2*np.log10(reco_jet_ee.mass/(reco_jet_ee.pt*0.8)))
                                    fill_hist(self.hists, "m_over_pt_u",
                                              dataset = dataset,
                                              ptreco = reco_jet_ee.pt,
                                              mpt_reco = 2*np.log10(reco_jet_ee.mass/(reco_jet_ee.pt*0.8)),
                                              mreco = reco_jet_ee.mass,
                                              systematic = syst,
                                              weight = w
                                              )
                                    
                                    fill_hist(self.hists, "m_over_pt_g",
                                              dataset = dataset,
                                              ptreco = reco_jet_ee.pt,
                                              mpt_reco = 2*np.log10(reco_jet_ee.msoftdrop/(reco_jet_ee.pt*0.8)),
                                              mreco = reco_jet_ee.msoftdrop,
                                              systematic = syst,
                                              weight = w)

                                    
    
                                    # fill_tunfold_hist_1d(hist = fill_hist(self.hists, "tunfold_reco_u"], dataset = dataset,
                                    #                      mass = reco_jet_ee.mass, pt = reco_jet_ee.pt,  systematic = syst, weight = w, recogen = 'reco')
                                    # fill_tunfold_hist_1d(hist = fill_hist(self.hists, "tunfold_reco_g"], dataset = dataset,
                                    #                      mass = reco_jet_ee.msoftdrop, pt = reco_jet_ee.pt, systematic = syst, weight = w, recogen = 'reco')
                                    ### data/MC plots
                                    if not self.minimal:
                                        fill_hist(self.hists, "jk_ptjet_mjet_u_reco",ptreco  = reco_jet_ee.pt, mreco = reco_jet_ee.mass, jk = jk_index, weight = w)
                                        fill_hist(self.hists, "jk_ptjet_mjet_g_reco",ptreco  = reco_jet_ee.pt, mreco = reco_jet_ee.msoftdrop, jk = jk_index, weight = w)

                                    
                                    if not self.minimal:
                                        fill_hist(self.hists, 'm_z_reco',dataset = dataset, mass = z_reco_ee.mass, weight = w, systematic = syst)
                                        fill_hist(self.hists, 'eta_z_reco',dataset = dataset,  eta = z_reco_ee.eta, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'pt_z_reco',dataset = dataset, pt = z_reco_ee.pt, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'phi_z_reco',dataset = dataset,  phi = z_reco_ee.phi, weight = w, systematic = syst)
                                        fill_hist(self.hists, 'eta_phi_z_reco',dataset = dataset,  eta =z_reco_ee.eta,  phi = z_reco_ee.phi, weight = w, systematic = syst)
                                        
                                        fill_hist(self.hists, 'm_jet_reco',dataset = dataset, mreco = reco_jet_ee.mass, weight = w, systematic = syst)
                                        fill_hist(self.hists, 'eta_jet_reco',dataset = dataset, eta = reco_jet_ee.eta, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'pt_jet_reco',dataset = dataset, pt = reco_jet_ee.pt, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'phi_jet_reco',dataset = dataset, phi = reco_jet_ee.phi, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'eta_phi_jet_reco',dataset = dataset,  eta =reco_jet_ee.eta,  phi = reco_jet_ee.phi, weight = w, systematic = syst)
                                    
    
                                    if self.do_gen:
                                        fill_hist(self.hists, "response_matrix_u", dataset=dataset, channel = 'ee',
                                                                           ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                           mreco=reco_jet_ee.mass, mgen=gen_jet_ee.mass, systematic = syst,  weight = w )
                            
                                        fill_hist(self.hists, "response_matrix_g", dataset=dataset, channel = 'ee',
                                                                           ptreco= reco_jet_ee.pt,
                                                                             ptgen = gen_jet_ee.pt,
                                                                             mreco= reco_jet_ee.msoftdrop, 
                                                                             mgen= groomed_gen_jet_ee.mass,
                                                                             systematic = syst, 
                                                                             weight = w )


                                        

                                        fill_hist(self.hists, "resp_mpt_u",dataset= dataset,
                                                                     ptreco = reco_jet_ee.pt,
                                                                     ptgen = gen_jet_ee.pt,
                                                                     mreco = reco_jet_ee.mass,
                                                                     mgen = gen_jet_ee.mass,
                                                                     mpt_reco = 2*np.log10(reco_jet_ee.mass/(reco_jet_ee.pt*0.8)),
                                                                     mpt_gen = 2*np.log10(gen_jet_ee.mass/(gen_jet_ee.pt*0.8)),
                                                                     systematic = syst, 
                                                                     weight = w)
                                        
                                        fill_hist(self.hists, "resp_mpt_g",dataset= dataset,
                                                                    ptreco = reco_jet_ee.pt,
                                                                    ptgen = gen_jet_ee.pt,
                                                                    mreco = reco_jet_ee.msoftdrop,
                                                                    mgen = groomed_gen_jet_ee.mass,
                                                                    mpt_reco = 2*np.log10(reco_jet_ee.msoftdrop/(reco_jet_ee.pt*0.8)),
                                                                    mpt_gen = 2*np.log10(groomed_gen_jet_ee.mass/(gen_jet_ee.pt*0.8)),
                                                                    systematic = syst, 
                                                                    weight = w)
                                        if not self.minimal:
                                            fill_hist(self.hists, "jk_response_matrix_u", dataset=dataset, 
                                                                               ptreco= reco_jet_ee.pt, ptgen = gen_jet_ee.pt, jk = jk_index,
                                                                               mreco= reco_jet_ee.mass, mgen=gen_jet_ee.mass, systematic = syst,  weight = w )
                                
                                            fill_hist(self.hists, "jk_response_matrix_g", dataset=dataset, 
                                                                               ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt, jk = jk_index,
                                                                               mreco=reco_jet_ee.msoftdrop, mgen=groomed_gen_jet_ee.mass, systematic = syst, weight = w )
                                            fill_hist(self.hists, "ptjet_mjet_g_gen",dataset=dataset, 
                                                                             ptgen = gen_jet_ee.pt, 
                                                                             mgen = groomed_gen_jet_ee.mass, 
                                                                             systematic = syst,
                                                                             weight = w) 
                                        
                                        # fill_tunfold_hist_2d(dataset = dataset, 
                                        #                      hist = fill_hist(self.hists, "tunfold_migration_u"], mass_gen = gen_jet_ee.mass, pt_gen = gen_jet_ee.pt, 
                                        #                      mass_reco = reco_jet_ee.mass, pt_reco = reco_jet_ee.pt, systematic = syst, weight  = w)
                                        # fill_tunfold_hist_2d(dataset = dataset, 
                                        #                      hist = fill_hist(self.hists, "tunfold_migration_g"], mass_gen = groomed_gen_jet_ee.mass, pt_gen = groomed_gen_jet_ee.pt, 
                                        #                      mass_reco = reco_jet_ee.msoftdrop, pt_reco = reco_jet_ee.pt, systematic = syst, weight  = w)

                                        # fill_tunfold_hist_2d(dataset = dataset, jk_index = jk_index,
                                        #                      hist = fill_hist(self.hists, "jackknife_response_u"], mass_gen = gen_jet_ee.mass, pt_gen = gen_jet_ee.pt, 
                                        #                      mass_reco = reco_jet_ee.mass, pt_reco = reco_jet_ee.pt, systematic = syst, weight  = w)
                                        # fill_tunfold_hist_2d(dataset = dataset, jk_index = jk_index,
                                        #                      hist = fill_hist(self.hists, "jackknife_response_g"], mass_gen = groomed_gen_jet_ee.mass, pt_gen = groomed_gen_jet_ee.pt, 
                                        #                      mass_reco = reco_jet_ee.msoftdrop, pt_reco = reco_jet_ee.pt, systematic = syst, weight  = w)
                                                
                                del weights_ee, z_reco_ee, reco_jet_ee
                                if self.do_gen:
                                    z_gen_ee, gen_jet_ee, groomed_gen_jet_ee, coffea_weights
                            
                        if cat == "mm":
    
                            events_mm = events[mm_sel]
                            weights_mm = weights[mm_sel]
                            if len(events_mm) > 0:
                                coffea_weights = Weights(size = len(events_mm), storeIndividual = True)
    
                                coffea_weights.add("init_weight", weights_mm)
    
                                coffea_weights.add(name = "pu", weight = events_mm.pu_nominal, weightUp = events_mm.pu_U, weightDown = events_mm.pu_D)
                                if not self.do_background:
                                    coffea_weights.add("q2", events_mm.q2_N, events_mm.q2_U, events_mm.q2_D)
                                coffea_weights.add("pdf", events_mm.pdf_N, events_mm.pdf_U, events_mm.pdf_D)
                                coffea_weights.add("prefiring", events_mm.prefiring_N, events_mm.prefiring_U, events_mm.prefiring_D)
                                
                                mureco_N = events_mm.mureco_N[:,0]*events_mm.mureco_N[:,1]
                                mureco_U = events_mm.mureco_U[:,0]*events_mm.mureco_U[:,1]
                                mureco_D = events_mm.mureco_D[:,0]*events_mm.mureco_D[:,1]
                                
                                coffea_weights.add(name = "mureco", weight = mureco_N, weightUp = mureco_U, weightDown = mureco_D)
                                
                                muid_N = events_mm.muid_N[:,0]*events_mm.muid_N[:,1]
                                muid_U = events_mm.muid_U[:,0]*events_mm.muid_U[:,1]
                                muid_D = events_mm.muid_D[:,0]*events_mm.muid_D[:,1]
    
                                coffea_weights.add(name = "muid", weight = muid_N, weightUp = muid_U, weightDown = muid_D)

                                muiso_N = events_mm.muiso_N[:,0]*events_mm.muiso_N[:,1]
                                muiso_U = events_mm.muiso_U[:,0]*events_mm.muiso_U[:,1]
                                muiso_D = events_mm.muiso_D[:,0]*events_mm.muiso_D[:,1]
    
                                coffea_weights.add(name = "muiso", weight = muiso_N, weightUp = muiso_U, weightDown = muiso_D)
    
                                mutrig_N = events_mm.mutrig_N[:,0]*events_mm.mutrig_N[:,1]
                                mutrig_U = events_mm.mutrig_U[:,0]*events_mm.mutrig_U[:,1]
                                mutrig_D = events_mm.mutrig_D[:,0]*events_mm.mutrig_D[:,1]
    
                                coffea_weights.add(name = "mutrig", weight = mutrig_N, weightUp = mutrig_U, weightDown = mutrig_D)
                                
                                coffea_weights.add(name = "elereco",
                                                   weight = ak.ones_like(events_mm.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_mm.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_mm.event, dtype = float ))
                                
                                coffea_weights.add(name = "eleid",
                                                   weight = ak.ones_like(events_mm.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_mm.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_mm.event, dtype = float))
                                
                                coffea_weights.add(name = "eletrig",
                                                   weight = ak.ones_like(events_mm.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_mm.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_mm.event, dtype = float))
      
                                del mureco_N, mureco_U, mureco_D, muid_N, muid_U, muid_D, mutrig_N, mutrig_U, mutrig_D
    
                                
    
                                
                                z_reco_mm = z_reco[mm_sel]
                                reco_jet_mm = reco_jet[mm_sel]
                                #weights = weights[allsel_reco]
                                if self.do_gen:
                                    z_gen_mm = z_gen[mm_sel]
                                    gen_jet_mm = gen_jet[mm_sel]
                                    groomed_gen_jet_mm = groomed_gen_jet[mm_sel]
    
                    
                                systematics = [syst for syst in self.systematics if syst not in mm_sys_var_list]
                                #print("for ee case ", systematics)
    
                                for syst in self.systematics:
                                    if syst == "nominal":
                                        w = coffea_weights.weight()

                                        dphi_mu1 = reco_jet_mm.delta_phi(ak.firsts(events_mm.Muon[:,:]))
                                        dphi_mu2 = reco_jet_mm.delta_phi(ak.firsts(events_mm.Muon[:,1:]))

                                        if not self.minimal:

                                            fill_hist(self.hists, "dphi_lep_jet_reco", dataset=dataset, 
                                                                                   dphi = dphi_mu1, 
                                                                                   weight=w)
    
                                            fill_hist(self.hists, "dphi_lep_jet_reco", dataset=dataset, 
                                                                                   dphi = dphi_mu2, 
                                                                                   weight=w)
                                        del dphi_mu1, dphi_mu2
                                        #w = weights
                                        # print("How many none is gen jet mm  2", ak.sum(ak.is_none(gen_jet_mm.pt)))
                                        # print("How many none is gen jet mm 2", ak.sum(ak.is_none(gen_jet_mm.mass)))
                                        # print("How many none is reco jet mm  2", ak.sum(ak.is_none(reco_jet_mm.pt)))
                                        # print("How many none is reco jet mm  2", ak.sum(ak.is_none(reco_jet_mm.mass)))
                                        # print("How many none in weight", ak.sum(ak.is_none(w)))
                                        if self.do_gen:
                                            fill_hist(self.hists, 'm_u_jet_reco_over_gen',dataset=dataset,
                                                                                     ptgen=gen_jet_mm.pt,
                                                                                     mgen=gen_jet_mm.mass,
                                                                                     frac = (reco_jet_mm.mass - gen_jet_mm.mass)/gen_jet_mm.mass,
                                                                                     weight = w)
                                            fill_hist(self.hists, 'm_g_jet_reco_over_gen',dataset=dataset,
                                                                                     ptgen= gen_jet_mm.pt,
                                                                                     mgen=groomed_gen_jet_mm.mass, 
                                                                                     frac= (reco_jet_mm.msoftdrop- groomed_gen_jet_mm.mass)/groomed_gen_jet_mm.mass,
                                                                                     weight = w)

                                            fill_hist(self.hists, 'delta_m_u', dataset=dataset, ptgen=gen_jet_mm.pt, mgen=gen_jet_mm.mass, diff =  reco_jet_mm.mass - gen_jet_mm.mass, weight = w)
                                            fill_hist(self.hists, 'delta_m_g', dataset=dataset, ptgen= gen_jet_mm.pt, mgen=groomed_gen_jet_mm.mass,  diff =  reco_jet_mm.mass - groomed_gen_jet_mm.mass, weight = w)

                                            fill_hist(self.hists, 'pt_u_jet_reco_over_gen',dataset=dataset,
                                                                                      ptgen=gen_jet_mm.pt,
                                                                                      mgen=gen_jet_mm.mass,
                                                                                      frac = (reco_jet_mm.pt-gen_jet_mm.pt) /gen_jet_mm.pt,
                                                                                      weight = w)
                                            fill_hist(self.hists, 'pt_g_jet_reco_over_gen',dataset=dataset,
                                                                                      ptgen= gen_jet_mm.pt,
                                                                                      mgen=groomed_gen_jet_mm.mass, 
                                                                                      frac=(reco_jet_mm.pt-gen_jet_mm.pt)/gen_jet_mm.pt,
                                                                                      weight = w)

                                            fill_hist(self.hists, 'delta_pt_u', dataset=dataset, ptgen=gen_jet_mm.pt, mgen=gen_jet_mm.mass, diff =  reco_jet_mm.pt - gen_jet_mm.pt, weight = w)
                                            fill_hist(self.hists, 'delta_pt_g', dataset=dataset, ptgen= gen_jet_mm.pt, mgen=groomed_gen_jet_mm.mass,  diff =  reco_jet_mm.pt - gen_jet_mm.pt, weight = w)
                                            if herwig:
                                                syst = 'herwig'
                                    else:
                                        #print(coffea_weights.variations)
                                        w = coffea_weights.weight(modifier = syst)
                                        #w = weights
                                    #print("Now weights")
                                    #print(w)    
                                    fill_hist(self.hists, "ptjet_mjet_u_reco",channel = 'mm', dataset=dataset, ptreco = reco_jet_mm.pt, mreco = reco_jet_mm.mass, systematic = syst, weight = w)
                                    fill_hist(self.hists, "ptjet_mjet_g_reco",channel = 'mm', dataset=dataset, ptreco = reco_jet_mm.pt, mreco = reco_jet_mm.msoftdrop, systematic = syst, weight = w)

                                    fill_hist(self.hists, "m_over_pt_u",
                                              dataset = dataset,
                                              ptreco = reco_jet_mm.pt,
                                              mpt_reco = 2*np.log10(reco_jet_mm.mass/(reco_jet_mm.pt*0.8)),
                                              mreco = reco_jet_mm.mass,
                                              systematic = syst,
                                              weight = w
                                              )
                                    
                                    fill_hist(self.hists, "m_over_pt_g",
                                              dataset = dataset,
                                              ptreco = reco_jet_mm.pt,
                                              mpt_reco = 2*np.log10(reco_jet_mm.msoftdrop/(reco_jet_mm.pt*0.8)),
                                              mreco = reco_jet_mm.msoftdrop,
                                              systematic = syst,
                                              weight = w)
                                    if not self.minimal:
                                        fill_hist(self.hists, "jk_ptjet_mjet_u_reco",ptreco  = reco_jet_mm.pt, mreco = reco_jet_mm.mass, jk = jk_index, weight = w)
                                        fill_hist(self.hists, "jk_ptjet_mjet_g_reco",ptreco  = reco_jet_mm.pt, mreco = reco_jet_mm.msoftdrop, jk = jk_index, weight = w)
    
                                    # fill_tunfold_hist_1d(hist = fill_hist(self.hists, "tunfold_reco_u"], dataset = dataset,
                                    #                      mass = reco_jet_mm.mass, pt = reco_jet_mm.pt,  systematic = syst, weight = w, recogen = 'reco')
                                    # fill_tunfold_hist_1d(hist = fill_hist(self.hists, "tunfold_reco_g"], dataset = dataset,
                                    #                      mass = reco_jet_mm.msoftdrop, pt = reco_jet_mm.pt, systematic = syst, weight = w, recogen = 'reco')
                                    if not self.minimal:
                                        fill_hist(self.hists, 'm_z_reco',dataset = dataset, mass = z_reco_mm.mass, weight = w, systematic = syst)
                                        fill_hist(self.hists, 'eta_z_reco',dataset = dataset,  eta = z_reco_mm.eta, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'pt_z_reco',dataset = dataset, pt = z_reco_mm.pt, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'phi_z_reco',dataset = dataset,  phi = z_reco_mm.phi, weight = w, systematic = syst)
                                        fill_hist(self.hists, 'eta_phi_z_reco',dataset = dataset,  phi = z_reco_mm.phi,eta = z_reco_mm.eta, weight = w, systematic = syst)
                                        
                                        
                                        fill_hist(self.hists, 'm_jet_reco',dataset = dataset, mreco = reco_jet_mm.mass, weight = w, systematic = syst)
                                        fill_hist(self.hists, 'eta_jet_reco',dataset = dataset, eta = reco_jet_mm.eta, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'pt_jet_reco',dataset = dataset, pt = reco_jet_mm.pt, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'phi_jet_reco',dataset = dataset, phi = reco_jet_mm.phi, weight = w, systematic = syst )
                                        fill_hist(self.hists, 'eta_phi_jet_reco',dataset = dataset,  phi = reco_jet_mm.phi,eta = reco_jet_mm.eta, weight = w, systematic = syst)
                                    
                                    if self.do_gen:
                                        fill_hist(self.hists, "response_matrix_u", dataset=dataset, channel = 'mm',
                                                                           ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                           mreco=reco_jet_mm.mass, mgen=gen_jet_mm.mass, systematic = syst,  weight = w )
                            
                                        fill_hist(self.hists, "response_matrix_g", dataset=dataset, channel = 'mm',
                                                                           ptreco=reco_jet_mm.pt,
                                                                           ptgen=gen_jet_mm.pt,
                                                                           mreco=reco_jet_mm.msoftdrop, 
                                                                           mgen=groomed_gen_jet_mm.mass,
                                                                           systematic = syst, weight = w)
                                    
                                        fill_hist(self.hists, "resp_mpt_u",dataset= dataset,
                                                                     ptreco = reco_jet_mm.pt,
                                                                     ptgen = gen_jet_mm.pt,
                                                                     mreco = reco_jet_mm.mass,
                                                                     mgen = gen_jet_mm.mass,
                                                                     mpt_reco = 2*np.log10(reco_jet_mm.mass/(reco_jet_mm.pt*0.8)),
                                                                     mpt_gen = 2*np.log10(gen_jet_mm.mass/(gen_jet_mm.pt*0.8)),
                                                                     systematic = syst, 
                                                                     weight = w)
                                        
                                        fill_hist(self.hists, "resp_mpt_g",dataset= dataset,
                                                                    ptreco = reco_jet_mm.pt,
                                                                    ptgen = gen_jet_mm.pt,
                                                                    mreco = reco_jet_mm.msoftdrop,
                                                                    mgen = groomed_gen_jet_mm.mass,
                                                                    mpt_reco = 2*np.log10(reco_jet_mm.msoftdrop/(reco_jet_mm.pt*0.8)),
                                                                    mpt_gen = 2*np.log10(groomed_gen_jet_mm.mass/(gen_jet_mm.pt*0.8)),
                                                                    systematic = syst, 
                                                                    weight = w)
                                        if not self.minimal:

                                            fill_hist(self.hists, "jk_response_matrix_u", dataset=dataset, 
                                                                               ptreco=reco_jet_mm.pt, ptgen=groomed_gen_jet_mm.pt,
                                                                               mreco=reco_jet_mm.mass, mgen=gen_jet_mm.mass, systematic = syst, jk = jk_index,  weight = w )
                                
                                            fill_hist(self.hists, "jk_response_matrix_g", dataset=dataset, 
                                                                               ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                               mreco=reco_jet_mm.msoftdrop, mgen=groomed_gen_jet_mm.mass, systematic = syst, jk = jk_index, weight = w )
                                            fill_hist(self.hists, "ptjet_mjet_g_gen",dataset=dataset, 
                                                                             ptgen = gen_jet_mm.pt, 
                                                                             mgen = groomed_gen_jet_mm.mass, 
                                                                             systematic = syst,
                                                                             weight = w) 

                                del weights_mm, z_reco_mm, reco_jet_mm
                                if self.do_gen:
                                    del z_gen_mm, gen_jet_mm, groomed_gen_jet_mm
                    else: ## running over outer loop of jet systematics
                        if cat == "ee":
    
                            events_ee = events[ee_sel]
                            weights_ee = weights[ee_sel]
                            if len(events_ee) > 0:
                                coffea_weights = Weights(size = len(events_ee), storeIndividual = True)
    
                                coffea_weights.add("init_weight", weights_ee)
    
                                coffea_weights.add(name = "pu", weight = events_ee.pu_nominal, weightUp = events_ee.pu_U, weightDown = events_ee.pu_D)
                                coffea_weights.add("q2", events_ee.q2_N, events_ee.q2_U, events_ee.q2_D)
                                coffea_weights.add("pdf", events_ee.pdf_N, events_ee.pdf_U, events_ee.pdf_D)
                                coffea_weights.add("prefiring", events_ee.prefiring_N, events_ee.prefiring_U, events_ee.prefiring_D)
                                
                                elereco_N = events_ee.elereco_N[:,0]*events_ee.elereco_N[:,1]
                                elereco_U = events_ee.elereco_U[:,0]*events_ee.elereco_U[:,1]
                                elereco_D = events_ee.elereco_D[:,0]*events_ee.elereco_D[:,1]
                                
                                coffea_weights.add(name = "elereco", weight = elereco_N, weightUp = elereco_U, weightDown = elereco_D)
                                
                                eleid_N = events_ee.eleid_N[:,0]*events_ee.eleid_N[:,1]
                                eleid_U = events_ee.eleid_U[:,0]*events_ee.eleid_U[:,1]
                                eleid_D = events_ee.eleid_D[:,0]*events_ee.eleid_D[:,1]
    
                                coffea_weights.add(name = "eleid", weight = eleid_N, weightUp = eleid_U, weightDown = eleid_D)
                                
                                eletrig_N = events_ee.eletrig_N[:,0]
                                eletrig_U = events_ee.eletrig_U[:,0]
                                eletrig_D = events_ee.eletrig_D[:,0]
    
                                coffea_weights.add(name = "eletrig", weight = eletrig_N, weightUp = eletrig_U, weightDown = eletrig_D)
                                
                                coffea_weights.add(name = "mureco",
                                                   weight = ak.ones_like(events_ee.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_ee.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_ee.event, dtype = float))
    
                                coffea_weights.add(name = "muid",
                                                   weight = ak.ones_like(events_ee.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_ee.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_ee.event, dtype = float))
    
                                coffea_weights.add(name = "mutrig",
                                                   weight = ak.ones_like(events_ee.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_ee.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_ee.event, dtype = float))

                                coffea_weights.add(name = "muiso",
                                                   weight = ak.ones_like(events_ee.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_ee.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_ee.event, dtype = float))
    
                                
                                del elereco_N, elereco_U, elereco_D, eleid_N, eleid_U, eleid_D
                                
                                z_reco_ee = z_reco[ee_sel]
                                reco_jet_ee = reco_jet[ee_sel]
                                #weights = weights[allsel_reco]
                                if self.do_gen:
                                    z_gen_ee = z_gen[ee_sel]
                                    gen_jet_ee = gen_jet[ee_sel]
                                    groomed_gen_jet_ee = groomed_gen_jet[ee_sel]
    
                                w = coffea_weights.weight()
    
                                fill_hist(self.hists, "ptjet_mjet_u_reco",channel = 'ee',dataset=dataset, ptreco = reco_jet_ee.pt, mreco = reco_jet_ee.mass, systematic = jet_syst, weight = w)
                                fill_hist(self.hists, "ptjet_mjet_g_reco",channel = 'ee',dataset=dataset, ptreco = reco_jet_ee.pt, mreco = reco_jet_ee.msoftdrop, systematic = jet_syst, weight = w)
                                
                                fill_hist(self.hists, "m_over_pt_u",
                                              dataset = dataset,
                                              ptreco = reco_jet_ee.pt,
                                              mpt_reco = 2*np.log10(reco_jet_ee.mass/(reco_jet_ee.pt*0.8)),
                                              mreco = reco_jet_ee.mass,
                                              systematic = jet_syst,
                                              weight = w
                                              )
                                    
                                fill_hist(self.hists, "m_over_pt_g",
                                          dataset = dataset,
                                          ptreco = reco_jet_ee.pt,
                                          mpt_reco = 2*np.log10(reco_jet_ee.msoftdrop/(reco_jet_ee.pt*0.8)),
                                          mreco = reco_jet_ee.msoftdrop,
                                          systematic = jet_syst,
                                          weight = w)
    
                                # fill_tunfold_hist_1d(hist = fill_hist(self.hists, "tunfold_reco_u"], dataset = dataset,
                                #                          mass = reco_jet_ee.mass, pt = reco_jet_ee.pt,  weight = w, recogen = 'reco', systematic = jet_syst)
                                # fill_tunfold_hist_1d(hist = fill_hist(self.hists, "tunfold_reco_g"], dataset = dataset,
                                #                          mass = reco_jet_ee.msoftdrop, pt = reco_jet_ee.pt, weight = w, recogen = 'reco', systematic = jet_syst)
                                if self.do_gen:
                                    fill_hist(self.hists, "response_matrix_u", dataset=dataset, channel = 'ee',
                                                                           ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                           mreco=reco_jet_ee.mass, mgen=gen_jet_ee.mass, systematic = jet_syst,  weight = w )
                            
                                    fill_hist(self.hists, "response_matrix_g", dataset=dataset, channel = 'ee',
                                                                           ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                           mreco=reco_jet_ee.msoftdrop, mgen=groomed_gen_jet_ee.mass, systematic = jet_syst, weight = w )

                                    fill_hist(self.hists, "resp_mpt_u",dataset= dataset,
                                                                     ptreco = reco_jet_ee.pt,
                                                                     ptgen = gen_jet_ee.pt,
                                                                     mreco = reco_jet_ee.mass,
                                                                     mgen = gen_jet_ee.mass,
                                                                     mpt_reco = 2*np.log10(reco_jet_ee.mass/(reco_jet_ee.pt*0.8)),
                                                                     mpt_gen = 2*np.log10(gen_jet_ee.mass/(gen_jet_ee.pt*0.8)),
                                                                     systematic = jet_syst, 
                                                                     weight = w)
                                        
                                    fill_hist(self.hists, "resp_mpt_g",dataset= dataset,
                                                                ptreco = reco_jet_ee.pt,
                                                                ptgen = gen_jet_ee.pt,
                                                                mreco = reco_jet_ee.msoftdrop,
                                                                mgen = groomed_gen_jet_ee.mass,
                                                                mpt_reco = 2*np.log10(reco_jet_ee.msoftdrop/(reco_jet_ee.pt*0.8)),
                                                                mpt_gen = 2*np.log10(groomed_gen_jet_ee.mass/(gen_jet_ee.pt*0.8)),
                                                                systematic = jet_syst, 
                                                                weight = w)
                                    
                                    # fill_hist(self.hists, "ptjet_mjet_g_gen",dataset=dataset, 
                                    #                                      ptgen = gen_jet_ee.pt, 
                                    #                                      mgen = groomed_gen_jet_ee.mass, 
                                    #                                      systematic = jet_syst,
                                    #                                      weight = w)    

                                del weights_ee, z_reco_ee, reco_jet_ee
                                if self.do_gen:
                                    del z_gen_ee, gen_jet_ee, groomed_gen_jet_ee
                                
                        if cat == "mm":
                            events_mm = events[mm_sel]
                            weights_mm = weights[mm_sel]
                            if len(events_mm) > 0:
                                coffea_weights = Weights(size = len(events_mm), storeIndividual = True)
    
                                coffea_weights.add("init_weight", weights_mm)
    
                                
                                coffea_weights.add(name = "pu", weight = events_mm.pu_nominal, weightUp = events_mm.pu_U, weightDown = events_mm.pu_D)
                                coffea_weights.add("q2", events_mm.q2_N, events_mm.q2_U, events_mm.q2_D)
                                coffea_weights.add("pdf", events_mm.pdf_N, events_mm.pdf_U, events_mm.pdf_D)
                                coffea_weights.add("prefiring", events_mm.prefiring_N, events_mm.prefiring_U, events_mm.prefiring_D)
                                
                                mureco_N = events_mm.mureco_N[:,0]*events_mm.mureco_N[:,1]
                                mureco_U = events_mm.mureco_U[:,0]*events_mm.mureco_U[:,1]
                                mureco_D = events_mm.mureco_D[:,0]*events_mm.mureco_D[:,1]
                                
                                coffea_weights.add(name = "mureco", weight = mureco_N, weightUp = mureco_U, weightDown = mureco_D)
                                
                                muid_N = events_mm.muid_N[:,0]*events_mm.muid_N[:,1]
                                muid_U = events_mm.muid_U[:,0]*events_mm.muid_U[:,1]
                                muid_D = events_mm.muid_D[:,0]*events_mm.muid_D[:,1]
    
                                coffea_weights.add(name = "muid", weight = muid_N, weightUp = muid_U, weightDown = muid_D)
    
                                mutrig_N = events_mm.mutrig_N[:,0]*events_mm.mutrig_N[:,1]
                                mutrig_U = events_mm.mutrig_U[:,0]*events_mm.mutrig_U[:,1]
                                mutrig_D = events_mm.mutrig_D[:,0]*events_mm.mutrig_D[:,1]
    
                                coffea_weights.add(name = "mutrig", weight = mutrig_N, weightUp = mutrig_U, weightDown = mutrig_D)


                                muiso_N = events_mm.muiso_N[:,0]*events_mm.muiso_N[:,1]
                                muiso_U = events_mm.muiso_U[:,0]*events_mm.muiso_U[:,1]
                                muiso_D = events_mm.muiso_D[:,0]*events_mm.muiso_D[:,1]
    
                                coffea_weights.add(name = "muiso", weight = muiso_N, weightUp = muiso_U, weightDown = muiso_D)
                                
                                coffea_weights.add(name = "elereco",
                                                   weight = ak.ones_like(events_mm.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_mm.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_mm.event, dtype = float))
                                
                                coffea_weights.add(name = "eleid",
                                                   weight = ak.ones_like(events_mm.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_mm.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_mm.event, dtype = float))
                                coffea_weights.add(name = "eletrig",
                                                   weight = ak.ones_like(events_mm.event, dtype = float), 
                                                   weightUp =  ak.ones_like(events_mm.event, dtype = float),
                                                   weightDown =  ak.ones_like(events_mm.event, dtype = float))
      
                                del mureco_N, mureco_U, mureco_D, muid_N, muid_U, muid_D, mutrig_N, mutrig_U, mutrig_D
    
                                
    
                                
                                z_reco_mm = z_reco[mm_sel]
                                reco_jet_mm = reco_jet[mm_sel]
                                #weights = weights[allsel_reco]
                                if self.do_gen:
                                    z_gen_mm = z_gen[mm_sel]
                                    gen_jet_mm = gen_jet[mm_sel]
                                    groomed_gen_jet_mm = groomed_gen_jet[mm_sel]
    
                    
                                systematics = [syst for syst in self.systematics if syst not in mm_sys_var_list]
                                #print("for ee case ", systematics)
    
    
                                w = coffea_weights.weight()
     
                                fill_hist(self.hists, "ptjet_mjet_u_reco",dataset=dataset, ptreco = reco_jet_mm.pt, mreco = reco_jet_mm.mass, channel = 'mm', systematic = jet_syst, weight = w)
                                fill_hist(self.hists, "ptjet_mjet_g_reco",dataset=dataset, ptreco = reco_jet_mm.pt, mreco = reco_jet_mm.msoftdrop,channel = 'mm',  systematic = jet_syst, weight = w)    

                                fill_hist(self.hists, "m_over_pt_u",
                                              dataset = dataset,
                                              ptreco = reco_jet_mm.pt,
                                              mpt_reco = 2*np.log10(reco_jet_mm.mass/(reco_jet_mm.pt*0.8)),
                                              mreco = reco_jet_mm.mass,
                                              systematic = syst,
                                              weight = w
                                              )
                                    
                                fill_hist(self.hists, "m_over_pt_g",
                                          dataset = dataset,
                                          ptreco = reco_jet_mm.pt,
                                          mpt_reco = 2*np.log10(reco_jet_mm.msoftdrop/(reco_jet_mm.pt*0.8)),
                                          mreco = reco_jet_mm.msoftdrop,
                                          systematic = syst,
                                          weight = w)
    
    
                                # fill_tunfold_hist_1d(hist = fill_hist(self.hists, "tunfold_reco_u"], dataset = dataset,
                                #                          mass = reco_jet_mm.mass, pt = reco_jet_mm.pt,   weight = w, recogen = 'reco', systematic = jet_syst)
                                # fill_tunfold_hist_1d(hist = fill_hist(self.hists, "tunfold_reco_g"], dataset = dataset,
                                #                          mass = reco_jet_mm.msoftdrop, pt = reco_jet_mm.pt,  weight = w, recogen = 'reco', systematic = jet_syst)
                                if self.do_gen:
                                    fill_hist(self.hists, "response_matrix_u", dataset=dataset, channel = 'mm',
                                                                       ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                       mreco=reco_jet_mm.mass, mgen=gen_jet_mm.mass, systematic = jet_syst,  weight = w )
                        
                                    fill_hist(self.hists, "response_matrix_g", dataset=dataset, channel = 'mm',
                                                                       ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                       mreco=reco_jet_mm.msoftdrop, mgen=groomed_gen_jet_mm.mass, systematic = jet_syst, weight = w )



                                    fill_hist(self.hists, "resp_mpt_u",dataset= dataset,
                                                                     ptreco = reco_jet_mm.pt,
                                                                     ptgen = gen_jet_mm.pt,
                                                                     mreco = reco_jet_mm.mass,
                                                                     mgen = gen_jet_mm.mass,
                                                                     mpt_reco = 2*np.log10(reco_jet_mm.mass/(reco_jet_mm.pt*0.8)),
                                                                     mpt_gen = 2*np.log10(gen_jet_mm.mass/(gen_jet_mm.pt*0.8)),
                                                                     systematic = jet_syst, 
                                                                     weight = w)
                                        
                                    fill_hist(self.hists, "resp_mpt_g",dataset= dataset,
                                                                ptreco = reco_jet_mm.pt,
                                                                ptgen = gen_jet_mm.pt,
                                                                mreco = reco_jet_mm.msoftdrop,
                                                                mgen = groomed_gen_jet_mm.mass,
                                                                mpt_reco = 2*np.log10(reco_jet_mm.msoftdrop/(reco_jet_mm.pt*0.8)),
                                                                mpt_gen = 2*np.log10(groomed_gen_jet_mm.mass/(gen_jet_mm.pt*0.8)),
                                                                systematic = jet_syst, 
                                                                weight = w)

                                    # fill_hist(self.hists, "ptjet_mjet_g_gen",dataset=dataset, 
                                    #                                      ptgen = gen_jet_mm.pt, 
                                    #                                      mgen = groomed_gen_jet_mm.mass, 
                                    #                                      systematic = jet_syst,
                                    #                                      weight = w) 
                                    # fill_tunfold_hist_2d(dataset = dataset, 
                                    #                          hist = fill_hist(self.hists, "tunfold_migration_u"], mass_gen = gen_jet_mm.mass, pt_gen = gen_jet_mm.pt, 
                                    #                          mass_reco = reco_jet_mm.mass, pt_reco = reco_jet_mm.pt, systematic = jet_syst, weight = w)
                                    # fill_tunfold_hist_2d(dataset = dataset, 
                                    #                          hist = fill_hist(self.hists, "tunfold_migration_g"], mass_gen = groomed_gen_jet_mm.mass, pt_gen = groomed_gen_jet_mm.pt, 
                                    #                          mass_reco = reco_jet_mm.msoftdrop, pt_reco = reco_jet_mm.pt, systematic = jet_syst, weight = w)
                                del weights_mm, z_reco_mm, reco_jet_mm
                                if self.do_gen:
                                    del z_gen_mm, gen_jet_mm, groomed_gen_jet_mm
    
                    
    
                    
                            
                del events
                if not self.do_gen:
                    break ###only doing nominal when dealing with data
            del events0
            if not self.do_jk:
                break ## break when not doing jk
        
        for name in sel.names:
            self.hists["cutflow"][dataset][ht_bin][name] += sel.all(name).sum()
        t1 = time.time()
        print('total time taken ', t1-t0)
        return self.hists

    
    def postprocess(self, accumulator):
        return accumulator

    

    
