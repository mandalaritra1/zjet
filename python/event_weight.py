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
        
        self.minimal = False
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


        ht_axis = hist.axis.StrCategory([],growth = True, name = "ht_bin", label = "h_T bin")
        
        weight_axis = hist.axis.Regular(100, 0, 10, name="corrWeight", label=r"Weight")
        #### weight to check what is causing this
        
        self.gen_binning = binning.gen_binning
        self.reco_binning = binning.reco_binning


        
        ### Plots of things during the selection process / for debugging with fine binning
        h_njet_gen = hist.Hist(dataset_axis, n_axis, storage="weight", label="Counts")
        h_njet_reco = hist.Hist(dataset_axis, n_axis, storage="weight", label="Counts")
        h_ptjet_gen_pre = hist.Hist(dataset_axis, pt_axis, storage="weight", label="Counts")
        h_ptjet_reco_over_gen = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_drjet_reco_gen = hist.Hist(dataset_axis, dr_fine_axis, storage="weight", label="Counts")
        h_ptz_gen = hist.Hist(dataset_axis, pt_axis, storage="weight", label="Counts")
        h_ptz_reco = hist.Hist(dataset_axis, pt_axis, storage="weight", label="Counts")        
        h_mz_gen = hist.Hist(dataset_axis, zmass_axis, storage="weight", label="Counts")
        h_mz_reco = hist.Hist(dataset_axis, zmass_axis, storage="weight", label="Counts")
        h_mz_reco_over_gen = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_dr_z_jet_gen = hist.Hist(dataset_axis, dr_axis, storage="weight", label="Counts")
        h_dr_z_jet_reco = hist.Hist(dataset_axis, dr_axis, storage="weight", label="Counts")
        h_dphi_z_jet_gen = hist.Hist(dataset_axis, dphi_axis, storage="weight", label="Counts")
        h_dphi_z_jet_reco = hist.Hist(dataset_axis, dphi_axis, storage="weight", label="Counts")

        h_dphi_lep_jet_reco = hist.Hist(dataset_axis, dphi_axis, storage="weight", label="Counts")

        
        h_ptasym_z_jet_gen = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_ptasym_z_jet_reco = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_ptfrac_z_jet_gen = hist.Hist(dataset_axis, ptreco_axis, frac_axis, storage="weight", label="Counts")
        h_ptfrac_z_jet_reco = hist.Hist(dataset_axis, ptreco_axis, frac_axis, storage="weight", label="Counts")
        h_dr_gen_subjet = hist.Hist(dataset_axis, dr_axis, storage="weight", label="Counts")
        h_dr_reco_to_gen_subjet = hist.Hist(dataset_axis, dr_axis, storage="weight", label="Counts")


        ### Plots to check ht_contributions
        h_jet_mass_u_presel = hist.Hist(dataset_axis, mreco_axis, ht_axis, storage = 'weight', label = 'Counts')
        h_jet_mass_u_postsel = hist.Hist(dataset_axis, mreco_axis, ht_axis, storage = 'weight', label = 'Counts')

        h_jet_mass_g_presel = hist.Hist(dataset_axis, mreco_axis, ht_axis, storage = 'weight', label = 'Counts')
        h_jet_mass_g_postsel = hist.Hist(dataset_axis, mreco_axis, ht_axis, storage = 'weight', label = 'Counts')
        
        ### 
        #### Newly added for data_MC plots
        
        #h_ptz_gen = hist.Hist(dataset_axis, pt_axis, storage="weight", label="Counts")
    
        #h_mz_gen = hist.Hist(dataset_axis, zmass_axis, storage="weight", label="Counts")
        #mass, eta, pt, phi

        
        h_m_z_reco = hist.Hist(dataset_axis, zmass_axis, syst_axis, storage="weight", label="Counts")
        h_eta_z_reco = hist.Hist(dataset_axis, eta_axis, syst_axis, storage="weight", label="Counts")
        h_pt_z_reco = hist.Hist(dataset_axis, pt_axis, syst_axis, storage="weight", label="Counts")
        h_phi_z_reco = hist.Hist(dataset_axis, phi_axis, syst_axis, storage="weight", label="Counts")

        h_m_jet_reco = hist.Hist(dataset_axis, mreco_axis, syst_axis, storage="weight", label="Counts")
        h_eta_jet_reco = hist.Hist(dataset_axis, eta_axis, syst_axis, storage="weight", label="Counts")
        h_pt_jet_reco = hist.Hist(dataset_axis, pt_axis, syst_axis, storage="weight", label="Counts")
        h_phi_jet_reco = hist.Hist(dataset_axis, phi_axis, syst_axis, storage="weight", label="Counts")
        
        
        ####################################
        # Fakes and misses
        ####################################
        h_fakes_u = hist.Hist(dataset_axis, ptreco_axis, mreco_axis,  storage="weight", label="Counts")
        h_misses_u = hist.Hist(dataset_axis, ptgen_axis, mgen_axis,  storage="weight", label="Counts")

        h_fakes_g = hist.Hist(dataset_axis, ptreco_axis, mreco_axis,  storage="weight", label="Counts")
        h_misses_g = hist.Hist(dataset_axis, ptgen_axis, mgen_axis,  storage="weight", label="Counts")
        
        ### Plots to be unfolded
        h_ptjet_mjet_u_reco = hist.Hist(dataset_axis, ptreco_axis, mreco_axis, syst_axis, storage="weight", label="Counts")
        h_ptjet_mjet_g_reco = hist.Hist(dataset_axis, ptreco_axis, mreco_axis, syst_axis, storage="weight", label="Counts")
        ### Plots for comparison
        h_ptjet_mjet_u_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, storage="weight", label="Counts")        
        h_ptjet_mjet_g_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, syst_axis,storage="weight", label="Counts")
        
        
        ### Plots to get JMR and JMS in MC
        h_m_u_jet_reco_over_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, frac_axis, storage="weight", label="Counts")
        h_m_g_jet_reco_over_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, frac_axis, storage="weight", label="Counts")

        h_delta_m_u = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, binning.diff_axis, storage="weight", label="Counts")
        h_delta_m_g = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, binning.diff_axis, storage="weight", label="Counts")

        

        ### Plots for JER and JEC in MC

        h_pt_u_jet_reco_over_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis,  frac_axis, storage="weight", label="Counts")
        h_pt_g_jet_reco_over_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis,  frac_axis, storage="weight", label="Counts")

        h_delta_pt_u = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, binning.diff_axis_large, storage="weight", label="Counts")
        h_delta_pt_g = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, binning.diff_axis_large, storage="weight", label="Counts")
        
        ### Plots for the analysis in the proper binning
        # h_response_matrix_u = hist.Hist(dataset_axis,
        #                                 ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis,
        #                                 storage="weight", label="Counts")
        # h_response_matrix_g = hist.Hist(dataset_axis,
        #                                 ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis,
        #                                 storage="weight", label="Counts")
        #mgen_axis_fine = hist.axis.Variable( [0,10,20,30,40,50,60,70,80,90,100,125,150,175,200,6200,13000], name="mgen", label=r"m_{GEN} (GeV)")
        h_response_matrix_u = hist.Hist(dataset_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis,
                                        storage="weight", label="Counts")
        h_response_matrix_g = hist.Hist(dataset_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis,
                                        storage="weight", label="Counts")

        # h_response_matrix_u = hist.Hist(dataset_axis,
        #                                 ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis,
        #                                 storage="weight", label="Counts")
        # h_response_matrix_g = hist.Hist(dataset_axis,
        #                                 ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis,
        #                                 storage="weight", label="Counts")
        
        h_jk_response_matrix_u = hist.Hist(dataset_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis, binning.jackknife_axis,
                                        storage="weight", label="Counts")
        h_jk_response_matrix_g = hist.Hist(dataset_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis, binning.jackknife_axis,
                                        storage="weight", label="Counts")
        
        # h_tunfold_gen_u = hist.Hist(dataset_axis, binning.gen_axis, syst_axis, storage = 'weight', label = 'Counts')
        # h_tunfold_gen_g = hist.Hist(dataset_axis, binning.gen_axis, syst_axis, storage = 'weight', label = 'Counts')
        
        # h_tunfold_reco_u = hist.Hist(dataset_axis, binning.reco_axis,  syst_axis, storage = 'weight', label = 'Counts')
        # h_tunfold_reco_g = hist.Hist(dataset_axis, binning.reco_axis, syst_axis, storage = 'weight', label = 'Counts')
        
        # h_tunfold_migration_u = hist.Hist(dataset_axis, binning.gen_axis, syst_axis, binning.reco_axis, storage = "weight", label = "Counts" )
        # h_tunfold_migration_g = hist.Hist(dataset_axis, binning.gen_axis, syst_axis, binning.reco_axis, storage = "weight", label = "Counts" )

        # jackknife_response_u = hist.Hist(dataset_axis, binning.jackknife_axis, binning.gen_axis, syst_axis, binning.reco_axis, storage = "weight", label = "Counts" )
        # jackknife_response_g = hist.Hist(dataset_axis, binning.jackknife_axis, binning.gen_axis, syst_axis, binning.reco_axis, storage = "weight", label = "Counts" )
        
        cutflow = {}
        jackknife_total = { '0': 0, '1': 0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0 }

        if self.minimal:
            self.hists = {
            
            "njet_gen":h_njet_gen,
            "total_weight": hist.Hist( weight_axis, label = 'Counts'),

            "fakes_u": h_fakes_u,
            "misses_u": h_misses_u,

            "m_u_jet_reco_over_gen":h_m_u_jet_reco_over_gen,
            "m_g_jet_reco_over_gen":h_m_g_jet_reco_over_gen,
            "delta_m_u":   h_delta_m_u ,
            "delta_m_g":   h_delta_m_g ,

            "pt_u_jet_reco_over_gen":h_pt_u_jet_reco_over_gen,
            "pt_g_jet_reco_over_gen":h_pt_g_jet_reco_over_gen,
            "delta_pt_u":   h_delta_pt_u ,
            "delta_pt_g":   h_delta_pt_g ,

            
            "response_matrix_u":h_response_matrix_u,
            "response_matrix_g":h_response_matrix_g,
            "cutflow":cutflow,

        }
            
        if self.minimal & (not self.do_gen):
            self.hists = {
            
            "njet_gen":h_njet_gen,
            "total_weight": hist.Hist( weight_axis, label = 'Counts'),


            "ptjet_mjet_u_reco":h_ptjet_mjet_u_reco, 

            "ptjet_mjet_g_reco":h_ptjet_mjet_g_reco, 


            "cutflow":cutflow,

        }
        
        self.hists = {
            # "tunfold_gen_u":h_tunfold_gen_u,
            # "tunfold_gen_g": h_tunfold_gen_g,
            # "tunfold_reco_u": h_tunfold_reco_u,
            # "tunfold_reco_g": h_tunfold_reco_g,
            # "tunfold_migration_u":h_tunfold_migration_u,
            # "tunfold_migration_g":h_tunfold_migration_g,
            
            "njet_gen":h_njet_gen,
            "total_weight": hist.Hist( weight_axis, label = 'Counts'),
            # "pdfweight" : hist.Hist(dataset_axis, weight_axis, label = 'Counts'),
            # "elerecoweight": hist.Hist( dataset_axis, weight_axis, label = 'Counts'),
            # "murecoweight": hist.Hist( dataset_axis, weight_axis, label = 'Counts'),
            # "muidweight": hist.Hist( dataset_axis, weight_axis, label = 'Counts'),
            # "q2weight": hist.Hist(dataset_axis, weight_axis, label = 'Counts'),
            # "mutrigweight": hist.Hist(dataset_axis, weight_axis, label = 'Counts'),
            # "prefiringweight": hist.Hist(dataset_axis, weight_axis, label = 'Counts'),
            "dphi_lep_jet_reco": h_dphi_lep_jet_reco,
            "njet_reco":h_njet_reco,
            "ptjet_gen_pre":h_ptjet_gen_pre, 
            "ptjet_mjet_u_gen":h_ptjet_mjet_u_gen, 
            "ptjet_mjet_u_reco":h_ptjet_mjet_u_reco, 
            "ptjet_mjet_g_gen":h_ptjet_mjet_g_gen, 
            "ptjet_mjet_g_reco":h_ptjet_mjet_g_reco, 
            "ptjet_reco_over_gen":h_ptjet_reco_over_gen,
            "drjet_reco_gen":h_drjet_reco_gen,
            "ptz_gen":h_ptz_gen,
            "ptz_reco":h_ptz_reco,
            "mz_gen":h_mz_gen,
            "mz_reco":h_mz_reco,            
            "mz_reco_over_gen":h_mz_reco_over_gen,
            "fakes_u": h_fakes_u,
            "misses_u": h_misses_u,
            "fakes_g": h_fakes_g,
            "misses_g": h_misses_g,
            "dr_z_jet_gen":h_dr_z_jet_gen,
            "dr_z_jet_reco":h_dr_z_jet_reco,            
            "dphi_z_jet_gen":h_dphi_z_jet_gen,
            "dphi_z_jet_reco":h_dphi_z_jet_reco,
            "ptasym_z_jet_gen":h_ptasym_z_jet_gen,
            "ptasym_z_jet_reco":h_ptasym_z_jet_reco,
            "ptfrac_z_jet_gen":h_ptfrac_z_jet_gen,
            "ptfrac_z_jet_reco":h_ptfrac_z_jet_reco,


            "m_u_jet_reco_over_gen":h_m_u_jet_reco_over_gen,
            "m_g_jet_reco_over_gen":h_m_g_jet_reco_over_gen,
            "delta_m_u":   h_delta_m_u ,
            "delta_m_g":   h_delta_m_g ,

            "pt_u_jet_reco_over_gen":h_pt_u_jet_reco_over_gen,
            "pt_g_jet_reco_over_gen":h_pt_g_jet_reco_over_gen,
            "delta_pt_u":   h_delta_pt_u ,
            "delta_pt_g":   h_delta_pt_g ,
            
            "dr_gen_subjet":h_dr_gen_subjet,
            "dr_reco_to_gen_subjet":h_dr_reco_to_gen_subjet,
            "response_matrix_u":h_response_matrix_u,
            "response_matrix_g":h_response_matrix_g,
            "cutflow":cutflow,
            'jackknife_total': jackknife_total,
            # 'jackknife_response_u': jackknife_response_u,
            # 'jackknife_response_g': jackknife_response_g,
            'jk_response_matrix_u': h_jk_response_matrix_u,
            'jk_response_matrix_g':h_jk_response_matrix_g,

            'm_z_reco':h_m_z_reco,
            'eta_z_reco':h_eta_z_reco,
            'pt_z_reco':h_pt_z_reco ,
            'phi_z_reco':h_phi_z_reco,
            'm_jet_reco':h_m_jet_reco,
            'eta_jet_reco':h_eta_jet_reco,
            'pt_jet_reco':h_pt_jet_reco,
            'phi_jet_reco':h_phi_jet_reco,
            ### Presel Histogram, not needed for now
            
            # 'jet_mass_u_presel': h_jet_mass_u_presel,
            # 'jet_mass_u_postsel': h_jet_mass_u_postsel,
            # 'jet_mass_g_presel': h_jet_mass_g_presel,
            # 'jet_mass_g_postsel': h_jet_mass_g_postsel
        }
        
        #self.systematics = ['nominal', 'puUp', 'puDown', "elerecoUp", "elerecoDown" ] 

        
        if do_syst and (syst_list == None): # in case systematic flag is enabled but no list of systematics is provied
            self.systematics = ['nominal', 'puUp', 'puDown' , 'elerecoUp', 'elerecoDown', 
                                'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp', 'murecoDown', 
                                'muidUp', 'muidDown', 'mutrigUp', 'mutrigDown', 
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
        return self.hists

    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events_all):

        print("Only weights")
        import time
        t0  = time.time()

        dataset = events_all.metadata['dataset']

        filename = events_all.metadata['filename']
        
        

            
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
            ht_bin = 'all'
            #print("IOV ", IOV, ", era ", era)
            lumi_mask = np.array(self.lumimasks[IOV](events_all.run, events_all.luminosityBlock), dtype=bool)
            events_all = events_all[lumi_mask]

            print("Events after lumi mask ", len(events_all))


        


        #### Setting up storage for cutflows ####
        
        if dataset not in self.hists["cutflow"]:
            self.hists["cutflow"][dataset] = {}
        if ht_bin not in self.hists["cutflow"][dataset]:
            self.hists['cutflow'][dataset][ht_bin] = defaultdict(int)

        
        print("Initial Total Event ", len(events_all))

        if self.do_gen:
            if "LHEWeight" in events_all.fields:
                weights = events_all["LHEWeight"].originalXWGTUP
            else:
                weights = np.full( len( events_all ), 1.0 )
    
            if ht_bin not in self.hists["cutflow"][dataset]:
                self.hists["cutflow"][dataset][ht_bin]["sumw"] = 0
            self.hists["cutflow"][dataset][ht_bin]["sumw"] += np.sum(weights)
        
        systematic_list = ['pdf_N', 'pdf_U', 'pdf_D', 'q2_N', 'q2_U', 'q2_D', "pu_nominal", "pu_U", "pu_D", "prefiring_N", "prefiring_U", "prefiring_D", 
                              "elereco_N", "elereco_U", "elereco_D", "eleid_N", "eleid_U", "eleid_D" ,"eletrig_N","eletrig_U","eletrig_D" ,"mureco_N", "mureco_U", "mureco_D",  "muid_N", "muid_U", "muid_D", "mutrig_N", "mutrig_U", "mutrig_D"]
        
        index_list = np.arange(len(events_all)) ## for jackknife resampling
        for jk_index in range(0,10):  ## loops from 0 to 10 in case do_jk flag is enabled, otherwise breaks at 0
            jk_sel = ak.where(index_list%10 == jk_index, False, True) ## splitting events into 10 distinct parts

            if self.do_jk:
                print("Now doing jackknife {}".format(jk_index))
                events0 = events_all[jk_sel]
            else:
                events0 = events_all 
            del jk_sel
                      
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
                events0["muid_N"] = GetMuonSF(IOV, "IDISO", np.abs(events0.Muon.eta), events0.Muon.pt)
                events0["muid_U"] = GetMuonSF(IOV, "IDISO", np.abs(events0.Muon.eta), events0.Muon.pt, "systup")
                events0["muid_D"] = GetMuonSF(IOV, "IDISO", np.abs(events0.Muon.eta), events0.Muon.pt, "systdown")
    
    
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

                
                for systematic in systematic_list:
                    if "mu" in systematic:
                        events0[systematic] = ak.ones_like(events0.Muon.pt)
                    elif "ele" in systematic:
                        events0[systematic] = ak.ones_like(events0.Electron.pt)
                    else:
                        events0[systematic]= ak.ones_like(events0.event, dtype=float) ## setting everything to 1 in case of DATA
    
            #####################################
            ### Initialize selection
            #####################################
            sel = PackedSelection()
            if not self.do_gen:
                sel.add("npv", events0.PV.npvsGood > 0)



            
            nominal_weights = [s for s in systematic_list if s.endswith('_N') or s == 'pu_nominal']
            total_weight = weights
            print("t weight",total_weight)
            print("p weight",events0['mureco_N'])
            print(ak.num(events0.Muon))
            mu_weight = ak.where(ak.num(events0.Muon) == 0, 1, ak.prod(events0['mureco_N'], axis = 1) *ak.prod(events0['muid_N'], axis = 1)*ak.prod(events0['mutrig_N'], axis = 1))
            ele_weight = ak.where(ak.num(events0.Electron) == 0, 1, ak.prod(events0['elereco_N'], axis = 1) *ak.prod(events0['eleid_N'], axis = 1)*ak.prod(events0['eletrig_N'], axis = 1))


            print("combi weight", total_weight*events0['pu_nominal'] )
            for s in nominal_weights:
                if s.startswith('ele') or s.startswith('mu'):
                    continue
                print(s)
                total_weight = total_weight*events0[s]
            total_weight = total_weight*mu_weight*ele_weight

            if ht_bin not in self.hists["cutflow"][dataset]:
                self.hists["cutflow"][dataset][ht_bin]["total_weight"] = 0
            self.hists["cutflow"][dataset][ht_bin]["total_weight"] += np.sum(total_weight)
            break

        t1 = time.time()
        print('total time taken ', t1-t0)
        return self.hists

    def postprocess(self, accumulator):
        return accumulator