import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
import os
import sys
import pandas as pd
import time
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.analysis_tools import PackedSelection
from collections import defaultdict
from smp_utils import *
import tokenize as tok
import re
from cms_utils import *
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory

from weight_class import Weights

class QJetMassProcessor(processor.ProcessorABC):
    '''
    Processor to run a Z+jets jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, do_gen=True, ptcut=200., etacut = 2.5, ptcut_ee = 40., ptcut_mm = 29., skimfilename=None):
        
        self.lumimasks = getLumiMaskRun2()
        
        # should have separate lower ptcut for gen
        self.do_gen=do_gen
        self.ptcut = ptcut
        self.etacut = etacut        
        self.lepptcuts = [ptcut_ee, ptcut_mm]
        
        if skimfilename != None: 
            if ".root" in skimfilename: 
                self.skimfilename = skimfilename.split(".root")[0]
            else: 
                self.skimfilename = skimfilename
                
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
        h_ptasym_z_jet_gen = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_ptasym_z_jet_reco = hist.Hist(dataset_axis, frac_axis, storage="weight", label="Counts")
        h_ptfrac_z_jet_gen = hist.Hist(dataset_axis, ptreco_axis, frac_axis, storage="weight", label="Counts")
        h_ptfrac_z_jet_reco = hist.Hist(dataset_axis, ptreco_axis, frac_axis, storage="weight", label="Counts")
        h_dr_gen_subjet = hist.Hist(dataset_axis, dr_axis, storage="weight", label="Counts")
        h_dr_reco_to_gen_subjet = hist.Hist(dataset_axis, dr_axis, storage="weight", label="Counts")
        
        ####################################
        # Fakes and misses
        ####################################
        h_fakes = hist.Hist(dataset_axis, ptreco_axis, mreco_axis,  storage="weight", label="Counts")
        h_misses = hist.Hist(dataset_axis, ptgen_axis, mgen_axis,  storage="weight", label="Counts")

        
        ### Plots to be unfolded
        h_ptjet_mjet_u_reco = hist.Hist(dataset_axis, ptreco_axis, mreco_axis, storage="weight", label="Counts")
        h_ptjet_mjet_g_reco = hist.Hist(dataset_axis, ptreco_axis, mreco_axis, storage="weight", label="Counts")
        ### Plots for comparison
        h_ptjet_mjet_u_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, storage="weight", label="Counts")        
        h_ptjet_mjet_g_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, storage="weight", label="Counts")
        
        
        ### Plots to get JMR and JMS in MC
        h_m_u_jet_reco_over_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, frac_axis, storage="weight", label="Counts")
        h_m_g_jet_reco_over_gen = hist.Hist(dataset_axis, ptgen_axis, mgen_axis, frac_axis, storage="weight", label="Counts")
        
        ### Plots for the analysis in the proper binning
        h_response_matrix_u = hist.Hist(dataset_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis,
                                        storage="weight", label="Counts")
        h_response_matrix_g = hist.Hist(dataset_axis,
                                        ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis,
                                        storage="weight", label="Counts")
        
        cutflow = {}
        
        self.hists = {
            "njet_gen":h_njet_gen,

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
            "fakes": h_fakes,
            "misses": h_misses,
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
            "dr_gen_subjet":h_dr_gen_subjet,
            "dr_reco_to_gen_subjet":h_dr_reco_to_gen_subjet,
            "response_matrix_u":h_response_matrix_u,
            "response_matrix_g":h_response_matrix_g,
            "cutflow":cutflow
        }
        
        #self.systematics = ['nominal', 'puUp', 'puDown', "elerecoUp", "elerecoDown" ] 
        self.systematics = ['nominal', 'puUp', 'puDown' , 'elerecoUp', 'elerecoDown', 'murecoUp', 'murecoDown'] 
        self.jet_systematics =  ["nominal", "JES_up", "JES_down", "JER_up", "JER_down"]
        
        
        ## This is for rejecting events with large weights
        self.means_stddevs = defaultdict()

        
    
    @property
    def accumulator(self):
        #return self._histos
        return self.hists

    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events_original):

        #print(events.metadata.keys())
        dataset = events_original.metadata['dataset']
        print(dataset)
        #lenprint(events.metadata['version'])
        filename = events_original.metadata['filename']

        
        if dataset not in self.hists["cutflow"]:
            self.hists["cutflow"][dataset] = defaultdict(int)
            
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'UL2016APV', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')


        
        #####################################
        #### Find the era from the file name
        #### Apply the good lumi mask
#         #####################################
        
        era = None

        #####################################
        ### Initialize selection
        #####################################
        






        #####################################
        ### Remove events with very large gen weights (>2 sigma)
        #####################################
            

        if dataset not in self.means_stddevs : 
            average = np.average( events_original["LHEWeight"].originalXWGTUP )
            stddev = np.std( events_original["LHEWeight"].originalXWGTUP )
            self.means_stddevs[dataset] = (average, stddev)            
        average,stddev = self.means_stddevs[dataset]
        vals = (events_original["LHEWeight"].originalXWGTUP - average ) / stddev
        self.hists["cutflow"][dataset]["all events"] += len(events_original)
        events_original = events_original[ np.abs(vals) < 2 ]
        self.hists["cutflow"][dataset]["weights cut"] += len(events_original)

        
        events_original = events_original[events_original.PV.npvsGood > 0]
        self.hists["cutflow"][dataset]["npv_v2"] += len(events_original) #good npv cut

        events_original = events_original[ak.sum( (events_original.GenJetAK8.pt > 136.) & (np.abs(events_original.GenJetAK8.eta) < 2.5), axis=1 ) >= 1]
        events_original = events_original[ak.sum( (events_original.FatJet.pt > 170.) & (np.abs(events_original.FatJet.eta) < 2.5), axis=1 ) >= 1]
        
        
        #####################################
        ### Initialize event weight to gen weight
        #####################################
        if len(events_original)>0:
            coffea_weights = Weights(size = len(events_original), storeIndividual = True)
        
            weights = events_original["LHEWeight"].originalXWGTUP
            coffea_weights.add("init_weight", weights)
    
            coffea_weights.add(name = "pu",
                               weight = GetPUSF(IOV, np.array(events_original.Pileup.nTrueInt)),
                              weightUp = GetPUSF(IOV, np.array(events_original.Pileup.nTrueInt), "up"),
                              weightDown = GetPUSF(IOV, np.array(events_original.Pileup.nTrueInt), "down"))
    
            
            ele0eta = ak.firsts(events_original.Electron[:,0:1].eta)
            ele0pt = ak.firsts(events_original.Electron[:,0:1].pt)
            ele1eta = ak.firsts(events_original.Electron[:,1:2].eta)
            ele1pt = ak.firsts(events_original.Electron[:,1:2].pt)
    
            mu0eta = ak.firsts(events_original.Muon[:,0:1].eta)
            mu0pt = ak.firsts(events_original.Muon[:,0:1].pt)
            mu1eta = ak.firsts(events_original.Muon[:,1:2].eta)
            mu1pt = ak.firsts(events_original.Muon[:,1:2].pt)
    
            elereco_N = ak.where( ak.num(events_original.Electron)>=2 , ak.flatten( GetEleSF(IOV, "RecoAbove20",  ele0eta, ele0pt)),  1 )[0] * ak.where( ak.num(events_original.Electron)>=2 , ak.flatten( GetEleSF(IOV, "RecoAbove20",  ele1eta, ele1pt)),  1 )[0]
            elereco_U = ak.where( ak.num(events_original.Electron)>=2 , ak.flatten(GetEleSF(IOV, "RecoAbove20",  ele0eta, ele0pt, "up")),  1 )[0] * ak.where( ak.num(events_original.Electron)>=2 , ak.flatten(GetEleSF(IOV, "RecoAbove20",  ele1eta, ele1pt, "up")),  1 )[0]
            elereco_D = ak.where( ak.num(events_original.Electron)>=2 , ak.flatten(GetEleSF(IOV, "RecoAbove20",  ele0eta, ele0pt, "down")), 1 )[0] * ak.where( ak.num(events_original.Electron)>=2 , ak.flatten(GetEleSF(IOV, "RecoAbove20",  ele1eta, ele1pt, "down")), 1 )[0]
    
            
    
    
            coffea_weights.add(name = "elereco",
                               weight = elereco_N,
                              weightUp = elereco_U,
                              weightDown = elereco_D)
            
            del ele0eta, ele0pt, ele1eta, ele1pt, elereco_N, elereco_U, elereco_D
    
    
            mureco_N = ak.where( ak.num(events_original.Muon)>=2 ,
                                ak.flatten( GetMuonSF(IOV, "mureco", np.abs(mu0eta), mu0pt)),  1 ) * ak.where( ak.num(events_original.Muon)>=2 , ak.flatten(GetMuonSF(IOV, "RecoAbove20",  np.abs(mu1eta), mu1pt)),  1 )
            mureco_U = ak.where( ak.num(events_original.Muon)>=2 , ak.flatten(GetMuonSF(IOV, "RecoAbove20",  np.abs(mu0eta), mu0pt, "systup")),  1 ) * ak.where( ak.num(events_original.Muon)>=2 , ak.flatten(GetMuonSF(IOV, "RecoAbove20", np.abs(mu1eta), mu1pt, "systup")),  1 )
            mureco_D = ak.where( ak.num(events_original.Muon)>=2 , ak.flatten(GetMuonSF(IOV, "RecoAbove20", np.abs(mu0eta), mu0pt, "systdown")), 1 ) * ak.where( ak.num(events_original.Muon)>=2 , ak.flatten(GetMuonSF(IOV, "RecoAbove20",  np.abs(mu1eta), mu1pt, "systdown")), 1 )
    
    
            coffea_weights.add(name = "mureco",
                               weight = mureco_N,
                              weightUp = mureco_U,
                              weightDown = mureco_D)
            del mu0eta, mu0pt, mu1eta, mu1pt, mureco_N, mureco_U, mureco_D
            
                
        
            ###########################
            ###adding jet corrections ##
            ###########################
            corr_jets = GetJetCorrections(events_original.FatJet, events_original, era, IOV, isData = not self.do_gen)
            
            # print("length of recojets JES up: " , len(corr_jets.JES_jes.up))
            # print("length of recojets JES down: " , len(corr_jets.JES_jes.down))
            # print("length of recojets JER up: " , len(corr_jets.JER.up))
            # print("length of recojets JER down: " , len(corr_jets.JER.down))
    
            #### Do one correction per job
            
            #print("length of original event: " , len(events_original))
            for jet_syst in self.jet_systematics:
                #print("length of event in loop: " , len(events))
                
                if jet_syst == "nominal":
                    events = ak.with_field(events_original, corr_jets, "FatJet")
                    
                    
                if jet_syst == "JES_up":
                    #print("length of recojets up: " , len(corr_jets.JES_jes.up))
                    events = ak.with_field(events_original, corr_jets.JES_jes.up, "FatJet")
    
                if jet_syst == "JES_down":
                    events = ak.with_field(events_original, corr_jets.JES_jes.down, "FatJet")
    
                if jet_syst == "JER_up":
                    events = ak.with_field(events_original, corr_jets.JER.up, "FatJet")
    
                if jet_syst == "JER_down":
                    events = ak.with_field(events_original, corr_jets.JER.down, "FatJet")
    
                sel = PackedSelection() ## initialise selection for MC
                #print(events.PV.npvsGood > 0)
                sel.add("npv", events.PV.npvsGood > 0)
                #####################################
                #####################################
                #####################################
                ### Gen selection
                #####################################
                #####################################
                #####################################
    
                
                #####################################
                ### Events with at least one gen jet
                #####################################
        
                    
        
                sel.add("oneGenJet", 
                      ak.sum( (events.GenJetAK8.pt > 136.) & (np.abs(events.GenJetAK8.eta) < 2.5), axis=1 ) >= 1
                )
                events.GenJetAK8 = events.GenJetAK8[(events.GenJetAK8.pt > 136.) & (np.abs(events.GenJetAK8.eta) < 2.5)]
        
                ###################################
                ### Events with no misses #########
                ###################################
        
                matches = ak.all(events.GenJetAK8.delta_r(events.GenJetAK8.nearest(events.FatJet)) < 0.2, axis = -1)
                misses = ~matches
        
                sel.add("matches", matches)
                #print(len(ak.flatten(events[misses].GenJetAK8.pt)))
                #print(len(ak.flatten(ak.broadcast_arrays( weights[misses], events[misses].GenJetAK8.pt)[0] )))
                
                self.hists["misses"].fill(dataset = dataset, ptgen= ak.flatten(events[misses].GenJetAK8.pt),
                                                  mgen = ak.flatten(events[misses].GenJetAK8.mass),  weight = ak.flatten(ak.broadcast_arrays( weights[misses], events[misses].GenJetAK8.pt)[0] ) )
        
                #####################################
                ### Make gen-level Z
                #####################################
                z_gen = get_z_gen_selection(events, sel, self.lepptcuts[0], self.lepptcuts[1] )
                z_ptcut_gen = ak.where( sel.all("twoGen_leptons") & ~ak.is_none(z_gen),  z_gen.pt > 90., False )
                z_mcut_gen = ak.where( sel.all("twoGen_leptons") & ~ak.is_none(z_gen),  (z_gen.mass > 71.) & (z_gen.mass < 111), False )
                sel.add("z_ptcut_gen", z_ptcut_gen)
                sel.add("z_mcut_gen", z_mcut_gen)
        
                    
        
                #####################################
                ### Get Gen Jet
                #####################################
                #print("zgen len ", len(z_gen))
                #print("events.GenJetAK8 len ", len(events.GenJetAK8))
                
                gen_jet, z_jet_dphi_gen = get_dphi( z_gen, events.GenJetAK8 )
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
        
                
                
                #####################################
                ### Make gen plots with Z and jet cuts
                #####################################
                kinsel_gen = sel.require(twoGen_leptons=True,oneGenJet=True,z_ptcut_gen=True,z_mcut_gen=True)
                sel.add("kinsel_gen", kinsel_gen)
                toposel_gen = sel.require( z_pt_asym_sel_gen=True, z_jet_dphi_sel_gen=True)
                sel.add("toposel_gen", toposel_gen)
                self.hists["ptz_gen"].fill(dataset=dataset,
                                          pt=z_gen[kinsel_gen].pt,
                                          weight=weights[kinsel_gen])
                self.hists["mz_gen"].fill(dataset=dataset,
                                          mass=z_gen[kinsel_gen].mass,
                                          weight=weights[kinsel_gen])
                self.hists["njet_gen"].fill(dataset=dataset,
                                            n=ak.num(events[kinsel_gen].GenJetAK8),
                                            weight = weights[kinsel_gen] )
    
                
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
    
                del kinsel_gen, toposel_gen
                # Making N-1 plots for these three
                self.hists["dr_z_jet_gen"].fill( dataset=dataset,
                                                  dr=z_jet_dr_gen2[z_pt_asym_sel_gen2],
                                                  weight=weights2[z_pt_asym_sel_gen2])
                self.hists["dphi_z_jet_gen"].fill(dataset=dataset, 
                                                   dphi=z_jet_dphi_gen2[z_pt_asym_sel_gen2], 
                                                   weight=weights2[z_pt_asym_sel_gen2])
                self.hists["ptasym_z_jet_gen"].fill(dataset=dataset, 
                                                     frac=z_pt_asym_gen2[z_jet_dphi_sel_gen2],
                                                     weight=weights2[z_jet_dphi_sel_gen2])
                self.hists["ptfrac_z_jet_gen"].fill(dataset=dataset, 
                                                     ptreco=z_gen[z_jet_dphi_sel_gen2].pt,
                                                     frac=z_pt_frac_gen2[z_jet_dphi_sel_gen2],
                                                     weight=weights2[z_jet_dphi_sel_gen2])
                
                del z_jet_dr_gen2, z_pt_asym_sel_gen2, z_pt_asym_gen2, z_jet_dphi_gen2, z_pt_frac_gen2,z_jet_dphi_sel_gen2, z_pt_asym_gen ,z_pt_frac_gen, z_pt_asym_sel_gen, z_jet_dphi_sel_gen
                #####################################
                ### Get gen subjets 
                #####################################
                gensubjets = events.SubGenJetAK8
                groomed_gen_jet, groomedgensel = get_groomed_jet(gen_jet, gensubjets, False)
        
                #####################################
                ### Convenience selection that has all gen cuts
                #####################################
                allsel_gen = sel.all("npv", "kinsel_gen", "toposel_gen" , "matches")
                sel.add("allsel_gen", allsel_gen)
        
                #####################################
                ### Plots for gen jets and subjets
                #####################################
                self.hists["ptjet_gen_pre"].fill(dataset=dataset, 
                                             pt=gen_jet[allsel_gen].pt, 
                                             weight=weights[allsel_gen])
                self.hists["dr_gen_subjet"].fill(dataset=dataset,
                                                 dr=groomed_gen_jet[allsel_gen].delta_r(gen_jet[allsel_gen]),
                                                 weight=weights[allsel_gen])
    
                del allsel_gen
                    #print("Zgen mass: ", z_gen[allsel_gen].mass)
                    #print("lep0 mass ", lep0.mass)
                    #print("lep1 mass ", lep1.mass)
                #####################################
                ### Make reco-level Z
                #####################################
                z_reco = get_z_reco_selection(events, sel, self.lepptcuts[0], self.lepptcuts[1])
                z_ptcut_reco = z_reco.pt > 90.
                z_mcut_reco = (z_reco.mass > 71.) & (z_reco.mass < 111.)
                sel.add("z_ptcut_reco", z_ptcut_reco)
                sel.add("z_mcut_reco", z_mcut_reco)
        
                #######################
                
                
        
                
                #####################################
                ### Reco jet selection
                #####################################
                recojets = events.FatJet[(events.FatJet.pt > 170.) & (np.abs(events.FatJet.eta) < 2.5)  ] # &  get_dR( z_reco, events.FatJet )>0.8
                sel.add("oneRecoJet", 
                     ak.sum( (events.FatJet.pt > 170.) & (np.abs(events.FatJet.eta) < 2.5), axis=1 ) >= 1
                )
        
                
                #####################################
                # Find reco jet opposite the reco Z
                #####################################
        
                #print("len recojets", len(recojets))
                #print("len z_reco", len(z_reco))
        
                reco_jet, z_jet_dphi_reco = get_dphi( z_reco, events.FatJet )
                z_jet_dr_reco = reco_jet.delta_r(z_reco)
                z_jet_dphi_reco_values = z_jet_dphi_reco
                
                #####################################
                ### Reco event topology sel
                #####################################
                z_jet_dphi_sel_reco = z_jet_dphi_reco > 1.57 #np.pi * 0.5
                z_pt_asym_reco = np.abs(z_reco.pt - reco_jet.pt) / (z_reco.pt + reco_jet.pt)
                z_pt_frac_reco = reco_jet.pt / z_reco.pt
                z_pt_asym_sel_reco = z_pt_asym_reco < 0.3
                sel.add("z_jet_dphi_sel_reco", z_jet_dphi_sel_reco)
                sel.add("z_pt_asym_sel_reco", z_pt_asym_sel_reco)
        
                kinsel_reco = sel.require(twoReco_leptons=True,oneRecoJet=True,z_ptcut_reco=True,z_mcut_reco=True)
                sel.add("kinsel_reco", kinsel_reco)
                toposel_reco = sel.require( z_pt_asym_sel_reco=True, z_jet_dphi_sel_reco=True)
                sel.add("toposel_reco", toposel_reco)
                
                
                # Note: Trigger is not applied in the MC, so this is 
                # applying the full gen selection here to be in sync with rivet routine
                if self.do_gen:
                    presel_reco = sel.all("npv", "allsel_gen", "kinsel_reco")
                else:
                    presel_reco = sel.all("npv", "trigsel", "kinsel_reco")
                allsel_reco = presel_reco & toposel_reco
                sel.add("presel_reco", presel_reco)
                sel.add("allsel_reco", allsel_reco)
        
                self.hists["mz_reco"].fill(dataset=dataset, mass=z_reco[presel_reco].mass, 
                                           weight=weights[presel_reco])
                if self.do_gen:
                    self.hists["mz_reco_over_gen"].fill(dataset=dataset, 
                                                        frac=z_reco[presel_reco].mass / z_gen[presel_reco].mass, 
                                                        weight=weights[presel_reco] )
        
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
                
                # Making N-1 plots for these three
                self.hists["dr_z_jet_reco"].fill( dataset=dataset,
                                                  dr=z_jet_dr_reco3[presel_reco3 & z_pt_asym_sel_reco3],
                                                  weight=weights3[presel_reco3 & z_pt_asym_sel_reco3])
                self.hists["dphi_z_jet_reco"].fill(dataset=dataset, 
                                                   dphi=z_jet_dphi_reco3[presel_reco3 & z_pt_asym_sel_reco3], 
                                                   weight=weights3[presel_reco3 & z_pt_asym_sel_reco3])
                self.hists["ptasym_z_jet_reco"].fill(dataset=dataset, 
                                                     frac=z_pt_asym_reco3[presel_reco3 & z_jet_dphi_sel_reco3],
                                                     weight=weights3[presel_reco3 & z_jet_dphi_sel_reco3])
                self.hists["ptfrac_z_jet_reco"].fill(dataset=dataset, 
                                                     ptreco=z_reco[presel_reco3 & z_jet_dphi_sel_reco3].pt,
                                                     frac=z_pt_frac_reco3[presel_reco3 & z_jet_dphi_sel_reco3],
                                                     weight=weights3[presel_reco3 & z_jet_dphi_sel_reco3])
    
                del kinsel_reco, toposel_reco, presel_reco, z_jet_dr_reco3, z_pt_asym_sel_reco3, z_pt_asym_reco3, z_pt_frac_reco3, z_jet_dphi_reco3 , z_jet_dphi_sel_reco3, z_pt_asym_reco ,z_pt_frac_reco, z_pt_asym_sel_reco, z_jet_dphi_sel_reco
                #####################################
                ### Make final selection plots here
                #####################################
                
                # For convenience, finally reduce the size of the arrays at the end
                #weights = weights[allsel_reco]
                # z_reco = z_reco[allsel_reco]
                # reco_jet = reco_jet[allsel_reco]
                #self.hists["ptjet_mjet_u_reco"].fill( dataset=dataset, ptreco=reco_jet.pt, mreco=reco_jet.mass, weight=weights )
                #self.hists["ptjet_mjet_g_reco"].fill( dataset=dataset, ptreco=reco_jet.pt, mreco=reco_jet.msoftdrop, weight=weights )
                
                if self.do_gen:
                    fakes = ak.any(ak.is_none(events.FatJet.matched_gen, axis = -1), axis = -1)
                    sel.add("fakes", fakes)
                    self.hists["fakes"].fill(dataset = dataset,ptreco = ak.flatten(events[fakes].FatJet.pt), mreco = ak.flatten(events[fakes].FatJet.mass))
        
                    
                    
                    matched_reco = sel.require(fakes = False)
                    sel.add("matched_reco", matched_reco)
        
                    allsel_reco = sel.all("allsel_reco")
            
                    
        
                    ee_sel = sel.all("allsel_reco", "twoReco_ee")
                    mm_sel = sel.all("allsel_reco", "twoReco_mm")
        
                    ##############################################
                    ee_sys_list = [syst for syst in coffea_weights.variations if "ele" in syst]
                    mm_sys_list = [syst for syst in coffea_weights.variations if "mu" in syst]
        
                    ee_sys_var_list = [syst for syst in self.systematics if "ele" in syst]
                    mm_sys_var_list = [syst for syst in self.systematics if "mu" in syst]
                    #print("size of sel {}".format(sys.getsizeof(sel)))
                    #print("size of event {}".format(sys.getsizeof(events)))
                    for cat in ["ee","mm"]:
                        #print("now doing category {}".format(cat))
                        if jet_syst == "nominal":
                            if cat == "ee":
        
                                weights_ee = weights[ee_sel]
                                z_reco_ee = z_reco[ee_sel]
                                reco_jet_ee = reco_jet[ee_sel]
                                #weights = weights[allsel_reco]
                                z_gen_ee = z_gen[ee_sel]
                                gen_jet_ee = gen_jet[ee_sel]
                                groomed_gen_jet_ee = groomed_gen_jet[ee_sel]
        
                    
                                systematics = [syst for syst in self.systematics if syst not in mm_sys_var_list]
                                #print("for ee case ", systematics)
        
                                for syst in self.systematics:
                                    #print("ee now doing {}".format(syst))
                                    if syst == "nominal":
                                        w = coffea_weights.partial_weight(exclude = mm_sys_list)[ee_sel]
                                        #print("sum of weight  {}".format(np.sum(w)))
                                        #w = weights
         
                                    else:
                                        #print(coffea_weights.variations)
                                        w = coffea_weights.partial_weight(exclude = mm_sys_list, modifier = syst)[ee_sel]
                                        #w = weights
                                        #print("sum of weight  {}".format(np.sum(w)))
        
                                    #print("sum of weight  {}".format(np.sum(w)))
                                    self.hists["response_matrix_u"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                       mreco=reco_jet_ee.mass, mgen=gen_jet_ee.mass, systematic = syst,  weight = w )
                        
                                    self.hists["response_matrix_g"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                       mreco=reco_jet_ee.msoftdrop, mgen=groomed_gen_jet_ee.mass, systematic = syst, weight = w )
                                del weights_ee, z_reco_ee, reco_jet_ee, z_gen_ee, gen_jet_ee, groomed_gen_jet_ee
                            if cat == "mm":
                                weights_mm = weights[mm_sel]
                                z_reco_mm = z_reco[mm_sel]
                                reco_jet_mm = reco_jet[mm_sel]
                                #weights = weights[allsel_reco]
                                z_gen_mm = z_gen[mm_sel]
                                gen_jet_mm = gen_jet[mm_sel]
                                groomed_gen_jet_mm = groomed_gen_jet[mm_sel]
                                
                                systematics = [syst for syst in self.systematics if syst not in ee_sys_var_list]
    
                                for syst in self.systematics:
                                    #print("now doing mm {}".format(syst))
                                    if syst == "nominal":
                                        w = coffea_weights.partial_weight(exclude = ee_sys_list)[mm_sel]
    
        
                                    else:
                                        w = coffea_weights.partial_weight(exclude = ee_sys_list, modifier = syst)[mm_sel]
    
                                    self.hists["response_matrix_u"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                       mreco=reco_jet_mm.mass, mgen=gen_jet_mm.mass, systematic = syst,  weight = w )
                        
                                    self.hists["response_matrix_g"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                       mreco=reco_jet_mm.msoftdrop, mgen=groomed_gen_jet_mm.mass, systematic = syst, weight = w )
                                del weights_mm, z_reco_mm, reco_jet_mm, z_gen_mm, gen_jet_mm, groomed_gen_jet_mm
     
                        else:
                            if cat == 'ee': 
                                weights_ee = weights[ee_sel]
                                z_reco_ee = z_reco[ee_sel]
                                reco_jet_ee = reco_jet[ee_sel]
                                #weights = weights[allsel_reco]
                                z_gen_ee = z_gen[ee_sel]
                                gen_jet_ee = gen_jet[ee_sel]
                                groomed_gen_jet_ee = groomed_gen_jet[ee_sel]
        
                    
                                #systematics = [syst for syst in self.systematics if syst not in mm_sys_var_list]
                                w = coffea_weights.partial_weight(exclude = mm_sys_list )[ee_sel]
                                
                                self.hists["response_matrix_u"].fill( dataset=dataset, 
                                                                   ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                   mreco=reco_jet_ee.mass, mgen=gen_jet_ee.mass, systematic = jet_syst,  weight = w )
                    
                                self.hists["response_matrix_g"].fill( dataset=dataset, 
                                                                   ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                   mreco=reco_jet_ee.msoftdrop, mgen=groomed_gen_jet_ee.mass, systematic = jet_syst, weight = w )  
                                del weights_ee, z_reco_ee, reco_jet_ee, z_gen_ee, gen_jet_ee, groomed_gen_jet_ee
                            if cat == 'mm':
                                weights_mm = weights[mm_sel]
                                z_reco_mm = z_reco[mm_sel]
                                reco_jet_mm = reco_jet[mm_sel]
                                #weights = weights[allsel_reco]
                                z_gen_mm = z_gen[mm_sel]
                                gen_jet_mm = gen_jet[mm_sel]
                                groomed_gen_jet_mm = groomed_gen_jet[mm_sel]
                                
                                #systematics = [syst for syst in self.systematics if syst not in mm_sys_var_list]
                                w = coffea_weights.partial_weight(exclude = ee_sys_list )[mm_sel]
                                
                                self.hists["response_matrix_u"].fill( dataset=dataset, 
                                                                   ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                   mreco=reco_jet_mm.mass, mgen=gen_jet_mm.mass, systematic = jet_syst,  weight = w )
                    
                                self.hists["response_matrix_g"].fill( dataset=dataset, 
                                                                   ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                   mreco=reco_jet_mm.msoftdrop, mgen=groomed_gen_jet_mm.mass, systematic = jet_syst, weight = w )   
                                del weights_mm, z_reco_mm, reco_jet_mm, z_gen_mm, gen_jet_mm, groomed_gen_jet_mm
                        
    
                del events
            for name in sel.names:
                self.hists["cutflow"][dataset][name] = sel.all(name).sum()
        
        return self.hists

    
    def postprocess(self, accumulator):
        return accumulator

    

    
