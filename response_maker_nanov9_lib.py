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
    def __init__(self, do_gen=True, ptcut=200., etacut = 2.5, ptcut_ee = 40., ptcut_mm = 29., skimfilename=None, jet_syst = "nominal"):
        
        self.lumimasks = getLumiMaskRun2()
        
        # should have separate lower ptcut for gen
        self.do_gen=do_gen
        self.ptcut = ptcut
        self.etacut = etacut        
        self.lepptcuts = [ptcut_ee, ptcut_mm]
        self.jet_syst = jet_syst

                
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
        self.systematics = ['nominal', 'puUp', 'puDown' , 'elerecoUp', 'elerecoDown', 'murecoUp', 'murecoDown', 'muidUp', 'muidDown', 'mutrigUp', 'mutrigDown', 'pdfUp', 'pdfDown', 'q2Up', 'q2Down', 'prefiringUp', 'prefiringDown'] 
        self.jet_systematics = ["nominal", "JERUp", "JERDown"]
        
        
        ## This is for rejecting events with large weights
        self.means_stddevs = defaultdict()

        
    
    @property
    def accumulator(self):
        #return self._histos
        return self.hists

    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events0):

        #print(events.metadata.keys())
        dataset = events0.metadata['dataset']
        print(dataset)
        #lenprint(events.metadata['version'])
        filename = events0.metadata['filename']
        jet_syst = self.jet_syst
        
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
        if (self.do_gen):
            era = None
        else:
            firstidx = filename.find( "store/data/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            era = fname_toks[ fname_toks.index("data") + 1]
            #print("IOV ", IOV, ", era ", era)
            lumi_mask = np.array(self.lumimasks[IOV](events0.run, events0.luminosityBlock), dtype=bool)
            events0 = events0[lumi_mask]

        
        #print("Luminosity working")
    
        #corrections = {"nominal": recojets, "JES_up":  recojets.JES_jes.up, "JES_down" : recojets.JES_jes.down}
        
        events0 = events0[events0.PV.npvsGood > 0]

        ## PU reweighting           
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




            

        #####################################
        ### Initialize selection
        #####################################
        sel = PackedSelection()

        #####################################
        ### Trigger selection for data
        #####################################       
        if not self.do_gen:
            if "UL2016" in dataset: 
                trigsel = events0.HLT.IsoMu24 | events0.HLT.Ele27_WPTight_Gsf | events0.HLT.Photon175
            elif "UL2017" in dataset:
                trigsel = events0.HLT.IsoMu27 | events0.HLT.Ele35_WPTight_Gsf | events0.HLT.Photon200
            elif "UL2018" in dataset:
                trigsel = events0.HLT.IsoMu24 | events0.HLT.Ele32_WPTight_Gsf | events0.HLT.Photon200
            else:
                raise Exception("Dataset is incorrect, should have 2016, 2017, 2018: ", dataset)
            sel.add("trigsel", trigsel)    
        

        # if self.do_gen:
        #     corrections.update({"jesUp": recojets.JES_jes.up})
        # for sys_name in corrections.keys():
             
        #     events = ak.with_field(events, corrections[sys_name], "FatJet")


        #############################
        ## adding jet corrections ##
        #############################
        corr_jets = GetJetCorrections(events0.FatJet, events0, era, IOV, isData = not self.do_gen)
        #print("Jet corrections working")
        # print("length of recojets JES up: " , len(corr_jets.JES_jes.up))
        # print("length of recojets JES down: " , len(corr_jets.JES_jes.down))
        # print("length of recojets JER up: " , len(corr_jets.JER.up))
        # print("length of recojets JER down: " , len(corr_jets.JER.down))

        for unc_src in (unc_src for unc_src in corr_jets.fields if "JES" in unc_src):
            #print("Uncertainty source: ", unc_src)
            #print(corr_jets[unc_src])
            self.jet_systematics.append(unc_src+"Up")
            self.jet_systematics.append(unc_src+"Down")


                
        
        #print("length of original event: " , len(events_original))
        for jet_syst in self.jet_systematics:
            #print("length of event in loop: " , len(events))
            # print("JES" in jet_syst)
            # print(jet_syst[:-2])
            # print(jet_syst[:-2]=="Up")
            # print("f")
            if jet_syst == "nominal":
                events = ak.with_field(events0, corr_jets, "FatJet")

            elif jet_syst == "JERUp":
                events = ak.with_field(events0, corr_jets.JER.up, "FatJet")

            elif jet_syst == "JERDown":
                events = ak.with_field(events0, corr_jets.JER.down, "FatJet")
            
            elif (jet_syst[-2:]=="Up" and "JES" in jet_syst):
                #print(jet_syst)
                field = jet_syst[:-2]
                #print(field)
                events = ak.with_field(events0, corr_jets[field].up, "FatJet")
            elif (jet_syst[-4:]=="Down" and "JES" in jet_syst):
                field = jet_syst[:-4]
                events = ak.with_field(events0, corr_jets[field].down, "FatJet")
            else:
                print("{} is not considered".format(jet_syst))

            #####################################
            ### Remove events with very large gen weights (>2 sigma)
            #####################################
    
            if self.do_gen:
                if dataset not in self.means_stddevs : 
                    average = np.average( events["LHEWeight"].originalXWGTUP )
                    stddev = np.std( events["LHEWeight"].originalXWGTUP )
                    self.means_stddevs[dataset] = (average, stddev)            
                average,stddev = self.means_stddevs[dataset]
                vals = (events["LHEWeight"].originalXWGTUP - average ) / stddev
                self.hists["cutflow"][dataset]["all events"] += len(events)
                events = events[ np.abs(vals) < 2 ]
                self.hists["cutflow"][dataset]["weights cut"] += len(events)

                sel = PackedSelection() ## initialise selection for MC
                weights = events["LHEWeight"].originalXWGTUP
                #####################################
                ### Initialize event weight to gen weight
                #####################################
                
            
                weights = events["LHEWeight"].originalXWGTUP
    
                
                
            else:
                weights = np.full( len( events ), 1.0 )

            # NPV selection
            #print(events.PV.npvsGood>0)
    
            
            #npv = np.array(events.PV.npvsGood >0)
            sel.add("npv", events.PV.npvsGood > 0)

    
    
            #####################################
            #####################################
            #####################################
            ### Gen selection
            #####################################
            #####################################
            #####################################
            if self.do_gen:
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
                lep0 = events[allsel_gen].GenDressedLepton[:,0]
                lep1 = events[allsel_gen].GenDressedLepton[:,1]
                
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

                events = events[allsel_reco]
                weights = weights[allsel_reco]
                z_reco = z_reco[allsel_reco]
                reco_jet = reco_jet[allsel_reco]
                #weights = weights[allsel_reco]
                z_gen = z_gen[allsel_reco]
                gen_jet = gen_jet[allsel_reco]
                groomed_gen_jet = groomed_gen_jet[allsel_reco]

                
                

                
                ###########################################
                ### Categorize events into ee and mm channel
                ###########################################

                ee_sel = sel.all("allsel_reco", "twoReco_ee")[allsel_reco]
                mm_sel = sel.all("allsel_reco", "twoReco_mm")[allsel_reco]
                
                

                cat_sel_list = {"ee":ee_sel, "mm": mm_sel}
                ##############################################
                # ee_sys_list = [syst for syst in coffea_weights.variations if "ele" in syst]
                # mm_sys_list = [syst for syst in coffea_weights.variations if "mu" in syst]

                ee_sys_var_list = [syst for syst in self.systematics if "ele" in syst]
                mm_sys_var_list = [syst for syst in self.systematics if "mu" in syst]
                
                for cat in cat_sel_list.keys():
                    
                    if jet_syst == "nominal":
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
                                
                                coffea_weights.add(name = "mureco",
                                                   weight = ak.ones_like(events_ee.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_ee.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_ee.Pileup.nTrueInt))

                                coffea_weights.add(name = "muid",
                                                   weight = ak.ones_like(events_ee.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_ee.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_ee.Pileup.nTrueInt))

                                coffea_weights.add(name = "mutrig",
                                                   weight = ak.ones_like(events_ee.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_ee.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_ee.Pileup.nTrueInt))

      
                                del elereco_N, elereco_U, elereco_D, eleid_N, eleid_U, eleid_D
    
                                
    
                                
                                z_reco_ee = z_reco[ee_sel]
                                reco_jet_ee = reco_jet[ee_sel]
                                #weights = weights[allsel_reco]
                                z_gen_ee = z_gen[ee_sel]
                                gen_jet_ee = gen_jet[ee_sel]
                                groomed_gen_jet_ee = groomed_gen_jet[ee_sel]
    
                    
                                #systematics = [syst for syst in self.systematics if syst not in mm_sys_var_list]
                                #print("for ee case ", systematics)
    
                                for syst in self.systematics:
                                    if syst == "nominal":
                                        w = coffea_weights.weight()
                                        #w = weights
                                    else:
                                        #print(coffea_weights.variations)
                                        w = coffea_weights.weight(modifier = syst)
                                        #w = weights
                                        
                                    self.hists["response_matrix_u"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                       mreco=reco_jet_ee.mass, mgen=gen_jet_ee.mass, systematic = syst,  weight = w )
                        
                                    self.hists["response_matrix_g"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                       mreco=reco_jet_ee.msoftdrop, mgen=groomed_gen_jet_ee.mass, systematic = syst, weight = w )
                                del weights_ee, z_reco_ee, reco_jet_ee, z_gen_ee, gen_jet_ee, groomed_gen_jet_ee, coffea_weights
                            
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
                                
                                coffea_weights.add(name = "elereco",
                                                   weight = ak.ones_like(events_mm.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_mm.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_mm.Pileup.nTrueInt))
                                
                                coffea_weights.add(name = "eleid",
                                                   weight = ak.ones_like(events_mm.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_mm.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_mm.Pileup.nTrueInt))
      
                                del mureco_N, mureco_U, mureco_D, muid_N, muid_U, muid_D, mutrig_N, mutrig_U, mutrig_D
    
                                
    
                                
                                z_reco_mm = z_reco[mm_sel]
                                reco_jet_mm = reco_jet[mm_sel]
                                #weights = weights[allsel_reco]
                                z_gen_mm = z_gen[mm_sel]
                                gen_jet_mm = gen_jet[mm_sel]
                                groomed_gen_jet_mm = groomed_gen_jet[mm_sel]
    
                    
                                systematics = [syst for syst in self.systematics if syst not in mm_sys_var_list]
                                #print("for ee case ", systematics)
    
                                for syst in self.systematics:
                                    if syst == "nominal":
                                        w = coffea_weights.weight()
                                        #w = weights
                                    else:
                                        #print(coffea_weights.variations)
                                        w = coffea_weights.weight(modifier = syst)
                                        #w = weights
                                        
                                    self.hists["response_matrix_u"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                       mreco=reco_jet_mm.mass, mgen=gen_jet_mm.mass, systematic = syst,  weight = w )
                        
                                    self.hists["response_matrix_g"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                       mreco=reco_jet_mm.msoftdrop, mgen=groomed_gen_jet_mm.mass, systematic = syst, weight = w )
                                del weights_mm, z_reco_mm, reco_jet_mm, z_gen_mm, gen_jet_mm, groomed_gen_jet_mm
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
                                
                                coffea_weights.add(name = "mureco",
                                                   weight = ak.ones_like(events_ee.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_ee.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_ee.Pileup.nTrueInt))

                                coffea_weights.add(name = "muid",
                                                   weight = ak.ones_like(events_ee.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_ee.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_ee.Pileup.nTrueInt))

                                coffea_weights.add(name = "mutrig",
                                                   weight = ak.ones_like(events_ee.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_ee.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_ee.Pileup.nTrueInt))
    
                                
                                del elereco_N, elereco_U, elereco_D, eleid_N, eleid_U, eleid_D
                                
                                z_reco_ee = z_reco[ee_sel]
                                reco_jet_ee = reco_jet[ee_sel]
                                #weights = weights[allsel_reco]
                                z_gen_ee = z_gen[ee_sel]
                                gen_jet_ee = gen_jet[ee_sel]
                                groomed_gen_jet_ee = groomed_gen_jet[ee_sel]
    
                                w = coffea_weights.weight()
                                
                                self.hists["response_matrix_u"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                       mreco=reco_jet_ee.mass, mgen=gen_jet_ee.mass, systematic = jet_syst,  weight = w )
                        
                                self.hists["response_matrix_g"].fill( dataset=dataset, 
                                                                       ptreco=reco_jet_ee.pt, ptgen=gen_jet_ee.pt,
                                                                       mreco=reco_jet_ee.msoftdrop, mgen=groomed_gen_jet_ee.mass, systematic = jet_syst, weight = w )
                                del weights_ee, z_reco_ee, reco_jet_ee, z_gen_ee, gen_jet_ee, groomed_gen_jet_ee
                                
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
                                
                                coffea_weights.add(name = "elereco",
                                                   weight = ak.ones_like(events_mm.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_mm.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_mm.Pileup.nTrueInt))
                                
                                coffea_weights.add(name = "eleid",
                                                   weight = ak.ones_like(events_mm.Pileup.nTrueInt), 
                                                   weightUp =  ak.ones_like(events_mm.Pileup.nTrueInt),
                                                   weightDown =  ak.ones_like(events_mm.Pileup.nTrueInt))
      
                                del mureco_N, mureco_U, mureco_D, muid_N, muid_U, muid_D, mutrig_N, mutrig_U, mutrig_D
    
                                
    
                                
                                z_reco_mm = z_reco[mm_sel]
                                reco_jet_mm = reco_jet[mm_sel]
                                #weights = weights[allsel_reco]
                                z_gen_mm = z_gen[mm_sel]
                                gen_jet_mm = gen_jet[mm_sel]
                                groomed_gen_jet_mm = groomed_gen_jet[mm_sel]
    
                    
                                systematics = [syst for syst in self.systematics if syst not in mm_sys_var_list]
                                #print("for ee case ", systematics)
    
    
                                w = coffea_weights.weight()
     
                                        
                                self.hists["response_matrix_u"].fill( dataset=dataset, 
                                                                   ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                   mreco=reco_jet_mm.mass, mgen=gen_jet_mm.mass, systematic = jet_syst,  weight = w )
                    
                                self.hists["response_matrix_g"].fill( dataset=dataset, 
                                                                   ptreco=reco_jet_mm.pt, ptgen=gen_jet_mm.pt,
                                                                   mreco=reco_jet_mm.msoftdrop, mgen=groomed_gen_jet_mm.mass, systematic = jet_syst, weight = w )
                                del weights_mm, z_reco_mm, reco_jet_mm, z_gen_mm, gen_jet_mm, groomed_gen_jet_mm

                

                
                        
                del events, weights
            

        
        for name in sel.names:
            self.hists["cutflow"][dataset][name] = sel.all(name).sum()
        
        return self.hists

    
    def postprocess(self, accumulator):
        return accumulator

    

    
