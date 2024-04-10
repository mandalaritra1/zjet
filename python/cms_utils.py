#############################################################################
# ### Author : Garvita Agarwal
# ############################################################################


import time
from coffea import nanoevents, util
import hist
import coffea.processor as processor
import awkward as ak
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import glob as glob
import re
import itertools
import vector as vec
#from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoAODSchema
from coffea.lumi_tools import LumiMask
# for applying JECs
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
#from jmeCorrections import ApplyJetCorrections, corrected_polar_met
from collections import defaultdict
import correctionlib

import os



def GetJetCorrections(FatJets, events, era, IOV, isData=False, uncertainties = None):
    if uncertainties != None:
        uncertainty_sources = uncertainties
    else:
        uncertainty_sources = ["AbsoluteMPFBias","AbsoluteScale","AbsoluteStat","FlavorQCD","Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF",
"PileUpPtRef","RelativeFSR","RelativeJEREC1","RelativeJEREC2","RelativeJERHF","RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF","RelativeBal","RelativeSample",
"RelativeStatEC","RelativeStatFSR","RelativeStatHF","SinglePionECAL","SinglePionHCAL","TimePtEta"]
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py
    jer_tag=None
    if (IOV=='2018'):
        jec_tag="Summer19UL18_V5_MC"
        jec_tag_data={
            "Run2018A": "Summer19UL18_RunA_V5_DATA",
            "Run2018B": "Summer19UL18_RunB_V5_DATA",
            "Run2018C": "Summer19UL18_RunC_V5_DATA",
            "Run2018D": "Summer19UL18_RunD_V5_DATA",
        }
        jer_tag = "Summer19UL18_JRV2_MC"
    elif (IOV=='2017'):
        jec_tag="Summer19UL17_V5_MC"
        jec_tag_data={
            "Run2017B": "Summer19UL17_RunB_V5_DATA",
            "Run2017C": "Summer19UL17_RunC_V5_DATA",
            "Run2017D": "Summer19UL17_RunD_V5_DATA",
            "Run2017E": "Summer19UL17_RunE_V5_DATA",
            "Run2017F": "Summer19UL17_RunF_V5_DATA",
        }
        jer_tag = "Summer19UL17_JRV3_MC"
    elif (IOV=='2016'):
        jec_tag="Summer19UL16_V7_MC"
        jec_tag_data={
            "Run2016F": "Summer19UL16_RunFGH_V7_DATA",
            "Run2016G": "Summer19UL16_RunFGH_V7_DATA",
            "Run2016H": "Summer19UL16_RunFGH_V7_DATA",
        }
        jer_tag = "Summer20UL16_JRV3_MC"
    elif (IOV=='2016APV'):
        jec_tag="Summer19UL16_V7_MC"
        ## HIPM/APV     : B_ver1, B_ver2, C, D, E, F
        ## non HIPM/APV : F, G, H

        jec_tag_data={
            "Run2016B": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016C": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016D": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016E": "Summer19UL16APV_RunEF_V7_DATA",
            "Run2016F": "Summer19UL16APV_RunEF_V7_DATA",
        }
        jer_tag = "Summer20UL16APV_JRV3_MC"
    else:
        print(f"Error: Unknown year \"{IOV}\".")


    #print("extracting corrections from files for " + jec_tag)
    ext = extractor()
    if not isData:
    #For MC
        ext.add_weight_sets([
            '* * '+'correctionFiles/JEC/{0}/{0}_L1FastJet_AK8PFPuppi.jec.txt'.format(jec_tag),
            '* * '+'correctionFiles/JEC/{0}/{0}_L2Relative_AK8PFPuppi.jec.txt'.format(jec_tag),
            '* * '+'correctionFiles/JEC/{0}/{0}_L3Absolute_AK8PFPuppi.jec.txt'.format(jec_tag),
            '* * '+'correctionFiles/JEC/{0}/{0}_UncertaintySources_AK8PFPuppi.junc.txt'.format(jec_tag),
            '* * '+'correctionFiles/JEC/{0}/{0}_Uncertainty_AK8PFPuppi.junc.txt'.format(jec_tag),
        ])
        #### Do AK8PUPPI jer files exist??
        if jer_tag:
            #print("File "+'correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)))
            #print("File "+'correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)))
            ext.add_weight_sets([
            '* * '+'correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag),
            '* * '+'correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)])

    else:       
        #For data, make sure we don't duplicat
        tags_done = []
        for run, tag in jec_tag_data.items():
            if not (tag in tags_done):
                ext.add_weight_sets([
                '* * '+'correctionFiles/JEC/{0}/{0}_L1FastJet_AK8PFPuppi.jec.txt'.format(tag),
                '* * '+'correctionFiles/JEC/{0}/{0}_L2Relative_AK8PFPuppi.jec.txt'.format(tag),
                '* * '+'correctionFiles/JEC/{0}/{0}_L3Absolute_AK8PFPuppi.jec.txt'.format(tag),
                '* * '+'correctionFiles/JEC/{0}/{0}_L2L3Residual_AK8PFPuppi.jec.txt'.format(tag),
                ])
                tags_done += [tag]
    ext.finalize()


    evaluator = ext.make_evaluator()

    if (not isData):
        jec_names = [
            '{0}_L1FastJet_AK8PFPuppi'.format(jec_tag),
            '{0}_L2Relative_AK8PFPuppi'.format(jec_tag),
            '{0}_L3Absolute_AK8PFPuppi'.format(jec_tag)]
        jec_names.extend(['{0}_UncertaintySources_AK8PFPuppi_{1}'.format(jec_tag, unc_src) for unc_src in uncertainty_sources])

        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_AK8PFPuppi'.format(jer_tag),
                              '{0}_SF_AK8PFPuppi'.format(jer_tag)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_AK8PFPuppi'.format(tag),
                '{0}_L3Absolute_AK8PFPuppi'.format(tag),
                '{0}_L2Relative_AK8PFPuppi'.format(tag),
                '{0}_L2L3Residual_AK8PFPuppi'.format(tag),]



    if not isData:
        jec_inputs = {name: evaluator[name] for name in jec_names}
    else:
        jec_inputs = {name: evaluator[name] for name in jec_names[era]}


    # print("jec_input", jec_inputs)
    jec_stack = JECStack(jec_inputs)


    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
    FatJets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]
    FatJets['pt_gen'] = ak.values_astype(ak.fill_none(FatJets.matched_gen.pt, 0), np.float32)
    
    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'rho'


    events_cache = events.caches[0]

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    corrected_jets = jet_factory.build(FatJets, lazy_cache=events_cache)
    # print("Available uncertainties: ", jet_factory.uncertainties())
    # print("Corrected jets object: ", corrected_jets.fields)
    return corrected_jets


def CorrectJetsRun2(IOV, Jets, JetsName="AK4PFchs", leptons_inJet=None):
    jer_tag=None
    if (IOV=='2018' or IOV == 'Test'):
        jec_tag="Summer19UL18_V5_MC"
        jec_tag_data={
            "RunA": "Summer19UL18_RunA_V5_DATA",
            "RunB": "Summer19UL18_RunB_V5_DATA",
            "RunC": "Summer19UL18_RunC_V5_DATA",
            "RunD": "Summer19UL18_RunD_V5_DATA",
        }
        jer_tag = "Summer19UL18_JRV2_MC"
    elif (IOV=='2017'):
        jec_tag="Summer19UL17_V5_MC"
        jec_tag_data={
            "RunB": "Summer19UL17_RunB_V5_DATA",
            "RunC": "Summer19UL17_RunC_V5_DATA",
            "RunD": "Summer19UL17_RunD_V5_DATA",
            "RunE": "Summer19UL17_RunE_V5_DATA",
            "RunF": "Summer19UL17_RunF_V5_DATA",
        }
        jer_tag = "Summer19UL17_JRV2_MC"
    elif (IOV=='2016'):
        jec_tag="Summer19UL16_V5_MC"
        jec_tag_data={
            "RunF": "Summer19UL16_RunFGH_V7_DATA",
            "RunG": "Summer19UL16_RunFGH_V7_DATA",
            "RunH": "Summer19UL16_RunFGH_V7_DATA",
        }
        jer_tag = "Summer20UL16_JRV3_MC"
    elif (IOV=='2016APV'):
        jec_tag="Summer19UL16_V5_MC"
        ## HIPM/APV     : B_ver1, B_ver2, C, D, E, F
        ## non HIPM/APV : F, G, H

        jec_tag_data={
            "RunB_ver1": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunB_ver2": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunC": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunD": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunE": "Summer19UL16APV_RunEF_V7_DATA",
            "RunF": "Summer19UL16APV_RunEF_V7_DATA",
        }
        jer_tag = "Summer20UL16APV_JRV3_MC"
    else:
        raise ValueError(f"Error: Unknown year \"{IOV}\".")
    
    extract = extractor()
    if (isMC):
        #For MC
        extract.add_weight_sets([
            '* * data/JEC/{0}/{0}_L1FastJet_{1}.txt'.format(jec_tag, JetsName),
            '* * data/JEC/{0}/{0}_L2Relative_{1}.txt'.format(jec_tag, JetsName),
            '* * data/JEC/{0}/{0}_L3Absolute_{1}.txt'.format(jec_tag, JetsName),
            '* * data/JEC/{0}/{0}_UncertaintySources_{1}.junc.txt'.format(jec_tag, JetsName),
            '* * data/JEC/{0}/{0}_Uncertainty_{1}.junc.txt'.format(jec_tag, JetsName),
        ])

        if jer_tag:
            extract.add_weight_sets([
            '* * data/JER/{0}/{0}_PtResolution_{1}.jr.txt'.format(jer_tag, JetsName),
            '* * data/JER/{0}/{0}_SF_{1}.jersf.txt'.format(jer_tag, JetsName)])
    else:       
        #For data, make sure we don't duplicate
        tags_done = []
        for run, tag in jec_tag_data.items():
            if not (tag in tags_done):
                extract.add_weight_sets([
                '* * data/JEC/{0}/{0}_L1FastJet_{1}.txt'.format(tag, JetsName),
                '* * data/JEC/{0}/{0}_L2Relative_{1}.txt'.format(tag, JetsName),
                '* * data/JEC/{0}/{0}_L3Absolute_{1}.txt'.format(tag, JetsName),
                '* * data/JEC/{0}/{0}_L2L3Residual_{1}.txt'.format(tag, JetsName),
                ])
                tags_done += [tag]
                
    extract.finalize()
    evaluator = extract.make_evaluator()
    
    if (isMC):
        jec_names = [
            '{0}_L1FastJet_{1}'.format(jec_tag, JetsName),
            '{0}_L2Relative_{1}'.format(jec_tag, JetsName),
            '{0}_L3Absolute_{1}'.format(jec_tag, JetsName),
            '{0}_Uncertainty_{1}'.format(jec_tag, JetsName)]
        if do_factorized_jec_unc:
            for name in dir(evaluator):
               #factorized sources
               if '{0}_UncertaintySources_{1}'.format(jec_tag, JetsName) in name:
                    jec_names.append(name)
        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_{1}'.format(jer_tag, JetsName),
                              '{0}_SF_{1}'.format(jer_tag, JetsName)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_{1}'.format(tag, JetsName),
                '{0}_L3Absolute_{1}'.format(tag, JetsName),
                '{0}_L2Relative_{1}'.format(tag, JetsName),
                '{0}_L2L3Residual_{1}'.format(tag, JetsName),]
    if isMC:
        jec_inputs = {name: evaluator[name] for name in jec_names}
    else:
        jec_inputs = {name: evaluator[name] for name in jec_names[era]}
    
    
    
    
    ## (1) Uncorrecting Jets
    CleanedJets = Jets
    debug(self.debugMode, "Corrected Jets (Before Cleaning): ", CleanedJets[0].pt)
    CleanedJets["rho"] = ak.broadcast_arrays(df.fixedGridRhoFastjetAll, Jets.pt)[0]
    CleanedJets["pt_raw"] = (1 - CleanedJets.rawFactor) * CleanedJets.pt
    CleanedJets["mass_raw"] = (1 - CleanedJets.rawFactor) * CleanedJets.mass
    CleanedJets["pt"] = CleanedJets.pt_raw
    CleanedJets["mass"] = CleanedJets.mass_raw
    CleanedJets["p4","pt"] = CleanedJets.pt_raw
    CleanedJets["p4","mass"] = CleanedJets.mass_raw
    if (isMC):
        CleanedJets["pt_gen"] = ak.values_astype(ak.fill_none(CleanedJets.matched_gen.pt, 0), np.float32)
    debug(self.debugMode, "Raw Jets (Before Cleaning):       ", CleanedJets[0].pt)

    ## (2) Removing leptons from jets

    if leptons_inJet != None :
        cleaned = (CleanedJets.p4).subtract(leptons_inJet)
        CleanedJets["p4","pt"] = cleaned.pt
        CleanedJets["p4","eta"] = cleaned.eta
        CleanedJets["p4","phi"] = cleaned.phi
        CleanedJets["pt"] = cleaned.pt
        CleanedJets["eta"] = cleaned.eta
        CleanedJets["phi"] = cleaned.phi
        CleanedJets["pt_raw"] = cleaned.pt
    
    jec_stack = JECStack(jec_inputs)
    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetEta'] = 'eta'
    name_map['JetPhi'] = 'phi'
    name_map['JetMass'] = 'mass'
    name_map['Rho'] = 'rho'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['METpt'] = 'pt'
    name_map['METphi'] = 'phi'
    name_map['UnClusteredEnergyDeltaX'] = 'MetUnclustEnUpDeltaX'
    name_map['UnClusteredEnergyDeltaY'] = 'MetUnclustEnUpDeltaY'
    if corr_type=='met': return CorrectedMETFactory(name_map)
    CleanedJets = CorrectedJetsFactory(name_map, jec_stack).build(CleanedJets, lazy_cache=df.caches[0])    
    return CleanedJets




## --------------------------------- MET Filters ------------------------------#
## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL

MET_filters = {'2016APV':["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "eeBadScFilter",
                          "hfNoisyHitsFilter"],
               '2016'   :["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "eeBadScFilter",
                          "hfNoisyHitsFilter"],
               '2017'   :["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "hfNoisyHitsFilter",
                          "eeBadScFilter",
                          "ecalBadCalibFilter"],
               '2018'   :["goodVertices",
                          "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter",
                          "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter",
                          "BadPFMuonDzFilter",
                          "hfNoisyHitsFilter",
                          "eeBadScFilter",
                          "ecalBadCalibFilter"]}

corrlib_namemap = {
    "2016APV":"2016preVFP_UL",
    "2016":"2016postVFP_UL",
    "2017":"2017_UL",
    "2018":"2018_UL"
}


def GetPUSF(IOV, nTrueInt, var='nominal'):
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM

    fname = "correctionFiles/POG/LUM/" + corrlib_namemap[IOV] + "/puWeights.json.gz"
    hname = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016"   : "Collisions16_UltraLegacy_goldenJSON",
        "2017"   : "Collisions17_UltraLegacy_goldenJSON",
        "2018"   : "Collisions18_UltraLegacy_goldenJSON"
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    return evaluator[hname[IOV]].evaluate(np.array(nTrueInt), var)

def GetL1PreFiringWeight(IOV, df, var="Nom"):
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe
    ## var = "Nom", "Up", "Dn"
    L1PrefiringWeights = ak.ones_like(df.event)
    if ("L1PreFiringWeight" in ak.fields(df)):
        L1PrefiringWeights = df["L1PreFiringWeight"][var]
    return L1PrefiringWeights

def HEMCleaning(IOV, JetCollection):
    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    isHEM = ak.ones_like(JetCollection.pt)
    if (IOV == "2018"):
        detector_region1 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -1.3) & (JetCollection.eta > -2.5))
        detector_region2 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -2.5) & (JetCollection.eta > -3.0))
        jet_selection    = ((JetCollection.jetId > 1) & (JetCollection.pt > 15))

        isHEM            = ak.where(detector_region1 & jet_selection, 0.80, isHEM)
        isHEM            = ak.where(detector_region2 & jet_selection, 0.65, isHEM)

    return isHEM

def GetEleSF(IOV, wp, eta, pt, var = ""):
    ## Reference:
    ##   - https://twiki.cern.ch/twiki/bin/view/CMS/EgammaUL2016To2018
    ##   - https://twiki.cern.ch/twiki/bin/view/CMS/EgammaSFJSON
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM
    fname = "correctionFiles/POG/EGM/" + corrlib_namemap[IOV] + "/electron.json.gz"
    year = {
        "2016APV" : "2016preVFP",
        "2016"    : "2016postVFP",
        "2017"    : "2017",
        "2018"    : "2018",
    }
    num = ak.num(pt)
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    
    ## if the eta and pt satisfy the requirements derive the eff SFs, otherwise set it to 1.
    mask = pt > 20
    pt = ak.where(mask, pt, 22)
    
    sf = evaluator["UL-Electron-ID-SF"].evaluate(year[IOV], "sf"+var, wp,
                                                 np.array(ak.flatten(eta)),
                                                 np.array(ak.flatten(pt)))
    sf = ak.where(np.array(ak.flatten(~mask)), 1, sf)
    return ak.unflatten(sf, ak.num(pt))

def GetMuonSF(IOV, corrset, abseta, pt, var="sf"):
    ## For reco and trigger SF for high pT muons
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MuonUL2016
    ##            https://twiki.cern.ch/twiki/bin/viewauth/CMS/MuonUL2017
    ##            https://twiki.cern.ch/twiki/bin/viewauth/CMS/MuonUL2018
    ## Using the JSONs created by MUO POG
    ## corrset = "RECO", "HLT", "IDISO"
    ## var = "sf", "systup", "systdown"
    
    tag = IOV
    if 'APV' in IOV:
        tag = '2016_preVFP'
    fname = "correctionFiles/muonSF/UL"+IOV+"/ScaleFactors_Muon_highPt_"+corrset+"_"+tag+"_schemaV2.json"
    
    num = ak.num(pt)
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    
    ## the correction for TuneP muons are avaiable for p > 50GeV and eta < 2.4,
    ## so for those cases I'm applying SFs form the next closest bin.
    pt = ak.where(pt > 50, pt, 50.1)
    abseta = ak.where(abseta < 2.4, abseta, 2.39)
    
    if corrset == "RECO":
        hname = "NUM_GlobalMuons_DEN_TrackerMuonProbes" # for RECO (p, eta)
        #we need to modify the pT into |p|
        pt = np.cosh(abseta)*pt
        pt = ak.where(pt < 3500, pt, 3499)
        
    if corrset == "HLT":
        hname = "NUM_HLT_DEN_HighPtTightRelIsoProbes" # for HLT (pt, eta)
        pt = ak.where(pt < 1000, pt, 999.9)
        
    if corrset == "IDISO":
        hname = "NUM_HighPtID_DEN_GlobalMuonProbes" # for IDISO (pt, eta)
        pt = ak.where(pt < 1000, pt, 999.9)
    
    sf = evaluator[hname].evaluate(np.array(ak.flatten(abseta)),
                                   np.array(ak.flatten(pt)),
                                   'nominal')
    syst = evaluator[hname].evaluate(np.array(ak.flatten(abseta)),
                                   np.array(ak.flatten(pt)),
                                   'syst')
    if "up" in var:
        sf = sf + syst
    elif "down" in var:
        sf = sf - syst

    return ak.unflatten(sf, ak.num(pt))



def GetEleTrigEff(IOV, lep0pT, var = ""):
    ## Most recent presentation avaible at: https://indico.cern.ch/event/1290491/#5-tt-resonances-2l-update
    eleSF = {
        #"2016APV":{"sf": 1.035, "sfup": 1.0971, "sfdown": 0.9729},
        "2016APV":{"sf": 1.034, "sfup": 1.0702934, "sfdown": 0.9977066}, # 3.51%
        #"2016"   :{"sf": 1.024, "sfup": 1.3424,  "sfdown": 1.01376},
        "2016"   :{"sf": 1.026, "sfup": 1.0391328,  "sfdown": 1.0128672}, #1.28%
        #"2017"   :{"sf": 0.983, "sfup": 1.00266, "sfdown": 0.96334},
        "2017"   :{"sf": 0.982, "sfup": 0.989856, "sfdown": 0.974144}, #0.80%
        # "2018"   :{"sf": 0.992, "sfup": 1.01184, "sfdown": 0.97216},
        "2018"   :{"sf": 0.994, "sfup": 1.0105998, "sfdown": 0.9774002}} #1.67%

    out_L = np.where(lep0pT<200, eleSF[IOV]["sf"+var], 1.0)
    out_M = np.where((lep0pT>=200)&(lep0pT<400), eleSF[IOV]["sf"+var], 1.0)
    out_H = np.where(lep0pT>=400, eleSF[IOV]["sf"+var], 1.0)
    return out_L, out_M, out_H

def GetPDFweights(df, var="nominal"):
    ## determines the pdf up and down variations
    pdf = ak.ones_like(df.Pileup.nTrueInt)
    if ("LHEPdfWeight" in ak.fields(df)):
        pdfUnc = ak.std(df.LHEPdfWeight,axis=1)/ak.mean(df.LHEPdfWeight,axis=1)
    if var == "up":
        pdf = pdf + pdfUnc
    elif var == "down":
        pdf = pdf - pdfUnc
    return pdf

def GetQ2weights(df, var="nominal"):
    ## determines the envelope of the muR/muF up and down variations
    ## Case 1:
    ## LHEScaleWeight[0] -> (0.5, 0.5) # (muR, muF)
    ##               [1] -> (0.5, 1)
    ##               [2] -> (0.5, 2)
    ##               [3] -> (1, 0.5)
    ##               [4] -> (1, 1)
    ##               [5] -> (1, 2)
    ##               [6] -> (2, 0.5)
    ##               [7] -> (2, 1)
    ##               [8] -> (2, 2)
                  
    ## Case 2:
    ## LHEScaleWeight[0] -> (0.5, 0.5) # (muR, muF)
    ##               [1] -> (0.5, 1)
    ##               [2] -> (0.5, 2)
    ##               [3] -> (1, 0.5)
    ##               [4] -> (1, 2)
    ##               [5] -> (2, 0.5)
    ##               [6] -> (2, 1)
    ##               [7] -> (2, 2)

    q2 = ak.ones_like(df.event)
    q2Up = ak.ones_like(df.event)
    q2Down = ak.ones_like(df.event)
    if ("LHEScaleWeight" in ak.fields(df)):
        if ak.all(ak.num(df.LHEScaleWeight)==9):
            nom = df.LHEScaleWeight[:,4]
            scales = df.LHEScaleWeight[:,[0,1,3,5,7,8]]
            q2Up = ak.max(scales,axis=1)/nom
            q2Down = ak.min(scales,axis=1)/nom 
        elif ak.all(ak.num(df.LHEScaleWeight)==9):
            scales = df.LHEScaleWeight[:,[0,1,3,4,6,7]]
            q2Up = ak.max(scales,axis=1)
            q2Down = ak.min(scales,axis=1)
            
    if var == "up":
        return q2Up
    elif var == "down":
        return q2Down
    else:
        return q2

    
def getLumiMaskRun2():

    golden_json_path_2016 = "correctionFiles/goldenJsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
    golden_json_path_2017 = "correctionFiles/goldenJsons/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
    golden_json_path_2018 = "correctionFiles/goldenJsons/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"

    masks = {"2016APV":LumiMask(golden_json_path_2016),
             "2016":LumiMask(golden_json_path_2016),
             "2017":LumiMask(golden_json_path_2017),
             "2018":LumiMask(golden_json_path_2018)
            }

    return masks
