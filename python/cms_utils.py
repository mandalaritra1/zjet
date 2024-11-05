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
import uproot
import hist
from coffea.lookup_tools.dense_lookup import dense_lookup
import os

import pickle



lumi = {'2018' : 59740,
        '2017': 41480,
        '2016': 36330 
       }

numentries = {
    'UL16NanoAODv9': 56287369,
    #'UL16NanoAODv9':27417103,
    #'UL16NanoAODAPVv9':28870266,
    'UL17NanoAODv9': 59243606,
    'UL18NanoAODv9': 84844279 ,
    
}



numentries_herwig = {
    'UL16NanoAODv9': 59608360,
    #'UL16NanoAODv9':30069348,
    #'UL16NanoAODAPVv9':29539012,
    'UL17NanoAODv9': 29578468,
    'UL18NanoAODv9': 29382595 ,
    
}

xs_scale_dic = {'UL16NanoAODv9': {'DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 2.991310948858253,
  'DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 7.138925260818154,
  'DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 3.686932374495151,
  'DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 6.214908946937683,
  'DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 8.011787385711257,
  'DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 5.978752909327263,
  'DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 14.617404276755694},
 'UL17NanoAODv9': {'DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 5.1030609511138065,
  'DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 5.9610666847447815,
  'DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 6.994267753016897,
  'DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 5.464222685936051,
  'DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 6.0985857085643485,
  'DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 12.484619059260753,
  'DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 2.5996094839386332},
 'UL18NanoAODv9': {'DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 5.737810943377211,
  'DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 5.336676322857477,
  'DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 5.729376480238347,
  'DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 10.072142643834628,
  'DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 4.8339327922549735,
  'DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 5.372392798310038,
  'DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 4.038484958833355},
 'UL16NanoAODAPVv9': {'DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 2.795146012551516,
  'DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 7.673615583818124,
  'DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 4.088296479338986,
  'DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 7.903701508961329,
  'DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 3.4002966860617256,
  'DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 4.441725939157714,
  'DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8': 4.461335661665623}}


### make a check with the number of events of herwig

def getXSweight(dataset, IOV, herwig = False):
    z_xs = 6077.22
    if IOV=='2016APV' or IOV=='2016':
        lum_val = lumi['2016']
    else:
        lum_val = lumi[IOV]
    if dataset == 'UL16NanoAODAPVv9': dataset = 'UL16NanoAODv9'
    if not herwig:
        print("Using PYTHIA")
        weight  = (lum_val*6077.22)/numentries[dataset]
    else:
        print("Using Herwig")
        weight  = (lum_val*6077.22)/numentries_herwig[dataset]
    return weight


def getpTweight(pt, herwig):
    pt_bins = np.array([140, 200, 260, 350, 460, 13000])
    #weights = np.array([1.3382529 , 1.01746086, 0.73766062, 0.50567517, 0.50315774]) 
    if herwig:
        weights =  np.array([       1, 1.044007  , 0.95259771, 0.84541505, 0.77269404])  ## Herwig
    else:
        weights = np.array([       1, 0.92659584 ,1.13264522, 1.46778435, 1.54480244])    ## Pythia
    corr = dense_lookup(weights,  [pt_bins])
    return corr(pt)



def ApplyVetoMap(IOV, jets, mapname='jetvetomap'):

    if IOV=="2016APV":
        IOV="2016"
    iov_map = {
        "2016" : "2016postVFP_UL",
        "2017" : "2017_UL",
        "2018" : "2018_UL"}
    fname = "correctionFiles/POG/JME/"+iov_map[IOV]+"/jetvetomaps.json.gz"
    hname = {
        "2016"   : "Summer19UL16_V1",
        "2017"   : "Summer19UL17_V1",
        "2018"   : "Summer19UL18_V1"
    }
    # print("Len of jets before veto", len(jets))
    # print("veto file name", fname)
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    jetphi = ak.where(jets.phi<3.141592, jets.phi, 3.141592)
    jetphi = ak.where(jetphi>-3.141592, jetphi, -3.141592)
    vetoedjets = np.array(evaluator[hname[IOV]].evaluate(mapname, jets.eta, jetphi), dtype = bool)
    # print("vetoed jets", vetoedjets)
    # print("Sum of vetoed jets ", ak.sum(vetoedjets), " len of veto jets ", len(vetoedjets))
    # print("Len of jets AFTER veto", len(jets[~vetoedjets]))
    return ~vetoedjets


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
            '* * '+'correctionFiles/JEC/{0}/{0}_UncertaintySources_AK8PFPuppi.junc.txt'.format(jec_tag),  ## Uncomment this later
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
        jec_names.extend(['{0}_UncertaintySources_AK8PFPuppi_{1}'.format(jec_tag, unc_src) for unc_src in uncertainty_sources])  ##uncomment this later

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

    ###### testing purposes

    # CleanedJets = FatJets
    # #debug(self.debugMode, "Corrected Jets (Before Cleaning): ", CleanedJets[0].pt)
    # CleanedJets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]
    # CleanedJets["pt_raw"] = (1 - CleanedJets.rawFactor) * CleanedJets.pt
    # CleanedJets["mass_raw"] = (1 - CleanedJets.rawFactor) * CleanedJets.mass
    # CleanedJets["pt"] = CleanedJets.pt_raw
    # CleanedJets["mass"] = CleanedJets.mass_raw

    #############

    #FatJets = CleanedJets

    ## Uncomment this when removing testing part
    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
    FatJets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]
    if not isData:
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



def GetJetCorrections_sd(FatJets, events, era, IOV, isData=False, uncertainties = None):
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
            '* * '+'correctionFiles/JEC/{0}/{0}_UncertaintySources_AK8PFPuppi.junc.txt'.format(jec_tag),  ## Uncomment this later
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
        jec_names.extend(['{0}_UncertaintySources_AK8PFPuppi_{1}'.format(jec_tag, unc_src) for unc_src in uncertainty_sources])  ##uncomment this later

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

    ###### testing purposes

    # CleanedJets = FatJets
    # #debug(self.debugMode, "Corrected Jets (Before Cleaning): ", CleanedJets[0].pt)
    # CleanedJets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]
    # CleanedJets["pt_raw"] = (1 - CleanedJets.rawFactor) * CleanedJets.pt
    # CleanedJets["mass_raw"] = (1 - CleanedJets.rawFactor) * CleanedJets.mass
    # CleanedJets["pt"] = CleanedJets.pt_raw
    # CleanedJets["mass"] = CleanedJets.mass_raw

    #############

    #FatJets = CleanedJets

    ## Uncomment this when removing testing part

    
    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['msoftdrop']
    
    FatJets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]
    if not isData:
        FatJets['pt_gen'] = ak.values_astype(ak.fill_none(FatJets.matched_gen.pt, 0), np.float32)
    
    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'msoftdrop'
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
    JetCollection = ak.with_field(JetCollection, JetCollection.pt*isHEM, "pt" )
    return JetCollection

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


def GetEleTrigEff(IOV, lep0pT, lep0eta, var = ""):
    num = ak.num(lep0pT)
    ceval = correctionlib.CorrectionSet.from_file('correctionFiles/eleSF/egammaEffi_EGM2D.json')
    
    if var=='up':
        wrap_c = ceval["pt_reweight_up"]

    if var=='down':
        wrap_c = ceval["pt_reweight_down"]

    else:
        wrap_c = ceval["pt_reweight"]
    print(lep0eta)    
    sf = wrap_c.evaluate(ak.flatten(lep0eta), ak.flatten(lep0pT))
    return ak.unflatten(sf, num)
    

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


def jmssf(IOV, FatJet,  var = ''):
    jmsSF = {

        "2016APV":{"sf": 1.00, "sfup": 1.0094, "sfdown": 0.9906}, 

        "2016"   :{"sf": 1.00, "sfup": 1.0094, "sfdown": 0.9906}, 

        "2017"   :{"sf": 0.982, "sfup": 0.986, "sfdown": 0.978},

        "2018"   :{"sf": 0.999, "sfup": 1.001, "sfdown": 0.997}} 
    
    out = jmsSF[IOV]["sf"+var]
    

    FatJet = ak.with_field(FatJet, FatJet.mass * out, 'mass')
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * out, 'msoftdrop')
    return FatJet

def jmrsf(IOV, FatJet, var = ''):
    jmrSF = {

       #"2016APV":{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 
        "2016APV":{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 
        "2016"   :{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 

        "2017"   :{"sf": 1.09, "sfup": 1.14, "sfdown": 1.04},

        "2018"   :{"sf": 1.108, "sfup": 1.142, "sfdown": 1.074}}     
    
    jmrvalnom = jmrSF[IOV]["sf"+var]
    
    recomass = FatJet.mass
    genmass = FatJet.matched_gen.mass


    # counts = ak.num(recomass)
    # recomass = ak.flatten(recomass)
    # genmass = ak.flatten(genmass)
    
    
    deltamass = (recomass-genmass)*(jmrvalnom-1.0)
    condition = ((recomass+deltamass)/recomass) > 0
    jmrnom = ak.where( recomass <= 0.0, 0 , ak.where( condition , (recomass+deltamass)/recomass, 0 ))


    FatJet = ak.with_field(FatJet, FatJet.mass * jmrnom, 'mass')
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * jmrnom, 'msoftdrop')

    return FatJet


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
