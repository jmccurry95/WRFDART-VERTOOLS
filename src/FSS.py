from datetime import datetime
import numpy as np
import os
import netCDF4 as nc
import pandas as pd
from scipy.spatial.distance import cdist
from netCDF4 import Dataset as netcdf_dataset
#import wrf
#from wrf import to_np
from datetime import datetime, timedelta
import sys
import os
import subprocess
import configparser
import matplotlib.pyplot as plt

class MissingEnvironmentVariable(Exception):
    pass
class ObsForecastDomainMismatch(Exception):
    pass
def _config_handler(setting): #gets settings from input files - dimsize, available vars, etc
    if not os.path.isfile("FSS.ini"):
        raise FileNotFoundError('cannot find necessary config file')
    else:
        try:
            config=configparser.ConfigParser()
            config.read('{}/FSS.ini'.format(os.getcwd()))
            return config['general']['{}'.format(setting)]
        except:
            raise MissingEnvironmentVariable('cannot find config variable: {}'.format(setting))
def evaluate_fss(model_probabilities, obs_probabilities):
    roi = [int(_config_handler('roi_1')),int(_config_handler('roi_2')),int(_config_handler('roi_3'))]  # About 3km, 6 km, 15 km, 45 km and 135km respectively
    FN = [] #false negatives
    FP = [] #false positives
    TN = [] #true negatives
    TP = [] #true positives
    O = [] #observed fractions
    M = [] #modeled fractions
    possible = np.zeros((len(roi)+1,np.shape(model_probabilities)[0],np.shape(model_probabilities)[1])) #field showing total non-NAN gridpoints for scaling 
    if _config_handler('mask_fss')=='True':
        MASK=np.load(FSS_MASK)
    for i in range(int(_config_handler('roi_3')),int(_config_handler('dimsize_X'))-int(_config_handler('roi_3'))): #buffer based on largest ROI
        for j in range(int(_config_handler('roi_3')),int(_config_handler('dimsize_Y'))-int(_config_handler('roi_3'))):   
                if _config_handler('mask_fss')=='True':
                    if(MASK[j,i]<0.5):
                       continue
                # Loop through roi's
                obcount = []
                modcount = []
                fneg = []
                fpos = []
                tneg = []
                tpos = []
                
                ri = 0
                for r in roi:    
                    # Take subset of points arround current one
                    a_in = obs_probabilities[max(j-r,0):min(j+r+1,int(_config_handler('dimsize_X'))),max(i-r,0):min(i+r+1,int(_config_handler('dimsize_Y')))]
                    b_in = model_probabilities[max(j-r,0):min(j+r+1,int(_config_handler('dimsize_X'))),max(i-r,0):min(i+r+1,int(_config_handler('dimsize_Y')))]
                    # calculate number of valid elements 
                    
                    possible[ri,j,i] = float(np.count_nonzero(~np.isnan(a_in)))
                    #if > 50% of FSS window is within boundaries and not outside radar FOI or otherwise NAN, record fractions
                    if(possible[ri,j,i]>=0.5*(((2*r)+1)**2)):
                        # Number of obs field points where threshold is exceeded
                 
                        obcount.append(float(np.nansum(a_in))/possible[ri,j,i])
                        
                        # Number of model points where threshold is exceeded

                        modcount.append(float(np.nansum(b_in))/possible[ri,j,i])

                        # Store obs fraction calculations for each roi                         
                        fneg.append(np.nansum(np.where((a_in==1)&(b_in<1),1-b_in,0))) 
                        fpos.append(np.nansum(np.where((a_in==0)&(b_in>0),b_in,0)))
                        tneg.append(np.nansum(np.where((a_in==0)&(b_in<1),1-b_in,0)))
                        tpos.append(np.nansum(np.where((a_in==1)&(b_in>0),b_in,0)))
                    # if < 50% of FSS window is valid, then record NAN's instead of fractions 
                    else:
                   
                        
                        obcount = [np.nan for r in roi]

                        # Number of model points where threshold is exceeded

                        modcount = [np.nan for r in roi]

                        # Store obs fraction calculations for each roi 
                          
                        #detection theory stuff
                        fneg = [np.nan for r in roi]
                        fpos = [np.nan for r in roi]
                        tneg = [np.nan for r in roi]
                        tpos = [np.nan for r in roi]
                        break 
                    ri = ri + 1                       

                O.append(obcount)
                M.append(modcount)
                FN.append(fneg)
                FP.append(fpos)
                TN.append(tneg)
                TP.append(tpos)
                    # Loop through roi's
   

    FN = np.array(FN)
    FP = np.array(FP)
    TN = np.array(TN)
    TP = np.array(TP)
    O = np.array(O)
    M = np.array(M)
    fss = []
    bias = []
    pod =[]
    false_alarm = []
    for i in range(np.size(roi)):
        pod.append(1 - np.nanmean(FN[:,i])/np.nanmean((FN[:,i]+TP[:,i])))
        false_alarm.append(np.nanmean(FP[:,i])/np.nanmean((FP[:,i]+TN[:,i])))
        MSE = np.nansum( (O[:,i] - M[:,i])**2 ) / np.count_nonzero(~np.isnan(O[:,i]))
        MSE_ref = ( np.nansum( O[:,i]**2 ) + np.nansum( M[:,i]**2 ) ) / np.count_nonzero(~np.isnan(O[:,i]))
        fss.append(1.0 - (MSE / MSE_ref))
        bias.append(np.nansum( (O[:,i] - M[:,i]) ) / np.count_nonzero(~np.isnan(O[:,i])))
    return fss,bias,false_alarm,pod   
def _find_files_with_numerical_suffix(basedir,extra_terms='N/A'):
    file_list = []
    for filename in os.listdir(basedir):
        if filename[-1].isdigit():  # Check if the last character is a digit
            if extra_terms!='N/A':
                file_list.append(os.path.join(basedir,filename))
            elif extra_terms in filename:
                file_list.append(os.path.join(basedir,filename))       
    return file_list
def _return_date_from_filename(filename):
    basename=os.path.basename(os.path.normpath(filename))
    date = re.findall('\d+',basename)
    return date
def _ensemble_getter(timestamp_path): #in: datetimes, base directories (global) out: filepaths
    file_list = _find_files_with_numerical_suffix(timestamp_path,extra_terms='wrfout_d01_forecast')
    return file_list
def _obs_getter(timestamp_path): #in: datetimes, base directories (global) out: filepaths
    file_list=[]
    base_directory=_config_handler('base_obs_directory')
    timestamp_date=_return_date_from_filename(timestamp_path)
    file_list.append(os.join(base_directory,timestamp_date))
    return file_list

def _init_getter(): #get valid forecast initiation times
    stride_mode=_config_handler('find_mode')
    parent_directory = _config_handler('parent_directory')
    folder_list=[]
    if stride_mode=='datetime':
        start_init=_config_handler('start_init')
        end_init=_config_handler('end_init')
        delta=timedelta(minutes=_config_handler('init_interval'))
        while start_init<end_init:
            folder_list.append(os.path.join(parent_directory,'{}{}'.format('WRFOUTS_FCST',start_init.strftime("%Y%m%d%H%M"))))
            start_init+=delta
    else:
        folder_list=_find_files_with_numerical_suffix(parent_directory,extra_terms='WRFOUTS_FCST')
        
    return folder_list
def _timestamp_getter(init_path): #get valid timestamps in each forecast init folder 
    stride_mode=_config_handler('find_mode')
    folder_list=[]
    if stride_mode=='datetime':
        delta_total=timedelta(minutes=_config_handler('forecast_length'))
        delta_interval=timedelta(minutes=_config_handler('forecast_output_interval'))
        start_timestamp=datetime.strptime(_return_date_from_filename(init_path),"%Y%m%d%H%M")
        end_timestamp=start_timestamp+delta_total
        while start_timestamp<end_timestamp:
            folder_list.append(os.path.join(init_path,'{}'.format(start_timestamp.strftime("%Y%m%d%H%M"))))
            start_timestamp+=delta_interval
    else:
        folder_list = _find_files_with_numerical_suffix(init_path)
    return folder_list
def _get_event_probability(infiles,variable,thresh,operation='none'):
    #set up empty array for calculations
    
    for n,infile in enumerate(infiles):  
        #try:
        reflh = np.asarray(netcdf_dataset(infile)[variable]).squeeze()
        #except:
        #    reflh = np.load(infile)
        #    reflh = reflh.squeeze()
        if operation == 'composite':
            reflh = np.amax(reflh,axis=0).squeeze()
        if (n==0):
            probabilities = np.zeros_like(reflh)
        probabilities += np.where(reflh>=thresh, 1, 0)

    probabilities[np.isnan(reflh)] = np.nan
    probabilities_final = probabilities/len(infiles)
    return probabilities_final

f __name__ == "__main__":
    init_paths=_init_geter()
    for init_path in init_paths:
        timestamp_paths=_timestamp_getter(init_path)
            for n,timestamp_path in enumerate(timestamp_paths):
                print('working on time: {} init: {}'.format(_get_date_from_filename(timestamp_path),_get_date_from_filename(init_path)))
                indie = n*int(eval(_config_handler('forecast_output_interval')))
                ensemble=_ensemble_getter(timestamp_path)
                obs=_obs_getter(timestamp_path)
                if _config_handler('mask_obs')=='True':
                    obs=_mask_obs(obs)
                ensemble_probs=_get_event_probability(ensemble,variable=eval(_config_handler('model_variable_of_interest')),eval(_config_handler('thresh')),operation=eval(_config_handler('operation')))
                obs_probs=_get_event_probability(obs,variable=eval(_config_handler('obs_variable_of_interest')),eval(_config_handler('thresh')),operation=eval(_config_handler('operation')))
                ensemble_probs[np.isnan(obs_probs)]=np.nan
                fss,bias,false_alarm,pod = evaluate_fss(ensemble_probs,obs_probs

                if(n==0):
                     df = pd.DataFrame(np.array([[datetime.strptime(_get_date_from_filename(init_path),"%Y%m%d%H%M"),indie,fss[0],fss[1],fss[2],bias[0],bias[1],bias[2],pod[0],pod[1],pod[2],falarm[0],falarm[1],falarm[2]]]),columns=['init','time','3km_FSS','15km_FSS','135km_FSS','3km_BIAS','15km_BIAS','135km_BIAS','3km_POD','15km_POD','135km_POD','3km_FALARM','15km_FALARM','135km_FALARM']).set_index('time')        
                else:
                     df = df.append(pd.DataFrame(np.array([[datetime.strptime(_get_date_from_filename(init_path),"%Y%m%d%H%M"),indie,fss[0],fss[1],fss[2],bias[0],bias[1],bias[2],pod[0],pod[1],pod[2],falarm[0],falarm[1],falarm[2]]]),columns=['init','time','3km_FSS','15km_FSS','135km_FSS','3km_BIAS','15km_BIAS','135km_BIAS','3km_POD','15km_POD','135km_POD','3km_FALARM','15km_FALARM','135km_FALARM']).set_index('time'))        

                    df= df.reset_index()
                    os.makedirs('{}'.format(_config_handler('savedir')),exist_ok=True)
                    df.to_pickle('{}/{}_{}'.format(_config_handler('savedir'),forecast_init,str(prescribed_thresh).replace(".", "")))