import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from data_preprocessing import zscore_norm

def load_training_data(path):
    signals_train = pd.read_csv(path)
    signals_train.dropna(inplace=True)
    return signals_train
    
    
def load_logs(path):

    logs_train = pd.read_csv(path)

    logs_train = logs_train[logs_train['Remark'].str.contains('A Ctrl:')|
              logs_train['Remark'].str.contains('A CtrlV.STD')|
              logs_train['Remark'].str.contains('Ambient temperature high:')|
              logs_train['Remark'].str.contains('B Ctrl:')|
              logs_train['Remark'].str.contains('Ch High res. load')|
              logs_train['Remark'].str.contains('Circuit breaker open')|
              logs_train['Remark'].str.contains('EmcPitchAvel:')|
              logs_train['Remark'].str.contains('Emergency circuit open')|
              logs_train['Remark'].str.contains('Encoder signal error')|
              logs_train['Remark'].str.contains('Error on all wind sensors')|
              logs_train['Remark'].str.contains('ExEx low voltage')|
              logs_train['Remark'].str.contains('Extr. low voltage')|            
              logs_train['Remark'].str.contains('Ext. High cur. Grid inv')|
              logs_train['Remark'].str.contains('ExtHighIRotorInv phase:')|
              logs_train['Remark'].str.contains('External RPM not Reset')|
              logs_train['Remark'].str.contains('Extra info. Err:')|   
              logs_train['Remark'].str.contains('Feedback=0,')|
              logs_train['Remark'].str.contains('Feedback = 0,')|
              logs_train['Remark'].str.contains('Feedback = 1, Brake')|
              logs_train['Remark'].str.contains('Frequency error')|
              logs_train['Remark'].str.contains('High cur.rotor inv.')|
              logs_train['Remark'].str.contains('High temp top ctrl.:')|
              logs_train['Remark'].str.contains('High temp. Aux.')|
              logs_train['Remark'].str.contains('High temp. Gen bearing')|
              logs_train['Remark'].str.contains('High temp. VCP Board')|
              logs_train['Remark'].str.contains('High temperature T53:')|
              logs_train['Remark'].str.contains('High windspeed:')|
              logs_train['Remark'].str.contains('Hydr max time')|
              logs_train['Remark'].str.contains('Internal sublogic error')|
              logs_train['Remark'].str.contains('Low workingpressure:')|
              logs_train['Remark'].str.contains('No RT, High Rotor Cur L')|
              logs_train['Remark'].str.contains('Oil leakage in Hub')|
              logs_train['Remark'].str.contains('OVPHwErr UDC')|
              logs_train['Remark'].str.contains('Pitch dev. min:')|
              logs_train['Remark'].str.contains('Q7 breaker open')|
              logs_train['Remark'].str.contains('Q8 breaker open')|
              logs_train['Remark'].str.contains('Remote Reboot')|
              logs_train['Remark'].str.contains('Rotor inv. HW error L')|
              logs_train['Remark'].str.contains('SignalError.')|
              logs_train['Remark'].str.contains('above limits')|
              logs_train['Remark'].str.contains('Thermo error, ventilators')|
              logs_train['Remark'].str.contains('Thermoerror yawmotor')|
              logs_train['Remark'].str.contains('Thermoerr. Nac. fan')|         
              logs_train['Remark'].str.contains('Too many auto-restarts:')|
              logs_train['Remark'].str.contains('Trip Q8 L1: 397 V')|
              logs_train['Remark'].str.contains('WS1 timeout err.')|
              logs_train['Remark'].str.contains('YawSpeedFault')]
    return logs_train
    
    
def load_labelling_data(path1, path2):
    
    signals_train = load_training_data(path1)
    logs_train = load_logs(path2)


    isResetLog = logs_train.dropna()
    noResetLog = logs_train[pd.isnull(logs_train["TimeReset"])]

    isResetLog.loc[:, 'TimeDetected'] = isResetLog['TimeDetected'].astype('datetime64[m]').dt.floor('10min')
    isResetLog.loc[:, 'TimeReset'] = isResetLog['TimeReset'].astype('datetime64[m]').dt.ceil('10min')

    noResetLog.loc[:, 'TimeReset'] = noResetLog['TimeDetected'].astype('datetime64[m]').dt.ceil('10min')
    noResetLog.loc[:, 'TimeDetected'] = noResetLog['TimeDetected'].astype('datetime64[m]').dt.floor('10min')

    logs_train = pd.concat([isResetLog, noResetLog])
    
    timeSet = []
    for i in range(1,logs_train.shape[0]):
        time1 = logs_train.iloc[i,0]
        time2 = logs_train.iloc[i,1]

        while time1 <= time2:
            timeSet.append(time1)
            time1 += timedelta(minutes=10)

    timeList = list(set(timeSet))
    
    signals_train.loc[:, "Timestamp"] = signals_train["Timestamp"].astype('datetime64[m]')
    
    signals_train['isFaulty'] = signals_train["Timestamp"].isin(timeList)
    map = {True : 1, False : 0}
    signals_train['isFaulty'] = signals_train['isFaulty'].map(map)
    
    return signals_train


def load_LDT_dataset(month):
    
    usecols=['TimeStamp',
     'GenSpeedRelay',
     'GenBearingtemp1',
     'GearBoxTemperature_DegC',
     'NacInsidetemp',
     'RotorSpeed_rpm',
     'WindSpeed1',
     'AmbTemp',
     'Power_kW','ReactivePower_kVAr',
     'SubPtchPrivHubtemp',
     'Pitch_Deg',
     'Frequency',
     'SubPcsPrivGridVoltagePhaseTR', 'SubPcsPrivGridVoltagePhaseST', 'SubPcsPrivGridVoltagePhaseRS',
     'SubPcsPrivGridActivePwr', 'SubPcsPrivGridReactivePwer',
     'NacelleOrientation_Deg']
    
    ldt_select = pd.read_csv("../dataset/LDT-1Hz-Tur-2019-0" + str(month) + ".csv", usecols=usecols)
    ldt_select['TimeStamp'] = ldt_select['TimeStamp'].astype('datetime64[m]').dt.floor('10min')
    
    # drop error data
    ldt_select.drop(ldt_select[ldt_select.Power_kW <= 0].index,inplace=True)
    ldt_select.drop(ldt_select[ldt_select.WindSpeed1 < 0].index,inplace=True)
    
    # unit conversion
    ldt_select['GenSpeedRelay'] *= 9.5493 
    ldt_select['SubPcsPrivGridActivePwr'] /= 1000
    ldt_select['SubPcsPrivGridReactivePwer'] /= 1000
    
    ldt_select = ldt_select.dropna()
    
    # add columns
    ldt_new = ldt_select.groupby('TimeStamp').mean()
    ldt_max = ldt_select.groupby('TimeStamp').max()
    ldt_min = ldt_select.groupby('TimeStamp').min()
    ldt_std = ldt_select.groupby('TimeStamp').std()
    
    ldt_new['GenSpeedRelay_Max'] = ldt_max['GenSpeedRelay']
    ldt_new['GenSpeedRelay_Min'] = ldt_min['GenSpeedRelay']
    ldt_new['GenSpeedRelay_Std'] = ldt_std['GenSpeedRelay']

    ldt_new['RotorSpeed_rpm_Max'] = ldt_max['RotorSpeed_rpm']
    ldt_new['RotorSpeed_rpm_Min'] = ldt_min['RotorSpeed_rpm']
    ldt_new['RotorSpeed_rpm_Std'] = ldt_std['RotorSpeed_rpm']

    ldt_new['WindSpeed1_Max'] = ldt_max['WindSpeed1']
    ldt_new['WindSpeed1_Min'] = ldt_min['WindSpeed1']
    ldt_new['WindSpeed1_Std'] = ldt_std['WindSpeed1']

    ldt_new['Power_Wh'] = ldt_new['Power_kW'] * 1000 / 6
    ldt_new['ReactivePower_VArh'] = ldt_new['ReactivePower_kVAr'] * 1000 / 6

    ldt_new['Pitch_Deg_Max'] = ldt_max['Pitch_Deg']
    ldt_new['Pitch_Deg_Min'] = ldt_min['Pitch_Deg']
    ldt_new['Pitch_Deg_Std'] = ldt_std['Pitch_Deg']

    ldt_new['SubPcsPrivGridActivePwr_Max'] = ldt_max['SubPcsPrivGridActivePwr'] 
    ldt_new['SubPcsPrivGridActivePwr_Min'] = ldt_min['SubPcsPrivGridActivePwr'] 

    ldt_new['SubPcsPrivGridReactivePwer_Max'] = ldt_max['SubPcsPrivGridReactivePwer'] 
    ldt_new['SubPcsPrivGridReactivePwer_Min'] = ldt_min['SubPcsPrivGridReactivePwer'] 
    ldt_new['SubPcsPrivGridReactivePwer_Std'] = ldt_std['SubPcsPrivGridReactivePwer'] 
    
    ldt_test = ldt_new[['GenSpeedRelay_Max','GenSpeedRelay_Min','GenSpeedRelay','GenSpeedRelay_Std',
     'GenBearingtemp1',
     'GearBoxTemperature_DegC',
     'NacInsidetemp',
     'RotorSpeed_rpm_Max', 'RotorSpeed_rpm_Min', 'RotorSpeed_rpm', 'RotorSpeed_rpm_Std',
     'WindSpeed1_Max', 'WindSpeed1_Min', 'WindSpeed1', 'WindSpeed1_Std',
     'AmbTemp',
     'Power_Wh','ReactivePower_VArh',
     'SubPtchPrivHubtemp',
     'Pitch_Deg_Min', 'Pitch_Deg_Max', 'Pitch_Deg', 'Pitch_Deg_Std',
     'Frequency',
     'SubPcsPrivGridVoltagePhaseTR', 'SubPcsPrivGridVoltagePhaseST', 'SubPcsPrivGridVoltagePhaseRS',
     'SubPcsPrivGridActivePwr_Max','SubPcsPrivGridActivePwr_Min', 'SubPcsPrivGridReactivePwer', 'SubPcsPrivGridReactivePwer_Max', 'SubPcsPrivGridReactivePwer_Min', 'SubPcsPrivGridReactivePwer_Std',
     'NacelleOrientation_Deg']]
    
    ldt_test = ldt_test.dropna()
    
    
    # label data
    model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.04),max_features=1.0)
    model.fit(ldt_test)
    ldt_test["isFaulty"] = model.predict(ldt_test)
    
    # visualize
    map = {-1 : 1, 1 : 0}
    ldt_test['isFaulty'] = ldt_test['isFaulty'].map(map)
    
#     ax = ldt_test.plot.scatter(x="WindSpeed1",y="Power_Wh",s=5, c="label", cmap='viridis',alpha=0.5, figsize=(12,8))
#     ax.set_xlabel("WindSpeed1")
    
#     Y_test = ldt_test['label']
#     X_test = ldt_test.drop('label', axis = 1)
    
    return ldt_test

def LDT_save_all():
    for i in range(1,7):
        LDT_data = load_LDT_dataset(i)
        if i == 1:
            LDT_data.to_csv("LDT_total_data.csv", header=True)
        else:
            LDT_data.to_csv("LDT_total_data.csv", mode='a', header=False)
