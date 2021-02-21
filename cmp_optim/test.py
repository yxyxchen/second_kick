  

wtwCeiling = min(delayMaxs)

# 
thisTrialData[thisTrialData['trialEarnings'] != smallReward]['timeWaited'] = \
thisTrialData[thisTrialData['trialEarnings'] != smallReward]['delay']

# initialize the trial-wise estimate of WTW
nTrial = thisTrialData.shape[0]
trialWTW = np.zeros(nTrial)


# loop over trial
lastQuitIdx = -1
for i in range(nTrial):
    if thisTrialData['trialEarnings'][i] == smallReward:
        currentWTW = thisTrialData['timeWaited'][i]
        lastQuitIdx = i
    else: 
        currentWTW = max(thisTrialData['timeWaited'][(lastQuitIdx + 1) : (i + 1)]) # count after the last quit index and include the current index
    trialWTW[i] = currentWTW

# impose a ceiling value, since WTW exceeding some value may be infrequent
trialWTW = np.minimum(currentWTW, wtwCeiling)

# figure out how to convert to time-wtw later 
