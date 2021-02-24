import numpy as np
import random as rd

# experiment design paramters
conditions = ["HP", "LP"]
delayMaxs = [20, 40]
nBlock = 3
blockMin = 7
blockSec = blockMin * 60
tokenValue = 10
smallReward = 0
iti = 2
# delay distributions 
pareto = {
    "k": 4,
    "mu": 0,
    "sigma": 2
}

# analyses parameters
tGrid = np.array(range(blockSec * nBlock))
kmGrid = np.linspace(0, min(delayMaxs) - 0.2, num = min(delayMaxs) / 0.2) # time grid for Kaplan-Meier survival curves

def expParas():
    return conditions, delayMaxs, blockMin, blockSec, nBlock, tokenValue, smallReward, iti, pareto, tGrid, kmGrid


def findOptim(dist):
    '''find the optimal give-up time based on the delay distribution
    
    key arguments:
        
        dist -- dist of delay, encoded at 1-s resolution     
    '''
    time = dist['time']
    bin = time[1] - time[0]
    cdf = dist['cdf']
    pdf = np.diff(np.append(0, cdf))

    # average delay durations for all policies 
    meanRewardDelays = np.cumsum((time - 0.5 * bin) * pdf) / np.cumsum(pdf)
    rewardRates = (tokenValue * cdf + smallReward * (1 - cdf)) / \
    ((meanRewardDelays * cdf + time * (1 - cdf)) + iti) 
    rewardRates = np.nan_to_num(rewardRates, 0)

    # 
    optimWaitThreshold = time[np.argmax(rewardRates)]
    optimRewardRate = max(rewardRates)
    return optimWaitThreshold, optimRewardRate

def drawSample(cond):
    ''' draw samples from the delay distribution
    
    key arguments:
    
    cond: environment condition
    
    '''
    k = pareto['k']
    mu = pareto['mu']
    sigma = pareto['sigma']
    
    if cond == "HP":
        sample = rd.uniform(0, delayMaxs[0])
    else:
        sample = min(mu + sigma * (rd.uniform(0, 1) ** (-k) - 1) / k, delayMaxs[1])
    
    return sample


def sample2dist(delays, delayMax, step):
    '''convert delay samples into a delay CDF
    
    key arguments:
    
    delays: delay samples
    
    delayMax: upper limit of delay durations
    
    step: sampling resolution of the output dist
    
    '''
    tempt = delayMax / step
    if not tempt.is_integer():
        print("delayMax should be divisble by step")
        return
    
    nBin = int(delayMax / step)
    time = np.linspace(step, step + nBin * step, num = nBin, endpoint = False) # right boundaries
    cdf = np.zeros_like(time)

    for i in range(int(nBin)):
        cdf[i] = np.sum(delays < time[i]) / len(delays)
        
    dist = {
        "time": time,
        "cdf": cdf
    }
    
    return dist


def empStoc(delays):
    nTrial = len(delays)

    # initialize 
    trialEarnings_ = np.repeat(0.0, nTrial)
    timeWaited_ = np.repeat(0.0, nTrial)
    sellTime_ = np.repeat(0.0, nTrial)
    elapsedTime = 0
    pastDelays = np.array([])

    # delays 
    for i in range(nTrial):
        elapsedTime = elapsedTime + iti
        scheduledDelay = delays[i]

        # determine the empirical average reward rate 
        if(i == 0):
            aveRewardRateHat = np.nan 
        else:
            aveRewardRateHat = np.sum(trialEarnings_[:i])/ np.sum(timeWaited_[:i] + iti)

        # make decisions on the fly
        timeWaited = 0 # reset timeWaited
        while(timeWaited <= delayMax):
            # for the following time interval [t, t + 1), predict the remaining delay      
            select = pastDelays >= timeWaited
            scheduledDelayHat = np.average(pastDelays[select]) 
            remainDelayHat = scheduledDelayHat - timeWaited
            # action selection
            if(i == 0):
                pWait = 1
                decValue = np.nan
            else:
                decValue = remainDelayHat * aveRewardRateHat - tokenValue
                pWait = 1 / (1  + np.exp(decValue * tau - eta))
            action = 'wait' if rd.uniform(0, 1) < pWait else 'quit'
            # check the trial-wise output 
            print([i, scheduledDelay, pWait, aveRewardRateHat, decValue, timeWaited, action]) # use this to check the performance of tau
            # check the status 
            tokenMature = (scheduledDelay >= timeWaited) & (scheduledDelay < (timeWaited + stepSec))  
            getToken = (action == 'wait' and tokenMature) 
            isTerminal = (getToken or action == "quit")
            if isTerminal:
                timeWaited_[i] = scheduledDelay if getToken else timeWaited 
                trialEarnings_[i] = tokenValue if getToken else smallReward
                sellTime_[i] = elapsedTime
                break
            # next time step
            timeWaited = timeWaited + stepSec
            elapsedTime = elapsedTime + stepSec
      # update pastDelays
        pastDelays = np.append(pastDelays, scheduledDelay)
            
    # outputs 
    outputs = pd.DataFrame({
        "trialNum": np.arange(0, nTrial), 
        "condition": np.repeat(condition, nTrial),
        "trialEarnings": trialEarnings_, 
        "timeWaited": timeWaited_,
        "sellTime": sellTime_,
        "delay": delays,
    })

    return(outputs)

