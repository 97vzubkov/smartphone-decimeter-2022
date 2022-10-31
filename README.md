# Smartphone-Decimeter-2022

Thanks to the hosts and to eveyone who took part and contributed to this very interesting and challenging competition.
In this competition, participants' task was to compute smartphones location down to the decimeter or even centimeter resolution which could enable services that require lane-level accuracy such as HOV lane ETA estimation.
My main goal in this competition was to gain new knowledge in the field of optimization and algorithms, so I started my journey by studying the domain area and the algorithms that are used there. After trying different methods, it turned out that RTK and WLS based solutions show themselves best.

# My solution

1. Carrier Smoothing + Robust WLS + Kalman Smoother which is based on @taroz1461 - https://www.kaggle.com/code/taroz1461/carrier-smoothing-robust-wls-kalman-smoother. Algorithms that used to improve this solution: LGBM model to remove car stopping points, LGBM model to remove outliers based on area, bias correction using ECEF coordinates, algorithm to adjust tectonic plate movement. Bayesian optimization is used for searching best hyperparameters.

2. RTKLIB solution by @timeverett - https://www.kaggle.com/code/timeverett/getting-started-with-rtklib. Fistly, I've manually tuned some parameters in configuration. Methods that helped to improve accuracy are usage of multiple base stations and then averaging results, filtering out RTKLIB solutions with hardware clock discontinuites, adjustment of tectonic plate movement, bias correction.

3. Ensemble - weighted average used to ensemble RTKLIB and WLS solution.

This solution gave me 67th place on the private leaderboard and fisrt medal on Kaggle platform. 

# Tried but not worked

- savgol filter 
- cost minimization 
- apply kalman filter to raw rtk 

Some of the submissions hit the silver area but unfortunately, I choose other ones. 
