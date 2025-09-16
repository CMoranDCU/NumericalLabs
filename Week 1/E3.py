import matplotlib.pyplot as plt
import numpy as np
import math as mth


def f_approx(x_0, t, t_step, t_max):
    ApproxList = [];
    StepList = [];
    
    for i in range(int(t_max / t_step)):
        
        StepList.append(t_step * i);
        
        if(i == 0):
            ApproxList.append(x_0);
            continue;
            
        ApproxList.append(ApproxList[i - 1] + (StepList[i - 1] - np.longdouble(ApproxList[i - 1]**2))*t_step);
    
    return ApproxList, StepList;


InitialValues = [3, 1, 0, -0.7, -0.75];
plt.title("Graph of Function: dx/dt = t - x^2")
#plt.xlabel("Time (Seconds)");
#plt.ylabel("Position (Meters)");

for i in range(len(InitialValues)):    
    Approx, Time = f_approx(InitialValues[i], 0, 0.05, 2);
    plt.plot(Time, Approx);
    continue;