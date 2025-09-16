import numpy as np
import matplotlib.pyplot as plt
import math

def x_approx(y_0, t_0, t_step, t_max):
    
    ApproxValue = [];
    TimeValue = [];
    
    for i in range(int(t_max / t_step)):
        
        TimeValue.append(t_step * i);
        
        if(i == 0):
            ApproxValue.append(0);
            continue;
            
        Value = ApproxValue[i - 1] - math.sin(TimeValue[i - 1])*t_step;
        ApproxValue.append(Value);
        continue;
        
        
    return ApproxValue, TimeValue;

def y_approx(x_0, t_0, t_step, t_max):
    
    ApproxValue = [];
    TimeValue = [];
    
    for i in range(int(t_max / t_step)):
        
        TimeValue.append(t_step * i);
        
        if(i == 0):
            ApproxValue.append(0);
            continue;
            
        Value = ApproxValue[i - 1] + math.cos(TimeValue[i - 1])*t_step;
        ApproxValue.append(Value);
        continue;
    
    return ApproxValue, TimeValue;


Y, X = x_approx(0, 0, 0.5, 9);
I, J = y_approx(0, 0, 0.5, 9);
plt.plot(Y, I);
plt.show();