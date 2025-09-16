import numpy as np
import matplotlib.pyplot as plt 

def f_approx(x_0, v_0, t_0, t_step, t_max):
    X = [x_0];
    V = [v_0];
    T = [t_0];
    
    for i in range(int(t_max / t_step)):
                
        if i == 0:
            continue;            
        
        T.append(i * t_step);
        
        Xinit = X[i - 1] + t_step * V[i - 1];
        Vinit = V[i - 1] - t_step * X[i - 1];
        
        X_Val = X[i - 1] + 0.5*t_step*(V[i-1] + Vinit);
        X.append(X_Val);
        
        V_Val = V[i - 1] - 0.5*t_step*(X[i - 1] + Xinit);
        V.append(V_Val);
        
    return X, V, T;

X, V, T = f_approx(0, 3, 0, 0.05, 50);
plt.plot(T, X);
plt.plot(T, V);
plt.show();