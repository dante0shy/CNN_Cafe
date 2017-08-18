


def out_log(time,cpu_u,gpu_u,gpu_m_u,train_loss,test_loss,test_acc):
    import pandas as pd
    import numpy as np
    data= np.array([cpu_u,gpu_u,gpu_m_u,train_loss,test_loss,test_acc])
    data = pd.DataFrame(data=data,index=["cpu_u","gpu_u","gpu_m_u","train_loss","test_loss","test_acc"])
    data.to_csv("./log/log"+str(time)+".csv")
    return


