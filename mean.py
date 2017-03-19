import pandas as pd

pred1 = pd.read_csv("simple_xgb_new.csv")
pred2 = pd.read_csv('sub.csv')

#pred = pred1.join(pred2,on='test_id')
pred1['is_duplicate'] = 0.95*pred1['is_duplicate']  + 0.05*pred2['is_duplicate']

pred1.to_csv('mean.csv',index=False)
