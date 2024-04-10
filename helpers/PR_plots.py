import pickle
import matplotlib.pyplot as plt

with open('pr_baseline_frcnn.pkl', 'rb') as f:
    frcnn_precision = pickle.load(f)
    
with open('pr_dgfrcnn_v4.pkl', 'rb') as f:
    dgfrcnn_precision = pickle.load(f)
    

plt.figure(figsize=(10, 8))

ious = [0.1, 0.5, 0.75]
colors = ['red', 'green', 'blue']
Rthres = [i / 100 for i in range(101)]
for index, PR in enumerate(frcnn_precision[:, :, 0, 0, -1]):
  plt.plot(Rthres, PR.numpy(), linestyle='dotted', color = colors[index], linewidth=4, label='FRCNN(IoU = '+str(ious[index])+')')
for index, PR in enumerate(dgfrcnn_precision[:, :, 0, 0, -1]):
  plt.plot(Rthres, PR.numpy(), color = colors[index], linewidth=4, label='DGFRCNN(IoU = '+str(ious[index])+')')
      
plt.xlabel('Recall', fontsize=22)
plt.xticks(fontsize=20)
plt.ylabel('Precision', fontsize=22)
plt.yticks(fontsize=20)
plt.title('PR Curves (DGFRCNN vs FRCNN)', fontsize=22)
plt.legend(loc='lower left', fontsize=10.5)
plt.savefig('PRCurves_FRCNN.png')  

