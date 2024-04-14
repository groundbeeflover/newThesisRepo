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
  plt.plot(Rthres, PR.numpy(), linestyle='dotted', color = colors[index], linewidth=4)
for index, PR in enumerate(dgfrcnn_precision[:, :, 0, 0, -1]):
  plt.plot(Rthres, PR.numpy(), color = colors[index], linewidth=4)
      
plt.xlabel('Recall', fontsize=30)
plt.xticks(fontsize=30)
plt.ylabel('Precision', fontsize=30)
plt.yticks(fontsize=30)
plt.title('DGFRCNN vs FRCNN', fontsize=30)
plt.tight_layout()

#dotted_patch = plt.Line2D([], [], color='black', linestyle='dotted', label='FRCNN', linewidth=4)
#solid_patch = plt.Line2D([], [], color='black', label='DGFRCNN', linewidth=4)
#plt.legend(loc='upper right', handles=[dotted_patch, solid_patch], fontsize=20)

#plt.legend(loc='best', fontsize=26)
plt.savefig('PRCurves_FRCNN.png')  

