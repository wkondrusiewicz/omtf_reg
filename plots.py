import matplotlib.pyplot as plt
import numpy as np
import json

with open('basic_info.json', 'r') as f:
    data = json.load(f)
    save_loc = data["save_loc"]
    ep_number = data['epochs']

dir_loc = save_loc.split('/')
dir_loc = dir_loc[:-2]
dir_loc = '/'.join(dir_loc)

# [()] is needed as np load returns 0-dim object
train_results = np.load(dir_loc + '/train.npy')[()]
test_results = np.load(dir_loc + '/test.npy')[()]

epochs = range(1,ep_number+1)

plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)
plt.plot(epochs, train_results['r2_scores_tr'], label='TRAIN', color='b')
plt.plot(epochs, train_results['r2_scores_val'], label='VALID', color='y')
plt.title(f'r2 score for training for {ep_number} epochs')
plt.xlabel('Epochs')
plt.ylabel('r2 score')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(epochs, train_results['losses_tr'], label='TRAIN', color='b')
plt.plot(epochs, train_results['losses_val'], label='VALID', color='y')
plt.title(f'Loss for training for {ep_number} epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(epochs, train_results['diff_tr'], label='TRAIN', color='b')
plt.plot(epochs, train_results['diff_val'], label='VALID', color='y')
#plt.fill_between(epochs, diff_tr + diff_tr_std, diff_tr - diff_tr_std, alpha=0.1, color= 'blue')
#plt.fill_between(epochs, diff_val + diff_val_std, diff_val - diff_val_std, alpha=0.1, color= 'yellow')

plt.title(f'Pull for training for {ep_number} epochs')
plt.xlabel('Epochs')
plt.ylabel('Pull')
plt.legend()

plt.tight_layout()
plt.savefig(dir_loc+'/train_scores.pdf')


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(test_results['pt_intervals'], test_results['eff'].mean(axis=0))
ax1.set_xlabel('$p_T$ [GeV]')

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(test_results['pt_intervals'][::2])
ax2.tick_params(colors='red')
ax2.set_xticklabels(range(1, len(test_results['pt_intervals']) + 1, 2))
ax2.set_xlabel('Corresponding $p_T$ codes', color='red')
ax2.spines['top'].set_color('red')


ax1.set_ylabel('Effectivness')
plt.title(f'Effectivness curve', fontsize=16)
plt.savefig(dir_loc+'/test_scores.pdf')
