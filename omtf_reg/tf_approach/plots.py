import matplotlib.pyplot as plt
import numpy as np
import json
import os

with open('basic_info.json', 'r') as f:
    data = json.load(f)
    ep_number = data['epochs']
    path_model = data['path_model']

path_up = os.path.dirname(path_model)
# [()] is needed as np load returns 0-dim object
train_results = np.load(path_up + '/train.npy')[()]
test_results = np.load(path_up + '/test.npy')[()]

ep_plot_thresh = 0

epochs = range(1,ep_number+1)[ep_plot_thresh:]

plt.figure(figsize=(8, 4))
plt.plot(epochs, train_results['r2_scores_tr'][ep_plot_thresh:], label='TRAIN', color='b')
plt.plot(epochs, train_results['r2_scores_val'][ep_plot_thresh:], label='VALID', color='y')
plt.title(f'r2 score for training for {ep_number} epochs')
plt.xlabel('Epochs')
plt.ylabel('r2 score')
plt.legend()
plt.savefig(path_up+'/train_scores_r2.pdf')
plt.tight_layout()

plt.close()
plt.figure(figsize=(8, 4))
plt.plot(epochs, train_results['losses_val'][ep_plot_thresh:], label='VALID', color='y')
plt.plot(epochs, train_results['losses_tr'][ep_plot_thresh:], label='TRAIN', color='b')
plt.title(f'Loss for training for {ep_number} epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(path_up+'/train_scores_loss.pdf')
plt.close()


plt.figure(figsize=(8, 4))
plt.plot(epochs, train_results['diff_tr'][ep_plot_thresh:], label='TRAIN', color='b')
plt.plot(epochs, train_results['diff_val'][ep_plot_thresh:], label='VALID', color='y')
#plt.fill_between(epochs, diff_tr + diff_tr_std, diff_tr - diff_tr_std, alpha=0.1, color= 'blue')
#plt.fill_between(epochs, diff_val + diff_val_std, diff_val - diff_val_std, alpha=0.1, color= 'yellow')

plt.title(f'Pull for training for {ep_number} epochs')
plt.xlabel('Epochs')
plt.ylabel('Pull')
plt.legend()
plt.tight_layout()
plt.savefig(path_up+'/train_scores_pull.pdf')
plt.close()



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
plt.savefig(path_up+'/test_scores_eff.pdf')
plt.close()
