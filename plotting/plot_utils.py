import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


from matplotlib.colors import LinearSegmentedColormap
from tueplots.constants.color import rgb
from matplotlib import ticker
from tueplots import bundles
import os


plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.dpi":200})

rw = LinearSegmentedColormap.from_list("rwg", colors=[(1, 1, 1), rgb.tue_red], N=1024)
colors = [rgb.tue_red, rgb.tue_blue, rgb.tue_green, rgb.tue_brown, rgb.tue_gold, rgb.tue_darkgreen,rgb.tue_violet,rgb.tue_mauve]

def plot_hessian(prior_prec):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.matshow(jnp.log10(jnp.abs(prior_prec)), cmap=rw)
    ax.set_xlim(0, prior_prec.shape[0])

    cax = fig.add_axes([1, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}(|H|)$")
    cb.locator = ticker.MultipleLocator(1)
    cb.update_ticks()

    plt.show()


def plot_accuracy_tasks(num_tasks, list_methods, acc_matrices, acc_mean,PLOT_DIR=None, max_val=0.5):
    plt.figure()
    plt.rcParams.update(bundles.beamer_moml(rel_width=.4 * (num_tasks / 5)))

    for method in range(len(list_methods)):
        if method == len(list_methods) - 1:
            c = 'black'
        else:
            c = 'grey'
        for i in range(num_tasks):
            plt.plot(np.arange(i + 1, num_tasks + 1), acc_matrices[method][i:, i], 'o-', color=c, alpha=0.6);
        plt.plot(np.arange(1, num_tasks + 1), acc_mean[method], 'o-', color=colors[method], label=list_methods[method]);

    plt.axhline(1 / 10, ls='-', label='chance level', linewidth=.5);
    plt.grid(axis='x')
    plt.ylim(max_val, 1);
    plt.xlim(0.9, num_tasks + 0.1);
    plt.xticks(np.arange(1, num_tasks + 1));
    plt.xlabel('Num. of Tasks');
    plt.ylabel('Accuracy');
    plt.legend();
    if PLOT_DIR is not None:
        plt.savefig(os.path.join(PLOT_DIR, 'accuracies.pdf'))
    plt.show()


def plot_accuracy_Q(num_tasks, Qs, acc_mean,PLOT_DIR=None):
    fig, ax = plt.subplots(2,1)
    plt.rcParams.update(bundles.beamer_moml(rel_width=.4 * (num_tasks / 5)))

    for qi in range(len(Qs)):
        ax[0].plot(np.arange(1, num_tasks + 1), acc_mean[qi][:num_tasks], 'o-', color=colors[qi], label=Qs[qi]);

    ax[0].grid(axis='x')
    ax[0].set_ylim(0.8, 0.98);
    ax[0].set_xlim(0.9, num_tasks + 0.1);
    ax[0].set_xticks(np.arange(1, num_tasks + 1));
    ax[0].set_xlabel('Num. of Tasks');
    ax[0].set_ylabel('Accuracy');
    ax[0].legend();

    for qi in [0]:
        ax[1].plot(np.arange(1, num_tasks + 1), acc_mean[qi][:num_tasks], 'o-', color=colors[qi], label=Qs[qi]);

    ax[1].grid(axis='x')
    ax[1].set_ylim(0.8, 0.98);
    ax[1].set_xlim(0.9, num_tasks + 0.1);
    ax[1].set_xticks(np.arange(1, num_tasks + 1));

    if PLOT_DIR is not None:
        plt.savefig(os.path.join(PLOT_DIR, 'accuracies_diff_Qs.pdf'))
    plt.show()

def plot_accuracy_lambdas(num_tasks, lambdas, acc_mean,PLOT_DIR=None):
    plt.figure()
    plt.rcParams.update(bundles.beamer_moml(rel_width=.4 * (num_tasks / 5)))

    for li in range(len(lambdas)):
        plt.plot(np.arange(1, num_tasks + 1), acc_mean[li][:num_tasks], 'o-', color=colors[li], label=lambdas[li]);

    plt.grid(axis='x')
    plt.ylim(0.5, 1);
    plt.xlim(0.9, num_tasks + 0.1);
    plt.xticks(np.arange(1, num_tasks + 1));
    plt.xlabel('Num. of Tasks');
    plt.ylabel('Accuracy');
    plt.legend();
    if PLOT_DIR is not None:
        plt.savefig(os.path.join(PLOT_DIR, 'accuracies_diff_lambdas.pdf'))
    plt.show()


def plot_accuracy_batches(num_tasks, batches, acc_mean,PLOT_DIR=None):
    plt.figure()
    plt.rcParams.update(bundles.beamer_moml(rel_width=.4 * (num_tasks / 5)))

    for bi in range(len(batches)):
        plt.plot(np.arange(1, num_tasks + 1), acc_mean[bi][:num_tasks], 'o-', color=colors[bi], label=batches[bi]);

    plt.grid(axis='x')
    plt.ylim(0.7, 1);
    plt.xlim(0.9, num_tasks + 0.1);
    plt.xticks(np.arange(1, num_tasks + 1));
    plt.xlabel('Num. of Tasks');
    plt.ylabel('Accuracy');
    plt.legend();
    if PLOT_DIR is not None:
        plt.savefig(os.path.join(PLOT_DIR, 'accuracies_diff_batches.pdf'))
    plt.show()


def plot_accuracy_nets(num_tasks, nets, acc_mean,PLOT_DIR=None):
    plt.figure()
    plt.rcParams.update(bundles.beamer_moml(rel_width=.4 * (num_tasks / 5)))

    for ni in range(len(nets)):
        plt.plot(np.arange(1, num_tasks + 1), acc_mean[ni][:num_tasks], 'o-', color=colors[ni], label=nets[ni]);

    plt.grid(axis='x')
    plt.ylim(0.7, 1);
    plt.xlim(0.9, num_tasks + 0.1);
    plt.xticks(np.arange(1, num_tasks + 1));
    plt.xlabel('Num. of Tasks');
    plt.ylabel('Accuracy');
    plt.legend();
    if PLOT_DIR is not None:
        plt.savefig(os.path.join(PLOT_DIR, 'accuracies_diff_nets.pdf'))
    plt.show()

def plot_Pi_t(Pi_t,PLOT_DIR=None):
    HOW_MANY = 10
    T = len(Pi_t)
    fig, ax = plt.subplots(len(Pi_t), 4, figsize=(15, 3*T))
    for task_id in range(T):
        D, U, C = Pi_t[task_id]['arr_0'], Pi_t[task_id]['arr_1'], Pi_t[task_id]['arr_2']

        low_rank = U[::HOW_MANY, :] @ C[:, :] @ U[::HOW_MANY, :].T
        diagonal = jnp.diag(D[0, ::HOW_MANY])
        object = diagonal + low_rank


        ax[task_id,0].matshow(jnp.abs(object),vmax=1e-3)
        ax[task_id,0].axis('off')
        ax[0,0].set_title('The full object')
        ax[task_id,1].matshow(diagonal, vmax=4)
        ax[task_id,1].axis('off')
        ax[0,1].set_title('The diagonal part')
        ax[task_id,2].matshow(jnp.abs(low_rank),vmax=1e-3)
        ax[task_id,2].axis('off')
        ax[0,2].set_title('Low-rank part')
        ax[task_id,3].matshow(C, vmax=4)
        ax[task_id,3].axis('off')
        ax[0,3].set_title('C')

    if PLOT_DIR is not None:
        plt.savefig(os.path.join(PLOT_DIR, 'Pi_t.pdf'), bbox_inches='tight')

    #plt.show()

def plot_eigenvals(Pi_t,PLOT_DIR=None):
    #HOW_MANY = 1000
    T = len(Pi_t)
    fig, ax = plt.subplots(len(Pi_t),1, figsize=(7, 3*T))
    for task_id in range(T):
        D, U, C = Pi_t[task_id]['arr_0'], Pi_t[task_id]['arr_1'], Pi_t[task_id]['arr_2']
        D_approx = jnp.mean(D)

        #low_rank = U @ C @ U.T
        #diagonal = jnp.diag(D[0, ::HOW_MANY])
        #object = diagonal + low_rank
        #eigenvalues = jnp.linalg.eigvalsh(object)

        ax[task_id].hist(jnp.diag(C+ D_approx), bins=35, range=(0,35))
        ax[task_id].set_ylim(0,10)
        ax[0].set_title('Eigenvalues')

    if PLOT_DIR is not None:
        plt.savefig(os.path.join(PLOT_DIR, 'eigenvals.pdf'), bbox_inches='tight')

    plt.show()

