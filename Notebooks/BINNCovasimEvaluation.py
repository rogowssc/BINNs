import numpy as np
import pandas as pd

from Modules.Utils.Imports import *
from Modules.Models.BuildBINNs import BINNCovasim
from Modules.Utils.ModelWrapper import ModelWrapper

import Modules.Utils.PDESolver as PDESolver
import Modules.Loaders.DataFormatter as DF
from utils import get_case_name, AIC_OLS, RSS
# helper functions
def to_torch(x):
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    return x.detach().cpu().numpy()


device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))
# instantiate BINN
path = '../Data/covasim_data/'
population = 50000
test_prob = 0.1
trace_prob = 0.1
keep_d = True
retrain = False
case_name = get_case_name(population, test_prob, trace_prob, keep_d)
params = DF.load_covasim_data(path, population, test_prob, trace_prob, keep_d, plot=False)
params['gamma'] /= 1.2
data = params['data']
data = (data / params['population']).to_numpy()
N = len(data)
t_max = N - 1
t = np.arange(N)
params.pop('data')

regression_coefs_cr = pd.read_csv('../Figures/covasim/' + case_name + '_regression_coef_cr.csv', index_col=0).to_numpy()
regression_coefs_qt = pd.read_csv('../Figures/covasim/' + case_name + '_regression_coef_qt.csv', index_col=0).to_numpy()
# regression_coefs = pd.read_csv('../Figures/covasim/' + '50000_0.1_0.1' + '_regression_coef.csv', index_col=0).to_numpy()
if test_prob == 0.5 and trace_prob == 0.5:
    yita_lb, yita_ub = 0.2, 0.4
    binn = BINNCovasim(params, t_max, yita_lb, yita_ub, keep_d=keep_d).to(device)
else:
    binn = BINNCovasim(params, t_max, keep_d=keep_d).to(device)
parameters = binn.parameters()
model = ModelWrapper(binn, None, None, save_name='')

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# markers = ['x', 'o', 's', 'd', '^']
# fig = plt.figure(figsize=(15 ,8))

# load model weights
model.save_name = '../Weights/'
model.save_name += case_name
if retrain:
    model.save_name += '_retrain'
model.save_name += '_best_val'
model.load(model.save_name + '_model', device=device)

# grab initial condition
u0 = data[0, :].copy()

# learned contact_rate function
def contact_rate(u):
    res = binn.contact_rate(to_torch(u)) # [:,[0,3,4]]
    res *= 1.4
    return to_numpy(res)

def quarantine_test(u):
    res = binn.quarantine_test_prob(to_torch(u))
    return to_numpy(res)

def contact_rate_regression(u):
    s, a, y = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None]
    features = [s, s ** 2]  # s related terms
    features += [a]
    features += [y]
    features = np.concatenate(features, axis=1)
    # res = features @ np.array([1.96726769, -0.98532934, -5.54030361, -7.32904966])[:, None]
    # 0.1 0.5 : [  3.37082762,  -2.50310168, -27.13545073, -14.13538669]
    res = features @ regression_coefs_cr
    res *= 1.4
    return res

def quarantine_test_regression(u):
    a, y = u[:, 0][:, None], u[:, 1][:, None]
    features = [a, y]
    features = np.concatenate(features, axis=1)
    res = features @ regression_coefs_qt
    return res

s_min, s_max = data[:,0].min(), data[:,0].max()
a_min, a_max = data[:,3].min(), data[:,3].max()
y_min, y_max = data[:,4].min(), data[:,4].max()

s_grid = np.arange(s_min, s_max, 0.05)
a_grid = np.arange(a_min, a_max, 0.01)
y_grid = np.arange(y_min, y_max, 0.01)



# simulate PDE
params['yita_lb'] = model.model.yita_lb
params['yita_ub'] = model.model.yita_ub
if keep_d:
    RHS = PDESolver.STEAYDQRF_RHS
    u_sim_NN = PDESolver.STEAYDQRF_sim(RHS, u0, t, contact_rate, quarantine_test, params)
    u_sim_NN *= population
    u_sim_regression = PDESolver.STEAYDQRF_sim(RHS, u0, t, contact_rate_regression, quarantine_test_regression, params)
    u_sim_regression *= population
else:
    RHS = PDESolver.STEAYQRF_RHS
    u_sim_NN = PDESolver.STEAYQRF_sim(RHS, u0, t, contact_rate, quarantine_test, params)
    u_sim_NN *= population
    u_sim_regression = PDESolver.STEAYQRF_sim(RHS, u0, t, contact_rate_regression, quarantine_test_regression, params)
    u_sim_regression *= population

data *= population
plot=True
if plot:
    # data = params['data']
    n = data.shape[1]
    col_names = list('STEAYDQRF') if keep_d else list('STEAYQRF')
    # t = np.arange(1, data.shape[0] + 1)
    # plot compartments
    fig = plt.figure(figsize=(10, 5))
    for i in range(1, n + 1):
        ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
        ax.plot(t, data[:, i - 1], '.k', label='Covasim Data')
        ax.plot(t, u_sim_NN[:, i - 1], '-*r', label='ODE-NN')
        ax.plot(t, u_sim_regression[:, i - 1], '-b', label='ODE-Regression')
        ax.set_title(col_names[i - 1])
        ax.legend(fontsize=8)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.savefig(model.save_folder + case_name + '.png')
        # plt.show()

# %% optional, add one comparison based on the results ODE-DE

ode_de_df = pd.read_csv('../Data/covasim_data/ODE_DE_50000_0.1_0.1.csv', index_col=0)
ode_de_array = ode_de_df.to_numpy()
# plot compartments
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(18, 9))
for i in range(1, n):
    ax = fig.add_subplot(2, 4, i) # int(np.ceil(n / 4))
    ax.plot(t, data[:, i - 1], '.k', label='Covasim Data')
    ax.plot(t, u_sim_NN[:, i - 1], 'r.-', label='ODE-NN')
    ax.plot(t, u_sim_regression[:, i - 1], 'b.-', label='ODE-SR')
    ax.plot(t, ode_de_array[:, i - 1], 'g.-', label='ODE-DE')
    if i > 5:
        ax.set_xlabel("Time (Days)")
    if i % 3 == 1:
        ax.set_ylabel("Count")
    ax.set_title(col_names[i - 1])
    if i == 1:
        ax.legend(loc="best")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad=2)
    plt.savefig(model.save_folder + case_name + '.png', dpi=300)
    # plt.show()

# %% save without R
fig = plt.figure(figsize=(18, 9))
for i in range(1, n + 1):
    if i == n - 1:
        continue
    ax = fig.add_subplot(2, 4, i) if i < n else fig.add_subplot(2, 4, i - 1) # int(np.ceil(n / 4))
    ax.plot(t, data[:, i - 1], '.k', label='Covasim Data')
    ax.plot(t, u_sim_NN[:, i - 1], 'r.-', label='ODE-NN')
    ax.plot(t, u_sim_regression[:, i - 1], 'b.-', label='ODE-SR')
    ax.plot(t, ode_de_array[:, i - 1], 'g.-', label='ODE-DE')
    if i > 4:
        ax.set_xlabel("Time (Days)")
    if i % 4 == 1:
        ax.set_ylabel("Count")
    ax.set_title(col_names[i - 1])
    if i == 1:
        ax.legend(loc="best")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad=2)
    plt.savefig(model.save_folder + case_name + '_without_R' + '.png', dpi=300)

# %% optional, calcualte the AIC values for ODE-Regression

aic_regression = AIC_OLS(u_sim_regression, data, 6)
aic_de = AIC_OLS(ode_de_array, data, 2)

# RSS for each variable
RSS_regression = RSS(u_sim_regression, data)
RSS_de = RSS(ode_de_array, data)

RSS_df = pd.DataFrame(data={'RSS_sparse': RSS_regression, 'RSS_de': RSS_de}, index=list('STEAYDQRF'))

RSS_df.to_csv('RSS_results.csv')

# %% optional, save compartment values
pd.DataFrame(data=data, columns=list('STEAYDQRF')).to_csv('Covasim-Data.csv')
pd.DataFrame(data=u_sim_regression, columns=list('STEAYDQRF')).to_csv('ODE-regression.csv')
pd.DataFrame(data=u_sim_NN, columns=list('STEAYDQRF')).to_csv('ODE-NN.csv')
pd.DataFrame(data=ode_de_array, columns=list('STEAYDQRF')).to_csv('ODE-DE.csv')

# %% optional, get the peak time
idx = 4
data[:, idx].argmax(), u_sim_regression[:,idx].argmax(), u_sim_NN[:, idx].argmax(), ode_de_array[:, idx].argmax()
