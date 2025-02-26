import matplotlib.pyplot as plt
import numpy as np

from Modules.Utils.Imports import *
from Modules.Models.BuildBINNs import BINNCovasim
from Modules.Utils.ModelWrapper import ModelWrapper

import Modules.Utils.PDESolver as PDESolver
import Modules.Loaders.DataFormatter as DF
from utils import get_case_name, lasso_parameter_fitting

from sklearn import linear_model
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
trace_prob = 0.3
keep_d = True
retrain = False
dynamic = True
case_name = get_case_name(population, test_prob, trace_prob, keep_d, dynamic=dynamic)

params = DF.load_covasim_data(path, population, test_prob, trace_prob, keep_d, plot=False, dynamic=dynamic)
data = params['data']
data = (data / params['population']).to_numpy()
N = len(data)
t_max = N - 1
t = np.arange(N)
params.pop('data')

binn = BINNCovasim(params, t_max, keep_d=keep_d).to(device)
parameters = binn.parameters()
model = ModelWrapper(binn, None, None, save_name='')

# load model weights
model.save_name = '../Weights/'
model.save_name += case_name
if retrain:
    model.save_name += '_retrain'
model.save_name += '_best_val'
model.load(model.save_name + '_model', device=device)
save_path = model.save_folder
# grab initial condition
u0 = data[0, :].copy()

# learned contact_rate function
def contact_rate(u):
    res = binn.contact_rate(to_torch(u)) # [:,[0,3,4]]
    return to_numpy(res)

def quarantine_test(u):
    res = binn.quarantine_test_prob(to_torch(u))
    return to_numpy(res)

s_min, s_max = data[:,0].min(), data[:,0].max()
a_min, a_max = 0.0, 0.05 # data[:,3].min(), data[:,3].max()
y_min, y_max = 0.0, 0.05 # data[:,4].min(), data[:,4].max()
chi_min, chi_max = 0.0, params['eff_ub']

s_grid = np.arange(s_min, s_max, 0.05)
a_grid = np.arange(a_min, a_max, 0.01)
y_grid = np.arange(y_min, y_max, 0.01)
chi_grid = np.arange(chi_min, chi_max, 0.1)

# 2d visualization for contact rate
fig = plt.figure(figsize=(10,7))
for i in range(3):
    if i == 0:
        X, Y = np.meshgrid(s_grid, a_grid)
        Z = np.ones_like(X) * 0.0
        x_label, y_label = 'S', 'A'
    elif i == 1:
        X, Z = np.meshgrid(s_grid, y_grid)
        Y = np.ones_like(X) * 0.0
        x_label, y_label = 'S', 'Y'
    else:
        Y, Z = np.meshgrid(a_grid, y_grid)
        X = np.ones_like(Y) * 0.5
        x_label, y_label = 'A', 'Y'
    u_grid = np.stack([np.ravel(X), np.ravel(Y), np.ravel(Z)], axis=1)
    res = contact_rate(u_grid)
    res = res[:,0].reshape(X.shape)
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    if i == 0:
        ax.plot_surface(X, Y, res, cmap=cm.coolwarm, alpha=1)
        ax.scatter(X.reshape(-1), Y.reshape(-1), res.reshape(-1), s=5, c='k')
    elif i == 1:
        ax.plot_surface(X, Z, res, cmap=cm.coolwarm, alpha=1)
        ax.scatter(X.reshape(-1), Z.reshape(-1), res.reshape(-1), s=5, c='k')
    else:
        ax.plot_surface(Y, Z, res, cmap=cm.coolwarm, alpha=1)
        ax.scatter(Y.reshape(-1), Z.reshape(-1), res.reshape(-1), s=5, c='k')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.tight_layout(pad=2)
plt.savefig(save_path + case_name + '_parameter_NN_cr' + '.png')
plt.close()

# 2d visualization for quarantine test prob
fig = plt.figure(figsize=(10,7))
X, Y = np.meshgrid(a_grid, y_grid)
x_label, y_label = 'A', 'Y'
u_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
res = quarantine_test(u_grid)
res = res[:,0].reshape(X.shape)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, res, cmap=cm.coolwarm, alpha=1)
ax.scatter(X.reshape(-1), Y.reshape(-1), res.reshape(-1), s=5, c='k')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.tight_layout(pad=2)
plt.savefig(save_path + case_name + '_parameter_NN_qt' + '.png')
plt.close()
# lasso regression to learn symbolic terms from parameter NN
def get_samples_ct(u):
    s, a, y = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None]
    candidates = [s, s**2] # s related terms
    candidates += [a]
    candidates += [y]
    candidates = np.concatenate(candidates, axis=1)
    return candidates

def get_samples_qt(u):
    a, y = u[:, 0][:, None], u[:, 1][:, None]
    candidates = [a, y]
    candidates = np.concatenate(candidates, axis=1)
    return candidates

train_x = np.array(np.meshgrid(s_grid, a_grid, y_grid)).T.reshape(-1,3)
data_x = get_samples_ct(train_x)
data_y = contact_rate(train_x)
data_y = data_y[:,0][:, None]
lasso_parameter_fitting(data_x, data_y, 'cr', save_path, case_name, False)

train_x = np.array(np.meshgrid(a_grid, y_grid)).T.reshape(-1,2)
data_x = get_samples_qt(train_x)
data_y = quarantine_test(train_x)
data_y = data_y[:,0][:, None]
lasso_parameter_fitting(data_x, data_y, 'qt', save_path, case_name, False)

# from sklearn.linear_model import LassoCV
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# start_time = time.time()
# model = make_pipeline( LassoCV(alphas=np.logspace(-6, -2), cv=5, max_iter=10000, fit_intercept=False)).fit(x_train, y_train)
# fit_time = time.time() - start_time
# ymin, ymax = 2300, 3800
# lasso = model[-1]
# plt.figure()
# plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
# plt.plot(
#     lasso.alphas_,
#     lasso.mse_path_.mean(axis=-1),
#     color="black",
#     label="Average across the folds",
#     linewidth=2,
# )
# plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")
#
# # plt.ylim(ymin, ymax)
# plt.xlabel(r"$\alpha$")
# plt.ylabel("Mean square error")
# plt.legend()
# _ = plt.title(
#     f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
# )
# plt.savefig(save_path + case_name + '_lassoCV' + '.png')
# plt.close()
#
# from sklearn.linear_model import lasso_path
# from itertools import cycle
# # Compute paths
#
# eps = 5e-3  # the smaller it is the longer is the path
#
# print("Computing regularization path using the lasso...")
# alphas_lasso, coefs_lasso, _ = lasso_path(x_train, y_train.reshape(-1), alphas=np.logspace(-6, -2),max_iter=10000, eps=eps)
# plt.figure()
# colors = cycle(["b", "r", "g", "c"])
# names = ['S', r'$S^2$', 'A', 'Y']
# neg_log_alphas_lasso = alphas_lasso # -np.log10(alphas_lasso)
# for coef_l, c, name in zip(coefs_lasso, colors, names):
#     l1 = plt.semilogx(neg_log_alphas_lasso, coef_l, c=c, label=name)
# plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")
# plt.xlabel(r"$\alpha$")
# plt.ylabel("coefficients")
# plt.title("Lasso Paths")
# plt.legend(loc="lower left")
# plt.axis("tight")
# plt.savefig(save_path + case_name + '_lassoPath' + '.png')
# plt.close()
#
# final_lasso = linear_model.Lasso(alpha=lasso.alpha_, max_iter=10000, fit_intercept=False)
# final_lasso.fit(x_train, y_train)
# final_lasso.coef_
# pd.DataFrame(final_lasso.coef_).to_csv(save_path + case_name + '_regression_coef.csv')

# fig = plt.figure(figsize=(10,7))
# for i in range(3):
#     if i == 0:
#         X, Y = np.meshgrid(s_grid, a_grid)
#         Z = np.ones_like(X) * 0.0
#         x_label, y_label = 'S', 'A'
#     elif i == 1:
#         X, Z = np.meshgrid(s_grid, y_grid)
#         Y = np.ones_like(X) * 0.0
#         x_label, y_label = 'S', 'Y'
#     else:
#         Y, Z = np.meshgrid(a_grid, y_grid)
#         X = np.ones_like(Y) * 0.5
#         x_label, y_label = 'A', 'Y'
#     u_grid = np.stack([np.ravel(X), np.ravel(Y), np.ravel(Z)], axis=1)
#     res_nn = contact_rate(u_grid)
#     u_grid_lasso = np.stack([np.ravel(X), np.ravel(X)**2, np.ravel(Y), np.ravel(Z)], axis=1)
#     res_lasso = final_lasso.predict(u_grid_lasso)
#     res_lasso = res_lasso.reshape(X.shape)
#     res_nn = res_nn[:,0].reshape(X.shape)
#     ax = fig.add_subplot(1, 3, i + 1, projection='3d')
#     if i == 0:
#         ax.plot_surface(X, Y, res_nn, cmap=cm.coolwarm, alpha=1)
#         ax.scatter(X.reshape(-1), Y.reshape(-1), res_nn.reshape(-1), s=5, c='k')
#         ax.plot_surface(X, Y, res_lasso, cmap=cm.coolwarm, alpha=1)
#         ax.scatter(X.reshape(-1), Y.reshape(-1), res_lasso.reshape(-1), s=5, c='r')
#     elif i == 1:
#         ax.plot_surface(X, Z, res_nn, cmap=cm.coolwarm, alpha=1)
#         ax.scatter(X.reshape(-1), Z.reshape(-1), res_nn.reshape(-1), s=5, c='k')
#         ax.scatter(X.reshape(-1), Z.reshape(-1), res_lasso.reshape(-1), s=5, c='r')
#     else:
#         ax.plot_surface(Y, Z, res_nn, cmap=cm.coolwarm, alpha=1)
#         ax.scatter(Y.reshape(-1), Z.reshape(-1), res_nn.reshape(-1), s=5, c='k')
#         ax.scatter(Y.reshape(-1), Z.reshape(-1), res_lasso.reshape(-1), s=5, c='r')
#
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# plt.tight_layout(pad=2)

