import torch
import torch
import torch.nn as nn
import sklearn.linear_model
from torch.nn.functional import normalize
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

@torch.no_grad()
def LR(encoder, support, support_ys, query, y_query, norm=False):
    """logistic regression classifier"""
    support = encoder(support).detach()
    query = encoder(query).detach()
    
    if norm:
        support = normalize(support)
        query = normalize(query)

    clf = sklearn.linear_model.LogisticRegression(random_state=0,
                                                  solver='lbfgs',
                                                  max_iter=1000,
                                                  C=1,
                                                  multi_class='multinomial')
    support_features_np = support.data.cpu().numpy()
    support_ys_np = support_ys.data.cpu().numpy()
    clf.fit(support_features_np, support_ys_np)

    query_features_np = query.data.cpu().numpy()
    query_ys_pred = clf.predict(query_features_np)

    pred = torch.from_numpy(query_ys_pred).to(support.device,
                                              non_blocking=True)


    # Visualize the features of the support set and the query set using t-SNE
    support_tsne = TSNE(n_components=2).fit_transform(support_features_np)
    query_tsne = TSNE(n_components=2).fit_transform(query_features_np)

    # Plot the t-SNE embeddings and save the figures to "./figs" directory
    if not os.path.exists("./figs"):
        os.mkdir("./figs")

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)

    plt.scatter(support_tsne[:,0], support_tsne[:,1], c=support_ys_np, cmap='Set1')
    plt.title("Support Set t-SNE Embedding")
    # plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    
    plt.subplot(1,2,2)
    incorrect_class_idx = torch.nonzero(pred != y_query).cpu()
    correct_class_idx = torch.nonzero(pred == y_query).cpu()

    # Scatter plot of query set with incorrect and correct classifications marked
    query_labels = y_query.data.cpu().numpy()
    query_labels_pred = query_ys_pred.copy()
    labels = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    # colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    
    # vmin = min(query_labels.min(), query_labels[incorrect_class_idx].min(), query_labels[correct_class_idx].min())
    # vmax = max(query_labels.max(), query_labels[incorrect_class_idx].max(), query_labels[correct_class_idx].max())

    # plt.scatter(query_tsne[incorrect_class_idx,0], query_tsne[incorrect_class_idx,1], c=query_labels[incorrect_class_idx], cmap='viridis', marker='x', vmin=vmin, vmax=vmax)
    # plt.scatter(query_tsne[correct_class_idx,0], query_tsne[correct_class_idx,1], c=query_labels[correct_class_idx], cmap='viridis', vmin=vmin, vmax=vmax)
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    vmin = min(query_labels.min(), query_labels[incorrect_class_idx].min(), query_labels[correct_class_idx].min())
    vmax = max(query_labels.max(), query_labels[incorrect_class_idx].max(), query_labels[correct_class_idx].max())

    # Map predicted labels to colors
    edge_colors = colors[query_labels_pred[incorrect_class_idx].astype(int)]
    edge_colors = np.squeeze(edge_colors)

    plt.scatter(query_tsne[incorrect_class_idx,0], 
                query_tsne[incorrect_class_idx,1], 
                c=query_labels[incorrect_class_idx],
                edgecolors=edge_colors, 
                linewidths=1.2, cmap='viridis', marker='D', 
                s=12,
                vmin=vmin, vmax=vmax 
                )
    plt.scatter(query_tsne[correct_class_idx,0], query_tsne[correct_class_idx,1], c=query_labels[correct_class_idx], s=12,cmap='viridis', vmin=vmin, vmax=vmax)

    
    scatters = [plt.scatter([], [], color=colors[i]) for i in range(len(labels))]
    plt.legend(scatters, labels, loc='lower right')

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("Query Set t-SNE Embedding")
    # plt.colorbar()


    plt.savefig("./figs/tsne_CropDisease_best_large.png")
    assert(1==0)
    return pred, query


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x


def set_bn_to_eval(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
