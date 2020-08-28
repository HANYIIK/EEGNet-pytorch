"""
4 classes
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import model
import utils
import visualization

# Hyper Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISUALIZATION = False
CATEGORY = 'fruits'           # or animals
CLASSES_NUM = 4
EPOCH = 10
BATCH_SIZE = 20
LR = 0.001


# get model
eeg_net = model.EEGNet(classes_num=CLASSES_NUM).to(DEVICE)

optimizer = torch.optim.Adam(eeg_net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss().to(DEVICE)

# get dataset
if CATEGORY == 'fruits':
    # fruits
    train_x, test_x = utils.get_fruits(dataset=True)
elif CATEGORY == 'animals':
    # animals
    train_x, test_x = utils.get_animals(dataset=True)
else:
    print('error!')
    exit()


train_y, test_y = utils.get_4_classes_labels()

dataset = Data.TensorDataset(train_x, train_y)
dataloader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

plt.ion()

# start train
print('use {}'.format("cuda" if torch.cuda.is_available() else "cpu"))

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(dataloader):
        b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
        output = eeg_net(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            test_out, last_layer = eeg_net(test_x.to(DEVICE))
            pred_y = torch.max(test_out.cpu(), 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.3f' % accuracy)
            if VISUALIZATION:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                visualization.plot_with_labels(low_dim_embs, labels)

plt.ioff()