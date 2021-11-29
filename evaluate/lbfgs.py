import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def encode_train_set(clftrainloader, device, net):
    net.eval()

    store = []
    with torch.no_grad():
        # t = tqdm(enumerate(clftrainloader), desc='Encoded: **/** ', total=len(clftrainloader),
        #          bar_format='{desc}{bar}{r_bar}')
        t = enumerate(clftrainloader)
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs)
            store.append((representation, targets))

            # t.set_description('Encoded %d/%d' % (batch_idx, len(clftrainloader)))

    X, y = zip(*store)
    X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)
    return X, y


def train_clf(X, y, representation_dim, num_classes, device, writer,epoch, reg_weight=1e-3):
    print('\nL2 Regularization weight: %g' % reg_weight)

    criterion = nn.CrossEntropyLoss()
    n_lbfgs_steps = 500

    # Should be reset after each epoch for a completely independent evaluation
    clf = nn.Linear(representation_dim, num_classes).to(device)
    clf_optimizer = optim.LBFGS(clf.parameters())
    clf.train()
    n_iter = 0

    # t = tqdm(range(n_lbfgs_steps), desc='Loss: **** | Train Acc: ****% ', bar_format='{desc}{bar}{r_bar}')
    t = range(n_lbfgs_steps)
    for _ in t:

        def closure():
            clf_optimizer.zero_grad()
            raw_scores = clf(X)
            loss = criterion(raw_scores, y)
            loss += reg_weight * clf.weight.pow(2).sum()
            loss.backward()

            _, predicted = raw_scores.max(1)
            correct = predicted.eq(y).sum().item()
            accuracy = 100. * correct / y.shape[0]
            # t.set_description('Loss: %.3f | Train Acc: %.3f%% ' % (loss, accuracy))

            writer.add_scalar(f'At Epoch{epoch} Classifier Train Loss', loss, global_step=n_iter)
            writer.add_scalar(f'At Epoch{epoch} Classifier Train Accuracy', accuracy, global_step=n_iter)
            return loss

        clf_optimizer.step(closure)
        n_iter += 1

    return clf


def test(testloader, device, net, clf, writer,epoch):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    clf.eval()
    test_clf_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # t = tqdm(enumerate(testloader), total=len(testloader), desc='Loss: **** | Test Acc: ****% ',
        #          bar_format='{desc}{bar}{r_bar}')
        t = enumerate(testloader)
        n_iter = 0
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs)
            # test_repr_loss = criterion(representation, targets)
            raw_scores = clf(representation)
            clf_loss = criterion(raw_scores, targets)

            test_clf_loss += clf_loss.item()
            _, predicted = raw_scores.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100. * correct / total
            # t.set_description(
            #     'Loss: %.3f | Test Acc: %.3f%% ' % (test_clf_loss / (batch_idx + 1), accuracy))
            writer.add_scalar(f'At Epoch{epoch} Classifier Test Loss', clf_loss, global_step=n_iter)
            writer.add_scalar(f'At Epoch{epoch} Classifier Test Accuracy', accuracy, global_step=n_iter)

            n_iter += 1

    acc = 100. * correct / total
    return acc
