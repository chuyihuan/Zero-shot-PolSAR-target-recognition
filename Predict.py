import torch
import util
from classifier_embed_contras import LINEAR_LOGSOFTMAX
from config import opt

# 1. Initialize model
def initialize_model(input_dim, nclass, model_path):
    classifier_model = LINEAR_LOGSOFTMAX(input_dim, nclass, opt)
    classifier_model.load_state_dict(torch.load(model_path))
    classifier_model.eval()
    return classifier_model

data = util.DATA_LOADER(opt)
input_dim = opt.embedSize

# 2. Load model
best_model_path = './models/' + opt.dataset + '/best_ZSLmodel_classifier_tensor(0.9250).pth'
nclass = data.unseenclasses.size(0)
# 3. Load data
test_X = data.test_unseen_feature.cuda()
test_label = data.test_unseen_label.cuda()
unseenclasses = data.unseenclasses.cuda()
mapping = {unseenclasses[0].item(): 0, unseenclasses[1].item(): 1}
for i in range(len(test_label)):
    test_label[i] = mapping.get(test_label[i].item(), test_label[i])

classifier_model = initialize_model(input_dim, nclass, best_model_path)
classifier_model.cuda()

# 3. Inference
predicted_labels_seen = []
predicted_labels_unseen = []
predicted_labels = []
batch_size = opt.batch_size

with torch.no_grad():
    for start in range(0, len(test_X), batch_size):
        end = min(start + batch_size, len(test_X))
        batch_input = test_X[start:end]

        output = classifier_model(batch_input)
        _, predicted = torch.max(output, 1)
        predicted_labels.append(predicted)
predicted_labels = torch.cat(predicted_labels)


def compute_accuracy(test_labels, predicted_labels, nclass):
    acc_per_class = torch.FloatTensor(nclass.shape[0]).fill_(0)
    k = 0
    for i in nclass:
        idx = (test_labels == i)
        acc_per_class[k] = float(torch.sum(test_labels[idx] == predicted_labels[idx])) / float(torch.sum(idx))
        k = k+1
    mean_accuracy = torch.sum(test_labels == predicted_labels).float() / len(test_labels)
    mean_accuracy = torch.tensor(mean_accuracy.item())
    return acc_per_class, mean_accuracy

nclass = torch.unique(test_label)
acc_per_class, mean_accuracy = compute_accuracy(test_label, predicted_labels, nclass)

print('unseen acc:', mean_accuracy)


