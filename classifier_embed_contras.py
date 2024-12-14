import torch
import torch.nn as nn
import torch.optim as optim
import util
import torch.nn.functional as F

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, netClassifier, embed_size, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True):
        # ---------------------------------------
        super(CLASSIFIER, self).__init__()
        # ---------------------------------------
        self.train_X =  _train_X
        self.train_Y = _train_Y 
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        # self.MapNet=map_net
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = embed_size
        self.cuda = _cuda
        # self.model =  LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        # self.model.apply(util.weights_init)
        self.model = netClassifier
        self.criterion = nn.NLLLoss()
        
        self.input = torch.FloatTensor(_batch_size, _train_X.size(1))
        self.label = torch.LongTensor(_batch_size) 
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()
    
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)

                batch_label[batch_label == self.unseenclasses[0]] = 0
                batch_label[batch_label == self.unseenclasses[1]] = 1

                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                # embed, _ = self.MapNet(self.input)
                # output = self.model(embed)
                output = self.model(self.input)
                # output = output.clone().detach().requires_grad_(True)
                loss = self.criterion(output, self.label)
                mean_loss += loss.data
                loss.backward()
                self.optimizer.step()
            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > best_acc:
                best_acc = acc
        print('Training classifier loss= %.4f' % (loss))
        return best_acc 

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                # embed, _ = self.MapNet(self.input)
                # output = self.model(embed)
                output = self.model(self.input)
                # output = output.clone().detach().requires_grad_(True)
                loss = self.criterion(output, self.label)

                loss.backward()
                self.optimizer.step()
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if (acc_seen+acc_unseen)==0:
                print('a bug')
                H=0
            else:
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen, best_unseen, best_H
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    # embed, _ = self.MapNet(test_X[start:end].cuda())
                    # output = self.model(embed)
                    output = self.model(test_X[start:end].cuda())
                else:
                    # embed, _ = self.MapNet(test_X[start:end])
                    # output = self.model(embed)
                    output = self.model(test_X[start:end])
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = torch.FloatTensor(target_classes.size(0)).fill_(0)
        k = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class[k] = float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
            k = k + 1
        return acc_per_class.mean()

    def compute_per_class_acc_gzsl_temp(self, test_label, predicted_label, target_classes):
        acc_per_class = torch.FloatTensor(target_classes.size(0)).fill_(0)
        k = 0

        for i in target_classes:
            idx = (test_label == i)
            acc_per_class[k] = float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
            k = k + 1
        return acc_per_class.mean()

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        output_temp = []
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    # embed, _ = self.MapNet(test_X[start:end].cuda())
                    # output = self.model(embed)
                    output = self.model(test_X[start:end].cuda())
                    output_temp.append(output)
                else:
                    # embed, _ = self.MapNet(test_X[start:end])
                    # output = self.model(embed)
                    output = self.model(test_X[start:end])
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        # acc, acc_per = self.compute_per_class_acc(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = float(torch.sum(test_label[idx]==predicted_label[idx])) / float(torch.sum(idx))
        return acc_per_class.mean()
    # def compute_per_class_acc(self, test_label, predicted_label, nclass):
    #     acc_per_class = torch.FloatTensor(nclass.size(0)).fill_(0)
    #     k = 0
    #     for i in range(nclass.size(0)):
    #         idx = (test_label == i)
    #         acc_per_class[k] = float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    #         k = k + 1
    #     return acc_per_class.mean(), acc_per_class

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass, opt):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

        self.embedding_net = Embedding_Net(opt)

    def forward(self, x):
        x, _ = self.embedding_net(x)
        x = self.fc(x)
        o = self.logic(x)
        return o

    def get_embedding(self, features):
        return self.embedding_net(features)

class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding = self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding, out_z

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
