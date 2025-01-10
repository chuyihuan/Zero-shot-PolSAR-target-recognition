import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import os
from logger import create_logger
import datetime


def initialize_exp(path, name):
	# """
	# Experiment initialization.
	# """
	# # dump parameters
	# params.dump_path = get_dump_path(params)
	# pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

	# create a logger
	time_stamp = datetime.datetime.now()

	time = time_stamp.strftime('%Y%m%d%H%M%S')

	logger = create_logger(os.path.join(path, name + '_' + time + '.log'))
	print('log_name:',name + '_' + time + '.log')
	# logger = create_logger(os.path.join(path, name +'.log'))
	logger.info('============ Initialized logger ============')
	# logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
	#                       in sorted(dict(vars(params)).items())))
	return logger


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


def map_label(label, classes):
	mapped_label = torch.LongTensor(label.size())
	for i in range(classes.size(0)):
		mapped_label[label == classes[i]] = i

	return mapped_label


class Logger(object):
	def __init__(self, filename):
		self.filename = filename
		f = open(self.filename + '.log', "a")
		f.close()

	def write(self, message):
		f = open(self.filename + '.log', "a")
		f.write(message)
		f.close()


class DATA_LOADER(object):
	def __init__(self, opt):
		if opt.matdataset:
			self.read_matdataset(opt)
		self.index_in_epoch = 0
		self.epochs_completed = 0

	def read_matdataset(self, opt):
		matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
		feature = matcontent['features']
		# self.all_file = matcontent['image_files']
		label = matcontent['labels'].astype(int).squeeze() - 1

		matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.dataset_loc + ".mat")
		trainval_loc = matcontent['train_loc'].squeeze() - 1
		# train_loc = matcontent['train_loc'].squeeze() - 1
		# val_unseen_loc = matcontent['val_loc'].squeeze() - 1
		test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
		test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
		val_seen_loc = matcontent['val_seen_loc'].squeeze() - 1
		val_unseen_loc = matcontent['val_unseen_loc'].squeeze() - 1

		matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_c.mat")
		self.attribute_c = torch.from_numpy(matcontent[opt.class_embedding + "_c"].T).float()

		matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_a.mat")
		self.attribute_a = torch.from_numpy(matcontent[opt.class_embedding + "_a"].T).float()

		matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_s.mat")
		self.attribute_s = torch.from_numpy(matcontent[opt.class_embedding + "_s"].T).float()

		self.attribute = torch.cat((self.attribute_c, self.attribute_a, self.attribute_s), dim=1)
		# self.attribute = self.attribute_c
		if not opt.validation:
			# self.train_image_file = self.all_file[trainval_loc]
			# self.test_seen_image_file = self.all_file[test_seen_loc]
			# self.test_unseen_image_file = self.all_file[test_unseen_loc]

			if opt.preprocessing:
				if opt.standardization:
					print('standardization...')
					scaler = preprocessing.StandardScaler()
				else:
					scaler = preprocessing.MinMaxScaler()

				_train_feature = scaler.fit_transform(feature[trainval_loc])
				_test_seen_feature = scaler.transform(feature[test_seen_loc])
				_test_unseen_feature = scaler.transform(feature[test_unseen_loc])
				self.train_feature = torch.from_numpy(_train_feature).float()
				mx = self.train_feature.max()
				self.train_feature.mul_(1 / mx)
				self.train_label = torch.from_numpy(label[trainval_loc]).long()
				self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
				self.test_unseen_feature.mul_(1 / mx)
				self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
				self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
				self.test_seen_feature.mul_(1 / mx)
				self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
			else:
				self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
				self.train_label = torch.from_numpy(label[trainval_loc]).long()
				self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
				self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
				self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
				self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
		else:
			if opt.preprocessing:
				if opt.standardization:
					print('standardization...')
					scaler = preprocessing.StandardScaler()
				else:
					scaler = preprocessing.MinMaxScaler()

				_train_feature = scaler.fit_transform(feature[trainval_loc])
				_test_seen_feature = scaler.transform(feature[val_seen_loc])
				_test_unseen_feature = scaler.transform(feature[val_unseen_loc])
				self.train_feature = torch.from_numpy(_train_feature).float()
				mx = self.train_feature.max()
				self.train_feature.mul_(1 / mx)
				self.train_label = torch.from_numpy(label[trainval_loc]).long()
				self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
				self.test_unseen_feature.mul_(1 / mx)
				self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
				self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
				self.test_seen_feature.mul_(1 / mx)
				self.test_seen_label = torch.from_numpy(label[val_seen_loc]).long()
			else:
				self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
				self.train_label = torch.from_numpy(label[trainval_loc]).long()
				self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
				self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
				self.test_seen_feature = torch.from_numpy(feature[val_seen_loc]).float()
				self.test_seen_label = torch.from_numpy(label[val_seen_loc]).long()

		self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
		self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
		self.ntrain = self.train_feature.size()[0]
		self.ntrain_class = self.seenclasses.size(0)
		self.ntest_class = self.unseenclasses.size(0)
		self.train_class = self.seenclasses.clone()
		self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
		self.attribute_seen = self.attribute[self.seenclasses]

		# collect the data of each class

		self.train_samples_class_index = torch.tensor([self.train_label.eq(i_class).sum().float() for i_class in self.train_class])
		#
		# import pdb
		# pdb.set_trace()

		# self.train_mapped_label = map_label(self.train_label, self.seenclasses)

	def next_batch_one_class(self, batch_size):
		if self.index_in_epoch == self.ntrain_class:
			self.index_in_epoch = 0
			perm = torch.randperm(self.ntrain_class)
			self.train_class[perm] = self.train_class[perm]

		iclass = self.train_class[self.index_in_epoch]
		idx = self.train_label.eq(iclass).nonzero().squeeze()
		perm = torch.randperm(idx.size(0))
		idx = idx[perm]
		iclass_feature = self.train_feature[idx]
		iclass_label = self.train_label[idx]
		self.index_in_epoch += 1
		return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

	def next_batch(self, batch_size):
		idx = torch.randperm(self.ntrain)[0:batch_size]
		batch_feature = self.train_feature[idx]
		batch_label = self.train_label[idx]
		batch_att = self.attribute[batch_label]
		return batch_feature, batch_label, batch_att

	# select batch samples by randomly drawing batch_size classes
	def next_batch_uniform_class(self, batch_size):
		batch_class = torch.LongTensor(batch_size)
		for i in range(batch_size):
			idx = torch.randperm(self.ntrain_class)[0]
			batch_class[i] = self.train_class[idx]

		batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
		batch_label = torch.LongTensor(batch_size)
		batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
		for i in range(batch_size):
			iclass = batch_class[i]
			idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
			idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
			idx_file = idx_iclass[idx_in_iclass]
			batch_feature[i] = self.train_feature[idx_file]
			batch_label[i] = self.train_label[idx_file]
			batch_att[i] = self.attribute[batch_label[i]]
		return batch_feature, batch_label, batch_att
