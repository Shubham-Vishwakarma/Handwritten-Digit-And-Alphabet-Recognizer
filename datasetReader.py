from os import listdir

train_data_path = 'dataset/training/'
test_data_path = 'dataset/test/'


def get_train_data_and_labels():
	arr = sorted(listdir(train_data_path))
	data = []
	labels = []
	for foldername in arr:
		for filename in listdir(train_data_path + foldername):
			data.append(train_data_path + foldername + '/' + filename)
			labels.append(ord(foldername)) # labels is the ASCII value of foldername

	return data, labels

# def get_test_data():
# 	data = []
# 	for filename in listdir(test_data_path):
# 		data.append(test_data_path + '/' + filename)

# 	return data