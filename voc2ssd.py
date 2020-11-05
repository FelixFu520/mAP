import os
import random 
 
xmlfilepath = r'./VOCdevkit/VOC2012/Annotations'
saveBasePath = r"./VOCdevkit/VOC2012/ImageSets/Main/"
 
train_percent = 1.0
trainval_percent = 0.9

temp_xml = os.listdir(xmlfilepath)  # len 17125
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num = len(total_xml)    # 17125
list = range(num)
tv = int(num*trainval_percent)  # 15412 训练集和验证集
tr = int(tv*train_percent)  # 15412 训练集
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
 
print("train and val size", tv)
print("train size", tr)
ftrainval = open(os.path.join(saveBasePath, 'mAP_trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'mAP_test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'mAP_train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'mAP_val.txt'), 'w')
 
for i in list:
    name = total_xml[i][:-4]+'\n'
    if i in trainval:  
        ftrainval.write(name)  # 放入训练集和验证集
        if i in train:  
            ftrain.write(name)  # 放入训练集
        else:  
            fval.write(name)  # 放入验证集
    else:  
        ftest.write(name)  # 放入测试集
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
