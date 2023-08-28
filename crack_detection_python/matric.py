import numpy as np
import torch 
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix) 
        IoU = intersection / union
        return IoU
    def recall(self):
        # recall = TP / (TP + FN)
        recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return recall
    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        #batch = imgPredict.shape[0]
        #for i in range(batch):
        #    imgPredict_rw,imgLabel_rw = imgPredict[i],imgLabel[i]
        #    self.confusionMatrix += self.genConfusionMatrix(imgPredict_rw, imgLabel_rw)
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def Evaluation(model,dataloader,device):
    metric = SegmentationMetric(2)
    ds_acc = 0.
    with torch.no_grad():
        for step,(features,labels,_) in enumerate(dataloader):
            # features,labels = features.unsqueeze(0).to(device),labels.to(device)
            features,labels = features.to(device),labels.to(device)
            out = model(features)

            out = torch.argmax(torch.softmax(out,dim=1),dim=1).squeeze()
            # labels = torch.argmax(torch.softmax(labels,dim=1),dim=1)
            pred, y = out.cpu().detach().numpy(),labels.cpu().detach().numpy()
            pred, y = pred.astype(np.int32), y.astype(np.int32) 
            _  = metric.addBatch(pred,y)
    
    pa = metric.classPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    recall = metric.recall()
    return pa,recall,IoU,mIoU


# # import segmentation_models_pytorch as smp
# from model.get_model import get_model
# import argparse
# from crack_dataset import crack_loader
# from torch.utils.data import DataLoader

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--val_set",default=r'/opt/data/private/code/scheme_set/val.txt')
#     args = parser.parse_args()
#     model = get_model(name= 'DDRNet' ,num_classes =2)
#     checkpoint = r'/opt/data/private/code/savedddr/generate_final.pkl'
#     model.load_state_dict(torch.load(checkpoint), strict = False)
#     model.eval()
#     device = torch.device("cuda")
#     model.to(device)
#     dl_val = DataLoader(crack_loader(args.val_set,256),
#                     batch_size=32,
#                     shuffle=False,
#                     num_workers=32,
#                     pin_memory=True,
#                     drop_last=True)
    
#     mpa,recall,IoU,mIoU = Evaluation(model=model,dataloader=dl_val,device= "cuda",aux=1)
#     print("MPA:{:.5f} , Recall:{} , IoU:{} , mIoU:{:.5f} ".format(mpa,recall,IoU[1],mIoU))
