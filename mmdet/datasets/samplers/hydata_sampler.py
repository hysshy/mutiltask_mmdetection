from __future__ import division
import math
import json
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
import random
from collections import defaultdict
import copy

class Hydata_DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

        ################################################################################################################
        self.detectproPosalIds = []
        self.facezitaiProposalIds = []
        self.facemohuProposalIds = []
        self.facekpProposalIds = []
        self.bodyDetectProposalIds = []
        self.clouseStyleProposalIds = []
        self.clouseColorProposalIds = []
        self.faceGenderProposalIds =[]
        self.carplateDetectProposalIds = []
        self.faceDetectProposalIds = []
        self.idList = []
        if hasattr(self.dataset, 'dataset'):
            ann_file = self.dataset.dataset.ann_file
        else:
            ann_file = self.dataset.ann_file
        with open(ann_file, 'rb') as f:
            data = f.read()
            data = json.loads(data.decode())
            annotations = data["annotations"]
            images = data["images"]
            for ann in annotations:
                image_id = ann["image_id"] - 1
                category_id = ann["category_id"] -1
                if image_id not in self.idList:
                    self.idList.append(image_id)

        for k in self.idList:
            filename = images[k]["file_name"]
            # print(filename)
            if "faceMohuImgs" == filename.split('/')[-2]:
                self.facemohuProposalIds.append(k)
            elif "faceZitaiImgs" == filename.split('/')[-2]:
                self.facezitaiProposalIds.append(k)
            elif 'detectImgs' == filename.split('/')[-2]:
                self.detectproPosalIds.append(k)
            elif 'faceKpImgs' == filename.split('/')[-2]:
                self.facekpProposalIds.append(k)
            elif 'bodyDetectImgs' == filename.split('/')[-2]:
                self.bodyDetectProposalIds.append(k)
            elif 'clouseStyleImgs' == filename.split('/')[-2]:
                self.clouseStyleProposalIds.append(k)
            elif 'clouseColorImgs' == filename.split('/')[-2]:
                self.clouseColorProposalIds.append(k)
            elif 'faceGenderImgs' == filename.split('/')[-2]:
                self.faceGenderProposalIds.append(k)
            elif 'carplateDetectImgs' == filename.split('/')[-2]:
                self.carplateDetectProposalIds.append(k)
            elif 'faceDetectImgs' == filename.split('/')[-2]:
                self.faceDetectProposalIds.append(k)
            else:
                print(filename.split('/')[-2])
                assert False

        print(len(self.detectproPosalIds))
        print(len(self.facezitaiProposalIds))
        print(len(self.facemohuProposalIds))
        print(len(self.facekpProposalIds))
        print(len(self.bodyDetectProposalIds))
        print(len(self.clouseStyleProposalIds))
        print(len(self.clouseColorProposalIds))
        print(len(self.faceGenderProposalIds))
        print(len(self.carplateDetectProposalIds))
        print(len(self.faceDetectProposalIds))

    def getIndices(self, proPosalIds, g):
        #查缺补漏
        size = len(proPosalIds)
        indices = []
        indice = np.array(proPosalIds)
        indice = indice[list(torch.randperm(int(size),
                                            generator=g))].tolist()
        extra = int(
            math.ceil(
                size * 1.0 / self.samples_per_gpu / self.num_replicas)
        ) * self.samples_per_gpu * self.num_replicas - len(indice)
        # pad indice
        tmp = indice.copy()
        if size > 0:
            for _ in range(extra // size):
                indice.extend(tmp)
            indice.extend(tmp[:extra % size])
        indices.extend(indice)
        return indices

    def getRangeIndices(self, indicesList, indices, single_indices):
        for i in range(len(indices)):
            sample_item = []
            for _ in range(self.samples_per_gpu):
                sample_item.append(single_indices.pop())
            indicesList[indices[i]] = sample_item

    def getShufful_indices(self, start_index, single_len, shufful_indices):
        end_index = start_index + single_len//self.samples_per_gpu
        indices = shufful_indices[start_index:end_index]
        return indices, end_index

    def __iter__(self):

        #查缺补漏
        detect_g = torch.Generator()
        detect_g.manual_seed(self.epoch)
        detectindices = self.getIndices(self.detectproPosalIds, detect_g)

        facekp_g = torch.Generator()
        facekp_g.manual_seed(self.epoch)
        faceKpindices = self.getIndices(self.facekpProposalIds, facekp_g)

        facemohu_g = torch.Generator()
        facemohu_g.manual_seed(self.epoch)
        facemohuindices = self.getIndices(self.facemohuProposalIds, facemohu_g)

        facezitai_g = torch.Generator()
        facezitai_g.manual_seed(self.epoch)
        facezitaiindices = self.getIndices(self.facezitaiProposalIds, facezitai_g)

        bodydetect_g = torch.Generator()
        bodydetect_g.manual_seed(self.epoch)
        bodydetect_indices = self.getIndices(self.bodyDetectProposalIds, bodydetect_g)

        clousestyle_g = torch.Generator()
        clousestyle_g.manual_seed(self.epoch)
        clousestyle_indices = self.getIndices(self.clouseStyleProposalIds, clousestyle_g)

        clousecolor_g = torch.Generator()
        clousecolor_g.manual_seed(self.epoch)
        clousecolor_indices = self.getIndices(self.clouseColorProposalIds, clousecolor_g)

        faceGender_g = torch.Generator()
        faceGender_g.manual_seed(self.epoch)
        faceGender_indices = self.getIndices(self.faceGenderProposalIds, faceGender_g)

        carplateDetect_g = torch.Generator()
        carplateDetect_g.manual_seed(self.epoch)
        carplateDetect_indices = self.getIndices(self.carplateDetectProposalIds, carplateDetect_g)

        faceDetect_g = torch.Generator()
        faceDetect_g.manual_seed(self.epoch)
        faceDetect_indices = self.getIndices(self.faceDetectProposalIds, faceDetect_g)

        #切片
        assert len(detectindices) % self.num_replicas == 0
        assert len(facezitaiindices) % self.num_replicas == 0
        assert len(facemohuindices) % self.num_replicas == 0
        assert len(faceKpindices) % self.num_replicas == 0
        assert len(bodydetect_indices) % self.num_replicas == 0
        assert len(clousestyle_indices)% self.num_replicas == 0
        assert len(clousecolor_indices)% self.num_replicas == 0
        assert len(faceGender_indices)% self.num_replicas == 0
        assert len(carplateDetect_indices)% self.num_replicas == 0
        assert len(faceDetect_indices)% self.num_replicas == 0

        print(len(detectindices), len(facezitaiindices), len(facemohuindices), len(faceKpindices), len(bodydetect_indices), len(clousestyle_indices), len(clousecolor_indices), len(faceGender_indices), len(carplateDetect_indices), len(faceDetect_indices))

        num_per_gpu_detect = len(detectindices)//self.num_replicas
        single_detectindices = detectindices[self.rank*num_per_gpu_detect:(self.rank+1)*num_per_gpu_detect]

        num_per_gpu_facemohu = len(facemohuindices)//self.num_replicas
        single_facemohuindices = facemohuindices[self.rank*num_per_gpu_facemohu:(self.rank+1)*num_per_gpu_facemohu]

        num_per_gpu_facezitai = len(facezitaiindices)//self.num_replicas
        single_facezitaiindices = facezitaiindices[self.rank*num_per_gpu_facezitai:(self.rank+1)*num_per_gpu_facezitai]

        num_per_gpu_facekp = len(faceKpindices)//self.num_replicas
        single_facekpindices = faceKpindices[self.rank*num_per_gpu_facekp:(self.rank+1)*num_per_gpu_facekp]

        num_per_gpu_bodydetect = len(bodydetect_indices)//self.num_replicas
        single_bodydetect_indices = bodydetect_indices[self.rank*num_per_gpu_bodydetect:(self.rank+1)*num_per_gpu_bodydetect]

        num_per_gpu_clouse_style = len(clousestyle_indices)//self.num_replicas
        single_clousestyle_indices = clousestyle_indices[self.rank*num_per_gpu_clouse_style:(self.rank+1)*num_per_gpu_clouse_style]

        num_per_gpu_clouse_color = len(clousecolor_indices)//self.num_replicas
        single_clousecolor_indices = clousecolor_indices[self.rank*num_per_gpu_clouse_color:(self.rank+1)*num_per_gpu_clouse_color]

        num_per_gpu_faceGender = len(faceGender_indices)//self.num_replicas
        single_faceGender_indices = faceGender_indices[self.rank*num_per_gpu_faceGender:(self.rank+1)*num_per_gpu_faceGender]

        num_per_gpu_carplateDetect = len(carplateDetect_indices)//self.num_replicas
        single_carplateDetect_indices = carplateDetect_indices[self.rank*num_per_gpu_carplateDetect:(self.rank+1)*num_per_gpu_carplateDetect]

        num_per_gpu_faceDetect = len(faceDetect_indices)//self.num_replicas
        single_faceDetect_indices = faceDetect_indices[self.rank*num_per_gpu_faceDetect:(self.rank+1)*num_per_gpu_faceDetect]

        #组合
        single_detect_len = len(single_detectindices)
        single_facemohu_len = len(single_facemohuindices)
        single_facezitai_len = len(single_facezitaiindices)
        single_facekp_len = len(single_facekpindices)
        single_bodydetect_len = len(single_bodydetect_indices)
        single_clousestyle_len = len(single_clousestyle_indices)
        single_clousecolor_len = len(single_clousecolor_indices)
        single_faceGender_len = len(single_faceGender_indices)
        single_carplateDetect_len = len(single_carplateDetect_indices)
        single_faceDetect_len = len(single_faceDetect_indices)

        single_allNums = single_detect_len  + single_facemohu_len + single_facezitai_len + single_facekp_len + single_bodydetect_len + single_clousestyle_len + single_clousecolor_len + single_faceGender_len + single_carplateDetect_len + single_faceDetect_len

        assert single_allNums % self.samples_per_gpu == 0
        assert single_detect_len % self.samples_per_gpu == 0
        assert single_facemohu_len % self.samples_per_gpu == 0
        assert single_facezitai_len % self.samples_per_gpu == 0
        assert single_facekp_len % self.samples_per_gpu == 0
        assert single_bodydetect_len % self.samples_per_gpu == 0
        assert single_clousestyle_len % self.samples_per_gpu == 0
        assert single_clousecolor_len % self.samples_per_gpu == 0
        assert single_faceGender_len % self.samples_per_gpu == 0
        assert single_carplateDetect_len % self.samples_per_gpu == 0
        assert single_faceDetect_len % self.samples_per_gpu == 0

        single_allsampers = int(single_allNums/self.samples_per_gpu)
        h = torch.Generator()
        h.manual_seed(self.epoch)
        shufful_indices = list(torch.randperm(int(single_allsampers),
                                            generator=h).tolist())
        start_index = 0
        detect_indices, end_index = self.getShufful_indices(start_index, single_detect_len, shufful_indices)
        facezitai_indices, end_index = self.getShufful_indices(end_index, single_facezitai_len, shufful_indices)
        facemohu_indices, end_index = self.getShufful_indices(end_index, single_facemohu_len, shufful_indices)
        facekp_indices, end_index = self.getShufful_indices(end_index, single_facekp_len, shufful_indices)
        bodydetect_indices, end_index = self.getShufful_indices(end_index, single_bodydetect_len, shufful_indices)
        clousestyle_indices, end_index = self.getShufful_indices(end_index, single_clousestyle_len, shufful_indices)
        clousecolor_indices, end_index = self.getShufful_indices(end_index, single_clousecolor_len, shufful_indices)
        faceGender_indices, end_index = self.getShufful_indices(end_index, single_faceGender_len, shufful_indices)
        carplateDetect_indices, end_index = self.getShufful_indices(end_index, single_carplateDetect_len, shufful_indices)
        faceDetect_indices, end_index = self.getShufful_indices(end_index, single_faceDetect_len, shufful_indices)

        indicesList = []
        for _ in range(single_allsampers):
            indicesList.append([])

        self.getRangeIndices(indicesList, facemohu_indices, single_facemohuindices)
        self.getRangeIndices(indicesList, facezitai_indices, single_facezitaiindices)
        self.getRangeIndices(indicesList, facekp_indices, single_facekpindices)
        self.getRangeIndices(indicesList, detect_indices, single_detectindices)
        self.getRangeIndices(indicesList, bodydetect_indices, single_bodydetect_indices)
        self.getRangeIndices(indicesList, clousestyle_indices, single_clousestyle_indices)
        self.getRangeIndices(indicesList, clousecolor_indices, single_clousecolor_indices)
        self.getRangeIndices(indicesList, faceGender_indices, single_faceGender_indices)
        self.getRangeIndices(indicesList, carplateDetect_indices, single_carplateDetect_indices)
        self.getRangeIndices(indicesList, faceDetect_indices, single_faceDetect_indices)

        indices = []
        for indice in indicesList:
            indices.extend(indice)
        return iter(indices)
