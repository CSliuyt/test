import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion, transform_tiny
from data import TensorDataset, ImageFolder, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from common import define_model, remove_aug, load_resized_data, diffaug
from test import test_data, load_ckpt
from misc import utils
from math import ceil
import glob
import random

def score(vector, matrix):
    vector_norm = vector / vector.norm()
    matrix_norm = matrix / matrix.norm(dim=1, keepdim=True)
    similarity = torch.mm(vector_norm.unsqueeze(0), matrix_norm.t())
    
    return similarity.squeeze(0)

def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    if len(labels.size()) < 2:
        labels = F.one_hot(labels, logits.size(-1))
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(
        1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


class Synthesizer():
    """Condensed data class
    """
    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                       w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor): # amplify four times, [2000, 3, 32, 32]
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])
        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
    
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=256):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target

    def loader(self, args, augment=True):
        """Data loader for condensed data
        """
        if args.dataset in ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']:
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)
        elif args.dataset == 'tinyimagenet':
            train_transform, _ = transform_tiny(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=8,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger, bench=True):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        convnet_result = test_data(args, loader, val_loader, test_resnet=False, logger=logger)

        return convnet_result
        # if bench and not (args.dataset in ['mnist', 'fashion']):
        #     resnet_result = test_data(args, loader, val_loader, test_resnet=True, logger=logger)
        #     return convnet_result, resnet_result
        # else:
        #     return convnet_result

'''
def innerloss(c, img_real, img_syn, model_init, model_real):
    """Matching losses (feature or gradient)
    """
    with torch.no_grad():
        pred_real, _ = model_real(img_real, return_features=True)
        _, feat_tg = model_init(img_real, return_features=True) 
    label_real = torch.tensor([c] * pred_real.size(0)).long().to(pred_real.device)
    mask = pred_real.softmax(1).argmax(1) == label_real
    feat_tg = feat_tg[mask]
    _, feat = model_init(img_syn, return_features=True)
    loss = torch.sum((feat.mean(0) - feat_tg.mean(0))**2)
    return loss, sum(mask)

'''
def innerloss(it, img_real, img_syn, model_init):
    """Matching losses (feature or gradient)
    """
    with torch.no_grad():
        _, feat_tg = model_init(img_real, return_features=True) 
    _, feat = model_init(img_syn, return_features=True)
    sc = score(feat.mean(0), feat_tg)
    sc_mean = sc.mean(0)
    sc_max = sc.max(0)[0]
    sc_min = sc.min(0)[0]
    pr_rate = it // 2 / 10000
    mask = sc > sc_min + pr_rate * (sc_mean - sc_min) # for low IPC [1, 10, 50, 100]
    # mask = sc < (sc_max - pr_rate * (sc_max - sc_mean)) # for high IPC  [200, 300, 500, 1000] 
    loss = torch.sum((feat.mean(0) - feat_tg[mask].mean(0))**2)
    return loss, sum(mask)


def interloss(c, img_real, img_syn, trained_model):
    logits = trained_model(img_syn, return_features=False)
    
    label = trained_model(img_real, return_features=False)
    label_real = torch.tensor([c] * label.size(0)).long().to(label.device)
    mask = label.softmax(1).argmax(1) == label_real
    num = sum(mask)
    label = label[mask].softmax(1).sum(0) / 0.01
    label = label.softmax(0).repeat(logits.size(0), 1)
    loss = cross_entropy(logits, label)
    
    return loss, num

def interloss(c, img_real, img_syn, trained_final, model_mid):
    logits = model_mid(img_syn, return_features=False)
    
    label = trained_final(img_real, return_features=False)
    label_real = torch.tensor([c] * label.size(0)).long().to(label.device)
    mask = label.softmax(1).argmax(1) == label_real
    num = sum(mask)
    label = label[mask].softmax(1).sum(0) / 0.01
    label = label.softmax(0).repeat(logits.size(0), 1)
    loss = cross_entropy(logits, label)
    
    return loss, num

def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """
    # Define real dataset and loader
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # Define syn dataset
    synset = Synthesizer(args, nclass, nch, hs, ws)
    synset.init(loader_real, init_type=args.init)
    print(args.init)


    # Define augmentation function
    aug, _ = diffaug(args)



    if not args.test:
        save_img(os.path.join(args.save_dir, 'init.png'),
            synset.data,
            unnormalize=False,
            dataname=args.dataset)
        torch.save([synset.data.detach().cpu(), synset.targets.cpu()],
        os.path.join(args.save_dir, 'data_0.pt'))
        # synset.test(args, val_loader, logger, bench=False)

    # Data distillation
    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)
    optim_img2 = torch.optim.SGD(synset.parameters(), lr=0.01, momentum=args.mom_img)
    ts = utils.TimeStamp(args.time)

    n_iter = args.niter
    it_log = 20

    it_test = np.arange(0, n_iter+1, 500).tolist()

    # it_test = [n_iter // 10, n_iter // 5, n_iter // 2, n_iter]

    logger(f"\n DANCE: Start condensing for {n_iter} iteration")


    best_convnet = -1
    best_resnet = -1
    model_init = define_model(args, nclass).to(device)
    model_final = define_model(args, nclass).to(device)
    model_interval = define_model(args, nclass).to(device)
    
    for it in range(n_iter):
        if args.num_premodel > 0:
            slkt_model_id = random.randint(0, args.num_premodel - 1)
            init_path = os.path.join(args.pretrain_dir, 'premodel{}_init.pth.tar'.format(slkt_model_id))
            final_path = os.path.join(args.pretrain_dir, 'premodel{}_trained.pth.tar'.format(slkt_model_id))
            model_init.load_state_dict(torch.load(init_path))
            model_final.load_state_dict(torch.load(final_path))
            l = (n_iter - it / 2) / n_iter
            for param_C, param_A, param_B in zip(model_interval.parameters(), model_init.parameters(), model_final.parameters()):
                param_C.data.copy_(l * param_A.data + (1 - l) * param_B.data)
        else:
            slkt_model_id = random.randint(0, 4)
            final_path = os.path.join(args.pretrain_dir, 'premodel{}_trained.pth.tar'.format(slkt_model_id))
            model_final.load_state_dict(torch.load(final_path))

        '''detach the model'''
        # for name, param in model.named_parameters():
        #     param = param.detach()

        loss_total = 0
        num_selected = 0.
        
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)

        ts.set()
    
        # Update synset (inner-class view)
        for c in range(nclass):
            img, _ = loader_real.class_sample(c)
            # print(len(img))
            img_syn, _ = synset.sample(c, max_size=args.batch_syn_max)
            ts.stamp("data")

            n = img.shape[0]
            img_aug = aug(torch.cat([img, img_syn]))
            ts.stamp("aug")

            loss, num = innerloss(it, img_aug[:n], img_aug[n:], model_interval)  # first input img_real then img_syn
            loss_total += loss.item()
            num_selected += num
            ts.stamp("loss")

            optim_img.zero_grad()
            loss.backward()

            '''print the range of gradients'''
            # logger('the grad of image range from [{}] to [{}]'.format(torch.min(torch.abs(synset.data.grad)),
            #                                                         torch.max(torch.abs(synset.data.grad))))

            optim_img.step()
            ts.stamp("backward")

        ts.flush()


        # Update syn set (inter-class view)
        calib_loss_total = 0
        label_selected = 0.
        if args.iter_calib > 0:
            for _ in range(args.iter_calib):
                for c in range(nclass):
                    img, _ = loader_real.class_sample(c)
                    img_syn, _ = synset.sample(c, max_size=args.batch_syn_max)
                    n = img.shape[0]
                    img_aug = aug(torch.cat([img, img_syn]))
                    loss, num = interloss(c, img_aug[:n], img_aug[n:], model_final, model_interval)
                    calib_loss_total += loss.item()
                    label_selected += num
                    optim_img2.zero_grad()
                    loss.backward()
                    optim_img2.step()

                    # if it % it_log == 0:
                    #     # check the logits
                    #     with torch.no_grad():
                    #         logits = F.softmax(model_final(img_aug, return_features=False), dim=-1)
                    #         prob, pred = torch.max(logits, dim=-1)
                    #     logger("CLASS {}:".format(c))
                    #     logger(prob)
                    #     logger(pred)
        else:
            pass

        # Logging
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) num-imgs: {num_selected/nclass:.2f} inter-loss: {calib_loss_total/nclass/args.iter_calib:.2f}  num-labels: {label_selected/nclass:.2f} inner-loss: {loss_total/nclass:.2f}")

        save_best = 0
        if ((it + 1) in it_test):
            conv_result = synset.test(args, val_loader, logger)

            if conv_result > best_convnet:
                best_convnet = conv_result
                save_best = 1

                logger("->->->->->->->->->->->->-> Best Result: {:.1f}".format(best_convnet))
            

            # It is okay to clamp data to [0, 1] at here.
            # synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)


            if not args.test:
                save_img(os.path.join(args.save_dir, f'img{it+1}.png'),
                     synset.data,
                     unnormalize=False,
                     dataname=args.dataset)
                torch.save(
                    [synset.data.detach().cpu(), synset.targets.cpu()],
                    os.path.join(args.save_dir, 'data_{}.pt'.format(it+1)))
                logger("img and data saved!")

                
                if save_best:
                    torch.save(
                    [synset.data.detach().cpu(), synset.targets.cpu()],
                    os.path.join(args.save_dir, 'data_best.pt'))
                    logger("best data saved")
            save_best = 0



if __name__ == '__main__':
    
    import torch.backends.cudnn as cudnn
    import json
    import os
    from misc.utils import Logger
    
    num_exp = 5
    for exp in range(num_exp):
        
        from arguments.arg_condense import args
        assert args.ipc > 0

        cudnn.benchmark = True
        if args.seed > 0:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            
        if not args.test:
            args.save_dir = os.path.join(args.save_dir, args.dataset, "IPC_" + str(args.ipc) + '_' + str(exp + 1))
            os.makedirs(args.save_dir, exist_ok=True)

            logger = Logger(args.save_dir)
            logger(f"Save dir: {args.save_dir}")

            with open(os.path.join(args.save_dir, 'args.log'), 'w') as f:
                json.dump(args.__dict__, f, indent=3)
        else:
            logger = print

        condense(args, logger)
