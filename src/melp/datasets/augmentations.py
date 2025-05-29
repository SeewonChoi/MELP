import random
import math
import numpy as np
import torch


class RandomLeadsMask(object):
    def __init__(
        self,
        p=1,
        mask_leads_selection="random",
        mask_leads_prob=0.5,
        mask_leads_condition=None,
        **kwargs,
    ):
        self.p = p
        self.mask_leads_prob = mask_leads_prob
        self.mask_leads_selection = mask_leads_selection
        self.mask_leads_condition = mask_leads_condition
    
    def __call__(self, sample):
        ''' Sample: torch.Tensor '''
        if self.p >= np.random.uniform(0, 1):
            new_sample = sample.new_zeros(sample.size())
            if self.mask_leads_selection == "random":
                survivors = np.random.uniform(0, 1, size=12) >= self.mask_leads_prob
                new_sample[survivors] = sample[survivors]
            elif self.mask_leads_selection == "conditional":
                (n1, n2) = self.mask_leads_selection
                assert (
                    (0 <= n1 and n1 <= 6) and
                    (0 <= n2 and n2 <= 6)
                ), (n1, n2)
                s1 = np.array(
                    random.sample(list(np.arange(6)), 6-n1)
                )
                s2 = np.array(
                    random.sample(list(np.arange(6)), 6-n2)
                ) + 6
                new_sample[s1] = sample[s1]
                new_sample[s2] = sample[s2]
        else:
            new_sample = sample.clone()

        return new_sample.float()


def Tinterpolate(data, marker):

    channels, timesteps = data.shape
    data = data.flatten()
    ndata = data.numpy()
    interpolation = torch.from_numpy(np.interp(np.where(ndata == marker)[0], np.where(ndata != marker)[0], ndata[ndata != marker]))
    data[data == marker] = interpolation.type(data.type())
    data = data.reshape(channels, timesteps)

    return data


class Transformation:
    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def get_params(self):
        return self.params


class TRandomResizedCrop(Transformation):
    """ Extract crop at random position and resize it to full size
    """
    
    def __init__(self, crop_ratio_range=[0.5, 1.0]):
        super().__init__()
        self.crop_ratio_range = crop_ratio_range
       
    def __call__(self, data):
        output = torch.full(data.shape, float("inf")).type(data.type())
        # timesteps, channels = output.shape
        channels, timesteps = data.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        data = TRandomCrop(int(crop_ratio * timesteps))(data)  # apply random crop
        cropped_timesteps = data.shape[1]
        indices = torch.sort((torch.randperm(timesteps-2)+1)[:cropped_timesteps-2])[0]
        indices = torch.cat([torch.tensor([0]), indices, torch.tensor([timesteps-1])])
        output[:, indices] = data  # fill output array randomly (but in right order) with values from random crop
        
        # use interpolation to resize random crop
        output = Tinterpolate(output, float("inf"))

        return output
    
    def __str__(self):
        return "RandomResizedCrop"


class TRandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size,annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, data):

        _, timesteps = data.shape
        assert(timesteps >= self.output_size)
        if(timesteps==self.output_size):
            start=0
        else:
            start = random.randint(0, timesteps - self.output_size-1) #np.random.randint(0, timesteps - self.output_size)

        data = data[:, start: start + self.output_size]
        
        return data
    
    def __str__(self):
        return "RandomCrop"


class TTimeOut(Transformation):
    """ replace random crop by zeros
    """

    def __init__(self, crop_ratio_range=[0.0, 0.5]):
        super(TTimeOut, self).__init__(crop_ratio_range=crop_ratio_range)
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, data):
        data = data.clone()
        timesteps, channels = data.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio*timesteps)
        start_idx = random.randint(0, timesteps - crop_timesteps-1)
        data[start_idx:start_idx+crop_timesteps, :] = 0
        return data

    def __str__(self):
        return "TimeOut"


def Tnoise_powerline(fs=100, N=1000,C=1,fn=50.,K=3, channels=1):
    '''powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    '''
    #C *= 0.333 #adjust default scale
    t = torch.arange(0,N/fs,1./fs)
    
    signal = torch.zeros(N)
    phi1 = random.uniform(0,2*math.pi)
    for k in range(1,K+1):
        ak = random.uniform(0,1)
        signal += C*ak*torch.cos(2*math.pi*k*fn*t+phi1)
    signal = C*signal[:,None]
    if(channels>1):
        channel_gains = torch.empty(channels).uniform_(-1,1)
        signal = signal*channel_gains[None]
    return signal

def Tnoise_baseline_wander(fs=100, N=1000, C=1.0, fc=0.5, fdelta=0.01,channels=1,independent_channels=False):
    '''baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale : 1)
    fc: cutoff frequency for the baseline wander (Hz)
    fdelta: lowest resolvable frequency (defaults to fs/N if None is passed)
    channels: number of output channels
    independent_channels: different channels with genuinely different outputs (but all components in phase) instead of just a global channel-wise rescaling
    '''
    if(fdelta is None):# 0.1
        fdelta = fs/N

    K = int((fc/fdelta)+0.5)
    t = torch.arange(0, N/fs, 1./fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    phase_k = torch.empty(K).uniform_(0, 2*math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    pre_cos = 2*math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(dim=0)
    return C*res


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wave, target):
        for t in self.transforms:
            wave, target = t(wave, target)
        return wave, target


class RandomCrop():
    def __init__(self, length, start, end):
        self.length = length
        self.start = start
        self.end = end
    
    def __call__(self, wave, target):
        start = random.randint(self.start, self.end-self.length)
        end = start + self.length
        return wave[:,start:end], target[:,start:end]


class ChannelResize():
    def __init__(self, magnitude_range=(0.5, 2)):
        self.log_magnitude_range = torch.log(torch.tensor(magnitude_range))

    def __call__(self, wave, target):
        channels, len_wave = wave.shape
        resize_factors = torch.exp(torch.empty(channels).uniform_(*self.log_magnitude_range)) 
        resize_factors = resize_factors.repeat(len_wave).view(wave.T.shape).T 
        wave = resize_factors * wave
        return wave, target
    
class GaussianNoise():
    def __init__(self, prob=1.0, scale=0.01):
        self.scale = scale
        self.prob = prob
    
    def __call__(self, wave, target):
        if random.random() < self.prob:
            wave += self.scale * torch.randn(wave.shape)
        return wave, target


class BaselineShift():
    def __init__(self, prob=1.0, scale=1.0):
        self.prob = prob
        self.scale = scale

    def __call__(self, wave, target):
        if random.random() < self.prob:
            shift = torch.randn(1)
            wave = wave + self.scale * shift
        return wave, target


class BaselineWander():
    def __init__(self, prob=1.0, freq=500):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave, target):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            wander = Tnoise_baseline_wander(fs=self.freq, N=len_wave) 
            wander = wander.repeat(channels).view(wave.shape)
            wave = wave + wander
        return wave, target


class PowerlineNoise():
    def __init__(self, prob=1.0, freq=500):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave, target):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            noise = Tnoise_powerline(fs=self.freq, N=len_wave, channels=channels).T 
            wave = wave + noise
        return wave, target
