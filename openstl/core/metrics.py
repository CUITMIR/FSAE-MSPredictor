import cv2
import numpy as np
import torch

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std


def SNR(pred, true):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    """
    signal = ((true)**2).mean()
    noise = ((true - pred)**2).mean()
    return 10. * np.log10(signal / noise)


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # Load images, which are min-max norm to [0, 1]
        img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


def prep_clf(sim, obs, threshold=0.1):
    # obs = np.asarray(obs.cpu().detach().numpy())
    # sim = np.asarray(sim.cpu().detach().numpy())
    obs = np.where(obs >= threshold, 1, 0)
    sim = np.where(sim >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (sim == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (sim == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (sim == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (sim == 0))

    return hits, misses, falsealarms, correctnegatives


def CSI(sim, obs, threshold=0.1):

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)
    results = hits / (hits + misses + falsealarms + 1)

    return results

def HSS(sim, obs, threshold=0.1):

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)
    results = 2 * (hits * correctnegatives - misses * falsealarms) / \
              ((hits + misses) * (misses + correctnegatives) + (hits + falsealarms) * (falsealarms + correctnegatives) + 1)

    return results

def POD(sim, obs, threshold=0.1):

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)
    results = hits / (hits + misses + 1)

    return results

def FAR(sim, obs, threshold=0.1):

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)
    results = falsealarms / (hits + falsealarms + 1)

    return results

class TimeSeries:
    @staticmethod
    def MAE(pred, true):
        return np.mean(np.abs(pred - true))

    @staticmethod
    def MSE(pred, true):
        return np.mean((pred - true) ** 2)

    @staticmethod
    def RMSE(pred, true):
        return np.sqrt(MSE(pred, true))

    @staticmethod
    def MAPE(pred, true):
        return np.mean(np.abs((pred - true) / true))

def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        true (tensor): The prediction values of output prediction.
        mean (tensor): The mean of the preprocessed video data.
        std (tensor): The std of the preprocessed video data.
        metric (str | list[str]): Metrics to be evaluated.
        clip_range (list): Range of prediction to prevent overflow.
        channel_names (list | None): The name of different channels.
        spatial_norm (bool): Weather to normalize the metric by HxW.
        return_log (bool): Whether to return the log string.

    Returns:
        dict: evaluation results
    """
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['#time_series', 'mae', 'mse', 'rmse', 'mape']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if '#time_series' in metrics:

        if 'mse' in metrics:
            eval_res['mse'] = TimeSeries.MSE(pred, true)
            eval_res['mse1/4'] = TimeSeries.MSE(pred[:, :pred.shape[1] // 4, :], true[:, :true.shape[1] // 4, :])
            eval_res['mse2/4'] = TimeSeries.MSE(pred[:, :pred.shape[1] // 2, :], true[:, :true.shape[1] // 2, :])
            eval_res['mse3/4'] = TimeSeries.MSE(pred[:, :pred.shape[1] // 4 * 3, :], true[:, :true.shape[1] // 4 * 3, :])
        if 'mae' in metrics:
            eval_res['mae'] = TimeSeries.MAE(pred, true)
            eval_res['mae1/4'] = TimeSeries.MAE(pred[:, :pred.shape[1] // 4, :], true[:, :true.shape[1] // 4, :])
            eval_res['mae2/4'] = TimeSeries.MAE(pred[:, :pred.shape[1] // 2, :], true[:, :true.shape[1] // 2, :])
            eval_res['mae3/4'] = TimeSeries.MAE(pred[:, :pred.shape[1] // 4 * 3, :], true[:, :true.shape[1] // 4 * 3, :])
        if 'rmse' in metrics:
            eval_res['rmse'] = TimeSeries.RMSE(pred, true)
            eval_res['rmse1/4'] = TimeSeries.RMSE(pred[:, :pred.shape[1] // 4, :], true[:, :true.shape[1] // 4, :])
            eval_res['rmse2/4'] = TimeSeries.RMSE(pred[:, :pred.shape[1] // 2, :], true[:, :true.shape[1] // 2, :])
            eval_res['rmse3/4'] = TimeSeries.RMSE(pred[:, :pred.shape[1] // 4 * 3, :], true[:, :true.shape[1] // 4 * 3, :])
        if 'mape' in metrics:
            eval_res['mape'] = TimeSeries.MAPE(pred, true)
            eval_res['mape1/4'] = TimeSeries.MAPE(pred[:, :pred.shape[1] // 4, :], true[:, :true.shape[1] // 4, :])
            eval_res['mape2/4'] = TimeSeries.MAPE(pred[:, :pred.shape[1] // 2, :], true[:, :true.shape[1] // 2, :])
            eval_res['mape3/4'] = TimeSeries.MAPE(pred[:, :pred.shape[1] // 4 * 3, :], true[:, :true.shape[1] // 4 * 3, :])

    else:

        raise ValueError(f'metric is not supported.')


    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log
