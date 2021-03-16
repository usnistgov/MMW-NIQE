import scipy.misc
import scipy.io
import scipy
from PIL import Image
import scipy.ndimage
import numpy as np
import scipy.special
import math
from scipy.stats import exponweib
from scipy.optimize import fmin
import scipy.stats
import scipy.signal
from numpy.lib.stride_tricks import as_strided as ast
import warnings
warnings.filterwarnings("ignore")


thresh = 75
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)


def aggd_features(imdata):
    #flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata*imdata
    left_data = imdata2[imdata<0]
    right_data = imdata2[imdata>=0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
        gamma_hat = np.inf
    #solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    #solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm)**2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    #mean parameter
    N = (br - bl)*(gam2 / gam1)#*aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def estimate_params_mv_ggd_mm(X):
    n, N = X.shape
    S = np.cov(X)
    gamma2_cap =  np.sum(np.sum(np.multiply(np.matmul(X.T, np.linalg.pinv(S)), X.T), axis = 1)**2)/N
    bb = np.arange(0.1, 1.0, 0.004)
    tt = (n**2)*scipy.special.gamma(n/(2*bb))*scipy.special.gamma((n+4)/(2*bb))/(scipy.special.gamma((n+2)/(2*bb)))**2 - gamma2_cap
    ind = np.argmin(np.abs(tt))
    beta = bb[ind]
    sigma = S*n*scipy.special.gamma(n/(2*beta))/(2**(1/beta)*scipy.special.gamma((n+2)/(2*beta)))
    return S, beta, sigma

def shifted_coefs(new_im):
    shifted = np.roll(new_im.copy(), 1, axis=0)
    return [new_im.flatten(), shifted.flatten()]

def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image), var_image, mu_image


def block_view(A, block=(8, 8)):
    shape= ( ( int(A.shape[0]/ block[0]), int(A.shape[1]/ block[1]))+ block)
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


def fitweibull(x):
    def optfun(theta):
        return -np.sum(np.log(exponweib.pdf(x, 1, theta[0], scale = theta[1], loc = 0)))
    logx = np.log(x)
    shape = 1.2 / np.std(logx)
    scale = np.exp(np.mean(logx) + (0.572 / shape))
    return fmin(optfun, [shape, scale], xtol = 0.01, ftol = 0.01, disp = 0)
    

def entropy(x, bins=None):
    N   = x.shape[0]
    if bins is None:
        counts = np.bincount(x)
    else:
        counts = np.histogram(x, bins=bins)[0] # 0th idx is counts
    
    p   = counts[np.nonzero(counts)]/float(N) # avoids log(0)
    H   = -np.dot( p, np.log2(p) )
    return H   


def _mmw_niqe_extract_subband_feats(coefs):
    
    ms = coefs[:,:,0]
    var = coefs[:,:,1]
    Ix = coefs[:,:,2]
    Iy = coefs[:,:,3]
    GM = coefs[:,:,4]
    patch = coefs[:,:,5]
    
    alpha_ms, _, bl_ms, br_ms, _, _ = aggd_features(ms.copy()) ## ms feats
    shifted_pair = shifted_coefs(ms.copy())
    S, beta, sigma = estimate_params_mv_ggd_mm(np.array(shifted_pair))
    eig1, eig2 = np.linalg.eigvals(sigma)
    shape_sig, scale_sig = fitweibull(var.flatten()+1e-6)
    alpha_Ix, _, bl_Ix, br_Ix, _, _ = aggd_features(Ix.copy()) ## Ix feats
    alpha_Iy, _, bl_Iy, br_Iy, _, _ = aggd_features(Iy.copy()) ## Iy feats
    shape_GM, scale_GM = fitweibull(GM.flatten())
    
    patch = np.pad(patch, ((0, int(np.ceil((patch.shape[0]/8.0))*8)-patch.shape[0]), 
                           (0, int(np.ceil((patch.shape[1]/8.0))*8)-patch.shape[1])), 'constant')
    
    blocked = block_view(patch.astype(int))
    
    ents = []
    for i in range(blocked.shape[0]):
        for j in range(blocked.shape[1]):
            ent = entropy(blocked[i][j].flatten(), 256)
            ents.append(ent)
    
    ents = np.array(ents)
    mean_ent = np.mean(ents)
    skew_ent = scipy.stats.skew(ents)
    
    return np.array([alpha_ms, (bl_ms+br_ms)/2.0,
            beta, min(eig1, eig2), max(eig1, eig2),  
            scale_sig, shape_sig,
            alpha_Ix, (bl_Ix+br_Ix)/2.0,  
            alpha_Iy, (bl_Iy+br_Iy)/2.0, 
            scale_GM, shape_GM,
            mean_ent, skew_ent
    ])


def get_patches_test_features(img, patch_size_h, patch_size_w, stride=8):
    return _get_patches_generic(img, patch_size_h, patch_size_w, 0, stride)

def extract_on_patches(data, patch_size_h, patch_size_w):
    h, w, map_nums  = data.shape
    patch_size_h = np.int(patch_size_h)
    patch_size_w = np.int(patch_size_w)
    patches = []
    patch_features = []
    sharpness = []
    for j in range(0, h-patch_size_h+1, patch_size_h):
        for i in range(0, w-patch_size_w+1, patch_size_w):
            patch = data[j:j+patch_size_h, i:i+patch_size_w, :]
            patch_features.append(_mmw_niqe_extract_subband_feats(patch))
            sharpness.append(np.mean(patch[:,:,1]))

    patch_features = np.array(patch_features)
    sharpness = np.array(sharpness)
    return patch_features, sharpness


def _get_patches_generic(img, patch_size_h, patch_size_w, is_train, stride):
    h, w = np.shape(img)

    hoffset = (h % patch_size_h)
    woffset = (w % patch_size_w)

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)

    sigmaForGauDerivative = 0.5
    scaleFactorForGaussianDer = 0.28
        
    feats = []
    
    for itr_scale in range(1, 3):
        
        ms, var, mu = compute_image_mscn_transform(img)
        sigmaGauss = sigmaForGauDerivative/(itr_scale**scaleFactorForGaussianDer)
        halfLength = np.ceil(3*sigmaGauss)
        num = int(2*halfLength+1)
        xv = np.linspace(-halfLength, halfLength, num)
        yv = np.linspace(-halfLength, halfLength, num)
        x, y = np.meshgrid(xv, yv) 
        dx = x*np.exp(-(x**2 + y**2)/2/sigmaGauss/sigmaGauss)
        dy = y*np.exp(-(x**2 + y**2)/2/sigmaGauss/sigmaGauss)    
        
        Ix = scipy.ndimage.convolve(img, dx)
        Iy = scipy.ndimage.convolve(img, dy)
        GM = np.sqrt(Ix**2 + Iy**2)+1e-16
        
        data = np.empty((img.shape[0], img.shape[1], 6),dtype=np.float32)
        data[:,:,0] = ms
        data[:,:,1] = var
        data[:,:,2] = Ix
        data[:,:,3] = Iy
        data[:,:,4] = GM
        data[:,:,5] = img

        feats_lvl, sharpness = extract_on_patches(data, patch_size_h/itr_scale, patch_size_w/itr_scale)
        
        if(itr_scale == 1):
            sharpness_first = sharpness

        feats.append(feats_lvl)
        #img = imresize(img, 0.5, interp='bicubic', mode='F')
        dsize = ( int(img.shape[1]/2) , int(img.shape[0]/2) )
        img = np.array(Image.fromarray(img).resize(dsize, Image.BICUBIC))
        # img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)

    out1 = np.hstack((feats[0], feats[1]))
    
    return(out1, sharpness_first)


def mmw_niqe(inputImgData):

    patch_size_row = 110
    patch_size_col = 84	
    thresh = 75

    params = scipy.io.loadmat('mmw-niqe_model_cf_25_bw_60.mat')
    pop_mu = np.ravel(params["mu_25_60"])
    pop_cov = params["cov_25_60"]

    feats, sharpness = get_patches_test_features(inputImgData, patch_size_row, patch_size_col)
    ind = sharpness  > np.percentile(sharpness, thresh)
    
    feats = feats[ind, :]
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score


if __name__ == "__main__":

    test_file = 'mmw_img.png'

    img = np.array(Image.open('../test_images/0.png'))[:,:,0] # 1
    mmw_niqe_score = mmw_niqe(img)

    print('-----------------------------------------------------')
    print('    The MMW-NIQE score of "{}" is: {:0.3f}'.format(test_file,mmw_niqe_score))
    print('-----------------------------------------------------')

