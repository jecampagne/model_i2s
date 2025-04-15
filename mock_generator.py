import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from tqdm import trange
import time


def sigmoid(x, x0, a):
    return 1.0 / (1 + np.exp(a * (x - x0)))


def fglob(x, a=1, y0=5, xm=400):
    """global shape"""
    xmax = x.max()
    y1 = y0 / 2 * (x / xm) ** 0.5
    y2 = y0 / xmax * (xmax - x)
    z = sigmoid(x, xm, 0.05)
    y = y1 * z + (1 - z) * y2
    return y * a


def fraie(x, a, mu, sig):
    """model de raie"""
    arg = (x - mu) / sig
    return -a * np.exp(-0.5 * arg * arg)


def make_psf(fwhm, flux=1):
    """model de PSF"""
    # The FWHM of a Gaussian is 2 sqrt(2 ln2) sigma
    _fwhm_factor = 2.3548200450309493
    _inv_twopi = 0.15915494309189535

    sigma = fwhm / _fwhm_factor
    _inv_sigsq = 1.0 / sigma**2
    W = np.linspace(-5 * sigma, 5 * sigma, int(10 * sigma), endpoint=True)
    H = np.linspace(-5 * sigma, 5 * sigma, int(10 * sigma), endpoint=True)

    X, Y = np.meshgrid(W, H)
    rsq = X**2 + Y**2
    _norm = flux * _inv_sigsq * _inv_twopi
    return _norm * np.exp(-0.5 * rsq * _inv_sigsq)


class Simul:
    def __init__(self, seed=42, img_H=128, img_W=1024, tot_1D_width=900):
        # spectrum numer of pixels
        self.tot_1D_width = tot_1D_width
        # Height/Width final image
        self.img_H = img_H
        self.img_W = img_W
        # for spectrum 1D (raw)
        self.x_shift_max = 10  # +/- pixels wrt x0
        self.y_shift_max = img_H//4  # +/- pixels wrt y0 # was 10 6 avril 25
        self.y_0 = int(img_H // 2)  # y positiion of the spectrum start
        self.x_0 = 2 * self.x_shift_max  # x position

        assert self.tot_1D_width + self.x_shift_max + self.x_0 < img_W
        self.rng = np.random.default_rng(seed)


    def get(self, debug=False):
        """
        return:
        - img_conv : final image
        - spectre  : original 1D spectrum
        - img      : image (bkgd+spectraum) before PSF convolution (debug)
        - y_spec_pos: row loc of spctrum in the image (debug)
        """

        # spectre 1D
        xvals = np.linspace(0, self.tot_1D_width, self.tot_1D_width, endpoint=True)
        yglob = fglob(xvals, a=15.0)
        yr_0 = 0.1 * self.rng.uniform() * fraie(xvals, a=yglob[300], mu=300, sig=70)
        yr_1 = self.rng.uniform() * fraie(xvals, a=yglob[600], mu=600, sig=3)
        yr_2 = self.rng.uniform() * fraie(xvals, a=yglob[750], mu=750, sig=5)

        spectre = yglob + yr_0 + yr_1 + yr_2

        spectre = self.rng.uniform(low=0.5) * spectre

        # img = start with a bkgd
        img = yglob.max() / 200.0 * self.rng.normal(size=(self.img_H, self.img_W))

        # define spectrum position
        x_shift = self.rng.integers(low=-self.x_shift_max, high=self.x_shift_max)
        y_shift = self.rng.integers(low=-self.x_shift_max, high=self.x_shift_max)

        x_spec_start = self.x_0 + x_shift
        y_spec_pos = self.y_0 + y_shift

        # fill img with spectrum
        img[y_spec_pos, x_spec_start : x_spec_start + self.tot_1D_width] = spectre

        # convolution
        #JEC 15/4/25 make variability in psf: here only fwhm
        psf_fwhm = 20 + 5 * self.rng.uniform(low=-1,high=1)
        psf = make_psf(fwhm=psf_fwhm)
        img_conv = signal.oaconvolve(img, psf, mode="same")

        # return
        if debug:
            return img_conv, spectre, img, y_spec_pos
        else:
            return img_conv, spectre


def main():
    root_dir = "/lustre/fswork/projects/rech/ixh/ufd72rp/spectrum_model_auxtel/"
    tag_dataset = "dataset_varpsf"
    train_img_dir = root_dir + tag_dataset + "/train/images/"
    train_spec_dir = root_dir + tag_dataset + "/train/spectra/"
    test_img_dir = root_dir + tag_dataset + "/test/images/"
    test_spec_dir = root_dir + tag_dataset + "/test/spectra/"

    for dname in [train_img_dir, train_spec_dir, test_img_dir, test_spec_dir]:
        try:
            os.makedirs(
                dname, exist_ok=False
            )  # avoid erase the existing directories (exit_ok=False)
        except OSError:
            pass

    mysimu = Simul(seed=15425)
    Ndata_train = 50_000
    Ndata_test  = 5_000
    data_names = ["train", "test"]

    t0 = time.time()
    for data_name in data_names:
        print("do ", data_name, "dataset")
        if data_name == "train":
            img_dir = train_img_dir
            spec_dir = train_spec_dir
            Ndata = Ndata_train
        else:
            img_dir = test_img_dir
            spec_dir = test_spec_dir
            Ndata = Ndata_test

        for i in trange(Ndata, disable=True):
            if i%1000 == 0:
                print(data_name,"i=",i,"time=",time.time()-t0)
            img_conv, spectre = mysimu.get()
            img_name = "img_" + str(i) + ".npy"
            np.save(img_dir + "/" + img_name, img_conv.astype(np.float32))
            spec_name = "spec_" + str(i) + ".npy"
            np.save(spec_dir + "/" + spec_name, spectre.astype(np.float32))

    # end
    tf = time.time()
    print("all done!", tf - t0)

################################
if __name__ == '__main__':
  main()
