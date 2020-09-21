import numpy as np
import cv2.cv2 as cv2
import math

***

The algorithms are optimized to perform matrix operations whenever possible.
All algorithms must have the same parameters, as they are called in a loop from an array.
The parameters used are:

m: Number of pixels per line
n: Number of lines per image
thres: Threshold value used, in case the algorithms has one
l: Number of bins in the histogram
sigma: Sigma value for First Gaussian derivative algorithm
mean: Mean value of all the pixels in the image
p: int64 codification image
cropped: Raw image
hist_range: Range for the histogram, based on pixel depth (0-255 for 8 bit images or 0-65525 for 16 bit images)

***

def abs_tenengrad(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    sum1 = 0
    sobelx = cv2.Sobel(cropped, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(cropped, cv2.CV_64F, 0, 1, ksize=3)

    sx = abs(np.int64(sobelx))
    sy = abs(np.int64(sobely))

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            sum1 = sum1 + (sx[i][j] + sy[i][j])

    return sum1


def brener_grad(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    sum3 = 0
    for j in range(n - 2):
        sum1 = abs(p[:, j + 1] - p[:, j])
        sum2 = abs(p[:, j + 2] - p[:, j])
        for i in range(m):
            if sum1[i] >= thres:
                sum3 = sum3 + sum2[i] ** 2

    return sum3


def entropy(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    sum1 = 0
    hist = np.histogram(p, bins=l, range=[0, hist_range])[0]
    prob_hist = hist / (m*n)

    for i in range(l):
        if prob_hist[i] > 0:
            sum1 = sum1 + (prob_hist[i] * (np.log2(prob_hist[i])))

    return -sum1


def first_gauss_der(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    sum1 = 0
    sum2 = 0
    gx = [None] * n
    gy = [None] * n
    for i in range(m):
        for j in range(n):
            gx[j] = ((-i) / (2 * math.pi * (sigma ** 4))) * math.exp(-(((i ** 2) + (j ** 2)) / (2 * (sigma ** 2))))
            gy[j] = ((-j) / (2 * math.pi * (sigma ** 4))) * math.exp(-(((i ** 2) + (j ** 2)) / (2 * (sigma ** 2))))

        sum1 = sum1 + (p[i] * gx) ** 2
        sum2 = sum2 + (p[i] * gy) ** 2

    sum3 = 0
    for i in range(n):
        sum3 = sum3 + sum1[i] + sum2[i]

    return sum3 / (m * n)


def img_power(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    ps = p ** 2
    sum1 = 0
    for i in range(m):
        for j in range(n):
            if p[i][j] >= thres:
                sum1 = sum1 + ps[i][j]

    return sum1


def laplacian(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    lap = cv2.Laplacian(cropped, cv2.CV_64F)
    lap = np.int64(lap)
    sum1 = 0
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            sum1 = sum1 + lap[i][j]**2

    return sum1


def norm_variance(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    pa = abs(p - mean)

    sum1 = 0
    for i in range(m):
        for j in range(n):
            sum1 = sum1 + pa[i][j]

    return sum1 / (m * n * mean)


def sq_grad(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    sum2 = 0
    for j in range(n - 1):
        sum1 = abs(p[:, j + 1] - p[:, j])
        for i in range(m):
            if sum1[i] >= thres:
                sum2 = sum2 + sum1[i] ** 2

    return sum2


def tenengrad(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    sum1 = 0
    sobelx = cv2.Sobel(cropped, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(cropped, cv2.CV_64F, 0, 1, ksize=3)

    sx = (np.int64(sobelx)) ** 2
    sy = (np.int64(sobely)) ** 2

    for i in range(1, m-1):
        for j in range(1, n-1):
            sum1 = sum1 + (sx[i][j] + sy[i][j])

    return sum1


def thres_abs_grad(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    sum2 = 0
    for j in range(n - 1):
        sum1 = abs(p[:, j + 1] - p[:, j])
        for i in range(m):
            if sum1[i] >= thres:
                sum2 = sum2 + sum1[i]

    return sum2


def thres_pix_count(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    pr = p - thres
    sum1 = 0
    for i in range(m):
        for j in range(n):
            if pr[i][j] < 0:
                sum1 = sum1 + 1

    return sum1


def variance(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    pa = abs(p - mean)

    sum1 = 0
    for i in range(m):
        for j in range(n):
            sum1 = sum1 + pa[i][j]

    return sum1 / (m * n)


def variance_log_hist(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    var = 0
    suma = 0
    hist = np.histogram(p, bins=l, range=[0, hist_range])[0]

    for i in range(l):
        if hist[i] == 0:
            hist[i] = 1
    log_hist = np.log10(hist)

    for i in range(l):
        if log_hist[i] > 0:
            suma = suma + log_hist[i]
        else:
            log_hist[i] = 0
    prob_log_hist = log_hist / suma

    for i in range(l):
        Elog = 0
        for j in range(i):
            Elog = Elog + (j * prob_log_hist[j])
        var = var + (((i - Elog) ** 2) * prob_log_hist[i])

    return var


def vollath4(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    sum1 = 0
    sum2 = 0
    for i in range(m - 1):
        sum1 = sum1 + (p[i] * p[i + 1])

    for i in range(m - 2):
        sum2 = sum2 + (p[i] * p[i + 2])

    sum3 = (sum1 - sum2)
    res = 0
    for i in range(n):
        res = res + sum3[i]

    return res


def vollath5(m, n, thres, l, sigma, mean, p, cropped, hist_range):
    sum1 = 0
    for i in range(m - 1):
        sum1 = sum1 + (p[i] * p[i + 1]) - (mean ** 2)

    res = 0
    for i in range(n):
         res = res + sum1[i]

    return res

