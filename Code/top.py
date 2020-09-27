import time
import scipy.signal
from algs_no_opt import *
from algs_opt import *
import numpy as np
import cv2.cv2 as cv2


# This part must be filled and uncomented 
***

m = 1040  # Number of lines per image
n = 1388  # Number of pixels per line
l = 256   # Number of bins in the histogram
sigma = 2   # Sigma value for First Gaussian derivative algorithm
num_criteria = 4   # Number of criteria evaluated: Accuracy, Range, False maxima, FWHW
num_algs = 15    # Number of algorithms to evaluate

direct_img = Path to the images
direct_res = Path to save the results of the analysis
tissue = ["Adipose/", "Stomach/", "Intestine/", "Kidney/"]
magnification = ["5x/", "10x/"]
depth = ["8 bits/", "16 bits/"]

***


global path
global path_res
global num_stacks


def play_algs():
    for k in range(num_stacks):
        alg_table = [[0 for i in range(num_imgs[k])] for j in range(num_algs)]
        norm_data_table = [[0. for i in range(num_imgs[k])] for j in range(num_algs)]
        time_table = [[0. for i in range(num_imgs[k])] for j in range(num_algs)]

		# Creates a .txt document for each stack with the contrast results of all the algorithms (data<>.txt)
        f2 = open(path_res + "data" + str(k) + ".txt", 'a')
		# Creates a .txt document for each stack with the criteria evaluation (crit_table<>.txt)
        f = open(path_res + "crit_table" + str(k) + ".txt", 'a')
        f.write('Algorithm;Accuracy;Range;False maxima;FWHW;Time\n') 

        for i in range(num_imgs[k]):
			# Opens an image
            cadena = path + "S" + str(k + 1) + "/" + str(i + 1) + ".tif"
            img = cv2.imread(cadena, -1)
            p = np.int64(img)
            mean = cv2.mean(img)[0]

			# Calculates the contrast value for all the algorithms, defining the contrast functions
            for j in range(num_algs):
                point1 = (time.time() * 1000)
                alg_table[j][i] = algorithms_opt[j](m, n, thres[j], l, sigma, mean, p, img, hist_range)
                point2 = (time.time() * 1000)
				
				# The execution time for each algorithm is measured, but will not be included in the analysis 
                time_table[j][i] = point2 - point1

            print("Done img ", i)
        print("Done stack ", k)

		# Evaluates the resulting contrast functions with the criteria and writes the results in crit_table<>.txt
        for j in range(num_algs):
            crit_table[k][j][:] = np.array((analysis(alg_table[j][:], real_maxs[k], k)))
            f.write(names[j])
            for c in range(num_criteria):
                f.write(';' + str(crit_table[k][j][c]))
            mean_t = np.mean(time_table[j])
            f.write(';' + str(mean_t) + '\n')
        f.close()

		# Normalizes the values of the contrast functions for a better graphic representation, and writes them in data<>.txt
        for j in range(num_algs):
            f2.write(names[j])
            mean2 = np.mean(alg_table[j])
            std2 = np.std(alg_table[j])
            if std2 > 0:
                norm_data_table[j] = (alg_table[j] - mean2) / std2
            for i in range(num_imgs[k]):
                f2.write(';' + str(norm_data_table[j][i]))
            f2.write('\n')
        f2.close()


# This function is used by analysis() to calculate the maximum range of the contrast functions
def range_max(alg, max_x):
    mins = (scipy.signal.argrelmin(alg))[0]

    izq = 0
    dcha = alg.size - 1
    for i in range(mins.size):
        if mins[i] > max_x:
            dcha = mins[i]
            break
        else:
            izq = mins[i]

    return dcha - izq


# This function applies the criteria to the calculated contrast functions
def analysis(alg, real_max, k):
    max_point_x = np.argmax(alg)
    max_point_y = max(alg)
    min_point_y = min(alg)
    h_tot = max_point_y - min_point_y
	
	"""
	The anlysis automatically gives the worst score to the contrast functions that have any of the following features:
	- Do not have a maximum
	- Even when they have a maximum, its value is not the maximum value in the series (maximum value != maximum point)
	- Even when the maximum is the maximum value in the series, its height is less than the 20% of the height range of the series
	
	"""

    array = np.array(alg)
    peaks, _ = scipy.signal.find_peaks(array)
    if peaks.size >= 1:
        max_x = peaks[(np.argmax(array[peaks]))]
        max_y = alg[peaks[(np.argmax(array[peaks]))]]
        max_y_base = ((scipy.signal.peak_widths(array, [np.array(max_x)], rel_height=1.0))[1])[0]
        h_peak = max_y - max_y_base

        if (max_x == max_point_x) and (h_peak >= (0.2 * h_tot)):
            acc = abs(real_max - max_x)
            ran = num_imgs[k] - 1 - range_max(array, max_x)
            fal = peaks.size - 1
            wid = ((scipy.signal.peak_widths(array, [np.array(max_x)], rel_height=0.5))[0])[0]
            return [acc, ran, fal, wid]
        else:
            return [num_imgs[k], num_imgs[k], num_imgs[k], num_imgs[k]]
    else:
        return [num_imgs[k], num_imgs[k], num_imgs[k], num_imgs[k]]


# This function ranks the each criterion in the criteria tables
def ranking(crit_table):
    table = [0 for j in range(num_algs)]
    arr_res = [0 for j in range(num_algs)]

    for k in range(num_stacks):
        for c in range(num_criteria):
            table = crit_table[k, :, c]

            arr = np.array(table)
            arr_copy = np.array(sorted(set(arr)))
            offset = 1

            for j in range(arr_copy.size):
                indexes = (np.where(arr == (arr_copy[j])))[0]
                for a in range(indexes.size):
                    arr_res[indexes[a]] = offset
                offset = offset + indexes.size

            rank_table[k, :, c] = arr_res


# This function performs the Semi Quantitative analysis, creates and writes the results in 
def semi_quant(crit_table):
    semi_table = np.array([[0. for a in range(num_criteria)] for b in range(num_algs)])
    results_semi = np.array([0. for b in range(num_algs)])
    ranking(crit_table)
    for k in range(num_stacks):
        semi_table[:, :] = semi_table[:, :] + rank_table[k]

    for c in range(num_criteria):
        results_semi[:] = results_semi[:] + semi_table[:, c]

    f = open(path_res + "semi_table.txt", 'a')

    f.write('Algorithm;Accuracy;Range;False maxima;FWHW\n')
    for j in range(num_algs):
        f.write(names[j])
        for c in range(num_criteria):
            f.write(';' + str(semi_table[j][c]))
        f.write('\n')
    f.close()


def quant(crit_table):
    euc_table = np.array([[0. for d in range(num_criteria)] for e in range(num_algs)])
    results_quant = np.array([0. for b in range(num_algs)])
    to_norm_table[:, :-1, :] = crit_table
    to_norm_table[:, -1, :] = ideal_func

    for k in range(num_stacks):
        for c in range(num_criteria):
            mean_crit = np.mean(to_norm_table[k, :, c])
            std_crit = np.std(to_norm_table[k, :, c])
            norm_table[k, :, c] = (to_norm_table[k, :, c] - mean_crit) / std_crit

    for k in range(num_stacks):
        for c in range(num_criteria):
            dist_table[k, :, c] = norm_table[k, :-1, c] - norm_table[k, -1, c]

    for k in range(num_stacks):
        euc_table = euc_table + (dist_table[k, :, :] ** 2)

    euc_table = np.sqrt(euc_table)

    for c in range(num_criteria):
        results_quant[:] = results_quant[:] + euc_table[:, c]

    f = open(path_res + "quant_table.txt", 'a')

    f.write('Algoritmo;Accuracy;Range;False maxima;FWHW\n')
    for j in range(num_algs):
        f.write(names[j])
        for c in range(num_criteria):
            f.write(';' + str(euc_table[j][c]))
        f.write('\n')
    f.close()


# -------------------------------- #

algorithms_opt = [abs_tenengrad_opt, brener_grad_opt, entropy_opt, first_gauss_der_opt, img_power_opt, laplacian_opt,
              norm_variance_opt, sq_grad_opt, tenengrad_opt, thres_abs_grad_opt, thres_pix_count_opt, variance_opt,
              variance_log_hist_opt, vollath4_opt, vollath5_opt]
algorithms = [abs_tenengrad, brener_grad, entropy, first_gauss_der, img_power, laplacian,
              norm_variance, sq_grad, tenengrad, thres_abs_grad, thres_pix_count, variance,
              variance_log_hist, vollath4, vollath5]

names = ('Absolute Tenengrad', 'Brener Gradient', 'Entropy', 'First Gaussian Derivative', 'Image Power', 'Laplacian',
         'Normalized Variance', 'Square Gradient', 'Tenengrad', 'Thresholded Absolute Gradient',
         'Thresholded Pixel Count', 'Variance',
         'Variance of log Histogram', 'Vollath4', 'Vollath5')

# ----------------------------------

			
imagenes = [[[17, 19, 20, 17, 18, 20, 17, 23, 20, 18], [11, 8, 10, 11, 10]],
            [[15, 19, 19, 19, 16, 16, 16, 14, 15, 14], [9, 12, 12, 12, 11, 11, 10, 11, 11, 12]],
            [[12, 14, 13, 13, 17, 16, 15, 14, 15, 13], [11, 12, 12, 12, 11, 11, 12, 12, 11, 12]],
            [[14, 15, 14, 15, 13, 13, 12, 14, 13, 12], [11, 13, 11, 11, 14, 10, 13, 12, 13, 11]]]


maximos = [[[10, 10, 9, 7, 9, 10, 8, 13, 12, 9], [7, 4, 5, 7, 5]],
            [[10, 15, 14, 14, 12, 10, 11, 8, 11, 10], [5, 8, 8, 7, 6, 7, 5, 5, 6, 7]],
            [[9, 12, 11, 10, 12, 12, 10, 10, 12, 9], [6, 6, 7, 6, 7, 6, 7, 7, 6, 7]],
            [[9, 9, 9, 10, 7, 8, 8, 9, 8, 8], [6, 8, 6, 6, 9, 5, 8, 7, 7, 7]]]

for tej in range(4):  # tejidos 4
    for mag in range(2):  # magnif 2
        for prof in range(2):  # profund 2
            if (tej == 0) and (mag == 1):
                num_stacks = 5
            else:
                num_stacks = 10

            crit_table = np.array([[[0. for a in range(num_criteria)] for b in range(num_algs)] for c in range(num_stacks)])
            rank_table = np.array([[[0. for a in range(num_criteria)] for b in range(num_algs)] for c in range(num_stacks)])

            ideal_func = np.array([[0. for a in range(num_criteria)] for c in range(num_stacks)])
            to_norm_table = np.array([[[0. for a in range(num_criteria)] for b in range(num_algs + 1)] for c in range(num_stacks)])
            norm_table = np.array([[[0. for a in range(num_criteria)] for b in range(num_algs + 1)] for c in range(num_stacks)])
            dist_table = np.array([[[0. for a in range(num_criteria)] for b in range(num_algs)] for c in range(num_stacks)])

            if prof == 0:
                thres = [0, 51, 0, 0, 51, 0, 0, 51, 0, 51, 51, 0, 0, 0, 0] # (20%)
                b = 8
                hist_range = (2 ** b) - 1
            else:
                thres = [0, 13105, 0, 0, 13105, 0, 0, 13105, 0, 13105, 13105, 0, 0, 0, 0] # (20%)
                b = 16
                hist_range = (2 ** b) - 1

            path = direct_img + tejido[tej] + magnificacion[mag] + profundidad[prof]
            path_res = direct_res + tejido[tej] + magnificacion[mag] + profundidad[prof]
            num_imgs = imagenes[tej][mag]
            real_maxs = maximos[tej][mag]
            start_analysis = (time.time() * 1000)

            start_algs = (time.time() * 1000)
            play_algs()
            end_algs = (time.time() * 1000)

            start_semi = (time.time() * 1000)
            semi_quant(crit_table)
            end_semi = (time.time() * 1000)

            start_quant = (time.time() * 1000)
            quant(crit_table)
            end_quant = (time.time() * 1000)

            end_analysis = (time.time() * 1000)

print("end_algs1 - start_algs1 =", end_algs - start_algs)
print("end_semi1 - start_semi1 =", end_semi - start_semi)
print("end_quant1 - start_quant1 = ", end_quant - start_quant)
print("end_analysis1 - start_analysis1 = ", end_analysis - start_analysis)


