# Autofocus URJC

This is an open source project to develop a fully automated focusing system.

INTRODUCTION

The original idea came up at the URJC with the QUASAR project, developed in collaboration with Cytognos S.L. and UNAV. The project aimed to create an hiperspectral histo-citometer which, in a few words, is like a fancy microscope.

This inspirated us to develop a general methodology and any-kind-of-application implementation for an autofocus solution. We are in the process of publishing a paper about this, but long story short, to focus a sample you need to sweep each field of view though the focal range to take a stack of images. Then, you use these images to calculate the contrast function of the stack, and its maximum would indicate the best focused position.

How do you calculate the contrast function of the stack? With a contrast calculation algorithm, of course! But there is a problem: there are many contrast calculation algorithms available, and the optimal one for a certain use varies with the sample and its content.

Then, we have to develop some kind of tool capable of selecting the best algorithms to use in a given application. And this is our first step: the creation of an automated analysis that studies a series of stacks with a set of algorithms. The analysis evaluates the performance of the algorithms and grades them, giving as a result a global ranking of those algorithms.

This is only the first step, and at this point our goal is to test the efectiveness of the analysis and its reliability. For this study we used 150 stacks of images, from 4 different types of mouse tissue (adipose, stomach, intestine and kidney), using 2 magnification settings (5x and 10x), and 2 bit-depth configurations (8 bits and 16 bits).

Those images were studied with 15 algorithms and evaluated according 4 criteria (accuracy, range, false maxima and FWHW) by 2 different analysis (semi-quantitative and quantitative), and the developed tool for this implementation, along with the datasets studied, and the results ara availabe here. More details can be found in the following paragraphs.


MATHERIALS

1- Images:

  The images used to test our project can be found in the following link:
  
  https://urjc-my.sharepoint.com/:f:/g/personal/marina_bsanz_urjc_es/EjYdL3WWrCBBpagt8AJ16Q0BFZuvUiw87Da-EJ6MptLGLw?e=qyPgA5
  
  There are 150 stacks of images, taken from 4 different types of mouse tissue (adipose, stomach, intestine and kidney), using 2 magnification settings (5x and 10x), and 2 bit-depth configurations (8 bits and 16 bits). The images are .tif in greyscale and have a resolution of 1388x1040.
  
  The filing tree used is as follows, and can be changed, but it is important to change the path-generating arrays in the code (tissue, magnification and depth). Also, notice that the stacks' folders are named "S" + the number of the stack, starting at 1.
  
  There are 10 stacks in each category, except for Adipose tissue 10x (both 8 and 16 bits), in which there are only 5. The stacks have a variable number of images, as they were taken manually, of arround 10 to 20 images per stack.
  
  - Tissue 1 (Adipose)
  	- Magnification 1 (5x)
		- Bit depth 1 (8 bits)
			- Stack 1 (S1)
			- Stack 2 (S2)
			- ...
		- Bit depth 2 (16 bits)
			- Stack 1 (S1)
			- Stack 2 (S2)
			- ...
	- Magnification 2 (10x)
		- Bit depth 1 (8 bits)
			- Stack 1 (S1)
			- Stack 2 (S2)
			- ...
		- Bit depth 2 (16 bits)
			- Stack 1 (S1)
			- Stack 2 (S2)
			- ...
  - Tissue 2 (Stomach)
  	- Magnification 1 (5x)
		- Bit depth 1 (8 bits)
		- Bit depth 2 (16 bits)
	- Magnification 2 (10x)
		- Bit depth 1 (8 bits)
		- Bit depth 2 (16 bits)

  ...

  
2- Code

All code has been developed in Python.
  1. algorithms.py
  
    Here are included the python codification of all the algorithms evaluated, optimized to perform matrix calculations when possible.
    More algorithms can be added, but some modifications should be made in the nexts script.
	The algorithms used are:
	Absolute Tenengrad, Brener Gradient, Entropy, First Gaussian Derivative, Image Power, Laplacian, Normalized Variance, Square Gradient, Tenengrad, Thresholded Absolute Gradient, Thresholded Pixel Count, Variance, Variance of log Histogram, Vollath4, Vollath5
	
  2. top.py
  
    The full analysis is performed here.
	It is important before running the analysis to check, fill and uncomment the parameters at the top, such as the size of the images, their path, the number of criteria under evaluation, the number of algoritms...
	There are also 4 arrays to check:
	
		- imagenes: it contains the number of images in each stack
		
		- maximos: it contains the positions of the maximums determined by an expert
		
		These both have the following structure:
		
		[ [Adipose, [5x, 10x]],
		
		  [Stomach, [5x, 10x]],
		  
		  [Intestine, [5x, 10x],
		  
		  [Kidney, [5x, 10x] ]
		  
		- algorithms: it contains the names of the functions given to the algorithms in algorithms.py. It is used to call the algorithms un a for loop
		
		- names: it contaims the names of the algorithms in the SAME ORDER as they are called. It is used to fill the .txt with the results
		
	If new algorithms are added, remember to update the arrays algorithms and names, and the variable num_algs
	
	If the path to the results is different than the path to the images, a folder structure as the one used for the images is expected by the script, and therefore, must be created.
	The results given by this stept are:
	
		- A data<>.txt for each stack, containing the normalized contrast values resulting of the application of the algorithms (contrast functions)
		
		- A crit_table<>.txt for each stack, with the evaluation of the contrast functions according to the evaluation criteria (Accuracy, Range, False maxima, FWHW)
		
		- A semi_table.txt for a set of stacks in the same tissue/magnification/bit depth category, with the results of the semi-quantitative analysis, its global scores and the ranking.
		
		- A quant_table.txt for a set of stacks in the same tissue/magnification/bit depth category, with the results of the quantitative analysis, its global scores and the ranking.
  
  3. excel.py
  
  This script transforms the .txt documents produced by the analysis into excel spreadsheets.
  The results will be the same, but they will be easier to read, as the spreadsheets feature a graphic of the contrast functions for each stack.
  
  
  
3- Results:

  As mentioned, for each tissue/magnification/bit depth category the top.py script will return:
  
	- A data<>.txt for each stack, containing the contrast values resulting of the application of the algorithms (contrast functions)
	
	- A crit_table<>.txt for each stack, with the evaluation of the contrast functions according to the evaluation criteria (Accuracy, Range, False maxima, FWHW)
	
	- A semi_table.txt with the results of the semi-quantitative analysis, its global scores and the ranking.
	
	- A quant_table.txt with the results of the quantitative analysis, its global scores and the ranking.
	
  And those can be transform into excel spreadsheets by excel.py. This will result in a file for each tissue/magnification category, with the following features:
  
	- A sheet for each stack, featuring the normalized contrast values of the contrast functions, the criteria evaluation of these functions, a graphic, the semi-quantitative evaluation of the single stack, and the calculations needed to obtain the squared euclidean distances for the quantitative analysis.
	
	- A summary sheet for each bit depth (8 bits and 16 bits), featuring the semi-quantitative evaluation of all the single stacks, the squared euclidean distances of all the single stacks, and the semi-quantitative and quantitative total soceres, global socres and ranking.
  
