-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-

                       DATA AND SAMPLE MATLAB CODE FOR THE
                      CHALEARN CAUSE-EFFECT PAIR CHALLENGE
   
              Isabelle Guyon -- isabelle@clopinet.com -- March-June 2013
                                  
-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-

DISCLAIMER: ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS" 
ISABELLE GUYON AND/OR OTHER CONTRIBUTORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR ANY PARTICULAR PURPOSE, AND THE WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S 
INTELLECTUAL PROPERTY RIGHTS. IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER CONTRIBUTORS 
BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, 
MATERIALS, PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

CONTENTS:

	SAMPLE SUBMISSIONS AN CODE

CEdata_baseline_submission.csv: Sample submission for https://www.kaggle.com/c/cause-effect-pairs
Sample_code.zip: Sample Matlab code.
Python sample code is found at: https://github.com/benhamner/CauseEffectPairsChallenge

	FINAL DATA RELEASE JUNE 2013

CEfinal_train_text.zip: Validation data in csv format 
CEfinal_train_split.zip: Training data in split files format (one file per pair of variables)	
CEfinal_valid_text.zip: Validation data in csv format 
CEfinal_valid_split.zip: Validation data in split files format

ENCRYPTED DATA (ENCRYPTION KEY TO BE RELEASED AT THE CHALLENGE DEADLINE
CEfinal_test_text.zip: Final evaluation data in csv format 
CEfinal_test_split.zip: Final evaluation data in split files format (one file per pair of variables)	
[-------- WARNING : text and split versions are identical, download only one -------]

	TRAINING/VALIDATION DATA MARCH 2013

The original data release was moved to the ORIG_RELEASE/ directory.

CEdata_matlab.zip: Original train/valid data in Matlabformat	 
CEdata_split.zip: Original train/valid data in split format (one file per pair of variables)	 
CEdata_text.zip: Original train/valid data in csv format
PUBLIC/	: same as compressed archive CEdata_text.zip
[-------- It was noted that data formatting artifacts exist in these data that were rectified in later releases: the distribution of values is uneven in the 4 causal categories -------]
[-------- WARNING : ALL 4 versions above are IDENTICAL, download only one -------]
	 
	MORE TRAINING DATA MAY 2013

SUP1 data: Numerical variables normalized to mean 0 and std 10000 then rounded to nearest integer
SUP1data_split.zip: Supplementary training data 1 in split format 
SUP1data_texr.zip: Supplementary training data 1 in csv format 

SUP2 data: Mixed variables (continuous, discrete, categorical, binary) normalized to mean 0 and std 10000 then rounded to nearest integer
The number of categories and unique values are balanced in the 4 causal categories.
SUP2data_split.zip: Supplementary training data 2 in split format 
SUP2data_text.zip: Supplementary training data 2 in csv format 
[-------- WARNING : text and split versions are identical, download only one -------]