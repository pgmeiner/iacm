-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-

                      		STARTING KIT FOR CHALEARN
				Fast Causation Coefficient Challenge
							March 2014
                                  
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

ce-mini-submission.zip: sample submission ready to submit to Codalab.
input_valid/CE*: Sample data similar to the data available on Codalab for testing.
output_valid/ref/CE*: Sample data similar to the target values (solution) available on Codalab for testing.
score.py: The scoring program used on the Codalab platform.

INSTRUCTIONS:
1) Submit the sample submission to the platform to check everything works fine. Make a copy of it for your records.

2) Install Anaconda 1.6.2 from http://repo.continuum.io/archive/.

3) Unzip ce-mini-submission.zip in the current directory (you should get the following files: basic_python_benchmark, predict.py, features.py, and metadata).

4) Start a shell terminal / OS prompt and type at the prompt:
> python predict.py input_valid output_valid/res

This command computes the predicted value for a simple causation coefficient for the data found in the input_valid directory. If you look at the code, you will notice that the code is looking for 2 types of files in input_valid/:
[basename]_pairs.csv: the cause-effect pairs.
[basename]_publicinfo.csv: information on the type of variables (binary, numerical, categorical).
You may name the files with any [basename], as long as they have rith correct ending and as long as you just have one of each type. DO NOT CHANGE THIS INTERFACE: this is what the Codalab server expects.

The results are deposited in the directory output_valid/res/, under the name [basename]_predict.csv. RESPECT THAT NOMENCLATURE: this is what the Codalab server expects.

5) Type at the prompt:
> python score.py output_valid score_valid
This command computes the score obtained by computing the AUC (forward, backward and averaged) using the predictions found in output_valid/res/ and comparing them with the solution found in output_valid/ref/. The results are scored in score_valid/scores.txt.

6) You may modify the code as you like as long as you do not change the input/output interface. To retrain or modify the training program, download the original code from https://github.com/benhamner/CauseEffectPairsChallenge.





