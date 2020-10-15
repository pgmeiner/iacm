#!/usr/bin/env python

# Score calculation script.
# Original code: Ben Hamner, Kaggle, March 2013
# https://github.com/benhamner/CauseEffectPairsChallenge
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013
# Modified by Isabelle Guyon, ChaLearn, March 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, KAGGLE, MOCROSOFT AND/OR OTHER ORGANIZERS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

import sys
import os
import os.path
import glob
import pandas as pd

from sklearn.metrics import auc_score

# Implement the symmetric score of the challenge
def forward_auc(labels, predictions):
    target_one = [1 if x==1 else 0 for x in labels]
    score = auc_score(target_one, predictions)
    return score

def reverse_auc(labels, predictions):
    target_neg_one = [1 if x==-1 else 0 for x in labels]
    neg_predictions = [-x for x in predictions]
    score = auc_score(target_neg_one, neg_predictions)
    return score

def bidirectional_auc(labels, predictions):
    score_forward = forward_auc(labels, predictions)
    score_reverse = reverse_auc(labels, predictions)
    score = (score_forward + score_reverse) / 2.0
    return score

if __name__=="__main__":

	# The input directory contains both the causation coefficients predicted (in subdirectory res)
	# and the truth values of the causal relationship or "solution" (in subdirectory ref)
	input_dir = sys.argv[1]

	# The output directory will contain the scores
	output_dir = sys.argv[2]

	# Get the solution from the ref subdirectory (must end with 'solution.csv')
	# There should be a single solution
	solution_name = glob.glob(os.path.join(input_dir, 'ref', '*solution.csv'))
	if len(solution_name)!=1:
		print('No or multiple solutions')
		exit(1)
	print("Reading " + solution_name[0]) 
	solution = pd.read_csv(solution_name[0], index_col="SampleID")
	
	# Get the submission from the ref subdirectory (must end with 'predict.csv')
	# There should be a single result.
	predict_name = glob.glob(os.path.join(input_dir, 'res', '*predict.csv'))
	if len(predict_name)!=1:
		print('No or multiple prediction files')
		exit(1)
	print("Reading " + predict_name[0]) 
	submission = pd.read_csv(predict_name[0], index_col="SampleID")

	# Compute scores (forward, backward, and symmetric)
	score_forward = forward_auc(solution.Target, submission.Target)
	print("Forward AUC: %0.6f" % score_forward)

	score_reverse = reverse_auc(solution.Target, submission.Target)
	print("Reverse AUC: %0.6f" % score_reverse)

	score = bidirectional_auc(solution.Target, submission.Target)
	print("Bidirectional AUC: %0.6f" % score)

	# Write scores to the output file
	try:
		os.stat(output_dir)
	except:
		os.mkdir(output_dir) 
	with open(os.path.join(output_dir, 'scores.txt'), 'wb') as f:
		f.write("ForwardAUC: %0.6f\n" % score_forward)
		f.write("ReverseAUC: %0.6f\n" % score_reverse)
		f.write("BidirectionalAUC: %0.6f\n" % score)

	exit(0)




