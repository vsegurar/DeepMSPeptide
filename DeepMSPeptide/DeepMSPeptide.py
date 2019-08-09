import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras


parser = argparse.ArgumentParser(description='''Predicts the detectability of input peptides using a single dimension
                                                Convolutionar Neural Network, based on Tensorflow 1.13.1
                                                Requierements: Tensorflow 1.13.1''')
parser.add_argument('infile', metavar='F', type=str, nargs='+',
                    help='File containing the peptides to be predicted, one per line (max length= 81)')
args = parser.parse_args()


def load_pep_and_codify(file, max_len):
    aa_dict={'A':1,'R':2,'N':3,'D':4,'C':5,'Q':6,'E':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,
        'P':15,'O':16,'S':17,'U':18,'T':19,'W':20,'Y':21,'V':22}
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
    pep_codes=[]
    long_pep_counter = 0
    newLines = []
    for pep in lines:
        if not len(pep) > max_len:
            current_pep=[]
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(current_pep)
            newLines.extend([pep])
        else:
            long_pep_counter += 1
    predict_data = keras.preprocessing.sequence.pad_sequences(pep_codes, value=0, padding='post', maxlen=max_len)
    return predict_data, long_pep_counter, newLines


print('Loading model...')
model_2_1D = keras.models.load_model('model_2_1D.h5')

print('Loading input peptides')
predict_data, skipped,  lines = load_pep_and_codify(args.infile[0], 81)
print('Succesfully loaded {0} peptides and skipped {1}'.format(len(lines), str(skipped)))

print('Making predictions')
model_2_1D_pred = model_2_1D.predict(predict_data)
model_2_1D_pred = np.hstack((np.array(lines).reshape(len(lines), 1),model_2_1D_pred)).tolist()

Pred_output = []
for pred in model_2_1D_pred:
    if float(pred[1]) > 0.5:
        # pred.extend('0')
        Pred_output.append([pred[0], str(1-float(pred[1])), '0'])
    else:
        Pred_output.append([pred[0], str(1-float(pred[1])), '1'])
        # pred.extend('1')

outFile = '{0}_Predictions.txt'.format(args.infile[0].split('.')[0])
print('Saving predictions to file {}'.format(outFile))
with open(outFile, 'w') as outf:
    outf.write('Peptide\tProb\tDetectability\n')
    outf.writelines('\t'.join(i) + '\n' for i in Pred_output)
