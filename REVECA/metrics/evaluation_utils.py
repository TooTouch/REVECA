import json

from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.cider.cider import Cider
from .pycocoevalcap.spice.spice import Spice


def split_pred_parts(pred):
    pred_dict = dict(
        subject=dict(),
        before=dict(),
        after=dict()
    )
    for bid, cap in pred.items():
        part_list = cap.split('//')

        if len(part_list) < 1:
            pred_dict['subject'][bid] = [' ']
        else:
            pred_dict['subject'][bid] = [part_list[0].split(':')[-1].strip()]
        if len(part_list) < 2:
            pred_dict['before'][bid] = [' ']
        else:
            pred_dict['before'][bid] = [part_list[1].split(':')[-1].strip()]
        if len(part_list) < 3:
            pred_dict['after'][bid] = [' ']
        else:
            pred_dict['after'][bid] = [part_list[2].split(':')[-1].strip()]
    return pred_dict


def split_gt_parts(gt):
    gt_dict = dict(
        subject=dict(),
        before=dict(),
        after=dict()
    )
    for bid, cap in gt.items():
        part_list = cap[0].split('//')
        gt_dict['subject'][bid] = [part_list[0].split(':')[-1].strip()]
        gt_dict['before'][bid] = [part_list[1].split(':')[-1].strip()]
        gt_dict['after'][bid] = [part_list[2].split(':')[-1].strip()]
    return gt_dict


def evaluate_on_caption(pred_dict, gt_dict, outfile=None):
    Eval = EvalCap(pred_dict, gt_dict, 'corpus')

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    Eval.evaluate()
    result = Eval.eval
    
    return result



class EvalCap:
    def __init__(self, pred_dict, gt_dict, df):
        self.evalBoundaries = []
        self.eval = dict()
        self.BoundariesToEval = dict()

        self.gts = gt_dict
        self.res = pred_dict
        self.df = df

    def tokenize(self):

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        self.gts = tokenizer.tokenize(self.gts)
        self.res = tokenizer.tokenize(self.res)

    def evaluate(self):
        self.tokenize()

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(self.df), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setBoundaryToEvalBoundaries(scs, self.gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setBoundaryToEvalBoundaries(scores, self.gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalBoundaries()

    def setEval(self, score, method):
        self.eval[method] = score

    def setBoundaryToEvalBoundaries(self, scores, b_ids, method):
        for b_id, score in zip(b_ids, scores):
            if not b_id in self.BoundariesToEval:
                self.BoundariesToEval[b_id] = dict()
                self.BoundariesToEval[b_id]["boundary_id"] = b_id
            self.BoundariesToEval[b_id][method] = score

    def setEvalBoundaries(self):
        self.evalBoundaries = [eval for imgId, eval in self.BoundariesToEval.items()]

