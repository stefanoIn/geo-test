import json
import math
from pathlib import Path
import numpy as np

METRIC_KEYS = ('accuracy', 'water_iou', 'nonwater_iou', 'miou', 'precision', 'recall',
               'f1', 'omission_error', 'commission_error', 'specificity', 'false_positive_rate',
               'balanced_accuracy', 'mcc', 'ground_truth_water_fraction', 'predicted_water_fraction',
               'TP', 'FP', 'FN', 'TN', 'valid_pixels')


def binary_metrics(tp, fp, fn, tn):
    tp, fp, fn, tn = map(int, (tp, fp, fn, tn))
    if min(tp, fp, fn, tn) < 0:
        raise ValueError('Confusion counts must be nonnegative.')
    def div(a, b): return a / b if b else float('nan')
    total = tp + fp + fn + tn
    wi, ni = div(tp, tp+fp+fn), div(tn, tn+fp+fn)
    recall, specificity = div(tp, tp+fn), div(tn, tn+fp)
    return dict(accuracy=div(tp+tn,total), water_iou=wi, nonwater_iou=ni, miou=(wi+ni)/2,
                precision=div(tp,tp+fp), recall=recall, f1=div(2*tp,2*tp+fp+fn),
                omission_error=div(fn,tp+fn), commission_error=div(fp,tp+fp),
                specificity=specificity, false_positive_rate=div(fp,fp+tn),
                balanced_accuracy=(recall+specificity)/2,
                mcc=div(tp*tn-fp*fn,math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),
                ground_truth_water_fraction=div(tp+fn,total), predicted_water_fraction=div(tp+fp,total),
                TP=tp, FP=fp, FN=fn, TN=tn, valid_pixels=total)


def confusion_counts(prediction, target, ignore_index=255):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError('Prediction and target shapes differ.')
    valid = (target != ignore_index) & (target != -1)
    if not np.isin(target[valid], [0,1]).all() or not np.isin(prediction, [0,1]).all():
        raise ValueError('Expected binary predictions and 0/1/ignored labels.')
    return tuple(int(np.count_nonzero(valid & (prediction==p) & (target==t)))
                 for p,t in [(1,1),(1,0),(0,1),(0,0)])


def clean_json(value):
    if isinstance(value, dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.generic): return clean_json(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def metric_table(split_metrics, output_dir=None):
    import pandas as pd
    table = pd.DataFrame.from_dict(split_metrics,orient='index')
    table.index.name='split'
    if output_dir is not None:
        output_dir=Path(output_dir); output_dir.mkdir(exist_ok=True,parents=True)
        table.to_csv(output_dir/'metrics_summary.csv')
        save_json(output_dir/'metrics_summary.json',split_metrics)
    return table


def group_metrics(rows, group_key):
    import pandas as pd
    frame=pd.DataFrame(rows)
    result=[]
    for name,group in frame.groupby(group_key,sort=True):
        counts=[int(group[key].sum()) for key in ('TP','FP','FN','TN')]
        result.append({group_key:name,'chips':len(group),**binary_metrics(*counts)})
    return pd.DataFrame(result)


def save_per_chip_metrics(rows, output_dir):
    import pandas as pd
    output_dir=Path(output_dir)
    pd.DataFrame(rows).to_csv(output_dir/'metrics_per_chip.csv',index=False)
    if rows: group_metrics(rows,'region').to_csv(output_dir/'metrics_per_region.csv',index=False)


def plot_metric_history(history, output_dir):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for key in ('precision','recall','f1','accuracy'):
        values=history.get('val_'+key,[])
        points=[(i+1,v) for i,v in enumerate(values) if v is not None]
        if points: axes[0].plot(*zip(*points),marker='o',label=key)
    for key in ('omission_error','commission_error'):
        values=history.get('val_'+key,[])
        points=[(i+1,v) for i,v in enumerate(values) if v is not None]
        if points: axes[1].plot(*zip(*points),marker='o',label=key)
    for ax in axes:
        ax.set(xlabel='Epoch',ylabel='Fraction',ylim=(0,1)); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(Path(output_dir)/'validation_all_metrics.png',dpi=160)
    return fig
