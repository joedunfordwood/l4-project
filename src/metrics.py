from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
from training import optimise_attribution
from utils import _batch_eval_probs, filter_nan
import torch.nn.functional as F

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# model metrics

def display_metrics(results, dataset, labels):
  assert len(results) == 2, f"results must be an array of size 2 where the first index corresponds to output logits and the second index is the ground truth values"
  print(f"{dataset} Metrics")
  logits = results[0]
  true = results[1]

  preds = np.argmax(logits.numpy(), axis=1)

  macro_f1 = metrics.f1_score(true, preds, average='macro')
  acc = metrics.accuracy_score(true, preds)
  macro_precision = metrics.precision_score(true, preds, average='macro')
  macro_recall = metrics.recall_score(true, preds, average='macro')

  weighted_f1 = metrics.f1_score(true, preds, average='weighted')
  balanced_acc= metrics.balanced_accuracy_score(true, preds)
  weighted_precision = metrics.precision_score(true, preds, average='weighted')
  weighted_recall = metrics.recall_score(true, preds, average='weighted')



  pc_precision = metrics.precision_score(true, preds, average=None)
  pc_recall = metrics.recall_score(true, preds, average=None)

  probs = F.softmax(logits, dim=1)
  auroc = metrics.roc_auc_score(true, probs, multi_class='ovr')

  cm = metrics.confusion_matrix(true, preds, labels=range(len(labels)))
  disp_cm = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
  disp_cm.plot()
  plt.title(f"{dataset} Confusion Matrix")

  plt.show()


  print(f"\n{dataset} Macro F1: {macro_f1:.4f}\t Weighted F1: {weighted_f1:.4f}\n Accuracy: {acc:.4f}\t Balanced Accuracy: {balanced_acc:.4f} \
          \n{dataset} Macro Precision: {macro_precision:.4f}\t Weighted Precision: {weighted_precision:.4f} \
          \n{dataset} Macro Recall: {macro_recall:.4f}\t Weighted Recall: {weighted_recall:.4f}\n{dataset} AUROC: {auroc:.4f} ")
  print(f"\nPer Class Precision: {pc_precision}\nPer Class Recall {pc_recall}")


  return {'macro_f1': macro_f1,
          'weighted_f1': weighted_f1,
          'acc': acc,
          'balanced_acc': balanced_acc,
          'macro_precision': macro_precision,
          'weighted_precision': weighted_precision,
          'macro_recall':macro_recall,
          'weighted_recall':weighted_recall,
          'auroc':auroc,
          'pc_precision': pc_precision,
          'pc_recall': pc_recall,
          'true':true,
          'preds': preds}


# sloc metrics

def deletion_insertion_auc(model,input,attr,device,target_class,steps=100,batch_size=64):


    C, T = input.shape
    N = C * T # number of total samples
    flat_attr = np.array(attr).flatten()
    # descending importance
    order = np.argsort(-flat_attr)

    fractions = np.linspace(0.0, 1.0, steps)
    ins_inputs = []
    del_inputs = []

    input_copy = input.clone().cpu()
    baseline = torch.full_like(input_copy, float(0))

    # Build inputs for each fraction (small memory footprint: steps * (C*T) floats)

    for f in fractions:
        k = int(np.round(f * N))

        # Insertion: start from baseline, copy top-k features from original


        ins = baseline.clone().reshape(-1)
        if k > 0:
            ins[order[:k]] = input_copy.reshape(-1)[order[:k]]
        ins_inputs.append(ins.reshape_as(input_copy).unsqueeze(0))

        # Deletion: start from original, set top-k features to baseline
        dele = input_copy.clone().reshape(-1)
        if k > 0:
            dele[order[:k]] = baseline.reshape(-1)[order[:k]]
        del_inputs.append(dele.reshape_as(input_copy).unsqueeze(0))

    ins_batch = torch.cat(ins_inputs, dim=0)  # (steps, C, T)
    del_batch = torch.cat(del_inputs, dim=0)

    # Evaluate model probabilities for target class
    ins_probs = _batch_eval_probs(model, ins_batch, device, target_class, batch_size=batch_size)
    del_probs = _batch_eval_probs(model, del_batch, device, target_class, batch_size=batch_size)

    # AUC via trapezoidal rule over fractions
    ins_auc = float(np.trapezoid(ins_probs, fractions))
    del_auc = float(np.trapezoid(del_probs, fractions))

    return {
        "fractions": fractions,
        "ins_curve": ins_probs,
        "del_curve": del_probs,
        "ins_auc": ins_auc,
        "del_auc": del_auc
    }

def gen_avgs(device, test_model, dataset, sloc_hyperparams, del_ins_params, log = True):

  ins_scores = np.zeros(len(dataset))
  del_scores = np.zeros(len(dataset))


  attributions = np.zeros((len(dataset), 12,500))


  for sample in range(len(dataset)):

    input = dataset[sample][0]
    truth = dataset[sample][1]


    # Optimise attribution
    attribution, loss_hist = optimise_attribution(device,input,test_model,
                                                  sloc_hyperparams["nmasks"],
                                                  sloc_hyperparams["batch_size"],
                                                  sloc_hyperparams["segsize"],
                                                  sloc_hyperparams["prob"],
                                                  sloc_hyperparams["epochs"],
                                                  sloc_hyperparams["lr"],
                                                  sloc_hyperparams["tv_eps"],
                                                  sloc_hyperparams["l1_eps"],
                                                  sloc_hyperparams["norm"],
                                                  label=truth)


    attr_np = attribution.attr_map.detach().cpu().numpy()


    res = deletion_insertion_auc(model=test_model,
                                 input=input,
                                 attr=attr_np,
                                 device=device,
                                 target_class=truth,
                                 steps=del_ins_params["steps"],
                                 batch_size=del_ins_params["batch_size"])

    ins_scores[sample] = res["ins_auc"]
    del_scores[sample] = res["del_auc"]

    attributions[sample,:,:] = attr_np
    if log and sample%10==0:
      print(f"Sample: {sample}\nINS AUC: {res["ins_auc"]:.4f}\tDEL AUC: {res["del_auc"]:.4f}")
    #print(f"Running AVG INS: {float(np.mean(ins_scores)):.4f}\tRunning AVG DEL: {float(np.mean(del_scores)):.4f}")

    del attribution, attr_np

  return {
      "insauc_avg": float(np.mean(ins_scores)),
      "insauc_std": float(np.std(ins_scores)),
      "ins_aucs": ins_scores,
      "delauc_avg": float(np.mean(del_scores)),
      "delauc_std": float(np.std(del_scores)),
      "del_aucs": del_scores,
      "attributions": attributions
  }

def lead_stats(attr_block):

    means = attr_block.mean(axis=(0, 2))  
    stds  = attr_block.std(axis=(0, 2))   
    order = np.argsort(-means)
    return {
        "means":   means,
        "stds":    stds,
        "ranking": [LEAD_NAMES[i] for i in order],
    }


def superpixel_stats(attr_block, sp_size=5):


    S, L, T = attr_block.shape
    assert T % sp_size ==0, 'superpixel size has to be multiple of number of samples'
    num_sp = int(T / sp_size)

    sp = attr_block[:, :, :num_sp * sp_size].reshape(S, L, num_sp, sp_size).mean(axis=-1)  

    return {
        "sp_size":        sp_size,
        "global_means":   sp.mean(axis=(0, 1)),   
        "global_stds":    sp.std(axis=(0, 1)),
        "per_lead_means": sp.mean(axis=0),         
        "per_lead_stds":  sp.std(axis=0),
    }





def analyse_saliency(attr_maps, sp_size=5, label_names=None):

    num_labels, num_samples, num_leads, T = attr_maps.shape
    if label_names is None:
        label_names = [f"Label {i}" for i in range(num_labels)]

    per_label = []
    clean_blocks = []
    for i in range(num_labels):
        block, nan_count, _ = filter_nan(attr_maps[i]) 
        clean_blocks.append(block)
        per_label.append({
            "name":             label_names[i],
            "nan_count":        nan_count,
            "nan_pct":          nan_count / num_samples * 100,
            "valid_count":      num_samples - nan_count,
            "avg_attr_map":     block.mean(axis=0),  
            "lead_stats":       lead_stats(block),
            "superpixel_stats": superpixel_stats(block, sp_size),
        })

    all_samples = np.concatenate(clean_blocks, axis=0)   
    total = num_labels * num_samples
    total_nan = sum(lr["nan_count"] for lr in per_label)
    global_stats = {
        "nan_count":        total_nan,
        "nan_pct":          total_nan / total * 100,
        "valid_count":      total - total_nan,
        "avg_attr_map":     all_samples.mean(axis=0), 
        "lead_stats":       lead_stats(all_samples),
        "superpixel_stats": superpixel_stats(all_samples, sp_size),
    }

    return {"per_label": per_label, "global": global_stats}

def std_per_label(attr_maps, label_names=None):

  num_labels, num_samples, _, _ = attr_maps.shape


  if label_names is None:
    label_names = [f'Label {i}' for i in range(num_labels)]

  per_label = []
  for i in range(num_labels):
    std_map = {}
    att_block, _, _ = filter_nan(attr_maps[i])
    std_map['name'] = label_names[i]
    std_map['std_map'] = att_block.std(axis=0)

    per_label.append(std_map)

  return per_label



def cross_label_discriminability(result):

  # subtracts the global average map from each label's avg map to isolate class-specific signal
  # coeff of variation across labels per pixel identifies most discriminative regions
  global_avg = result['global']['avg_attr_map']                               
  label_maps = np.stack([lr['avg_attr_map'] for lr in result['per_label']])  

  diff_maps   = label_maps - global_avg   
  across_std  = label_maps.std(axis=0)   
  across_mean = label_maps.mean(axis=0) 
  cov_map     = across_std / (across_mean + 1e-12)  

  return {
    'diff_maps': diff_maps,  
    'cov_map':   cov_map,    
    'std_map':   across_std, 
  }