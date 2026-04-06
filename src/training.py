import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from utils import normalise_attr

from models.sloc import AttributionMap, GenMaskResponse, TotalVariationLoss

# Model training

def train_model(model, optimiser, loss_fn, train_dl, valid_dl, regularisation_type=None, lam=0.01, num_epochs=50):
    loss_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs

    best_loss = 1000

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:

            x_batch = x_batch.cuda(non_blocking=True)
            y_batch = y_batch.cuda(non_blocking=True)

            pred = model(x_batch)

            #print(pred)
            #print(pred.shape)

            loss = loss_fn(pred, y_batch)


            # Apply L1 regularization
            if regularisation_type == 'L1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += lam * l1_norm

            # Apply L2 regularization
            elif regularisation_type == 'L2':
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                loss += lam * l2_norm


            loss.backward() # compute gradient based on results of all predictions made by the model given the inputs, in this case the batches

            optimiser.step()

            optimiser.zero_grad() # resets gradient

            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()

        loss_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()

        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.cuda(non_blocking=True)
                y_batch = y_batch.cuda(non_blocking=True)

                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

                 # Apply L1 regularization
                if regularisation_type == 'L1':
                  l1_norm = sum(p.abs().sum() for p in model.parameters())
                  loss += lam * l1_norm

                # Apply L2 regularization
                elif regularisation_type == 'L2':
                  l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                  loss += lam * l2_norm

                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)

        if loss_hist_valid[epoch] < best_loss:
          best_loss = loss_hist_valid[epoch]
          best_epoch = epoch

          best_model = model.state_dict()
          print('Model Saved')



        print(f'Epoch {epoch} Train_loss = {loss_hist_train[epoch]}, Valid_loss = {loss_hist_valid[epoch]}')


        if not loss_hist_valid[epoch]:
          del y_batch, x_batch, loss, pred
          torch.cuda.empty_cache()
          return loss_hist_train, loss_hist_valid, best_loss, best_epoch, best_model


        #if epoch > 10: torch.save(model.state_dict(), 'model.pth')

        del y_batch, x_batch, loss, pred
        torch.cuda.empty_cache()


    #torch.save(model.state_dict(), 'model.pth')
    #torch.save(best_model, 'best_model.pth')


    return loss_hist_train, loss_hist_valid, best_loss, best_epoch, best_model

def fit(train_dataset, valid_dataset, weighting, model, optimiser, batch_size, regularisation, device, num_epochs=50):
    
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True)
    
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weighting.to(device))

    loss_hist_train, loss_hist_valid, best_loss, best_epoch, best_model = train_model(model, optimiser, loss_fn, train_dl, valid_dl, regularisation_type=regularisation, num_epochs=num_epochs, lam=0.0005)

    return model, loss_hist_train, loss_hist_valid, best_loss, best_epoch


  
# sloc training


def map_train(responses, attribution, masks, epochs,lr, tv_eps, l1_eps, norm=False, score=1):
  # responses are all the probability responses from the models corresponding to each mask, [M]
  # masks are the masks used
  # attribution is a class
  # Masks come as [M,L,N] Where M is num of masks L is lead and N is number of samples


  optimiser = torch.optim.Adam(attribution.parameters(), lr)

  attribution.train()

  num_masks = masks.flatten(start_dim=1).sum(dim=1) * 2 #l1 norm of all masks *2?

  loss_hist = [0] * epochs


  tv = TotalVariationLoss()

  for epoch in range(epochs):


    masked_attr = attribution(masks)

    # maybe normalise attribution map wrt score.

    if norm:

      attr_map = normalise_attr(attribution.attr_map,score)

    else:
      attr_map = attribution.attr_map

    #comp_loss = (((responses-masked_attr)**2) / (attr_map.numel() * num_masks)).mean()

    comp_loss = F.mse_loss(masked_attr, responses)

    l1_norm = attr_map.abs().sum() # they do mean in the source code

    tv_loss = tv(attr_map)

    total_loss = comp_loss + tv_eps * tv_loss + l1_eps * l1_norm

    total_loss.backward()

    optimiser.step();

    optimiser.zero_grad();



    loss_hist[epoch] += total_loss.item()

  return loss_hist

def optimise_attribution(device, input, model, nmasks, batch_size, segsize, prob, epochs, lr, tv_eps, l1_eps, norm, initial_value=None, label=None):


  assert nmasks % batch_size == 0, "batch_size has to be a multiple of nmasks"

  inp_shape = input.shape

  attribution = AttributionMap(inp_shape[-2], inp_shape[-1], initial_value).to(device)

  model.eval()

  logits = torch.Tensor.cpu(model(input.unsqueeze(0).to(device)).detach_())

  if label is None:

    pred = np.argmax(logits.numpy())

    label = np.argmax(logits.numpy(),axis=1)


  score = F.softmax(logits, dim=1)[:,label]


  mr_generator = GenMaskResponse(segsize,label,device,prob,(inp_shape[-2], inp_shape[-1]))

  mr_generator.gen_mask_resp(model, input, nmasks, batch_size)

  masks = torch.stack(mr_generator.all_masks).flatten(0,1).to(device).detach() # shape [M, L, N], CPU
  responses = torch.stack(mr_generator.all_responses).flatten(0,1).squeeze().to(device).detach() # shape [M], CPU

  #print(masks.shape, responses.shape)

  loss_hist = map_train(responses, attribution, masks, epochs, lr, tv_eps, l1_eps, norm, score)

  return attribution, loss_hist
