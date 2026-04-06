import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class MaskedDataset(Dataset):
  def __init__(self, data):
    self.data = data;

  def __getitem__(self, index):
    x = self.data[index]
    return x

  def __len__(self):
    return len(self.data)

class AttributionMap(nn.Module):
  def __init__(self, L=12, N=500, initial_value=None):
    super().__init__();

    if initial_value is not None:
      self.attr_map = nn.Parameter(initial_value)

    else:
      self.attr_map = nn.Parameter(torch.zeros(L,N))

  def forward(self, x):
    y = (x * self.attr_map).flatten(start_dim=1).sum(dim=1) # dot product as torch.dot only supports 1d tensors. dim=1 for leaving batches alone. batches in this case being the number of masks !

    return y


class TotalVariationLoss(nn.Module):
    def __init__(self, beta=2):
        super().__init__()
        self.beta = beta

    def forward(self, x):

        # Calculate the variation in the x and y directions
        x_diff = torch.abs(x[ :, :-1] - x[ :, 1:]).pow(self.beta)
        y_diff = torch.abs(x[ :-1, :] - x[ 1:, :]).pow(self.beta)

        # Sum the variations to get the total variation loss
        tv_loss = torch.mean(x_diff) + torch.mean(y_diff)
        return tv_loss


class GenMaskResponse():
  def __init__(self, segsize, cls, device, prob=0.7, shape=(12,500)):
    self.segsize= segsize
    self.prob = prob
    self.shape = shape
    self.leads = shape[0]
    self.samples = shape[1]
    self.device= device

    self.all_masks = []
    self.all_responses = []

    self.cls = cls


    self.softmax = nn.Softmax(dim=1)


  def gen_masks(self, nmasks):

    n_segments = self.samples / self.segsize
    masks = torch.ones((nmasks,self.leads, self.samples), dtype=torch.bool)


    for j in range(nmasks):

      mask = (torch.rand((12,int(n_segments))) > self.prob)

      for i in range(int(n_segments)):
        seg_end = (i+1)*self.segsize+(j%500) # offset masks so its not the same segment locations, implemented wrap around also.
        if seg_end > self.samples:
          wseg_end = seg_end-self.samples
          masks[j,mask[:,i],max(wseg_end-self.segsize,0):wseg_end] = False

        masks[j,mask[:,i],seg_end-self.segsize:min(seg_end,self.samples)] = False # takes vertical slice of masks from t_mask to determine which leads have segments that need to be
        """
        # testing non wrap around masks
        seg_end = (i+1)*self.segsize
        masks[j, mask[:,i], i*self.segsize:seg_end] = False
        """

    return masks



  def gen_mask_resp(self, model, input, nmasks, batch_size):


    itr = nmasks//batch_size

    remainder = nmasks % batch_size

    model.eval()
    with torch.no_grad():
      for i in range(itr):
        masks = self.gen_masks(batch_size)


        masked_inputs = input.unsqueeze(0) * masks



        logits = model(masked_inputs.to(self.device))

        response = self.softmax(logits)[:,self.cls]

        # scale response test to get more obvious variation in results
        #response = response*10

        self.all_masks.append(masks)
        self.all_responses.append(response)

        del logits, masks, response, masked_inputs



      if remainder:
        masks = self.gen_masks(remainder)

        masked_inputs = input.unsqueeze(0) * masks

        logits = model(masked_inputs.to(self.device))

        response = self.softmax(logits)[:,self.cls]


        self.all_masks.append(masks)
        self.all_responses.append(response)

        del logits, masks, response, masked_inputs

      #print(len(self.all_masks))

    """
    masks = self.gen_masks(nmasks)

    masked_inputs = input.unsqueeze(0) * masks

    masked_dataset = MaskedDataset(masked_inputs)

    masked_input_loader = DataLoader(masked_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    model.eval()
    with torch.no_grad():
      for x_batch in masked_input_loader:
        logits = model(x_batch.to(device))
        responses = self.softmax(logits)[:,self.cls]
        self.all_responses.append(responses.flatten())

        del logits, responses, x_batch


    self.all_masks.append(masks)"""

    torch.cuda.empty_cache()



