import torch


def _criterioning(pred, truth, criterion):
    """Handle criterion ideosyncracies."""
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        truth = (
            truth.squeeze() if len(truth.shape) == len(pred.shape) else truth
        )
        return criterion(
            pred,
            truth.long().to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ),
        )
    if isinstance(
        criterion,
        (
            torch.nn.modules.loss.BCEWithLogitsLoss,
            torch.nn.MSELoss,
            torch.nn.L1Loss,
        ),
    ):
        return criterion(
            pred,
            truth.float().to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ),
        )


def recon_weighted_sum(modal_loss_funcs, weights):
    """Create wrapper function that computes the weighted model reconstruction loss."""

    def _actualfunc(recons, origs):
        totalloss = 0.0
        for i in range(len(recons)):
            trg = (
                origs[i].view(recons[i].shape[0], recons[i].shape[1])
                if len(recons[i].shape) != len(origs[i].shape)
                else origs[i]
            )
            totalloss += modal_loss_funcs[i](recons[i], trg) * weights[i]
        return torch.mean(totalloss)

    return _actualfunc


def MFM_objective(
    ce_weight,
    modal_loss_funcs,
    recon_weights,
    criterion=torch.nn.CrossEntropyLoss(),
):
    """Define objective for MFM.

    :param ce_weight: the weight of simple supervised loss
    :param model_loss_funcs: list of functions that takes in reconstruction and input of each modality and compute reconstruction loss
    :param recon_weights: list of float values indicating the weight of reconstruction loss of each modality
    :param criterion: the loss function for supervised loss (default CrossEntropyLoss)
    """
    recon_loss_func = recon_weighted_sum(modal_loss_funcs, recon_weights)

    def _actualfunc(pred, truth, args):
        ints = args["intermediates"]
        reps = args["reps"]
        fused = args["fused"]
        decoders = args["decoders"]
        inps = args["inputs"]
        recons = []
        for i in range(len(reps)):
            recons.append(
                decoders[i](torch.cat([ints[i](reps[i]), fused], dim=1))
            )
        ce_loss = _criterioning(pred, truth, criterion)
        inputs = [
            i.float().to(
                torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )
            )
            for i in inps
        ]
        recon_loss = recon_loss_func(recons, inputs)
        return ce_loss * ce_weight + recon_loss

    return _actualfunc

class NeuroSymbolicLoss(torch.nn.Module):
    """
    Neuro-Symbolic Regularized Loss.
    Combines the primary task loss (e.g., L1, MSE) with a soft-logic penalty
    derived from the Knowledge Graph node embedding.
    """
    def __init__(self, task_criterion=torch.nn.L1Loss(), lambda_logic=0.1, kg_dim=64):
        super(NeuroSymbolicLoss, self).__init__()
        self.task_criterion = task_criterion
        self.lambda_logic = lambda_logic
        
        # Projection layer to map the contextualized KG node (h_kg) to a logical prior [-1, 1]
        self.kg_to_prior = torch.nn.Sequential(
            torch.nn.Linear(kg_dim, 1),
            torch.nn.Tanh()
        )

    def forward(self, pred, truth, features):
        """
        pred: (B, 1) model prediction
        truth: (B, 1) ground truth labels
        features: dictionary containing 'h_kg' of shape (B, T, d)
        """
        # 1. Calculate standard task loss
        l_task = self.task_criterion(pred.squeeze(), truth.squeeze())
        
        # 2. Calculate logical prior from KG node
        # Average pool over temporal dimension: (B, T, d) -> (B, d)
        h_kg = features['h_kg'].mean(dim=1)
        
        # Project to logic space [-1, 1] representing positive/negative sentiment prior
        logic_prior = self.kg_to_prior(h_kg) # (B, 1)
        
        # 3. Soft-Logic Penalty (Hinge Loss)
        # We want the prediction to align with the logic_prior.
        # Penalty is high if prediction and logic prior have opposite signs and high magnitude.
        # Preds are typically unbounded or in [-3, 3] for sentiment.
        # We penalize: max(0, -pred * logic_prior)
        logic_penalty = torch.relu(-pred * logic_prior).mean()
        
        # Total Loss
        loss = l_task + (self.lambda_logic * logic_penalty)
        return loss, l_task, logic_penalty
