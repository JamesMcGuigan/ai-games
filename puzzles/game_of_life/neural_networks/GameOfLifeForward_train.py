import torch
import torch.nn as nn
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.engine import Events
from ignite.metrics import Accuracy
from ignite.metrics import Loss

from neural_networks.GameOfLifeForward import gameOfLifeForward

model = gameOfLifeForward()
train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
optimizer    = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion    = nn.NLLLoss()
log_interval = 10
trainer      = create_supervised_trainer(model, optimizer, criterion)

val_metrics = {
    "accuracy": Accuracy(),
    "nll":      Loss(criterion)
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(trainer):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))

trainer.run(train_loader, max_epochs=100)
