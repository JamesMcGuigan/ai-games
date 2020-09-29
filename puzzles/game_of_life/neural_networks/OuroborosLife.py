import atexit
import gc
import time
from typing import List
from typing import Union

import numpy as np
import torch
import torch as pt
import torch.nn as nn

from neural_networks.FocalLoss import FocalLoss
from neural_networks.GameOfLifeBase import GameOfLifeBase
from utils.game import generate_random_board
from utils.game import life_step_3d


class OuroborosLife(GameOfLifeBase):
    """
    The idea of the Ouroboros Network is that rather than just predicting the next or previous state,
    we want to past, present and future simultaneously in the same network.

    The dataset is a sequence of 3 consecutive board states generated by life_step().

    The network takes the middle/present board state and attempts to predict all Past, Present and Future states

    The loss function computes the loss against the original training data, but also feeds back in upon itself.
    The output for Future is fed back in and it's Past is compared with the Present, likewise in reverse with the Paspt.
    """
    @property
    def filename(self) -> str:
        """ ./models/OuroborosLife3.pth || ./models/OuroborosLife5.pth """
        return super().filename.replace('.pth', f'{self.out_channels}.pth')


    def __init__(self, in_channels=1, out_channels=3):
        assert out_channels % 2 == 1, f'{self.__class__.__name__}(out_channels={out_channels}) must be odd'

        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels  # Past, Present and Future

        self.relu    = nn.LeakyReLU()     # combines with nn.init.kaiming_normal_()
        self.dropout = nn.Dropout(p=0.2)

        # 2**9 = 512 filters and kernel size of 3x3 to allow for full encoding of game rules
        # Pixels can see distance 5 neighbours, (hopefully) sufficient for delta=2 timesteps or out_channels=5
        # https://www.youtube.com/watch?v=H3g26EVADgY&feature=youtu.be&t=1h39m410s&ab_channel=JeremyHoward
        self.cnn_layers = nn.ModuleList([
            # Previous pixel state requires information from distance 2, so we need two 3x3 convolutions
            nn.Conv2d(in_channels=in_channels, out_channels=512,  kernel_size=(5,5), padding=2, padding_mode='circular'),
            nn.Conv2d(in_channels=512,   out_channels=256,  kernel_size=(1,1)),
            nn.Conv2d(in_channels=256,   out_channels=128,  kernel_size=(1,1)),

            nn.Conv2d(in_channels=1+128, out_channels=128,  kernel_size=(3,3), padding=1, padding_mode='circular'),
            nn.Conv2d(in_channels=128,   out_channels=512,  kernel_size=(1,1)),
            nn.Conv2d(in_channels=512,   out_channels=256,  kernel_size=(1,1)),
            nn.Conv2d(in_channels=256,   out_channels=128,  kernel_size=(1,1)),

            # # Deconvolution + Convolution allows neighbouring pixels to share information to simulate forward play
            # # This creates a 52x52 grid of interspersed cells that can then be downsampled back down to 25x25
            nn.ConvTranspose2d(in_channels=1+128, out_channels=512,  kernel_size=(3,3), stride=2, dilation=1),
            nn.Conv2d(in_channels=512,   out_channels=256,   kernel_size=(1,1)),
            nn.Conv2d(in_channels=256,   out_channels=64,    kernel_size=(1,1)),
            nn.Conv2d(in_channels=64,    out_channels=128,   kernel_size=(3,3), stride=2),  # undo deconvolution

            nn.Conv2d(in_channels=1+128, out_channels=64,    kernel_size=(1,1)),
            nn.Conv2d(in_channels=64,    out_channels=32,    kernel_size=(1,1)),
            nn.Conv2d(in_channels=32,    out_channels=16,    kernel_size=(1,1)),
            nn.Conv2d(in_channels=1+16,  out_channels=out_channels, kernel_size=(1,1)),
        ])
        self.batchnorm_layers = nn.ModuleList([
            nn.BatchNorm2d(cnn_layer.out_channels)
            for cnn_layer in self.cnn_layers
        ])


        # self.criterion = nn.BCELoss()
        self.criterion = FocalLoss()
        # self.criterion = nn.MSELoss()
        self.optimizer = pt.optim.RMSprop(self.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            max_lr=1e-3,
            base_lr=1e-5,
            step_size_up=10,
            mode='exp_range',
            gamma=0.8
        )

    # def load(self):
    #     super().load()
    #     self.apply(self.weights_init)


    def forward(self, x):
        x = input = self.cast_inputs(x)
        for n, (cnn_layer, batchnorm_layer) in enumerate(zip(self.cnn_layers, self.batchnorm_layers)):
            if cnn_layer.in_channels > 1 and cnn_layer.in_channels % 2 == 1:   # autodetect 1+in_channels == odd number
                x = torch.cat([ x, input ], dim=1)                     # passthrough original cell state
            x = cnn_layer(x)
            if n != len(self.cnn_layers)-1:
                x = self.relu(x)
                if n != 1:               # Don't apply dropout to the first layer
                    x = self.dropout(x)  # BatchNorm eliminates the need for Dropout in some cases cause BN provides similar regularization benefits as Dropout intuitively"
                x = batchnorm_layer(x)   # batchnorm goes after activation
            else:
                x = torch.sigmoid(x)  # output requires sigmoid activation
        return x


    # DOCS: https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca
    # pytorch requires:    contiguous_format = (batch_size, channels, height, width)
    # tensorflow requires: channels_last     = (batch_size, height, width, channels)
    def cast_inputs(self, x: Union[List[np.ndarray], np.ndarray, torch.Tensor]) -> torch.Tensor:
        x = self.cast_to_tensor(x)
        if   x.dim() == 4: pass
        elif x.dim() == 3 and x.shape[0] == self.out_channels:           # x.shape = (channels, height, width)
            x = x.view(1, self.in_channels, x.shape[1], x.shape[2])   # x.shape = (batch_size, channels, height, width)
        else:
            x = super().cast_inputs(x)
        return x


    def loss(self, outputs, timeline, inputs):
        dataset_loss    = self.loss_dataset(outputs, timeline, inputs)
        ouroboros_loss  = self.loss_ouroboros(outputs, timeline, inputs)
        arithmetic_mean = ( dataset_loss + ouroboros_loss ) / 2.0
        geometric_mean  = ( dataset_loss * ouroboros_loss ) ** (1/2)
        harmonic_mean   = 2.0 / ( 1.0/dataset_loss + 1.0/ouroboros_loss )
        return arithmetic_mean


    def loss_dataset(self, outputs, timeline, inputs):
        # dataset_loss = torch.mean(torch.mean(( (timeline-outputs)**2 ).flatten(1), dim=1))  # average MSE per timeframe

        # FocalLoss(outputs, target) need to be in correct order
        # dataset_loss = torch.sum(torch.tensor([
        #     self.criterion(outputs[b][t], timeline[b][t])
        #     for b in range(timeline.shape[0])
        #     for t in range(timeline.shape[1])
        # ], requires_grad=True))
        dataset_loss = self.criterion(outputs, timeline)
        return dataset_loss


    def loss_ouroboros(self, outputs, timeline, inputs):
        """
        Now we feed the output back into the input and compare second order losses
        t1=0 -> t2 = {2,3,4} | t1=1 -> t2 = {1,2,3,4} | t1=2 -> t2 = {0,1,2,3,4} | t1=3 -> t2 = {0,1,2,3} | t1=4 -> t2 = {0,1,2}
        outputs = [ Past2, Past1, Present, Future1, Future2 ]
        at t1=0: reinput == Past2   | reoutput = [ _,       _,       Past2,   Past1,   Present ]
        at t1=1: reinput == Past1   | reoutput = [ _,       Past2,   Past1,   Present, Future1 ]
        at t1=2: reinput == Present | reoutput = [ Past2,   Past1,   Present, Future1, Future2 ]
        at t1=3: reinput == Future1 | reoutput = [ Past1,   Present, Future1, Future2, _       ]
        at t1=4: reinput == Future2 | reoutput = [ Present, Future1, Future2, _,       _       ]
        """
        ouroboros_losses = []
        for t1 in range(self.out_channels):
            # reinput.shape = (batch_size, 1,            width, height)
            # reinput.shape = (batch_size, out_channels, width, height)
            reinput  = outputs[:,t1,:,:].unsqueeze(1)
            reoutput = self(reinput)
            t2_range = ( max(0, self.out_channels//2-t1), max(self.out_channels+1-t1, self.out_channels-1) )
            ouroboros_loss_per_timeline = []
            for b in range(timeline.shape[0]):
                for delta in range(-self.out_channels//2+1, self.out_channels//2+1):
                    t2 = t1 + delta
                    if not ( t2_range[0] <= t2 <= t2_range[1] ): continue  # invalid index
                    ouroboros_loss_per_timeline.append(
                        # pt.mean( (timeline[b][t1] - reoutput[b][t2])**2 )
                        self.criterion(reoutput[b][t2], timeline[b][t1])
                    )
            ouroboros_losses += ouroboros_loss_per_timeline

        ouroboros_loss = pt.mean(pt.tensor(ouroboros_losses)).requires_grad_(True)
        return ouroboros_loss


    def accuracy(self, outputs, timeline, inputs) -> float:
        """
        Count the number of 100% accuracy boards predicted
        accuracy == 0.666 + output_channels == 3 means:
            Present and Future boards have been correctly predicted, but Past is still not fully solved
        """
        # noinspection PyTypeChecker
        accuracies = pt.tensor([
            pt.all( self.cast_bool(outputs[b][t]) == self.cast_bool(timeline[b][t]) )                      # percentage boards correct
            # pt.mean(( self.cast_bool(outputs[b][t]) == self.cast_bool(timeline[b][t]) ).to(pt.float32))  # percentage pixels correct
            for b in range(timeline.shape[0])
            for t in range(timeline.shape[1])
        ])
        accuracy = pt.mean(accuracies.to(pt.float))
        return accuracy.detach().item()


    def fit(self, epochs=100_000, batch_size=25, max_delta=25, timeout=0):
        gc.collect()
        torch.cuda.empty_cache()
        atexit.register(model.save)
        self.train()
        self.unfreeze()
        print(self)
        try:
            # timelines_batch = np.array([
            #     life_step_3d(generate_random_board(), max_delta)
            #     for _ in range(batch_size)
            # ])
            time_start  = time.perf_counter()
            board_count = 0
            dataset_accuracies = [0]
            for epoch in range(1, epochs+1):
                if np.min(dataset_accuracies[-10:]) == 1.0: break  # we have reached 100% accuracy
                if timeout and timeout < time.perf_counter() - time_start: break

                epoch_start = time.perf_counter()
                timelines_batch = np.array([
                    life_step_3d(generate_random_board(), max_delta)
                    for _ in range(batch_size)
                ])
                epoch_losses     = []
                dataset_losses   = []
                ouroboros_losses = [0]
                epoch_accuracies = []
                d = self.out_channels // 2  # In theory this should work for 5 or 7 channels
                for t in range(d, max_delta - d):
                    inputs_np   = timelines_batch[:, np.newaxis, t,:,:]  # (batch_size=10, channels=1,  width=25, height=25)
                    timeline_np = timelines_batch[:, t-d:t+d+1,    :,:]  # (batch_size=10, channels=10, width=25, height=25)
                    inputs   = pt.tensor(inputs_np).to(self.device).to(pt.float32)
                    timeline = pt.tensor(timeline_np).to(self.device).to(pt.float32)

                    self.optimizer.zero_grad()
                    outputs         = self(inputs)
                    accuracy        = model.accuracy(outputs, timeline, inputs)  # torch.sum( outputs.to(torch.cast_bool) == expected.to(torch.cast_bool) ).cpu().numpy() / np.prod(outputs.shape)
                    dataset_loss    = self.loss_dataset(outputs, timeline, inputs)
                    ouroboros_loss  = self.loss_ouroboros(outputs, timeline, inputs)
                    loss            = ouroboros_loss + dataset_loss / (epoch ** 0.5)  # fade out dataset_loss heuristic
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    board_count += batch_size
                    epoch_losses.append(loss.detach().item())
                    dataset_losses.append(dataset_loss.detach().item())
                    ouroboros_losses.append(ouroboros_loss.detach().item())
                    epoch_accuracies.append(accuracy)
                    torch.cuda.empty_cache()
                dataset_accuracies.append( min(epoch_accuracies) )

                epoch_time = time.perf_counter() - epoch_start
                time_taken = time.perf_counter() - time_start
                print(f'epoch: {epoch:4d} | boards: {board_count:5d} | loss: {np.mean(epoch_losses):.6f} | ouroboros: {np.mean(ouroboros_losses):.6f} | dataset: {np.mean(dataset_losses):.6f} | accuracy = {np.mean(epoch_accuracies):.6f} | time: {1000*epoch_time//batch_size}ms/board | {time_taken//60:.0f}:{time_taken%60:02.0f}')
        except KeyboardInterrupt: pass
        finally:
            model.save()
            atexit.unregister(model.save)
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == '__main__':
    # PYTHONUNBUFFERED=1 time python3 ./neural_networks/GameOfLifeReverseOneStep.py | tee ./neural_networks/models/GameOfLifeReverseOneStep.log
    model = OuroborosLife()
    model.fit()
    # train(model, reverse_input_output=True)
