import torch
from torch.autograd import Variable
import torch.utils.data as Data
import cPickle as pickle
import multiprocessing
import matplotlib.pyplot as plt

'''
class OzoneNet(torch.nn.Module):
    def __init__(self, input_size, time_step):
        super(OzoneNet, self).__init__()
        self.dim = H
        self.lstm1 = torch.nn.LSTMCell(input_size, time_step).cuda()
        #self.hidden_lstm = torch.nn.LSTMCell(H, H)
        self.lstm2 = torch.nn.LSTMCell(time_step, 1).cuda()
        #self.lstm = torch.nn.LSTM(H, 2*H, 1).cuda()

    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), self.dim), requires_grad=False).cuda()
        c_t = Variable(torch.zeros(input.size(0), self.dim), requires_grad=False).cuda()
        h_t2 = Variable(torch.zeros(input.size(0), 1), requires_grad=False).cuda()
        c_t2 = Variable(torch.zeros(input.size(0), 1), requires_grad=False).cuda()

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            outputs += [c_t2]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(c_t2, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            outputs += [c_t2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

'''

class OzoneNet(torch.nn.Module):
    def __init__(self, input_size, time_step):
        super(OzoneNet, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size = input_size,
            hidden_size = 32,
            num_layers = 3,
            batch_first = True
        )
        self.out = torch.nn.Linear(32, 1)

    def forward(self, input):
        r_out, (h_n, h_c) = self.lstm(input, None)
        out = self.out(r_out[:, -1, :])
        return out        

if __name__ == '__main__':

    samples = pickle.load(open('samples', 'rb'))
    targets = pickle.load(open('targets', 'rb'))
    targets = [x[-1] for x in targets]

    INPUT_SIZE = 1
    TIME_STEP = len(samples[0])
    BATCH_SIZE = int(len(samples)*0.9)
    EPOCH = 1000
    FUTURE = 1
    LR = 0.5

    dtype = torch.cuda.FloatTensor

    dataset = Data.TensorDataset(
        data_tensor = torch.FloatTensor(samples).view(len(samples), TIME_STEP, INPUT_SIZE),
        target_tensor = torch.FloatTensor(targets).view(len(targets), 1, 1)
    )

    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = multiprocessing.cpu_count(),
        #sampler = Data.sampler.RandomSampler(dataset)
    )

    ozonenet = OzoneNet(INPUT_SIZE, TIME_STEP).cuda()
    
    print(ozonenet)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.LBFGS(ozonenet.parameters(), lr = LR)
    #optimizer = torch.optim.SGD(ozonenet.parameters(), lr = 0.5)

    plt.ion()
    plt.show()


    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            inputs = Variable(batch_x, requires_grad = False).cuda()
            targets = Variable(batch_y, requires_grad = False).cuda()

            if step == 0:
                '''
                # SGD optimizer
                optimizer.zero_grad()
                out = ozonenet(inputs)
                loss = []
                for fut in range(FUTURE):
                    temp = criterion(out[:,-(fut+1)], targets[:,-(fut+1)])
                    loss.append(temp)
                loss = sum(loss)
                loss.backward()
                optimizer.step()
                '''

                
                
                # LBFGS optimizer
                def closure():
                    optimizer.zero_grad()
                    out = ozonenet(inputs)
                    #loss = criterion(out[:,-1], target[:,-1])
                    #print('loss:', loss.data.cpu().numpy()[0])
                    loss = []
                    for fut in range(FUTURE):
                        temp = criterion(out[:,-(fut+1)], targets[:,-(fut+1)])
                        loss.append(temp)
                    loss = sum(loss)
                    loss.backward()
                    return loss

                optimizer.step(closure)
                

            else:
                pred = ozonenet(inputs, future = FUTURE)
                loss = []
                for fut in range(FUTURE):
                    temp = criterion(pred[:,-(fut+1)], targets[:,-(fut+1)])
                    loss.append(torch.sqrt(temp).cpu().data.numpy()[0])
                
                if epoch % 2 == 0:
                    plt.cla()
                    plt.scatter(pred[:,-1].cpu().data.numpy(), targets[:,-1].cpu().data.numpy())
                    plt.pause(0.1)

            

        print 'EPOCH   '+str(epoch) +'\t\t' + '\t'.join('%.7f'%x for x in loss)
    
    plt.ioff()
    plt.show()