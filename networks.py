import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable



class res_block(nn.Module):
    def __init__(self, in_fmap, mid_fmap=32):
        super(res_block, self).__init__()
        self.conv = nn.Conv1d(in_fmap, mid_fmap, 5, stride=1, padding=2)
        self.transp_conv = nn.ConvTranspose1d(mid_fmap, in_fmap, 5, padding=2)
        self.recurrent = nn.GRU(mid_fmap, mid_fmap, num_layers=1, batch_first=True, dropout=0.2)

    def forward(self, inp):
        mid = self.conv(inp)
        mid, hidden = self.recurrent(mid.permute(0,2,1))
        mid = mid.permute(0,2,1)
        #out = self.transp_conv(mid)
        out = inp + self.transp_conv(mid)
        return out



class exp_net(nn.Module):
    def __init__(self, input_size, output_size, freq=513):
        super(exp_net, self).__init__()
        self.down_f0 = nn.Conv1d(freq, freq-1, 3, stride=1, padding=1)
        self.down_f1 = nn.Conv1d(freq-1, (freq-1)/2, 5, stride=1, padding=2)
        self.down_f2 = nn.Conv1d((freq-1)/2, (freq-1)/4, 5, stride=1, padding=2)
        self.down_f3 = nn.Conv1d((freq-1)/4, (freq-1)/8, 5, stride=1, padding=2)
        self.down_t0 = nn.Conv1d(freq-1, freq-1, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq-1)/2, (freq-1)/2, 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq-1)/4, (freq-1)/4, 5, stride=2, padding=2)

        self.activ = nn.LeakyReLU()

        self.inp_size = input_size
        self.out_size = output_size
        self.lstm = nn.GRU((freq-1)/4, (freq-1)/4, num_layers=1, batch_first=True, dropout=0.1)
        

    def forward(self, inp):
        # Encode
        C1 = self.activ(  self.down_f0(inp)   )
        C1 = self.activ(  self.down_t0(C1)  )
        C2 = self.activ(  self.down_f1(C1)  )
        C3 = self.activ(  self.down_t1(C2)  )
        C4 = self.activ(  self.down_f2(C3)  )

        C4, hidden = self.lstm(C4.permute(0,2,1))
        C4 = C4.permute(0,2,1)

        C5 = self.activ(  self.down_t2(C4)  )
        output = self.activ(  self.down_f3(C5)  )
        
        return output






class net_base(nn.Module):
    def __init__(self, input_size, output_size, freq=513):
        super(net_base, self).__init__()
        self.down_f0 = nn.Conv1d(freq, freq-1, 3, stride=1, padding=1)
        self.down_f1 = nn.Conv1d(freq-1, (freq-1)/2, 5, stride=1, padding=2)
        self.down_f2 = nn.Conv1d((freq-1)/2, (freq-1)/4, 5, stride=1, padding=2)
        self.down_f3 = nn.Conv1d((freq-1)/4, (freq-1)/8, 5, stride=1, padding=2)
        self.down_t0 = nn.Conv1d(freq-1, freq-1, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq-1)/2, (freq-1)/2, 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq-1)/4, (freq-1)/4, 5, stride=2, padding=2)

        self.up_f1 = nn.ConvTranspose1d((freq-1)/4, (freq-1)/4, 5, padding=2)
        self.up_f2 = nn.ConvTranspose1d((freq-1)/4, (freq-1)/2, 5, padding=2)
        self.up_f3 = nn.ConvTranspose1d((freq-1)/2, freq-1, 5, padding=2)
        self.up_f4 = nn.ConvTranspose1d(freq-1, freq, 5, padding=2)
        self.up_t1 = nn.ConvTranspose1d((freq-1)/4, (freq-1)/4, 3, stride=2, padding=1, output_padding=1)
        self.up_t2 = nn.ConvTranspose1d((freq-1)/2, (freq-1)/2, 3, stride=2, padding=1, output_padding=1)
        self.up_t3 = nn.ConvTranspose1d(freq-1, freq-1, 3, stride=2, padding=1, output_padding=1)

        self.activ = nn.LeakyReLU()

        self.inp_size = input_size
        self.out_size = output_size
        self.lstm_1 = nn.GRU((freq-1)/4, (freq-1)/4, num_layers=1, batch_first=True, dropout=0.3)
        self.lstm_2 = nn.GRU(freq-1, freq-1, num_layers=1, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.05)
        self.freq = freq



    def forward(self, inp, pitch_code):
        # Encode
        #print inp.shape, pitch_code.shape
        C1 = self.activ( self.down_f0(inp)  )
        C1 = self.activ( self.down_t0(C1)   )
        C2 = self.activ( self.down_f1(C1)   )
        C3 = self.activ( self.down_t1(C2)   )
        C4 = self.activ( self.down_f2(C3)   )

        C4, hidden = self.lstm_1(C4.permute(0,2,1))
        C4 = C4.permute(0,2,1)
        C5 = self.activ(  self.down_t2(C4)  )
        encoding1 = self.activ(  self.down_f3(C5)  )

        encoding = torch.cat((encoding1, pitch_code), 1)

        # Decode
        C6 = self.activ(  self.up_f1(encoding)  )
        C5 = self.activ(  self.up_t1(C6)    )
        C4 = self.activ(  self.up_f2(C5)  )
        C3 = self.activ(  self.up_t2(C4)  )
        C2 = self.activ(  self.up_f3(C3)  )
        output = self.activ(  self.up_t3(C2)  )

        #output = self.activ(  self.up_f4(output)   )

        output, hidden = self.lstm_2(output.permute(0,2,1))

        output = self.activ(  self.up_f4(output.permute(0,2,1))   )
        
        #return output
        return output, encoding1







class net_in_v2(nn.Module):
    def __init__(self, input_size, output_size, freq=513):
        super(net_in_v2, self).__init__()
        self.down_f0 = nn.Conv1d(freq, freq-1, 3, stride=1, padding=1)
        self.down_f1 = nn.Conv1d(freq-1, (freq-1)/2, 5, stride=1, padding=2)
        self.down_f2 = nn.Conv1d((freq-1)/2, (freq-1)/4, 5, stride=1, padding=2)
        self.down_f3 = nn.Conv1d((freq-1)/4, (freq-1)/8, 5, stride=1, padding=2)
        self.down_t0 = nn.Conv1d(freq-1, freq-1, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq-1)/2, (freq-1)/2, 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq-1)/4, (freq-1)/4, 5, stride=2, padding=2)

        self.up_f1 = nn.ConvTranspose1d((freq-1)/4, (freq-1)/4, 5, padding=2)
        self.up_f2 = nn.ConvTranspose1d((freq-1)/4, (freq-1)/2, 5, padding=2)
        self.up_f3 = nn.ConvTranspose1d((freq-1)/2, freq-1, 5, padding=2)
        self.up_f4 = nn.ConvTranspose1d(freq-1, freq, 5, padding=2)
        self.up_t1 = nn.ConvTranspose1d((freq-1)/4, (freq-1)/4, 3, stride=2, padding=1, output_padding=1)
        self.up_t2 = nn.ConvTranspose1d((freq-1)/2, (freq-1)/2, 3, stride=2, padding=1, output_padding=1)
        self.up_t3 = nn.ConvTranspose1d(freq-1, freq-1, 3, stride=2, padding=1, output_padding=1)

        self.activ = nn.LeakyReLU()

        self.inp_size = input_size
        self.out_size = output_size
        self.lstm_1 = nn.GRU((freq-1)/4, (freq-1)/4, num_layers=1, batch_first=True, dropout=0.3)
        self.lstm_2 = nn.GRU(freq-1, freq-1, num_layers=1, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.05)
        self.freq = freq

        self.norm_f = nn.InstanceNorm1d(freq, track_running_stats=False)
        self.norm_0 = nn.InstanceNorm1d(freq-1, track_running_stats=False)
        self.norm_1 = nn.InstanceNorm1d((freq-1)/2, track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq-1)/4, track_running_stats=False)
        self.norm_3 = nn.InstanceNorm1d((freq-1)/8, track_running_stats=False)

        self.res_block1 = res_block((freq-1)/1, 32)
        self.res_block2 = res_block((freq-1)/2, 32)
        self.res_block22 = res_block((freq-1)/2, 32)
        self.res_block3 = res_block((freq-1)/4, 32)
        self.res_block32 = res_block((freq-1)/4, 32)



    def forward(self, inp, pitch_code):
        # Encode
        #print inp.shape, pitch_code.shape
        C1 = self.activ( self.down_f0(inp)  )
        C1 = self.activ( self.down_t0(C1)  )
        C2 = self.activ( self.down_f1(C1)  )
        C3 = self.activ( self.down_t1(C2)  )
        C4 = self.norm_2( self.down_f2(C3) )

        C4, hidden = self.lstm_1(C4.permute(0,2,1))
        C4 = C4.permute(0,2,1)
        C5 = self.activ(  self.down_t2(C4)  )
        encoding1 = self.activ(  self.down_f3(C5)  )

        encoding = torch.cat((encoding1, pitch_code), 1)

        # Decode
        C6 = self.activ( self.up_f1(encoding)   )
        C5 = self.activ( self.up_t1(C6 + self.res_block3(C5))    )
        C4 = self.activ( self.up_f2(C5)   )
        C3 = self.activ( self.up_t2(C4 + self.res_block2(C3))  )
        C2 = self.activ( self.norm_0( self.up_f3(C3)  ) )
        output = self.norm_0( self.up_t3(C2 + self.res_block1(C1)) )

        #output = self.activ(  self.up_f4(output)   )

        output, hidden = self.lstm_2(output.permute(0,2,1))

        output = self.activ(  self.up_f4(output.permute(0,2,1))   )
        
        #return output
        return output, C2







