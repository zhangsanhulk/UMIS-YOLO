
import torch
import torch.nn as nn
import torch.fft
class FourierBlock(nn.Module):
    def __init__(self, channels1, channels2):
        super(FourierBlock, self).__init__()
        

        self.W_L1 = None
        self.W_L2 = None
        

        self.magnitude_weight1 = nn.Parameter(torch.randn(channels1, 1, 1))
        self.magnitude_weight2 = nn.Parameter(torch.randn(channels2, 1, 1))
        self.phase_weight1 = nn.Parameter(torch.randn(channels1, 1, 1))
        self.phase_weight2 = nn.Parameter(torch.randn(channels2, 1, 1))

    def forward(self, f_G):
        x1, x2 = f_G


        batch_size, channels1, height, width = x1.shape
        batch_size, channels2, height, width = x2.shape


        if self.W_L1 is None or self.W_L1.shape[0] != channels1 or self.W_L1.shape[1] != height or self.W_L1.shape[2] != width:
            device = x1.device
            self.W_L1 = nn.Parameter(torch.randn(channels1, height, width, dtype=torch.cfloat, device=device))
            nn.init.xavier_uniform_(self.W_L1.real)
            nn.init.zeros_(self.W_L1.imag)


        if self.W_L2 is None or self.W_L2.shape[0] != channels2 or self.W_L2.shape[1] != height or self.W_L2.shape[2] != width:
            device = x2.device
            self.W_L2 = nn.Parameter(torch.randn(channels2, height, width, dtype=torch.cfloat, device=device))
            nn.init.xavier_uniform_(self.W_L2.real)
            nn.init.zeros_(self.W_L2.imag)


        f_SF1 = torch.fft.fft2(x1, dim=(-2, -1))
        f_FW1 = f_SF1 * self.W_L1
        f_FS1 = torch.fft.ifft2(f_FW1, dim=(-2, -1))
        x_FS1 = f_FS1.real + x1
        x_FS11 = torch.fft.fft2(x_FS1, dim=(-2,-1))


        f_SF2 = torch.fft.fft2(x2, dim=(-2, -1))
        f_FW2 = f_SF2 * self.W_L2
        f_FS2 = torch.fft.ifft2(f_FW2, dim=(-2, -1))
        x_FS2 = f_FS2.real + x2
        x_FS22 = torch.fft.fft2(x_FS2, dim=(-2, -1))


        magnitude1 = torch.abs(x_FS11)
        phase1 = torch.angle(x_FS11)
        magnitude2 = torch.abs(x_FS22)
        phase2 = torch.angle(x_FS22)


        fused_magnitude = (self.magnitude_weight1 * magnitude1) + (self.magnitude_weight2 * magnitude2)
        fused_phase = (self.phase_weight1 * phase1) + (self.phase_weight2 * phase2)


        fused_complex = fused_magnitude * torch.exp(1j * fused_phase)


        fused_x = torch.fft.ifft2(fused_complex, dim=(-2, -1)).real

        return fused_x