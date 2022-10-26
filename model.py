from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class Fourier:
    """
    Helper class storing Fourier mappings
    """

    @staticmethod
    def shift_fourier(x: torch.Tensor, n_dims: int) -> torch.tensor:
        """
        Applies fourier transform and shifts the zero-frequency component to the 
        center of the spectrum. Only the spatial dimensions get transformed.

        Args:
            x: image to transform.
            n_dims: number of spatial dimensions.
        Returns
            k: k-space data.
        """
        k: torch.tensor = torch.fft.fftshift(
            torch.fft.fftn(x, dim=tuple(range(-n_dims, 0))), 
            dim=tuple(range(-n_dims, 0)))
        return k
    
    @staticmethod
    def inv_shift_fourier(k: torch.Tensor, n_dims: int) -> torch.tensor:
        """
        Applies inverse shift and fourier transform. Only the spatial
        dimensions are transformed.

        Args:
            k: k-space data.
            n_dims: number of spatial dimensions. 
        Returns:
            x: tensor in image space.
        """
        x: torch.tensor = torch.fft.ifftn(
            torch.fft.ifftshift(k, dim=tuple(range(-n_dims, 0))), 
            dim=tuple(range(-n_dims, 0))).real
        return x
    

class GibbsNoiseLayer(nn.Module, Fourier):
    """
    The layer applies Gibbs noise to 2D/3D images. 

    Args:
        alpha: Parametrizes the intensity of the Gibbs noise filter applied. 
            Takes values in the interval [0,1] with alpha = 1.0 acting as the 
            identity mapping.                
    """

    def __init__(self, alpha: Optional[float] = None) -> None:

        nn.Module.__init__(self)

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        if alpha is None:
            self.alpha = torch.rand(1, requires_grad = True, device = self.device)
        else:
            alpha = min(max(alpha,0.), 1.)
            self.alpha = torch.tensor([alpha], requires_grad = True, 
                                        device = self.device)

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        n_dims = len(img.shape[1:])
        # FT
        k = self.shift_fourier(img, n_dims)
        # build and apply mask
        k = self._apply_mask(k)
        # map back
        img = self.inv_shift_fourier(k, n_dims)

        return img

    def _apply_mask(self, k: torch.Tensor) -> torch.Tensor:
        """Builds and applies a mask on the spatial dimensions.

        Args:
            k: k-space version of the image.
        Returns:
            masked version of the k-space image.
        """
        shape = k.shape[1:]
    
        center = (torch.tensor(shape, dtype=torch.float, device=self.device, requires_grad=True) - 1) / 2
        coords = torch.meshgrid(list(torch.linspace(0, i-1, i) for i in shape))
        # need to subtract center coord and then square for Euc distance
        dist_from_center = torch.sqrt(sum([(coord.to(self.device) - c)**2 for (coord, c) in zip(coords, center)]))
       
        alpha_norm = self.alpha * dist_from_center.max()
        norm_dist = dist_from_center / alpha_norm
        mask = norm_dist.where(norm_dist < 1, torch.zeros_like(alpha_norm))
        mask = mask.where(norm_dist > 1, torch.ones_like(alpha_norm))
        # add channel dimension into mask
        mask = torch.repeat_interleave(mask[None], k.size(0), 0)
        # apply binary mask
        k_masked = k * mask
        
        return k_masked
    
    
class Gibbs_UNet(nn.Module):
    """ResUnet with Gibbs layer"""

    def __init__(self, alpha = Optional[None]):
        super().__init__()

        self.gibbs = GibbsNoiseLayer(alpha)

        self.res_unet = UNet(
            dimensions = 3,
            in_channels = 1,
            out_channels = 1,
            channels = (16, 32, 64, 128, 256),
            strides = (2, 2, 2, 2),
            num_res_units = 2
            )

    def forward(self,img):
        img = self.gibbs(img)
        img = self.res_unet(img)
        return img