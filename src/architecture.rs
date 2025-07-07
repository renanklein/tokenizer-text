use std::f64::consts::PI;

use tch::{nn::{self, VarStore}, Device, Kind, Tensor};

pub struct LayerNorm {
    eps: f64,
    shift: Tensor,
    scale: Tensor,
}

pub struct GELU;

impl LayerNorm {
    pub fn new(emb_dim: i64) -> Self {
        let vs = VarStore::new(Device::cuda_if_available());

        let root = vs.root();

        let eps = 1e-5;
        let scale = root.ones("scale", &[emb_dim]);
        let shift = root.zeros("shift", &[emb_dim]);

        LayerNorm { eps, shift, scale }
    }

    pub fn forward(&self, x: Tensor) -> Tensor {
        let mean = x.mean_dim(Some(&[-1i64][..]), true, Kind::Float);
        let var = x.var_dim(Some(&[-1i64][..]), false, true);

        let norm_x = (x - mean)/Tensor::sqrt(&(self.eps + var));

        &self.scale * norm_x + &self.shift
    }
}

impl GELU {
    pub fn new() -> Self {
        GELU
    }

    pub fn foward(&self, x: Tensor) -> Tensor{
        let part = Tensor::from(2.0 / PI);

        0.5 * &x * (1.0 + Tensor::tanh(&(&part * &(&x + 0.044715 + x.pow_tensor_scalar(3.0)))))
    }
}
